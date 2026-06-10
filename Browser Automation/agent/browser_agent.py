# agent/browser_agent.py
"""
Core Browser Agent.
Loop: Observe (accessibility tree) → Think (LLM) → Act (Playwright) → Repeat

Anti-bot measures applied:
- playwright-stealth patches fingerprints
- Random human-like delays between actions
- Slow typing instead of instant fill
- DuckDuckGo as default (no bot detection)
"""

import asyncio
import random
from playwright.async_api import async_playwright, Page, Browser

from agent.llm import HFAgent
from utils.accessibility import get_accessibility_tree, get_page_meta
from utils.parser import parse_action, BrowserAction
from config.settings import MAX_STEPS, HEADLESS
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SYSTEM_PROMPT = """You are a browser automation agent.
You are given:
- A TASK to complete
- The current PAGE URL and TITLE
- An ACCESSIBILITY TREE describing interactive elements on the page

Your job is to decide the NEXT SINGLE ACTION to take.

Always reply in EXACTLY this format (no extra text):
ACTION: <click|fill|goto|scroll|wait|done>
SELECTOR: <CSS selector or accessible name>
VALUE: <URL for goto, text for fill, pixels for scroll — omit if not needed>
REASON: <one short sentence explaining why>

Rules:
- Use ACTION: done when the task is complete.
- Use ACTION: goto with VALUE: <url> to navigate.
- Use ACTION: fill to type into inputs; always include VALUE.
- Use ACTION: click for buttons and links.
- Use ACTION: wait if the page is still loading.
- Use ACTION: scroll with VALUE: <pixels> to scroll down.
- Prefer accessible names (the text of a button/link) over brittle CSS selectors.
- Never repeat the same failed action twice — try something different.
"""
class BrowserAgent:
    def __init__(self):
        self.llm = HFAgent()
        self.history: list[dict] = []
    async def run(self, task: str, start_url: str = "https://www.duckduckgo.com"):
        console.print(Panel(f"[bold yellow]Task:[/] {task}", title="Browser Agent"))
        async with async_playwright() as p:
            browser: Browser = await p.chromium.launch(
                headless=HEADLESS,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-infobars",
                    "--disable-dev-shm-usage",
                ],
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            page: Page = await context.new_page()
            # ── Apply stealth patches ─────────────────────────────────────
            await self._apply_stealth(page)
            await page.goto(start_url)
            await asyncio.sleep(random.uniform(1.0, 2.0))
            for step in range(1, MAX_STEPS + 1):
                console.rule(f"[bold blue]Step {step}/{MAX_STEPS}")
                # ── 1. OBSERVE ───────────────────────────────────────────
                meta = await get_page_meta(page)
                tree = await get_accessibility_tree(page)
                console.print(f"[dim]URL:[/] {meta['url']}")
                console.print(f"[dim]Title:[/] {meta['title']}")
                # ── 2. THINK ─────────────────────────────────────────────
                user_prompt = f"""TASK: {task}

CURRENT URL: {meta['url']}
PAGE TITLE: {meta['title']}

ACCESSIBILITY TREE:
{tree}

What is the next action?"""

                console.print("[bold magenta]Asking LLM...[/]")
                llm_response = self.llm.chat(SYSTEM_PROMPT, user_prompt)
                console.print(Panel(llm_response, title="LLM Response", style="magenta"))
                # ── 3. PARSE ─────────────────────────────────────────────
                action = parse_action(llm_response)
                self.history.append({
                    "step":   step,
                    "url":    meta["url"],
                    "action": action,
                })
                if action.action_type == "error":
                    console.print("[red]Could not parse LLM output. Skipping step.[/]")
                    continue
                if action.action_type == "done":
                    console.print("[bold green]Task complete![/]")
                    break
                # ── 4. ACT ───────────────────────────────────────────────
                success = await self._execute(page, action)
                if not success:
                    console.print("[yellow]Action failed. LLM will try again next step.[/]")
                # Human-like random delay between actions
                await asyncio.sleep(random.uniform(1.5, 3.5))
            else:
                console.print(f"[red]Reached max steps ({MAX_STEPS}) without completing task.[/]")
            self._print_summary()
            await browser.close()
    async def _apply_stealth(self, page: Page):
        """Patch browser properties that reveal automation."""
        try:
            from playwright_stealth import stealth_async
            await stealth_async(page)
            console.print("[green]Stealth mode active (playwright-stealth)[/]")
            return
        except ImportError:
            pass
        # Manual stealth patches as fallback
        await page.add_init_script("""
            // Remove webdriver flag
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // Fake plugins list
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // Fake languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });

            // Fake chrome object
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };

            // Fix permissions query
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters)
            );
        """)
        console.print("[yellow]Stealth mode active (manual patches)[/]")
    async def _execute(self, page: Page, action: BrowserAction) -> bool:
        """Execute a parsed action. Returns True on success."""
        try:
            atype = action.action_type
            if atype == "goto":
                url = action.value or action.selector or ""
                if not url.startswith("http"):
                    url = "https://" + url
                console.print(f"[cyan]→ goto:[/] {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            elif atype == "click":
                console.print(f"[cyan]→ click:[/] {action.selector}")
                await asyncio.sleep(random.uniform(0.3, 0.8))  # pause before click
                try:
                    await page.get_by_role("button", name=action.selector).click(timeout=5000)
                except Exception:
                    try:
                        await page.get_by_text(action.selector).click(timeout=5000)
                    except Exception:
                        await page.click(action.selector, timeout=5000)
            elif atype == "fill":
                console.print(f"[cyan]→ fill:[/] {action.selector}  =  \"{action.value}\"")
                await asyncio.sleep(random.uniform(0.3, 0.7))
                # Try to find the input
                input_el = None
                try:
                    input_el = page.get_by_label(action.selector)
                    await input_el.wait_for(timeout=3000)
                except Exception:
                    try:
                        input_el = page.get_by_placeholder(action.selector)
                        await input_el.wait_for(timeout=3000)
                    except Exception:
                        input_el = None
                # Slow human-like typing
                if input_el:
                    await input_el.click()
                    await asyncio.sleep(random.uniform(0.2, 0.5))
                    await input_el.type(action.value or "", delay=random.randint(40, 120))
                else:
                    await page.click(action.selector, timeout=5000)
                    await asyncio.sleep(random.uniform(0.2, 0.4))
                    await page.type(action.selector, action.value or "", delay=random.randint(40, 120))
                await asyncio.sleep(random.uniform(0.3, 0.7))
                await page.keyboard.press("Enter")
            elif atype == "scroll":
                pixels = int(action.value or 500)
                console.print(f"[cyan]→ scroll:[/] {pixels}px")
                await page.evaluate(f"window.scrollBy(0, {pixels})")
            elif atype == "wait":
                wait_time = random.uniform(2.0, 3.5)
                console.print(f"[cyan]→ wait:[/] {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            return True
        except Exception as e:
            console.print(f"[red]Action error:[/] {e}")
            return False
    def _print_summary(self):
        table = Table(title="Agent History", show_lines=True)
        table.add_column("Step", style="bold")
        table.add_column("URL")
        table.add_column("Action")
        table.add_column("Selector")
        table.add_column("Value")
        for entry in self.history:
            a = entry["action"]
            table.add_row(
                str(entry["step"]),
                entry["url"][:50],
                a.action_type,
                a.selector or "",
                a.value or "",
            )
        console.print(table)
