import os
import time
import glob
import socket
import subprocess
import requests as _requests

from openai import OpenAI
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ── API Client ─────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

NIM_MODEL = "meta/llama-3.1-8b-instruct"

# ── Chrome profile config ──────────────────────────────────────────────────────
CHROME_USER_DATA = r"C:\Users\faiza\AppData\Local\Google\Chrome\User Data"
PROFILE_NAME     = "Profile 23"
DEBUG_PORT       = 9222

# ── Prompt ─────────────────────────────────────────────────────────────────────
BROWSER_AGENT_PROMPT = """
You are a web automation agent. You are given the current HTML structure of a webpage and a goal to achieve.
Your job is to analyze the HTML and decide the next single action to take to progress toward the goal.

You will receive:
- GOAL: what needs to be achieved
- CURRENT URL: the current page
- HTML: the simplified interactive elements of the page

You must respond EXACTLY in this format and nothing else:
ACTION: action_name
TARGET: css_selector
VALUE: value (or None)

Available actions:
1. click       — click a button, link, or element
2. type        — type text into an input field
3. select      — select an option from a dropdown
4. scroll      — scroll down the page
5. navigate    — go to a URL directly (use TARGET as the URL, VALUE as None)
6. wait        — wait for page to load
7. done        — goal has been achieved

Rules:
- Only one action per response
- Use precise CSS selectors (id > class > tag)
- The browser is already logged in — NEVER try to log in, type an email, or type a password
- If you see a login page, respond: ACTION: wait / TARGET: None / VALUE: None
- If goal is already achieved, respond ACTION: done
- Never repeat the same action twice in a row
- If a selector does not work, try a different one
- Never navigate away from the intended website domain
"""

# ── HTML cleaning ──────────────────────────────────────────────────────────────

def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "meta", "noscript", "svg", "img"]):
        tag.decompose()
    interactive = soup.find_all([
        "input", "button", "a", "select",
        "form", "textarea", "label", "nav",
        "h1", "h2", "h3", "li"
    ])
    cleaned = []
    for tag in interactive:
        attrs_to_keep = ["id", "class", "name", "type", "href",
                         "placeholder", "value", "action"]
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in attrs_to_keep}
        cleaned.append(str(tag))
    return "\n".join(cleaned)[:4000]


# ── LLM interaction ────────────────────────────────────────────────────────────

def get_next_action(goal: str, current_url: str, html: str, history: list) -> str:
    messages = [
        {"role": "system", "content": BROWSER_AGENT_PROMPT},
        *history,
        {"role": "user", "content": f"GOAL: {goal}\nCURRENT URL: {current_url}\nHTML:\n{html}"}
    ]
    response = client.chat.completions.create(
        model=NIM_MODEL,
        messages=messages,
        max_tokens=200,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def parse_action(response: str) -> tuple:
    try:
        lines  = [l.strip() for l in response.strip().split("\n") if l.strip()]
        action = lines[0].split("ACTION:")[-1].strip().lower()
        target = lines[1].split("TARGET:")[-1].strip()
        value  = lines[2].split("VALUE:")[-1].strip()
        value  = None if value.lower() == "none" else value
        return action, target, value
    except Exception as e:
        print(f"Parse error: {e} | Response: {response}")
        return "wait", None, None


# ── Browser control ────────────────────────────────────────────────────────────

def execute_action(page, action: str, target: str, value: str) -> bool:
    try:
        if action == "click":
            page.click(target, timeout=5000)
        elif action == "type":
            page.focus(target)
            page.type(target, value, delay=80)
        elif action == "select":
            page.select_option(target, value, timeout=5000)
        elif action == "scroll":
            page.evaluate("window.scrollBy(0, 500)")
        elif action == "navigate":
            page.goto(target, wait_until="domcontentloaded", timeout=15000)
        elif action == "wait":
            time.sleep(2)
        elif action == "done":
            return True
        time.sleep(1.5)
        return True
    except Exception as e:
        print(f"Action failed: {e}")
        return False


def is_port_free(port: int) -> bool:
    """Check whether a TCP port is available on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def kill_chrome():
    """Kill all Chrome processes and wait until none remain."""
    print("[Browser] Killing existing Chrome instances...")
    for _ in range(3):
        subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], capture_output=True)
        time.sleep(1)

    # Wait until no chrome.exe process remains
    print("[Browser] Waiting for Chrome processes to exit...")
    for _ in range(20):
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
            capture_output=True, text=True
        )
        if "chrome.exe" not in result.stdout:
            print("[Browser] All Chrome processes exited.")
            break
        time.sleep(0.5)
    else:
        print("[Browser] WARNING: Some Chrome processes may still be running.")


def clear_profile_locks():
    """Remove stale Chrome profile lock files that block reuse."""
    patterns = [
        os.path.join(CHROME_USER_DATA, PROFILE_NAME, "*.lock"),
        os.path.join(CHROME_USER_DATA, PROFILE_NAME, "SingletonLock"),
        os.path.join(CHROME_USER_DATA, PROFILE_NAME, "SingletonCookie"),
        os.path.join(CHROME_USER_DATA, "SingletonLock"),
        os.path.join(CHROME_USER_DATA, "SingletonCookie"),
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                print(f"[Browser] Removed lock file: {f}")
            except Exception as e:
                print(f"[Browser] Could not remove {f}: {e}")


def wait_for_debug_port(timeout: int = 30):
    """Poll until Chrome's remote-debug port responds or timeout is reached."""
    print(f"[Browser] Waiting for Chrome debug port {DEBUG_PORT}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _requests.get(
                f"http://127.0.0.1:{DEBUG_PORT}/json/version", timeout=1
            )
            if r.status_code == 200:
                elapsed = timeout - (deadline - time.time())
                print(f"[Browser] Chrome ready after {elapsed:.1f}s")
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(
        f"Chrome did not expose its debug port on :{DEBUG_PORT} within {timeout}s"
    )


def find_chrome_exe() -> str:
    candidates = [
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    exe = next((p for p in candidates if os.path.exists(p)), None)
    if not exe:
        raise FileNotFoundError(
            "chrome.exe not found in any standard location. "
            "Set the path manually in find_chrome_exe()."
        )
    return exe


def launch_browser(playwright):
    """
    Kill Chrome, clear locks, start a fresh instance with remote debugging,
    wait for the debug port, then connect Playwright over CDP.
    """
    kill_chrome()
    clear_profile_locks()

    # Make sure the port is actually free before we launch
    if not is_port_free(DEBUG_PORT):
        print(f"[Browser] Port {DEBUG_PORT} still in use — waiting up to 10s...")
        for _ in range(20):
            if is_port_free(DEBUG_PORT):
                break
            time.sleep(0.5)
        else:
            raise OSError(
                f"Port {DEBUG_PORT} is still occupied after waiting. "
                "Kill the process holding it manually."
            )

    chrome_exe = find_chrome_exe()
    print(f"[Browser] Launching Chrome: {chrome_exe}")
    print(f"[Browser] Profile: {PROFILE_NAME}")

    proc = subprocess.Popen([
        chrome_exe,
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={CHROME_USER_DATA}",
        f"--profile-directory={PROFILE_NAME}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-background-networking",
    ])
    print(f"[Browser] Chrome PID: {proc.pid}")

    wait_for_debug_port(timeout=30)

    browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{DEBUG_PORT}")
    context = browser.contexts[0] if browser.contexts else browser.new_context()
    print(f"[Browser] Connected. Open contexts: {len(browser.contexts)}")
    return browser, context


# ── Main agent loop ────────────────────────────────────────────────────────────

def run_browser_agent(goal: str, start_url: str, max_steps: int = 20):
    print(f"\nGoal: {goal}")
    print(f"Starting at: {start_url or 'https://www.google.com'}")
    print("=" * 50)

    history      = []
    last_actions = []

    with sync_playwright() as p:
        browser, context = launch_browser(p)

        url = start_url.strip() if start_url.strip() else "https://www.google.com"

        # Reuse existing page or open a new one
        if context.pages:
            page = context.pages[0]
            print(f"[Browser] Reusing existing page: {page.url}")
        else:
            page = context.new_page()

        print(f"[Browser] Navigating to {url} ...")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            print(f"[Browser] goto warning (continuing): {e}")

        try:
            page.wait_for_selector("body", state="visible", timeout=15000)
        except Exception:
            pass

        print(f"[Browser] Page ready. URL: {page.url}")
        time.sleep(1)

        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            current_url = page.url

            if "accounts.google.com" in current_url and "signin" in current_url:
                print("[WARNING] Redirected to Google login — cookies may be expired.")
                break

            try:
                raw_html = page.content()
            except Exception as e:
                print(f"Page closed or crashed: {e}")
                break

            clean = clean_html(raw_html)
            print(f"URL: {current_url}")

            response              = get_next_action(goal, current_url, clean, history)
            action, target, value = parse_action(response)
            print(f"Action: {action} | Target: {target} | Value: {value}")

            last_actions.append(f"{action}:{target}")
            if len(last_actions) > 3:
                last_actions.pop(0)
            if len(last_actions) == 3 and len(set(last_actions)) == 1:
                print("Loop detected — same action 3 times in a row. Stopping.")
                break

            history.append({"role": "assistant", "content": response})

            if action == "done":
                print("\n" + "=" * 50)
                print("Goal achieved!")
                print("=" * 50)
                break

            success = execute_action(page, action, target, value)
            if not success:
                history.append({
                    "role": "user",
                    "content": (
                        f"The action failed: {action} on '{target}'. "
                        "Try a different selector or approach."
                    )
                })

        else:
            print("\nMax steps reached — goal not achieved.")

        input("\nPress Enter to close browser...")
        browser.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    goal      = input("Enter your goal: ")
    start_url = input("Enter starting URL (leave blank for Google): ")
    run_browser_agent(goal, start_url)