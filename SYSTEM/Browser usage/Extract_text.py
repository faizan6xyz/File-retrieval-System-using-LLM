import json
import os
import re
import time
import glob
import requests
import threading
from openai import OpenAI


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)
NIM_MODEL = "meta/llama-3.1-8b-instruct"
MCP_BASE = "http://localhost:3000/mcp"
OUTPUT_DIR = "extracted_md"
CHUNK_SIZE = 12000  # chars per LLM call, keeps us well under context limits


# ---------------------------------------------------------------------------
# MCP CLIENT (minimal — navigate + snapshot only)
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(self, base_url: str = MCP_BASE):
        self.base_url = base_url
        self._req_id = 0
        self._session_id = None
        self._lock = threading.Lock()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    def _do_post(self, payload: dict) -> requests.Response:
        return requests.post(self.base_url, json=payload,
                              headers=self._headers(), timeout=30)

    def _parse_sse(self, text: str) -> dict:
        results = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
                if "error" in data:
                    raise RuntimeError(f"MCP error: {data['error']}")
                r = data.get("result", {})
                if r:
                    results.append(r)
            except json.JSONDecodeError:
                continue
        merged = {}
        for r in results:
            merged.update(r)
        return merged

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        with self._lock:
            payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method}
            if params:
                payload["params"] = params
            resp = self._do_post(payload)
            if resp.status_code == 404:
                print("[MCP] 404 — reconnecting...")
                self._session_id = None
                self._handshake()
                resp = self._do_post(payload)
            resp.raise_for_status()
            if "mcp-session-id" in resp.headers:
                self._session_id = resp.headers["mcp-session-id"]
            return self._parse_sse(resp.text)

    def _handshake(self):
        payload = {
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "llm-md-extractor", "version": "1.0"},
            },
        }
        resp = self._do_post(payload)
        resp.raise_for_status()
        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers["mcp-session-id"]
        self._do_post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        print(f"[MCP] Session: {self._session_id}")

    def start(self):
        self._handshake()

    def call_tool(self, name: str, arguments: dict) -> str:
        for attempt in range(3):
            try:
                result = self._rpc("tools/call", {"name": name, "arguments": arguments})
                return self._extract_text(result)
            except Exception as e:
                if "404" in str(e) or "Session not found" in str(e):
                    print(f"    [WARN] Session lost (attempt {attempt+1}). Reconnecting...")
                    self._session_id = None
                    self._handshake()
                    time.sleep(2)
                    continue
                return f"### Error\n{e}"
        return "### Error\nFailed after retries."

    def _extract_text(self, result: dict) -> str:
        content = result.get("content", [])
        parts = []
        for c in content:
            if c.get("type") == "text":
                parts.append(c["text"])
            elif c.get("type") == "resource" and "resource" in c:
                res_text = c["resource"].get("text", "")
                if res_text:
                    parts.append(res_text)
        return "\n".join(parts) if parts else ""


def find_and_read_latest_snapshot() -> str:
    current_dir = os.getcwd()
    latest_file, latest_mtime = None, 0
    while True:
        for f in glob.glob(os.path.join(current_dir, ".playwright-mcp", "page-*.yml")):
            mtime = os.path.getmtime(f)
            if mtime > latest_mtime:
                latest_mtime, latest_file = mtime, f
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent
    if latest_file:
        time.sleep(0.2)
        with open(latest_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

PRECLEAN_RE = re.compile(
    r'^\s*-?\s*(?P<role>[a-zA-Z]+)'
    r'(?:\s+"(?P<text>(?:[^"\\]|\\.)*)")?'
    r'(?P<attrs>(?:\s*\[[^\]]+\])*)'
)
LEVEL_RE = re.compile(r'\[level=(\d+)\]')


def preclean_snapshot(snapshot: str) -> str:
    """
    Strips refs, cursor tags, and other raw attributes from the accessibility
    tree BEFORE it goes anywhere near the LLM. This is the main slowdown fix —
    raw snapshots are 80%+ structural noise; this keeps only role + text.
    """
    out = []
    for line in snapshot.splitlines():
        if not line.strip():
            continue
        m = PRECLEAN_RE.match(line)
        if not m:
            continue
        text = (m.group("text") or "").strip()
        if not text:
            continue
        role = m.group("role").lower()
        level_match = LEVEL_RE.search(m.group("attrs") or "")
        if role == "heading" and level_match:
            out.append(f"HEADING(L{level_match.group(1)}): {text}")
        elif role == "listitem":
            out.append(f"LIST_ITEM: {text}")
        elif role == "link":
            out.append(f"LINK: {text}")
        else:
            out.append(text)
    return "\n".join(out)

# ---------------------------------------------------------------------------
# LLM-DRIVEN MARKDOWN STRUCTURING
# ---------------------------------------------------------------------------

CHUNK_PROMPT = """You are converting raw web-page accessibility-tree data into clean,
human-readable Markdown. You will be given ONE CHUNK of a larger page snapshot
(it may start or end mid-section — that's fine).

Rules:
- Ignore noise: refs like [ref=e12], cursor/pointer tags, ARIA roles, raw attributes.
- Turn headings into proper Markdown headings (##, ###, etc.) based on their hierarchy.
- Turn lists into Markdown bullet or numbered lists.
- Turn links into plain readable text (skip raw URLs unless they're clearly meaningful, like article links).
- Merge fragmented text nodes into natural, readable paragraphs/sentences.
- Drop purely structural or repeated UI chrome (nav bars, cookie banners, "Skip to content", footers with boilerplate links) unless it's the only content present.
- Do NOT invent content. Only restructure what's actually present.
- Output ONLY the Markdown for this chunk. No preamble, no explanation, no code fences.
"""

MERGE_PROMPT = """You are given several Markdown sections, each produced independently
from consecutive chunks of the same web page. Merge them into ONE clean, coherent,
human-readable Markdown document.

Rules:
- Remove duplicate headings, repeated boilerplate, or content that clearly appears in
  multiple chunks (e.g. nav menus repeated at chunk boundaries).
- Fix heading hierarchy so it makes sense as a single document (one main title, then
  logically nested subsections).
- Keep the actual page content's meaning and order intact — don't rewrite the substance,
  just clean up structure and remove duplication/noise.
- Output ONLY the final Markdown document. No preamble, no explanation, no code fences.
"""


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def llm_structure_chunk(chunk: str) -> str:
    response = client.chat.completions.create(
        model=NIM_MODEL,
        messages=[
            {"role": "system", "content": CHUNK_PROMPT},
            {"role": "user", "content": chunk},
        ],
        max_tokens=1500,
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()
    return re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text)


def llm_merge_sections(sections: list[str], url: str, title_hint: str) -> str:
    combined_input = "\n\n---CHUNK BOUNDARY---\n\n".join(sections)
    response = client.chat.completions.create(
        model=NIM_MODEL,
        messages=[
            {"role": "system", "content": MERGE_PROMPT},
            {"role": "user", "content": f"Page source: {url}\n\n{combined_input}"},
        ],
        max_tokens=3000,
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text)

    header = (
        f"<!-- Source: {url} -->\n"
        f"<!-- Extracted: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n\n"
    )
    return header + text.strip() + "\n"


from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 6
MAX_CHUNKS = 15  # safety cap — past this, truncate rather than melt your token budget


def structure_snapshot_with_llm(snapshot: str, url: str) -> str:
    cleaned = preclean_snapshot(snapshot)
    print(f"[Clean] {len(snapshot)} chars → {len(cleaned)} chars after noise removal")

    chunks = chunk_text(cleaned, CHUNK_SIZE)
    if len(chunks) > MAX_CHUNKS:
        print(f"[Warn] {len(chunks)} chunks found, truncating to first {MAX_CHUNKS} "
              f"(likely hitting references/footer boilerplate beyond this point)")
        chunks = chunks[:MAX_CHUNKS]

    print(f"[LLM] Processing {len(chunks)} chunk(s) with {MAX_WORKERS} parallel workers...")

    sections = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(llm_structure_chunk, chunk): i for i, chunk in enumerate(chunks)}
        done_count = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                sections[i] = future.result()
            except Exception as e:
                print(f"    [ERROR] Chunk {i+1} failed: {e}")
                sections[i] = ""
            done_count += 1
            print(f"    → {done_count}/{len(chunks)} chunks done")

    if len(sections) == 1:
        header = (
            f"<!-- Source: {url} -->\n"
            f"<!-- Extracted: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n\n"
        )
        return header + sections[0].strip() + "\n"

    print("    → merging chunks into final document")
    return llm_merge_sections(sections, url, url.split("//")[-1].split("/")[0])

def save_markdown(markdown: str, title_hint: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", title_hint)[:50] or "page"
    path = os.path.join(OUTPUT_DIR, f"{safe_name}_{int(time.time())}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"[Saved] {path}")
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def extract_page_to_markdown(url: str) -> str:
    mcp = MCPClient()
    mcp.start()

    print(f"[Navigate] {url}")
    mcp.call_tool("browser_navigate", {"url": url})
    mcp.call_tool("browser_wait_for", {"time": 2})

    snapshot = mcp.call_tool("browser_snapshot", {})
    if not snapshot or len(snapshot) < 20:
        snapshot = find_and_read_latest_snapshot()
    if not snapshot:
        raise RuntimeError("Could not obtain a snapshot of the page.")

    markdown = structure_snapshot_with_llm(snapshot, url)

    title_hint = url.split("//")[-1].split("/")[0]
    path = save_markdown(markdown, title_hint)
    return path


if __name__ == "__main__":
    url = input("Enter the URL to extract : ").strip()
    output_path = extract_page_to_markdown(url)
    print(f"\nDone. Markdown saved to: {output_path}")