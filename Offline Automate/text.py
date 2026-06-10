"""
Desktop LLM Controller
Uses Qwen2.5-3B-Instruct locally to control your PC:
- Open apps
- Find files
- Copy / move files
- List directory contents
"""

import os
import re
import sys
import json
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OS = platform.system()  # "Windows" | "Linux" | "Darwin"

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def open_app(app_name: str) -> str:
    """Open an application by name."""
    app_name = app_name.strip()
    try:
        if OS == "Windows":
            # Try direct command first, then common paths
            common_apps = {
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "paint": "mspaint.exe",
                "explorer": "explorer.exe",
                "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "vscode": "code",
                "cmd": "cmd.exe",
                "terminal": "cmd.exe",
                "word": "winword",
                "excel": "excel",
                "powerpoint": "powerpnt",
                "vlc": r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            }
            cmd = common_apps.get(app_name.lower(), app_name)
            subprocess.Popen(cmd, shell=True)
        elif OS == "Linux":
            subprocess.Popen([app_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif OS == "Darwin":
            subprocess.Popen(["open", "-a", app_name])
        return f"✅ Opened '{app_name}' successfully."
    except Exception as e:
        return f"❌ Could not open '{app_name}': {e}"


def find_file(filename: str, search_root: str = None) -> str:
    """Search the entire PC for a file by name (supports wildcards)."""
    filename = filename.strip()
    if search_root is None:
        # Start from user home by default; for full PC scan use "/"
        search_root = str(Path.home())

    matches = []
    pattern = filename.lower()
    use_glob = "*" in pattern or "?" in pattern

    print(f"  🔍 Searching in {search_root} for '{filename}' ...")

    try:
        for root, dirs, files in os.walk(search_root, followlinks=False):
            # Skip system/hidden folders to speed up search
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d not in {
                    "proc", "sys", "dev", "run",
                    "Windows", "System32", "SysWOW64",
                    "$Recycle.Bin", "node_modules", "__pycache__"
                }
            ]
            for f in files:
                if use_glob:
                    import fnmatch
                    if fnmatch.fnmatch(f.lower(), pattern):
                        matches.append(os.path.join(root, f))
                else:
                    if pattern in f.lower():
                        matches.append(os.path.join(root, f))
                if len(matches) >= 20:  # cap results
                    break
            if len(matches) >= 20:
                break
    except PermissionError:
        pass

    if not matches:
        return f"❌ No files found matching '{filename}' under {search_root}."

    result = f"✅ Found {len(matches)} match(es):\n"
    for m in matches:
        result += f"  📄 {m}\n"
    return result.strip()


def copy_file(source: str, destination: str) -> str:
    """Copy a file or folder from source to destination."""
    source = source.strip()
    destination = destination.strip()

    # Expand ~ and env vars
    source = os.path.expandvars(os.path.expanduser(source))
    destination = os.path.expandvars(os.path.expanduser(destination))

    src = Path(source)
    dst = Path(destination)

    if not src.exists():
        return f"❌ Source not found: {source}"

    try:
        if src.is_dir():
            shutil.copytree(str(src), str(dst / src.name), dirs_exist_ok=True)
            return f"✅ Copied folder '{src.name}' → {destination}"
        else:
            dst.mkdir(parents=True, exist_ok=True) if dst.suffix == "" else dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            return f"✅ Copied '{src.name}' → {destination}"
    except Exception as e:
        return f"❌ Copy failed: {e}"


def move_file(source: str, destination: str) -> str:
    """Move a file or folder from source to destination."""
    source = os.path.expandvars(os.path.expanduser(source.strip()))
    destination = os.path.expandvars(os.path.expanduser(destination.strip()))

    src = Path(source)
    dst = Path(destination)

    if not src.exists():
        return f"❌ Source not found: {source}"

    try:
        dst.mkdir(parents=True, exist_ok=True) if dst.suffix == "" else dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"✅ Moved '{src.name}' → {destination}"
    except Exception as e:
        return f"❌ Move failed: {e}"


def list_directory(path: str = None) -> str:
    """List contents of a directory."""
    if path is None or path.strip() == "":
        path = str(Path.home())
    path = os.path.expandvars(os.path.expanduser(path.strip()))

    p = Path(path)
    if not p.exists():
        return f"❌ Path not found: {path}"
    if not p.is_dir():
        return f"❌ Not a directory: {path}"

    try:
        items = list(p.iterdir())
        dirs = sorted([i.name for i in items if i.is_dir()])
        files = sorted([i.name for i in items if i.is_file()])
        result = f"📁 {path}\n"
        for d in dirs:
            result += f"  📂 {d}/\n"
        for f in files:
            result += f"  📄 {f}\n"
        return result.strip()
    except PermissionError:
        return f"❌ Permission denied: {path}"


def delete_file(path: str) -> str:
    """Delete a file or empty folder."""
    path = os.path.expandvars(os.path.expanduser(path.strip()))
    p = Path(path)
    if not p.exists():
        return f"❌ Not found: {path}"
    try:
        if p.is_dir():
            shutil.rmtree(str(p))
            return f"✅ Deleted folder: {path}"
        else:
            p.unlink()
            return f"✅ Deleted file: {path}"
    except Exception as e:
        return f"❌ Delete failed: {e}"


# ─────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────

TOOLS = {
    "open_app": open_app,
    "find_file": find_file,
    "copy_file": copy_file,
    "move_file": move_file,
    "list_directory": list_directory,
    "delete_file": delete_file,
}

TOOL_DESCRIPTIONS = """
You are a desktop controller AI. You interpret user commands and respond with a JSON tool call.

Available tools:
1. open_app(app_name) - Open an application. Example: open_app("chrome")
2. find_file(filename, search_root?) - Search for a file. Example: find_file("resume.pdf")
3. copy_file(source, destination) - Copy file/folder. Example: copy_file("C:/file.txt", "D:/backup/")
4. move_file(source, destination) - Move file/folder. Example: move_file("~/Downloads/doc.pdf", "~/Documents/")
5. list_directory(path?) - List folder contents. Example: list_directory("~/Desktop")
6. delete_file(path) - Delete a file or folder. Example: delete_file("~/Downloads/old.zip")

IMPORTANT RULES:
- Always respond with ONLY a JSON object, no extra text.
- Format: {"tool": "tool_name", "args": {"arg1": "value1", ...}}
- For find_file, only use search_root if the user specifies a location.
- If you cannot determine the tool, respond: {"tool": "unknown", "args": {}}

Examples:
User: "open notepad" → {"tool": "open_app", "args": {"app_name": "notepad"}}
User: "find my resume" → {"tool": "find_file", "args": {"filename": "resume"}}
User: "find resume.pdf on desktop" → {"tool": "find_file", "args": {"filename": "resume.pdf", "search_root": "~/Desktop"}}
User: "copy report.docx to D:/backup" → {"tool": "copy_file", "args": {"source": "report.docx", "destination": "D:/backup"}}
User: "show what's in downloads" → {"tool": "list_directory", "args": {"path": "~/Downloads"}}
"""

# ─────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────

def load_model():
    print(f"⚙️  Loading {MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    load_kwargs = {
        "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    # 4GB VRAM: load in 4-bit if bitsandbytes available
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs.pop("torch_dtype", None)
        print("  Using 4-bit quantization (bitsandbytes)")
    except ImportError:
        print("  bitsandbytes not found — loading in fp16")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()
    print("✅ Model loaded.\n")
    return tokenizer, model


def call_llm(tokenizer, model, user_input: str) -> dict:
    """Call the LLM and parse a tool call from its response."""
    messages = [
        {"role": "system", "content": TOOL_DESCRIPTIONS},
        {"role": "user", "content": user_input},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return response


def parse_tool_call(response: str) -> Optional[dict]:
    """Extract JSON from LLM response."""
    # Try to find JSON in the response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing entire response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def execute_tool(tool_call: dict) -> str:
    """Execute the parsed tool call."""
    tool_name = tool_call.get("tool", "unknown")
    args = tool_call.get("args", {})

    if tool_name == "unknown" or tool_name not in TOOLS:
        return f"❓ I couldn't understand that command. Available tools: {', '.join(TOOLS.keys())}"

    try:
        fn = TOOLS[tool_name]
        result = fn(**args)
        return result
    except TypeError as e:
        return f"❌ Tool call error: {e}"


# ─────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🖥️  Desktop LLM Controller")
    print("  Powered by Qwen2.5-3B-Instruct")
    print("=" * 55)
    print("Commands: 'quit' to exit | 'help' for examples\n")

    tokenizer, model = load_model()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("👋 Bye!")
            break

        if user_input.lower() == "help":
            print("""
📖 Example commands:
  • open chrome
  • open notepad
  • find resume.pdf
  • find *.py files in downloads
  • copy C:/report.docx to D:/backup
  • move ~/Downloads/photo.jpg to ~/Pictures
  • list downloads folder
  • list C:/Users
  • delete ~/Downloads/old_file.zip
""")
            continue

        print("🤖 Thinking...")
        raw_response = call_llm(tokenizer, model, user_input)
        print(f"   [LLM raw]: {raw_response}")

        tool_call = parse_tool_call(raw_response)

        if tool_call is None:
            print("❓ Could not parse a tool call. Try rephrasing your command.\n")
            continue

        print(f"   [Tool]: {tool_call['tool']} | [Args]: {tool_call.get('args', {})}")
        result = execute_tool(tool_call)
        print(f"\n{result}\n")


if __name__ == "__main__":
    main()