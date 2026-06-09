import json
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI

# ── CONFIG ───────────────────────────────────────────
NVIDIA_API_KEY = "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"
EMAIL_ADDRESS  = "faizanclaudeuser1@gmail.com"
EMAIL_PASSWORD = "azvf btak acmi npwn"   # Gmail app password

NIM_MODEL = "meta/llama-3.1-8b-instruct"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# ── EMAIL TOOLS ──────────────────────────────────────
def send_email(to: str, subject: str, body: str) -> str:
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"]   = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        return f"✅ Email sent to {to} with subject '{subject}'"
    except Exception as e:
        return f"❌ Failed to send: {e}"


def read_emails(folder: str = "INBOX", limit: int = 5) -> str:
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select(folder)

        _, message_ids = mail.search(None, "ALL")
        ids = message_ids[0].split()[-limit:]

        emails = []
        for eid in reversed(ids):
            _, msg_data = mail.fetch(eid, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            emails.append(
                f"From: {msg['from']}\n"
                f"Subject: {msg['subject']}\n"
                f"Body: {body[:300]}"
            )

        mail.logout()
        return "\n\n---\n\n".join(emails) if emails else "No emails found."
    except Exception as e:
        return f"❌ Failed to read: {e}"


def search_emails(keyword: str) -> str:
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("INBOX")

        _, message_ids = mail.search(None, f'SUBJECT "{keyword}"')
        ids = message_ids[0].split()[-5:]

        results = []
        for eid in reversed(ids):
            _, msg_data = mail.fetch(eid, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            results.append(f"From: {msg['from']}\nSubject: {msg['subject']}")

        mail.logout()
        return "\n\n".join(results) if results else "No emails found."
    except Exception as e:
        return f"❌ Search failed: {e}"


# ── TOOLS REGISTRY ───────────────────────────────────
TOOLS = {
    "send_email":    lambda p: send_email(p["to"], p["subject"], p["body"]),
    "read_emails":   lambda p: read_emails(p.get("folder", "INBOX"), int(p.get("limit", 5))),
    "search_emails": lambda p: search_emails(p["keyword"]),
}

# ── SYSTEM PROMPT ────────────────────────────────────
SYSTEM_PROMPT = """
You are an email assistant agent. You have access to these tools:

1. send_email    — params: to, subject, body
2. read_emails   — params: folder (default INBOX), limit (default 5)
3. search_emails — params: keyword

To call a tool respond EXACTLY like this (nothing else):
TOOL: tool_name
INPUT: {"param": "value"}

When the task is complete respond EXACTLY like this:
FINAL ANSWER: your final response to the user

Always think step by step before acting.
"""

# ── AGENT LOOP ───────────────────────────────────────
def parse_response(text: str):
    if "FINAL ANSWER:" in text:
        return "final", text.split("FINAL ANSWER:")[-1].strip()
    if "TOOL:" in text and "INPUT:" in text:
        tool   = text.split("TOOL:")[-1].split("\n")[0].strip()
        input_ = text.split("INPUT:")[-1].split("\n")[0].strip()
        return "tool", (tool, input_)
    return "unknown", text


def run_email_agent(user_goal: str):
    print(f"\n🎯 Goal: {user_goal}\n")
    messages = [{"role": "user", "content": user_goal}]

    for step in range(10):
        print(f"--- Step {step + 1} ---")

        response = client.chat.completions.create(
            model=NIM_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=1000,
            temperature=0.7
        )

        reply = response.choices[0].message.content
        print(f"🤖 Agent: {reply}\n")

        action_type, action_data = parse_response(reply)

        if action_type == "final":
            print(f"✅ Done: {action_data}")
            return action_data

        elif action_type == "tool":
            tool_name, tool_input = action_data
            try:
                params = json.loads(tool_input)
                print(f"🔧 Calling: {tool_name}({params})")
                result = TOOLS[tool_name](params)
                print(f"📤 Result: {result}\n")
            except Exception as e:
                result = f"Tool error: {e}"

            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",      "content": f"Tool result: {result}"})

        else:
            print("⚠️ Could not parse response.")
            break

    print("⚠️ Max steps reached.")


# ── RUN ──────────────────────────────────────────────
if __name__ == "__main__":
    run_email_agent("Read my last 3 emails and give me a summary")