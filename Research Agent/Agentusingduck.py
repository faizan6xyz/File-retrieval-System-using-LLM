import json
import os
from duckduckgo_search import DDGS
from openai import OpenAI
from Toolsusingduck import search_deep , search_news , search_web
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)
NIM_MODEL = "meta/llama-3.1-8b-instruct"
TOOLS = {
    "search_web":  lambda p: search_web(p["query"], int(p.get("max_results", 5))),
    "search_deep": lambda p: search_deep(p["query"]),
    "search_news": lambda p: search_news(p["query"], int(p.get("max_results", 5))),
}
SYSTEM_PROMPT = """
You are an expert research agent. Your job is to research topics
thoroughly and produce clear, well-structured markdown reports.

You have access to these tools:

1. search_web(query, max_results)   — quick broad search, use first
2. search_deep(query)               — deeper search, more content per result
3. search_news(query, max_results)  — recent news only

RESEARCH STRATEGY — always follow this:
Step 1: Start with a broad search_web to get an overview
Step 2: Identify key sub-topics from the results
Step 3: Use search_deep on each important sub-topic
Step 4: Use search_news for any recent developments
Step 5: Only write the final report after at least 4-5 searches

To call a tool respond EXACTLY like this and nothing else:
TOOL: tool_name
INPUT: {"param": "value"}

When research is fully complete write the report like this:
FINAL ANSWER:
# [Topic] — Research Report

## Overview
...

## [Section 2]
...

## Recent Developments
...

## Key Takeaways
...

## Sources
- url1
- url2
"""
def parse_response(text: str):
    if "FINAL ANSWER:" in text:
        return "final", text.split("FINAL ANSWER:")[-1].strip()
    if "TOOL:" in text and "INPUT:" in text:
        tool   = text.split("TOOL:")[-1].split("\n")[0].strip()
        input_ = text.split("INPUT:")[-1].split("\n")[0].strip()
        return "tool", (tool, input_)
    return "unknown", text
def save_report(topic: str, content: str):
    folder = "results"
    filename = topic[:40].replace(" ", "_") + "_report.md"
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(content)
    print(f"\n Report saved → {filename}")
    return filename
def run_research_agent(topic: str, max_steps: int = 15):
    print(f"\n Researching: {topic}")
    print("=" * 50)
    messages     = [{"role": "user", "content": f"Research this topic thoroughly: {topic}"}]
    search_count = 0
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        response = client.chat.completions.create(
            model=NIM_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=2000,
            temperature=0.4
        )
        reply = response.choices[0].message.content
        action_type, action_data = parse_response(reply)
        if action_type == "final":
            print("\n" + "=" * 50)
            print(" FINAL REPORT READY")
            print("=" * 50)
            print(action_data)
            save_report(topic, action_data)
            return action_data
        elif action_type == "tool":
            tool_name, tool_input = action_data
            try:
                params = json.loads(tool_input)
                query  = params.get("query", "")
                print(f" [{tool_name}] → '{query}'")
                result = TOOLS[tool_name](params)
                search_count += 1
                print(f" Got results ({search_count} searches done)")
            except Exception as e:
                result = f"Tool error: {e}"
                print(f" Error: {e}")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",      "content": f"Search result:\n{result}"})
        else:
            # Agent said something but didn't use a tool or finish , push it to continue
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",      "content": "Continue your research using the tools."})
    print("\n Max steps reached — forcing final report.")
    return None
if __name__ == "__main__":
    while True:
        prompt = input("Enter the query for research (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        run_research_agent(prompt)