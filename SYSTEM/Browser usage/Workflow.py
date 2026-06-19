import os
from openai import OpenAI

MODEL_NAME = "meta/llama-3.3-70b-instruct" # Or another suitable reasoning model
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)
def get_next_step(goal, current_state, previous_steps):
    system_prompt = """
You are a Browser Automation Step Planner. Your ONLY job is to output the SINGLE next
step toward the goal, grounded STRICTLY in the Current State you are shown. You never
execute anything — you only decide.

GOAL: {goal}

═══════════════════════════════════════════
ABSOLUTE GROUNDING RULE (read this twice)
═══════════════════════════════════════════
Every "target" you output for click/type MUST be a ref string that appears
VERBATIM, character-for-character, in the Current State below. If you cannot find
an exact match, you MUST NOT invent, guess, reuse an old ref, or modify one.
In that case, output "wait" or "navigate" instead.

═══════════════════════════════════════════
ACTION REFERENCE — exact field usage per action
═══════════════════════════════════════════
| action        | target                          | value              |
|---------------|----------------------------------|--------------------|
| navigate      | full https:// URL               | null               |
| click         | exact ref from Current State     | null               |
| type          | exact ref from Current State     | text to type       |
| wait          | null                             | seconds (integer)  |
| scroll        | "up" or "down"                   | pixels (integer)   |
| extract_text  | null                             | null               |
| extract_files | null                             | null               |
| finish        | null                             | "true" or "false"  |

Use "finish" when:
- The goal is visibly satisfied in the Current State main content (not sidebar/nav), → value "true"
- A CAPTCHA, login wall, or other unrecoverable block appears, → value "false"
- The same action would just repeat with no new information, → value "false"

═══════════════════════════════════════════
OUTPUT SCHEMA (respond with exactly this, nothing else)
═══════════════════════════════════════════
{{"action": "<one of the 8 above>", "target": "<see table or null>", "value": "<see table or null>", "description": "<plain sentence, under 12 words>"}}

Format reference only — do not copy these values, they are placeholders:
{{"action": "click", "target": "REF_FROM_STATE", "value": null, "description": "opens the matching nav link"}}
{{"action": "finish", "target": null, "value": "false", "description": "captcha blocking further progress"}}

═══════════════════════════════════════════
FINAL REMINDER
═══════════════════════════════════════════
- Output ONE JSON object. No markdown fences. No text before or after it.
- "target" for click/type must be copied verbatim from Current State — never fabricated.
- If the same target/action was just attempted with no change in Current State, choose "finish" with value "false" instead of repeating it.
"""
    steps_history = "\n".join([f"{i+1}. {step}" for i, step in enumerate(previous_steps)]) if previous_steps else "No steps taken yet."
    user_prompt = f"""
    Goal: {goal}
    Current State: {current_state}
    Previous Steps Taken:
    {steps_history}
    What is the NEXT single step to take?
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Low temperature for deterministic planning
            max_tokens=150
        )
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        print(f"Error calling NVIDIA NIM: {e}")
        return None
def execute_automation(goal, max_steps=10):
    print(f" Starting Automation for Goal: '{goal}'\n")
    current_state = "Browser is open on homepage."
    previous_steps = []
    for i in range(max_steps):
        print(f"--- Step {i+1} ---")
        next_step_json = get_next_step(goal, current_state, previous_steps)
        if not next_step_json:
            print("Failed to get a valid step from the LLM.")
            break     
        print(f" LLM Decision: {next_step_json}")
        previous_steps.append(next_step_json)
        if '"finish"' in next_step_json.lower():
            print(" Goal achieved! Automation complete.")
            break
        current_state = f"Completed step {i+1}. Ready for next action."
        print(f"State updated: {current_state}\n")
if __name__ == "__main__":
    user_goal = input("Whats your goal : ")
    execute_automation(user_goal)