import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from RAG_Load import retrieve

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

HISTORY_FILE = "chat_history.json"

if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        messages = json.loads(content) if content else []
else:
    messages = [{"role": "system", "content": ("You are a helpful AI assistant. "
                "Answer clearly and use the provided context when available.")}]

print("Assistant ready.")
print("Type 'exit' to quit.\n")

SCORE_THRESHOLD = 0.8

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        print("Conversation saved.")
        break

    retrieved = retrieve(user_input)
    retrieved_chunks = [chunk for chunk, score in retrieved]
    top_score = max(score for chunk, score in retrieved)

    if top_score >= SCORE_THRESHOLD:
        context = "\n\n".join([str(c) for c in retrieved_chunks])
        print(f"[RAG active — top score: {top_score:.4f}]")
        rag_prompt = (
            f"You are given the following retrieved context:\n\n{context}\n\n"
            f"Using only the relevant parts of the context above, "
            f"answer this question clearly and concisely:\n{user_input}\n\n"
            f"If the context does not contain the answer, say so and answer from your own knowledge."
        )
    
    else:
        print(f"[RAG skipped — top score: {top_score:.4f} below threshold {SCORE_THRESHOLD}]")
        rag_prompt = user_input

    messages.append({"role": "user", "content": rag_prompt})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print()

    messages[-1] = {"role": "user", "content": user_input}
    messages.append({"role": "assistant", "content": response})

    if len(messages) > 21:
        messages = [messages[0]] + messages[-20:]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)