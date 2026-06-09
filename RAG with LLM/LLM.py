import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from RAG import retrieve
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map="auto") # Load the Qwen 2.5B model with half-precision (float16) for faster inference and reduced memory usage, and automatically map the model to available devices (e.g., GPU) for optimal performance during generation. This allows the model to generate responses efficiently while maintaining a balance between speed and quality, especially when running on hardware with limited resources.
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) # Initialize a TextStreamer to handle the output of generated text from the model, configured to skip the original prompt and any special tokens in the output. This allows for real-time streaming of the generated response directly to the console without including unnecessary tokens or the input prompt, providing a cleaner and more user-friendly output during interactions with the assistant.
HISTORY_FILE = "chat_history.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f) # Load the conversation history from a JSON file if it exists, allowing the assistant to maintain continuity in the dialogue across sessions. This enables the user to exit and return to the conversation later without losing any context, as the entire history of messages (both user and assistant) is preserved and can be used to inform future responses from the model.
else:
    messages = [{"role": "system","content": ("You are a helpful AI assistant. "
                "Answer clearly and use the provided context when available.")}]
print("Assistant ready.")
print("Type 'exit' to quit.\n")
SCORE_THRESHOLD = 0.9  #  Set a threshold for the relevance score to decide when to use retrieved context. If the top score from retrieval is below this threshold, the model will answer without using the retrieved context, treating the query as a normal question. Adjust this value based on experimentation to find the right balance between using relevant context and avoiding irrelevant information.
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        print("Conversation saved.")
        break
    retrieved = retrieve(user_input) # Call the retrieve function to get relevant chunks of text based on the user's query. The function returns a list of tuples, where each tuple contains a retrieved chunk and its corresponding relevance score.
    retrieved_chunks = [chunk for chunk, score in retrieved] # Extract just the text chunks from the retrieved results, ignoring the scores for now. This will be used to construct the context for the RAG prompt if the relevance score is above the defined threshold.
    top_score = max(score for chunk, score in retrieved) # Get the highest relevance score from the retrieved results to determine if it meets the threshold for including context in the prompt. This score will be used to decide whether to activate RAG (include retrieved context) or to skip it and answer based solely on the user's query.
    if top_score >= SCORE_THRESHOLD:
        context = "\n\n".join([str(c) for c in retrieved_chunks])
        print(f"[RAG active — top score: {top_score:.4f}]")
        rag_prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{user_input}\n\n"
            f"Answer using the context above. "
            f"If the answer is not in the context, answer normally.")
    else:
        print(f"[RAG skipped — top score: {top_score:.4f} below threshold {SCORE_THRESHOLD}]")
        rag_prompt = user_input
    messages.append({"role": "user", "content": rag_prompt}) # Add the user's query (with or without context) to the conversation history as a new message. This allows the model to generate a response based on the entire conversation history, including the latest query and any retrieved context if applicable.
    prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True) # Convert the conversation history into a single prompt string formatted according to the chat template expected by the model. This includes adding any necessary special tokens or formatting to indicate the roles of the messages (user vs assistant) and to signal the start of the generation phase for the model.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # Tokenize the constructed prompt and convert it into a format suitable for input to the model. The resulting tensors are moved to the same device as the model (e.g., GPU) to ensure efficient processing during generation. This step prepares the input data for the model to generate a response based on the user's query and any relevant context that was included in the prompt.
    outputs = model.generate(**inputs,max_new_tokens=300,do_sample=True,temperature=0.7,top_p=0.9,repetition_penalty=1.1,eos_token_id=tokenizer.eos_token_id,streamer=streamer) # Generate a response from the model using the provided prompt, with parameters set for sampling to create more diverse and natural responses. The streamer allows for real-time output of the generated text as it is produced by the model.
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],skip_special_tokens=True) # Decode the generated tokens into a human-readable string, skipping any special tokens that are not part of the natural language response. The slicing [inputs.input_ids.shape[1]:] ensures that only the newly generated tokens (the response) are decoded, excluding the original prompt tokens.
    print()
    messages[-1] = {"role": "user", "content": user_input} # Update the last message in the conversation history to contain only the original user input, removing the context from the stored history. This ensures that the conversation history remains clean and focused on the user's queries without including potentially large amounts of retrieved context, which can be re-retrieved as needed for future queries.
    messages.append({"role": "assistant", "content": response}) # Add the assistant's generated response to the conversation history as a new message. This allows the model to maintain a complete record of the conversation, which can be used for generating future responses that take into account the entire dialogue history.
    if len(messages) > 21:
        messages = [messages[0]] + messages[-20:]   # Limit the conversation history to the most recent 20 messages plus the initial system message to prevent the prompt from becoming too long for the model to handle effectively. This ensures that the model has enough context to generate relevant responses while avoiding issues with excessively long prompts that can lead to truncation or memory constraints.
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)  # Save the updated conversation history to a JSON file after each interaction, ensuring that the entire dialogue is preserved for future reference or continuation of the conversation. This allows the user to exit and return to the conversation later without losing any context.