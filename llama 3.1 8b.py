from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)
while True:
  completion = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[{"role":"user","content":f"{input('Enter your prompt: ')}"}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=False
  )

  # Handle both content and tool calls for non-streaming
  if completion.choices[0].message.content is not None:
    print(completion.choices[0].message.content)