import os
from openai import OpenAI
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from config import NVIDIA_API_KEY
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key= NVIDIA_API_KEY
)
user_input = input("Enter the Query: ")

response = client.chat.completions.create(
    model="mistralai/mistral-small-4-119b-2603",  # needs vendor prefix
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ],
    temperature=0.7,
    max_tokens=512,
)

print(response.choices[0].message.content)