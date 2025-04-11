from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

with open('minigun.json', 'r') as f:
    guninfo = json.load(f)

# Initialize conversation
messages = guninfo

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    stream=True #
)
# messages.append(response.choices[0].message) 

# print("Message", messages)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)