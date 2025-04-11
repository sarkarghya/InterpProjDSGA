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
    model="deepseek-reasoner",
    messages=messages
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("\n\n Reasoning Content:", reasoning_content)
print("\n\n Final Answer:", content if content else "No answer provided.")
