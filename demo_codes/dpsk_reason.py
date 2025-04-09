from openai import OpenAI

client = OpenAI(api_key="sk-ceef1b68fcbd4adcb52d40fe1a61d02d", base_url="https://api.deepseek.com")

# Initialize conversation
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]

# First turn
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
print("First response reasoning:", reasoning_content)
print("First response answer:", content)

# Add assistant's response to the conversation
messages.append({'role': 'assistant', 'content': content})

# Add user's next question
messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})

# Second turn
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

print("Second response reasoning:", response.choices[0].message.reasoning_content)
print("Second response answer:", response.choices[0].message.content)

