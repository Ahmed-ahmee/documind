from ollama import chat

response = chat(
    model='glm-5:cloud',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print(response.message.content)