from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral")
response = llm.invoke("Say hello in one sentence.")
print(response.content)