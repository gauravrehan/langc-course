from dotenv import load_dotenv
from importlib.metadata import version
from langchain_ollama import ChatOllama

load_dotenv()


core_version = version("langchain-core")
lg_version = version("langgraph")

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    llm = ChatOllama(model="llama3.2:latest", temperature=0)
    response = llm.invoke("In French, can you write a story about a cat who learns to play the piano? in about 500 words?")
    print(f"Response from ollama {response}")


if __name__ == "__main__":
    main()
