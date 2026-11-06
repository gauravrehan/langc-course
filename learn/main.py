# This script demonstrates how to use LangChain to request a story from an
# Ollama language model. It first loads environment variables from a .env file,
# if available, and prints the installed langchain-core and langgraph versions.
# The main function connects to the llama3.2:latest model through ChatOllama,
# using a temperature of zero to reduce randomness in the generated response.
# It asks the model to write approximately 500 words in French about a cat
# learning to play the piano, then prints the returned message object.
# Ollama must be running with the requested model available, and the required
# Python packages must be installed. The story request runs only when this
# file is executed directly; environment loading and version printing also
# happen when the file is imported. Use response.content to print only the story.

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
