from fastapi import FastAPI
from chat_response import ChatResponse

app = FastAPI()


@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, World!"}


@app.get("/chat", response_model=ChatResponse)
def chat() -> ChatResponse:
    return ChatResponse(serverResponse="Hello from the chat endpoint!")
