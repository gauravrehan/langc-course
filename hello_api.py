from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException

from chat_graph import get_chat_state, resume_chat, start_chat
from chat_models import ChatMessageRequest, ChatRequest
from chat_response import ChatResponse

app = FastAPI(title="LangGraph Chat Companion API")


def _response_status(interrupt_kind: str) -> str:
    # Keep internal graph node names separate from the public API status values.
    if interrupt_kind == "confirm_action":
        return "confirm"
    return interrupt_kind


# Sample execution:
#   GET /
# Sample response:
#   {"message": "Hello, World!"}
@app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, World!"}


def _to_response(session_id: UUID, result: dict) -> ChatResponse:
    values = result
    interrupts = values.get("__interrupt__", ())
    if interrupts:
        interrupt_data = interrupts[0].value
        kind = interrupt_data["kind"]
        return ChatResponse(
            sessionId=session_id,
            status=_response_status(kind),
            serverResponse=interrupt_data["message"],
            pendingAction=interrupt_data.get("action"),
            pendingParameters=interrupt_data.get("parameters"),
            availableTools=values.get("available_tools"),
        )

    command = values.get("execute_command")
    return ChatResponse(
        sessionId=session_id,
        status=values.get("status", "unsupported"),
        serverResponse=values.get("message", ""),
        executeCommand=command,
        availableTools=values.get("available_tools"),
    )


# Sample execution:
#   POST /chat
#   Content-Type: application/json
#   {"query": "Delete my draft id XSSD1231.",
#    "screenName": "Home",
#    "userContext": {"userId": "u-123"},
#    "screenContext": {"page": "drafts"}}
# Sample response (before execution, awaiting confirmation):
#   {"sessionId": "<server-generated-uuid>",
#    "serverResponse": "Do you want to delete draft XSSD1231? Reply yes or no.",
#    "status": "confirm",
#    "pendingAction": "delete_draft",
#    "pendingParameters": {"draftId": "XSSD1231"},
#    "executeCommand": null}
# For an unmatched query on a configured screen, the response also includes
# availableTools with the actions configured for that screen.
# Example request: {"query": "resubmit drafts", "screenName": "Home"}
# Example response includes:
#   "availableTools": [{"toolName": "delete_draft",
#                        "displayName": "Delete Draft",
#                        "description": "Delete a draft from the application.",
#                        "requiredParameters": [{"name": "draftId",
#                                                "type": "string",
#                                                "pattern": "^[A-Za-z0-9]+$",
#                                                "description": "The alphanumeric identifier of the draft to delete."}],
#                        "uiFunction": "delete_draft"}]
@app.post("/chat", response_model=ChatResponse)
def start_chat_endpoint(request: ChatRequest) -> ChatResponse:
    session_id = uuid4()
    result = start_chat(
        str(session_id),
        request.query,
        request.screenName,
        request.userContext,
        request.screenContext,
    )
    return _to_response(session_id, result)


# Sample execution to provide a missing draft ID:
#   POST /chat/<session-id>/message
#   Content-Type: application/json
#   {"message": "XSSD1231"}
# Sample execution to approve the deletion:
#   POST /chat/<session-id>/message
#   Content-Type: application/json
#   {"message": "Yes, delete it"}
# Sample response after approval:
#   {"sessionId": "<same-session-uuid>",
#    "serverResponse": "Confirmed. Run the requested action in the web application.",
#    "status": "ready_to_execute",
#    "pendingAction": null,
#    "pendingParameters": null,
#    "executeCommand": {"tool": "delete_draft",
#                        "functionName": "delete_draft",
#                        "parameters": {"draftId": "XSSD1231"}}}
# A "No" response returns status "cancelled" and no executeCommand.
@app.post("/chat/{session_id}/message", response_model=ChatResponse)
def continue_chat_endpoint(
    session_id: UUID, request: ChatMessageRequest
) -> ChatResponse:
    current = get_chat_state(str(session_id))
    if not current.values or not current.next:
        raise HTTPException(status_code=404, detail="Active chat session not found")

    result = resume_chat(str(session_id), request.message)
    return _to_response(session_id, result)


# Sample execution:
#   GET /chat/<session-id>
# Sample response while waiting for the draft ID:
#   {"sessionId": "<same-session-uuid>",
#    "serverResponse": "What is the draft ID? It must contain only letters and numbers.",
#    "status": "provide_draft_id",
#    "pendingAction": null,
#    "pendingParameters": null,
#    "executeCommand": null}
# The response reflects the session's current prompt or completed result.
@app.get("/chat/{session_id}", response_model=ChatResponse)
def get_chat_endpoint(session_id: UUID) -> ChatResponse:
    current = get_chat_state(str(session_id))
    if not current.values:
        raise HTTPException(status_code=404, detail="Chat session not found")

    interrupts = current.tasks[0].interrupts if current.tasks else ()
    if interrupts:
        interrupt_data = interrupts[0].value
        return ChatResponse(
            sessionId=session_id,
            status=_response_status(interrupt_data["kind"]),
            serverResponse=interrupt_data["message"],
            pendingAction=interrupt_data.get("action"),
            pendingParameters=interrupt_data.get("parameters"),
        )

    return ChatResponse(
        sessionId=session_id,
        status=current.values.get("status", "unsupported"),
        serverResponse=current.values.get("message", ""),
        executeCommand=current.values.get("execute_command"),
        availableTools=current.values.get("available_tools"),
    )
