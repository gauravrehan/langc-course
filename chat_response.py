from uuid import UUID, uuid4

from typing import Literal

from pydantic import BaseModel, Field

from chat_models import AvailableTool, UICommand


class ChatResponse(BaseModel):
    sessionId: UUID = Field(default_factory=uuid4)
    serverResponse: str
    status: Literal[
        "provide_draft_id",
        "confirm",
        "cancelled",
        "ready_to_execute",
        "unsupported",
        "tool_unavailable",
    ] = "unsupported"
    pendingAction: str | None = None
    pendingParameters: dict[str, str] | None = None
    executeCommand: UICommand | None = None
    availableTools: list[AvailableTool] | None = None
