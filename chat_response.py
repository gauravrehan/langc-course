from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    sessionId: UUID = Field(default_factory=uuid4)
    serverResponse: str
