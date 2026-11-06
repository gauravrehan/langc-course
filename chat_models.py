from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    screenName: str = Field(min_length=1)
    userContext: dict[str, Any] | None = None
    screenContext: dict[str, Any] | None = None


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1)


class UICommand(BaseModel):
    tool: str
    functionName: str
    parameters: dict[str, str]


class ToolParameterInfo(BaseModel):
    name: str
    type: str
    pattern: str | None = None
    description: str


class AvailableTool(BaseModel):
    toolName: str
    displayName: str
    description: str
    requiredParameters: list[ToolParameterInfo]
    uiFunction: str


