"""The LLM used by the PAT graphs.

Every LLM call goes through this module, so tests can patch `pat.llm.llm`
with a fake. The model is configurable through PAT_LLM_PROVIDER and
PAT_LLM_MODEL, so moving to another provider is a configuration change.
"""

import json
import logging
import os
from typing import Any, TypeVar

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("pat")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

llm = init_chat_model(
    os.getenv("PAT_LLM_MODEL", "llama3.2:latest"),
    model_provider=os.getenv("PAT_LLM_PROVIDER", "ollama"),
    temperature=0,
)

Schema = TypeVar("Schema", bound=BaseModel)


def invoke_structured(schema: type[Schema], messages: list[tuple[str, str]], phase: str) -> Schema | None:
    """Structured LLM call. Returns None if the call or parsing fails, so callers can fall back."""
    logger.info("LLM request [%s]: %s", phase, json.dumps(messages, default=str)[:4000])
    try:
        result = llm.with_structured_output(schema, method="json_schema").invoke(messages)
    except Exception as error:  # noqa: BLE001 - any model failure means "fall back"
        logger.warning("LLM call failed [%s]: %s", phase, error)
        return None
    if isinstance(result, dict):
        try:
            result = schema.model_validate(result)
        except Exception as error:  # noqa: BLE001
            logger.warning("LLM output invalid [%s]: %s", phase, error)
            return None
    if not isinstance(result, schema):
        logger.warning("LLM output missing [%s]", phase)
        return None
    logger.info("LLM response [%s]: %s", phase, result.model_dump_json())
    return result


def bind_tools(tools: list[Any]) -> Any:
    return llm.bind_tools(tools)
