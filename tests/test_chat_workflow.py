"""Offline unit tests for the chat graph's interrupt and resume workflow."""

import json
import logging
import os
import re
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from uuid import uuid4

# Avoid initializing the production MongoDB checkpointer when importing the app.
_original_mongodb_uri = os.environ.get("MONGODB_URI")
os.environ["MONGODB_URI"] = ""
import chat  # noqa: E402
import chat_graph  # noqa: E402

if _original_mongodb_uri is None:
    os.environ.pop("MONGODB_URI", None)
else:
    os.environ["MONGODB_URI"] = _original_mongodb_uri

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402


class _FakeSelector:
    def __init__(self, tool_name="delete_draft"):
        self.tool_name = tool_name

    def invoke(self, _messages):
        return chat_graph.ToolSelection(
            toolName=self.tool_name, confidence=0.99
        )


class _FakeExtractor:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        evidence = messages[1][1]
        match = re.search(
            r"\b(?:draft id|draft identifier)\b\s*"
            r"(?:is|:|=|#)?\s*([A-Za-z0-9]+)\b",
            evidence,
            re.IGNORECASE,
        )
        value = match.group(1) if match else None
        return self.schema.model_validate(
            {name: value if name == "draftId" else None
             for name in self.schema.model_fields}
        )


class _FakeLLM:
    def with_structured_output(self, schema, **_kwargs):
        return _FakeExtractor(schema)

    def invoke(self, messages):
        system_prompt = messages[0][1]
        user_prompt = messages[1][1]
        if "Decide the next conversational step" not in system_prompt:
            raise AssertionError("Unexpected LLM invocation in this test")

        missing_text = user_prompt.split("Missing required parameters:\n", 1)[1]
        missing = json.loads(missing_text)
        next_step = (
            "ask_for_missing_parameters" if missing else "ask_for_confirmation"
        )
        return SimpleNamespace(content=json.dumps({"nextStep": next_step}))


class _FakeConfirmationClassifier:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)

    def invoke(self, _messages):
        return chat_graph.ConfirmationInterpretation(
            decision=next(self.decisions), confidence=0.99
        )


class ChatWorkflowTests(TestCase):
    def setUp(self):
        old_log_level = chat_graph.logger.level
        chat_graph.logger.setLevel(logging.WARNING)
        self.addCleanup(chat_graph.logger.setLevel, old_log_level)

        self.patches = [
            patch.object(chat_graph, "tool_selector", _FakeSelector()),
            patch.object(chat_graph, "llm", _FakeLLM()),
            patch.object(
                chat_graph,
                "confirmation_classifier",
                _FakeConfirmationClassifier("confirmed"),
            ),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.graph = chat_graph.builder.compile(checkpointer=InMemorySaver())

    def _config(self):
        return {"configurable": {"thread_id": str(uuid4())}}

    def _start(self, query, config):
        return self.graph.invoke(
            {
                "query": query,
                "screen_name": "Home",
                "user_context": None,
                "screen_context": None,
            },
            config,
        )

    @staticmethod
    def _interrupt_data(result):
        return result["__interrupt__"][0].value

    def test_unreadable_parameter_reply_prompts_again_then_accepts_valid_id(self):
        config = self._config()
        result = self._start("Delete my draft", config)

        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "provide_draft_id")
        self.assertEqual(prompt["missingParameters"], ["draftId"])

        result = self.graph.invoke(
            Command(resume={"message": "draft id ???"}), config
        )
        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "provide_draft_id")
        self.assertIn("couldn't read a valid value", prompt["message"])

        result = self.graph.invoke(
            Command(resume={"message": "draft id IAB33129"}), config
        )
        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "confirm_action")
        self.assertEqual(prompt["parameters"], {"draftId": "IAB33129"})

    def test_alias_does_not_extract_value_without_llm_result(self):
        class EmptyExtractor:
            def __init__(self, schema):
                self.schema = schema

            def invoke(self, _messages):
                return self.schema.model_validate({"draftId": None})

        class EmptyLLM:
            def with_structured_output(self, schema, **_kwargs):
                return EmptyExtractor(schema)

        required_parameters = [
            {
                "name": "draftId",
                "type": "string",
                "pattern": "^[A-Za-z0-9]+$",
                "aliases": ["draft id", "draft identifier"],
                "description": "The draft to delete.",
            }
        ]
        with patch.object(chat_graph, "llm", EmptyLLM()):
            extracted = chat_graph._extract_configured_parameters(
                required_parameters, "draft id XSSD1231"
            )

        self.assertIsNone(extracted["draftId"])

    def test_empty_llm_values_remain_missing(self):
        class PartialExtractor:
            def __init__(self, schema):
                self.schema = schema

            def invoke(self, _messages):
                return self.schema.model_validate(
                    {
                        "eventName": "Pitch Book for Goldman Sachs",
                        "eventDescription": "",
                    }
                )

        class PartialLLM:
            def with_structured_output(self, schema, **_kwargs):
                return PartialExtractor(schema)

        required_parameters = [
            {"name": "eventName", "type": "string", "description": "Event name."},
            {
                "name": "eventDescription",
                "type": "string",
                "description": "Event description.",
            },
        ]
        with patch.object(chat_graph, "llm", PartialLLM()):
            extracted = chat_graph._extract_configured_parameters(
                required_parameters,
                "can you create a new request with eventName Pitch book for Goldman Sachs?",
            )

        self.assertEqual(extracted["eventName"], "Pitch Book for Goldman Sachs")
        self.assertIsNone(extracted["eventDescription"])

    def test_unclear_confirmation_reprompts_until_user_confirms(self):
        chat_graph.confirmation_classifier = _FakeConfirmationClassifier(
            "unclear", "confirmed"
        )
        config = self._config()
        result = self._start("Delete my draft id XSSD1231", config)

        self.assertEqual(
            self._interrupt_data(result)["kind"], "confirm_action"
        )

        result = self.graph.invoke(
            Command(resume={"message": "Maybe, I need to think"}), config
        )
        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "confirm_action")
        self.assertEqual(prompt["parameters"], {"draftId": "XSSD1231"})

        result = self.graph.invoke(Command(resume={"message": "Yes"}), config)
        self.assertEqual(result["status"], "ready_to_execute")
        self.assertEqual(
            result["execute_command"],
            {
                "tool": "delete_draft",
                "functionName": "delete_draft",
                "parameters": {"draftId": "XSSD1231"},
            },
        )

    def test_clear_rejection_cancels_without_an_execute_command(self):
        chat_graph.confirmation_classifier = _FakeConfirmationClassifier("rejected")
        config = self._config()
        result = self._start("Delete my draft id XSSD1231", config)

        result = self.graph.invoke(Command(resume={"message": "No"}), config)

        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(result["execute_command"])

    def test_api_response_exposes_missing_fields_from_graph_interrupt(self):
        session_id = uuid4()
        interrupt = SimpleNamespace(
            value={
                "kind": "provide_draft_id",
                "message": "Please provide the draft ID.",
                "missingParameters": ["draftId"],
                "parameters": {},
            }
        )

        response = chat._to_response(
            session_id,
            {"__interrupt__": [interrupt], "available_tools": []},
        )

        self.assertEqual(response.sessionId, session_id)
        self.assertEqual(response.status, "provide_draft_id")
        self.assertEqual(response.missingParameters, ["draftId"])

    def test_api_response_maps_confirmation_and_execute_command(self):
        session_id = uuid4()
        confirmation = chat._to_response(
            session_id,
            {
                "__interrupt__": [
                    SimpleNamespace(
                        value={
                            "kind": "confirm_action",
                            "message": "Confirm deletion?",
                            "action": "delete_draft",
                            "parameters": {"draftId": "XSSD1231"},
                        }
                    )
                ]
            },
        )
        self.assertEqual(confirmation.status, "confirm")

        completed = chat._to_response(
            session_id,
            {
                "status": "ready_to_execute",
                "message": "Confirmed.",
                "execute_command": {
                    "tool": "delete_draft",
                    "functionName": "delete_draft",
                    "parameters": {"draftId": "XSSD1231"},
                },
            },
        )
        self.assertEqual(completed.status, "ready_to_execute")
        self.assertEqual(
            completed.executeCommand.parameters, {"draftId": "XSSD1231"}
        )

    def test_create_request_tool_is_selected_from_config(self):
        chat_graph.tool_selector = _FakeSelector("create_request")
        result = chat_graph.identify_tool(
            {
                "query": "can you create a new request?",
                "screen_name": "Home",
                "user_context": None,
                "screen_context": None,
            }
        )

        self.assertEqual(result["tool"], "create_request")


if __name__ == "__main__":
    import unittest

    unittest.main()
