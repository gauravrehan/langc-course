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

_REAL_BUILD_TOOL_SELECTOR = chat_graph._build_tool_selector


class _FakeSelector:
    def __init__(self, tool_name="delete_draft", confidence=0.99):
        self.tool_name = tool_name
        self.confidence = confidence

    def invoke(self, _messages):
        return chat_graph.ToolSelection(
            toolName=self.tool_name, confidence=self.confidence
        )


class _FakeExtractor:
    """Deterministic stand-in for LLM extraction.

    Reads only the user's text (after "User reply:" for follow-ups), finds
    "draft id X" or "name=value" pairs separated by ";", and maps a few
    phrases to reply intents.
    """

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        evidence = messages[1][1]
        if "User reply:\n" in evidence:
            evidence = evidence.split("User reply:\n", 1)[1]
        values = {
            key.strip(): value.strip()
            for key, value in re.findall(r"(\w+)=([^;\n]+)", evidence)
        }
        match = re.search(
            r"\b(?:draft id|draft identifier)\b\s*"
            r"(?:is|:|=|#)?\s*([A-Za-z0-9]+)\b",
            evidence,
            re.IGNORECASE,
        )
        if match:
            values["draftId"] = match.group(1)
        data = {}
        for name, field in self.schema.model_fields.items():
            if name == "replyIntent":
                continue
            value = values.get(name)
            # List fields are written as "name=a, b, c".
            if value is not None and "list" in str(field.annotation):
                value = [item.strip() for item in value.split(",")]
            data[name] = value
        if "replyIntent" in self.schema.model_fields:
            lowered = evidence.casefold()
            if "never mind" in lowered:
                data["replyIntent"] = "cancel"
            elif "instead" in lowered:
                data["replyIntent"] = "change_request"
            else:
                data["replyIntent"] = "answer"
        return self.schema.model_validate(data)


class _FakeLLM:
    def with_structured_output(self, schema, **_kwargs):
        return _FakeExtractor(schema)

    def invoke(self, _messages):
        raise AssertionError("The graph should not make free-form LLM calls")


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

        self.selector = _FakeSelector()
        self.patches = [
            patch.object(
                chat_graph,
                "_build_tool_selector",
                lambda _names: self.selector,
            ),
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
        self.selector = _FakeSelector("create_request")
        result = chat_graph.identify_tool(
            {
                "query": "can you create a new request?",
                "screen_name": "Home",
                "user_context": None,
                "screen_context": None,
            }
        )

        self.assertEqual(result["tool"], "create_request")

    # --- deterministic routing -------------------------------------------

    def test_next_step_is_decided_without_an_llm_call(self):
        # _FakeLLM.invoke raises, so any free-form LLM call would fail here.
        result = chat_graph.decide_next_step(
            {
                "tool_config": {"requiredParameters": [{"name": "draftId"}]},
                "parameters": {"draftId": None},
            }
        )
        self.assertEqual(result["next_step"], "ask_for_missing_parameters")
        self.assertEqual(result["missing_parameters"], ["draftId"])

    # --- follow-up replies -------------------------------------------------

    def test_cancel_while_collecting_parameters_ends_without_command(self):
        config = self._config()
        self._start("Delete my draft", config)

        result = self.graph.invoke(
            Command(resume={"message": "never mind"}), config
        )

        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(result["execute_command"])
        self.assertIn("Delete Draft cancelled", result["message"])

    def test_switching_request_while_collecting_reidentifies_the_tool(self):
        config = self._config()
        self._start("Delete my draft", config)

        self.selector = _FakeSelector("create_request")
        result = self.graph.invoke(
            Command(resume={"message": "create a request instead"}), config
        )

        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "provide_parameters")
        self.assertNotIn("draftId", prompt["parameters"])
        self.assertIn("eventName", prompt["missingParameters"])

    def test_repeated_unusable_replies_give_up(self):
        config = self._config()
        result = self._start("Delete my draft", config)

        for _ in range(chat_graph.MAX_FAILED_PARAMETER_REPLIES):
            result = self.graph.invoke(
                Command(resume={"message": "I don't know"}), config
            )

        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(result["execute_command"])
        self.assertIn("several attempts", result["message"])

    def test_reply_can_correct_a_value_given_earlier(self):
        self.selector = _FakeSelector("create_request")
        config = self._config()
        result = self._start(
            "create request eventName=Tech Summit; eventCity=Paris", config
        )
        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["parameters"]["eventCity"], "Paris")
        # Only the first few missing fields are named in one prompt.
        self.assertIn("eventDescription", prompt["message"])
        self.assertNotIn("primarySpeaker", prompt["message"])

        result = self.graph.invoke(
            Command(
                resume={
                    "message": "eventCity=Berlin; eventDescription=Annual meetup"
                }
            ),
            config,
        )

        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["parameters"]["eventCity"], "Berlin")
        self.assertEqual(
            prompt["parameters"]["eventDescription"], "Annual meetup"
        )
        self.assertEqual(prompt["parameters"]["eventName"], "Tech Summit")

    def test_values_from_the_question_are_not_accepted(self):
        class EchoExtractor:
            def __init__(self, schema):
                self.schema = schema

            def invoke(self, _messages):
                # Pretend the model copied "The" from "Draft ID: The ..."
                return self.schema.model_validate(
                    {"replyIntent": "answer", "draftId": "The"}
                )

        class EchoLLM:
            def with_structured_output(self, schema, **_kwargs):
                return EchoExtractor(schema)

        tool_config = chat_graph.load_tools_config()["screens"][0]["tools"][0]
        with patch.object(chat_graph, "llm", EchoLLM()):
            intent, updates = chat_graph._interpret_parameter_reply(
                tool_config,
                "Please provide the missing information: Draft ID: The id.",
                ["draftId"],
                {},
                "I'm not sure",
            )

        self.assertEqual(intent, "answer")
        self.assertEqual(updates, {})

    def test_confirmation_reply_can_modify_a_value(self):
        chat_graph.confirmation_classifier = _FakeConfirmationClassifier(
            "modify", "confirmed"
        )
        config = self._config()
        self._start("Delete my draft id XSSD1231", config)

        result = self.graph.invoke(
            Command(resume={"message": "use draft id ABC999 instead"}), config
        )
        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "confirm_action")
        self.assertEqual(prompt["parameters"], {"draftId": "ABC999"})

        result = self.graph.invoke(Command(resume={"message": "yes"}), config)
        self.assertEqual(
            result["execute_command"]["parameters"], {"draftId": "ABC999"}
        )

    def test_repeated_unclear_confirmations_give_up(self):
        chat_graph.confirmation_classifier = _FakeConfirmationClassifier(
            *["unclear"] * chat_graph.MAX_UNCLEAR_CONFIRMATIONS
        )
        config = self._config()
        self._start("Delete my draft id XSSD1231", config)

        for _ in range(chat_graph.MAX_UNCLEAR_CONFIRMATIONS):
            result = self.graph.invoke(
                Command(resume={"message": "hmm"}), config
            )

        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(result["execute_command"])

    # --- evidence matching -------------------------------------------------

    def test_evidence_modes(self):
        explicit = chat_graph._value_is_explicit
        self.assertTrue(explicit("Dr Smith", "speaker is Dr. Smith", "loose"))
        self.assertFalse(explicit("Dr Jones", "speaker is Dr. Smith", "loose"))
        self.assertFalse(explicit("XSSD12", "draft XSSD1231", "exact"))
        self.assertTrue(explicit("XSSD1231", "draft XSSD1231.", "exact"))
        self.assertTrue(explicit("A yearly meetup", "annual meetup", "none"))

    # --- list parameters ---------------------------------------------------

    _SECURITY_LIST = {
        "name": "eventSecurityList",
        "type": "array",
        "items": {"type": "string"},
        "description": "Companies or securities.",
    }

    def test_list_parameter_schema_is_a_list_of_strings(self):
        fields = chat_graph._parameter_fields([self._SECURITY_LIST])
        schema = chat_graph.create_model("S", **fields)
        parsed = schema.model_validate(
            {"eventSecurityList": ["Microsoft Corp", "SpaceX"]}
        )
        self.assertEqual(parsed.eventSecurityList, ["Microsoft Corp", "SpaceX"])

    def test_list_items_are_validated_individually(self):
        parsed = chat_graph._validate_values(
            [self._SECURITY_LIST],
            {
                "eventSecurityList": [
                    "Microsoft Corp",
                    "SpaceX",
                    "spacex",  # duplicate
                    "Tesla",  # not mentioned by the user
                    " ",
                ]
            },
            "securities are Microsoft Corp., SpaceX and 1Password",
        )
        self.assertEqual(parsed["eventSecurityList"], ["Microsoft Corp", "SpaceX"])

    def test_list_with_no_supported_items_is_missing(self):
        parsed = chat_graph._validate_values(
            [self._SECURITY_LIST], {"eventSecurityList": ["Tesla"]}, "no idea"
        )
        self.assertIsNone(parsed["eventSecurityList"])
        self.assertTrue(chat_graph._parameter_is_missing([]))
        self.assertTrue(chat_graph._parameter_is_missing(["", " "]))
        self.assertFalse(chat_graph._parameter_is_missing(["SpaceX"]))

    def test_reply_can_add_to_an_existing_list(self):
        class AppendingExtractor:
            def __init__(self, schema):
                self.schema = schema

            def invoke(self, _messages):
                return self.schema.model_validate(
                    {
                        "replyIntent": "answer",
                        "eventSecurityList": [
                            "Microsoft Corp",
                            "SpaceX",
                            "1Password",
                            "Tesla",  # hallucinated: neither said nor collected
                        ],
                    }
                )

        class AppendingLLM:
            def with_structured_output(self, schema, **_kwargs):
                return AppendingExtractor(schema)

        tool_config = {
            "displayName": "Create Request",
            "requiredParameters": [self._SECURITY_LIST],
        }
        with patch.object(chat_graph, "llm", AppendingLLM()):
            intent, updates = chat_graph._interpret_parameter_reply(
                tool_config,
                "Do you want to proceed?",
                ["eventSecurityList"],
                {"eventSecurityList": ["Microsoft Corp", "SpaceX"]},
                "also add 1Password",
            )

        self.assertEqual(intent, "answer")
        self.assertEqual(
            updates["eventSecurityList"], ["Microsoft Corp", "SpaceX", "1Password"]
        )

    def test_create_request_collects_security_list_end_to_end(self):
        self.selector = _FakeSelector("create_request")
        config = self._config()
        result = self._start(
            "create request eventName=Tech Summit; eventDescription=Annual "
            "meetup; eventDate=12/05/2026; eventCity=Paris; eventCountry=France; "
            "primarySpeaker=Jane Doe",
            config,
        )
        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["missingParameters"], ["eventSecurityList"])

        result = self.graph.invoke(
            Command(
                resume={
                    "message": "eventSecurityList=Microsoft Corp, SpaceX, 1Password"
                }
            ),
            config,
        )
        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "confirm_action")
        self.assertEqual(
            prompt["parameters"]["eventSecurityList"],
            ["Microsoft Corp", "SpaceX", "1Password"],
        )
        self.assertIn("Microsoft Corp, SpaceX, 1Password", prompt["message"])

        result = self.graph.invoke(Command(resume={"message": "yes"}), config)
        self.assertEqual(
            result["execute_command"]["parameters"]["eventSecurityList"],
            ["Microsoft Corp", "SpaceX", "1Password"],
        )

    # --- tool selection ----------------------------------------------------

    def test_tool_selector_schema_only_allows_configured_tools(self):
        captured = {}

        class CapturingLLM:
            def with_structured_output(self, schema, **_kwargs):
                captured["schema"] = schema
                return None

        with patch.object(chat_graph, "llm", CapturingLLM()):
            _REAL_BUILD_TOOL_SELECTOR(["delete_draft", "create_request"])

        schema = captured["schema"]
        schema.model_validate({"toolName": "delete_draft", "confidence": 0.9})
        schema.model_validate({"toolName": None, "confidence": 0.9})
        with self.assertRaises(Exception):
            schema.model_validate({"toolName": "drop_db", "confidence": 0.9})

    def test_low_confidence_selection_asks_for_clarification_once(self):
        self.selector = _FakeSelector("delete_draft", confidence=0.3)
        config = self._config()
        result = self._start("get rid of it", config)

        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "clarify_action")
        self.assertEqual(prompt["action"], "delete_draft")
        self.assertIn("Delete Draft", prompt["message"])

        # Still low confidence, but the clarification budget is spent, so the
        # best guess is used; the confirmation step still guards execution.
        result = self.graph.invoke(
            Command(resume={"message": "yes, draft id QW12"}), config
        )
        prompt = self._interrupt_data(result)
        self.assertEqual(prompt["kind"], "confirm_action")
        self.assertEqual(prompt["parameters"], {"draftId": "QW12"})


if __name__ == "__main__":
    import unittest

    unittest.main()
