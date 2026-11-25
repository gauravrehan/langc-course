# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Uses `uv` with Python 3.13.

```bash
uv sync                                         # install deps
uv run uvicorn chat:app --reload                # run the API
uv run python -m unittest discover -s tests     # run all tests (offline, no LLM/Mongo needed)
uv run python -m unittest tests.test_chat_workflow.ChatWorkflowTests.test_evidence_modes   # single test
```

There is no linter or formatter configured.

## Secrets

`.env` holds confidential values (e.g. `MONGODB_URI` credentials). Never stage, commit or push it: do not use `git add .`, `git add -A` or `git commit -a`; stage files by name and check `git status` before committing. Do not copy its values into code, docs, tests, commit messages or logs.

## Architecture

A FastAPI service that turns a natural-language request from a web app screen into a confirmed UI command. The server never executes actions itself: the final output is an `executeCommand` (`tool`, `functionName`, `parameters`) that the front end runs.

- `chat.py`: FastAPI endpoints. `POST /chat` starts a session (a new UUID used as the LangGraph `thread_id`), `POST /chat/{id}/message` resumes an interrupted graph, `GET /chat/{id}` reads current state. Translates graph interrupts into `ChatResponse`, mapping the internal interrupt kind `confirm_action` to the public status `confirm`.
- `chat_graph.py`: the LangGraph `StateGraph` (flow diagram in `langgraph_workflow.mmd`; keep it in sync when changing edges). Nodes: `identify_tool` → (`clarify_tool` | `unsupported` | `parse_parameters`) → `decide_next_step` → (`collect_missing_parameters` | `confirm`) → `classify_confirmation` → (`prepare_command` | `cancel` | `give_up`). Human-in-the-loop steps use `interrupt()` with a payload dict (`kind`, `message`, `action`, `parameters`, `missingParameters`); resume via `Command(resume={"message": ...})`.
- `chat_models.py` / `chat_response.py`: Pydantic request/response models. The `ChatResponse.status` `Literal` must include every status a node can set.
- `tools_config.json`: the per-screen tool catalogue, re-read from disk on each `identify_tool` call. Tools have `intentExamples`, `negativeExamples`, `requiredParameters` (with optional `pattern`, `aliases`, `items` for `array` types, and `evidenceMatch`), and `uiFunction`. Adding a tool or screen is normally a config-only change.
- `learn/`: **off-limits.** Personal tutorial scripts unrelated to the app. Never read, search, or modify anything in it (including via Bash such as `cat`, `grep`, `sed`), and never use it as context for any task. If a task seems to need it, ask the user first. `.claude/settings.json` also denies Read/Edit/Write on `learn/**`.

### LLM usage and validation

- A single module-level `llm` (`init_chat_model("llama3.2:latest", model_provider="ollama")`) is used only via `with_structured_output(..., method="json_schema")`. Tool selection builds a dynamic Pydantic schema whose `toolName` is a `Literal` of only that screen's tools (`_build_tool_selector`). Confirmation replies go through `confirmation_classifier`.
- Extracted parameter values are not trusted as-is. `_validate_values` checks each against `pattern` and against the user's own text, using `evidenceMatch`: `exact` (verbatim token; the default when a `pattern` is set), `loose` (same words in order; the default otherwise), or `none` (paraphrase allowed). Only the user's reply text counts as evidence, never the server's question, so the LLM cannot pick up values from the prompt.
- `decide_next_step` is deterministic (no LLM call). Retry and threshold limits are constants at the top of `chat_graph.py` (`TOOL_CONFIDENCE_THRESHOLD`, `MAX_FAILED_PARAMETER_REPLIES`, `MAX_UNCLEAR_CONFIRMATIONS`, `MAX_FIELDS_PER_PROMPT`, etc.).

### Checkpointing

`chat_graph.py` picks the checkpointer at import time: `MongoDBSaver` when `MONGODB_URI` is set (also reads `MONGODB_DATABASE`, `MONGODB_TIMEOUT_MS` from `.env`), otherwise `InMemorySaver`, which loses sessions on restart.

### Tests

`tests/test_chat_workflow.py` clears `MONGODB_URI` *before* importing `chat`/`chat_graph` so that importing doesn't connect to Mongo. In `setUp` it patches `chat_graph._build_tool_selector`, `chat_graph.llm` and `chat_graph.confirmation_classifier` with deterministic fakes, then compiles a fresh graph from `chat_graph.builder` with its own `InMemorySaver`. New LLM call sites must be patchable the same way, or the tests will try to reach Ollama.
