# PAT AI Intelligence: prototype plan

## Status

Last updated 2026-11-24, branch `feature/prototype`.

| Milestone | Status |
|---|---|
| M1 Models, repository, synthetic data, seed | Done |
| M2 Prediction engine and agent tools | Done |
| M3 Submission Advisor graph and `/pat/advise` | Done |
| M4 Approval Intelligence Agent graph, tick/events/health/resume | Done |
| M5 Notifications (stored + logged), monitor run, demo scenarios | Done |
| M6 Reply interpretation, learning loop, backtest script | Not started |
| Teams delivery | Future |

Changes from the original plan, found while building:

- Synthetic data models deferred reviews, last-minute rushes and occasionally missed (stalled) PATs. Result: about 2.5% terminations (about 70% blocked at CG) and about 18% last-minute approvals. Terminations are higher than the proposal's 1% so the model has enough failure cases to learn from.
- Gate-duration groups also use a complexity flag (sovereign entities or 3+ securities), and terminated PATs are kept as censored samples.
- Extra guardrails: a critical PAT always notifies at least the submitter, urgency has a floor by health, and any "N hours left" phrase must match the real time left.
- The agent's facts start with plain-English key facts, because the small local model confused "hours pending" with "hours left".
- Endpoints added: `GET /pat` (list) and `GET /pat/{id}/coverage`.

## 1. Goal and scope

Build a working prototype of the two capabilities in the PAT AI Intelligence proposal, inside this repo and alongside the existing chat API:

- **Submission Advisor**: before a PAT is submitted, predict the approval outlook, explain the risks and recommend actions.
- **Approval Intelligence Agent**: after submission, monitor the PAT, reason about its risk, choose an intervention and draft a notification message.

**Design rule:** LLM nodes do the understanding, reasoning, decisions and writing. Code computes the numbers (probabilities, durations, countdowns, gate states) and checks every LLM output before it is used.

**In scope:** synthetic data stored in MongoDB, a prediction engine, two LangGraph graphs, REST endpoints, channel-neutral notification payloads, and offline tests.

**Out of scope:** real PAT data, Teams or email delivery (left as an extension point, see section 8), authentication, and a front-end UI.

### Decisions

| Topic | Decision |
|---|---|
| Code layout | New `pat/` package |
| Storage | MongoDB for PAT data, with an in-memory fallback for tests and for running without `MONGODB_URI` |
| Delivery | No Teams integration yet. The agent produces a notification payload and stores it; a `Notifier` interface leaves room for Teams later |
| LLM | Keep the local model (`llama3.2` via Ollama). The model is configured in one place so the provider can change later |

## 2. Code layout

The existing chat flow changes by one line.

```
pat/
  models.py          # Pydantic: PatRequest, GateEvent, Outlook, Health, Intervention, Notification, ...
  repository.py      # PatRepository interface; MongoPatRepository + InMemoryPatRepository
  data.py            # synthetic history generator
  seed.py            # CLI: generate synthetic data and load it into MongoDB
  predict.py         # gate-duration model, Monte Carlo outlook, similar-PAT lookup
  tools.py           # LangChain @tool wrappers over predict/repository (for the agent)
  llm.py             # module-level llm + structured-output builders (patchable in tests)
  guards.py          # grounding checks, action allowlist, policy rules
  advisor_graph.py   # Submission Advisor StateGraph
  monitor_graph.py   # Approval Intelligence Agent StateGraph
  notify.py          # Notification payload builder + Notifier interface (LogNotifier for now)
  api.py             # FastAPI APIRouter: /pat/...
  pat_config.json    # gates, thresholds, action catalogue, approval-group coverage calendar
tests/test_pat_predict.py
tests/test_pat_advisor.py
tests/test_pat_monitor.py
docs/pat_prototype_plan.md
```

`chat.py` gains only `app.include_router(pat_router)`. Both graphs reuse the checkpointer from `chat_graph.py`.

## 3. Storage (MongoDB)

Same database as the checkpointer (`MONGODB_DATABASE`), with separate collections:

| Collection | Contents |
|---|---|
| `pat_history` | Synthetic historical PATs with their full gate journeys (used for prediction and similar-case lookup) |
| `pat_requests` | In-flight PATs submitted through the API |
| `pat_gate_events` | Workflow events (gate started or approved, reminder sent, terminated) |
| `pat_interventions` | Agent decisions: inputs, tool calls, reasoning, chosen action, notification payload |
| `pat_lessons` | Learning-loop post-mortems (stretch goal) |

- `repository.py` defines a `PatRepository` interface. `MongoPatRepository` is used when `MONGODB_URI` is set; otherwise `InMemoryPatRepository` is used. This matches how the checkpointer is chosen.
- Tests always use the in-memory repository, so they stay offline.
- `uv run python -m pat.seed` generates the synthetic history and loads it. It is idempotent: it clears and reloads the PAT collections only, never the checkpoint collections.
- Indexes: `pat_history` on `(event_type, region)` and `submitted_at`; `pat_gate_events` on `(pat_id, timestamp)`; `pat_interventions` on `(pat_id, created_at)`.
- The prediction engine loads `pat_history` into memory once at startup. At about 5,000 records this is fast and keeps the prediction functions pure.

## 4. Data model (synthetic)

- **PatRequest:** id, title, description (free text), event type (media / conference / client / internal), region, city and time zone, event time, submit time, speaker role, securities and issuers mentioned, sovereign entities.
- **Gate journey:** ordered gates (RM, SA, CG, CC), each with a status, start and end time, and reminders sent.
- **Generator patterns:** CG is slow under high queue load; weekends and after-hours are slow; media events and sovereign entities add review time; about 20% of PATs are submitted late. This should roughly reproduce the proposal's shape: about 20% of approvals in the last 3 hours and under 1% terminations.
- **Volume:** about 5,000 historical PATs plus about 20 in-flight demo PATs at different risk levels. A fixed random seed makes it reproducible.

## 5. Prediction engine (`predict.py`, code only, no LLM)

- `gate_duration_distribution(gate, segment)`: empirical durations for each segment (event type × weekday/hour bucket × queue-load band), falling back to coarser segments when data is thin.
- `simulate_outlook(pat, now)`: Monte Carlo over the remaining gates. Returns `P(complete before deadline)`, the P50 and P90 ETA, and confidence based on sample size.
- `similar_pats(pat, k)`: nearest neighbours on the encoded features. Returns the matches plus their on-time rate and median approval time.
- `blocking_gate(pat)` and `time_remaining(pat)`: computed deterministically from gate state.
- These functions are wrapped as LangChain tools in `tools.py` so the agent can call them.

## 6. Graph A: Submission Advisor

```
extract_signals      LLM  structured output: topics, securities, sovereigns, media?, cross-region?
validate_signals     code every entity must appear in the request text (reuse the evidence-match idea)
compute_outlook      code probability, ETA, expected journey, similar PATs
assess_risk          LLM  ranked risk drivers + impact, citing only provided facts
recommend_actions    LLM  pick from action catalogue: before-submit / after-submit
summarise            LLM  "AI Summary" paragraph + suggested monitoring preferences
validate_output      code numbers/gates in text match computed facts; else retry once, then fall back to template
respond              code AdvisorResponse
```

It is a single pass with no interrupts, run with `POST /pat/advise`.

## 7. Graph B: Approval Intelligence Agent (one thread per PAT, `thread_id = "pat:<id>"`)

```
load_state           code current PAT, gate events, prior interventions (from repository + checkpoint)
agent_reason         LLM + tools (ReAct loop, max N steps):
                       get_gate_status, predict_remaining_time, get_reminder_history,
                       get_similar_cases, get_group_coverage, get_past_lessons
assess_health        LLM  structured: health (on_track/at_risk/critical), blocker, explanation
                     code overrides health if it contradicts computed probability bands
choose_intervention  LLM  structured: action ∈ allowlist, target, urgency, reason
policy_check         code rate limits, quiet hours, dedupe, max escalations
  ├─ no action    → record
  ├─ escalate     → interrupt() for human confirmation (reuses the existing HITL pattern)
  └─ notify/nudge → draft_message
draft_message        LLM  notification title/body + optional note to the approval group
emit_notification    code build Notification payload, hand to Notifier (LogNotifier for now)
record               code store intervention in pat_interventions (audit trail)
```

How it gets triggered:

- `POST /pat/{id}/tick` runs one monitoring pass by hand.
- `POST /pat/monitor/run` runs a pass over every in-flight PAT, for demos.
- `POST /pat/{id}/events` feeds in workflow changes (gate approved, reminder sent).

**Reply interpretation (stretch goal):** `POST /pat/{id}/reply` uses the LLM to classify a free-text approver or submitter reply into an intent (OOO, needs info, will review by X, approved elsewhere). That intent updates the state and starts a new tick.

## 8. Notifications (future delivery integration)

- `Notification` is a channel-neutral model: PAT id, health, termination risk, time remaining, blocker, explanation, recommended next steps, reminder history, and action links.
- `Notifier` is a small interface (`send(notification) -> DeliveryResult`). The only implementation for now is `LogNotifier`, which logs the payload and marks it as stored.
- Every notification is saved with its intervention in `pat_interventions` and can be read with `GET /pat/{id}/notifications`.
- **Future:** a `TeamsNotifier` maps `Notification` to an Adaptive Card and posts it through a webhook or bot. Email or other channels plug in the same way. No graph changes are needed.

## 9. Learning loop (stretch goal)

When a PAT reaches a final state, an LLM node writes a structured post-mortem: root cause, the gate that blocked it, which intervention helped, and a lesson. These are stored in `pat_lessons` and returned by `get_past_lessons` for similar PATs.

## 10. API summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/pat/advise` | Submission Advisor outlook |
| POST | `/pat` | Submit a PAT (stored, starts monitoring) |
| POST | `/pat/{id}/events` | Workflow state change |
| POST | `/pat/{id}/tick` | Run the agent once |
| POST | `/pat/{id}/resume` | Confirm or cancel an escalation (interrupt) |
| POST | `/pat/monitor/run` | Run the agent over all in-flight PATs |
| GET | `/pat/{id}/health` | Latest health, blocker, countdown, explanation, history |
| GET | `/pat/{id}/notifications` | Notification payloads produced for this PAT |

## 11. Guardrails

- **Grounding:** numbers, gate names and entities in LLM output must match the given facts, or the output is rejected.
- **Allowlist:** the LLM chooses actions only from `pat_config.json`. It can never approve a PAT or change a gate.
- **Code overrides:** health and policy are checked in code, and code wins any conflict.
- **Human in the loop:** escalations go through `interrupt()`.
- **Bounded loops:** the ReAct loop has a step limit, and the agent falls back to a deterministic template if the LLM fails.
- **Audit trail:** every decision is stored with its inputs, the tool calls made and the reasoning.

## 12. LLM setup

- `pat/llm.py` builds the model from environment variables (`PAT_LLM_PROVIDER`, `PAT_LLM_MODEL`), defaulting to `llama3.2:latest` on Ollama.
- Structured outputs use `with_structured_output(..., method="json_schema")`, like the chat graph.
- **Local-model limitation:** small local models are unreliable at multi-step tool calling. If tool calls fail or are malformed, `agent_reason` falls back to one pass that pre-computes all tool results in code and gives them to the LLM as context. The LLM still assesses health, chooses the intervention and writes the message.

## 13. Testing (offline, same approach as the existing tests)

- Tests clear `MONGODB_URI` before import and use `InMemoryPatRepository` and `InMemorySaver`.
- **Prediction:** deterministic seed, plus checks that the outputs look right (late submissions get a lower probability, CG blocks under load).
- **Graphs:** patch `pat.llm` with fakes returning scripted structured outputs and tool calls.
- **Guardrail tests:** an invented number is rejected; an action outside the allowlist is blocked; a duplicate notification is suppressed; an escalation pauses at the interrupt; a failed tool call falls back to the single-pass path.
- **Backtest script (no CI):** replay the history and report calibration (Brier score), how far ahead of termination alerts fire, and alert precision.

## 14. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M1 | `models`, `repository` (Mongo + in-memory), `data`, `seed` | Synthetic history loads into MongoDB and reproduces the proposal's patterns; tests pass |
| M2 | `predict` + `tools` | Outlook, similar PATs and blocker computed; unit tests pass |
| M3 | Advisor graph + `/pat/advise` | Grounded advisor response end to end with the local LLM; fake-LLM tests pass |
| M4 | Agent graph + tick/events/health/resume | Agent reasons (with tool loop or fallback), chooses an action, checks policy, handles the interrupt |
| M5 | `notify` + monitor run + demo script | Demo shows a PAT moving from on-track to at-risk to intervention, with stored notifications |
| M6 (stretch) | Reply interpretation, learning loop, backtest | Lessons are reused; backtest metrics reported |
| Future | `TeamsNotifier` | Notifications delivered to Teams as Adaptive Cards |
