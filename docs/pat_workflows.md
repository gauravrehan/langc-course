# PAT LangGraph workflows

Two graphs in `pat/`. Edges match the compiled graphs (`builder.compile().get_graph()`); keep this file in sync when edges change.

Legend:

- **Purple, bold border**: node calls the LLM.
- **Grey**: code only (no LLM).
- **Orange**: human-in-the-loop `interrupt()`.
- **Dotted border**: code checks that run on LLM output.

## Submission Advisor (`pat/advisor_graph.py`)

Single pass, no checkpointer. Endpoint: `POST /pat/advise`.

```mermaid
flowchart TD
    START([START]) --> extract_signals

    extract_signals["extract_signals<br/><i>LLM: securities, sovereigns, topics from request text</i>"]
    validate_signals["validate_signals<br/><i>keep only names found in the user's text</i>"]
    compute_outlook["compute_outlook<br/><i>Monte Carlo probability, ETA, journey,<br/>similar PATs, candidate risk drivers</i>"]
    assess_risk["assess_risk<br/><i>LLM: rank risk drivers + explain each</i>"]
    recommend_actions["recommend_actions<br/><i>LLM: pick actions from catalogue + reasons</i>"]
    summarise["summarise<br/><i>LLM: AI summary paragraph</i>"]
    validate_output["validate_output<br/><i>numbers and gates must come from facts</i>"]
    respond["respond<br/><i>AdvisorResponse + suggested monitoring</i>"]

    extract_signals --> validate_signals
    validate_signals --> compute_outlook
    compute_outlook --> assess_risk
    assess_risk --> recommend_actions
    recommend_actions --> summarise
    summarise --> validate_output
    validate_output -->|Grounded| respond
    validate_output -->|Invented numbers: retry once with feedback| summarise
    validate_output -->|Still invalid: template summary| respond
    respond --> END([END])

    classDef llm fill:#e9e3ff,stroke:#5b3fd6,stroke-width:3px,color:#1d1240
    classDef code fill:#f4f4f5,stroke:#9ca3af,color:#111827
    classDef guard fill:#f4f4f5,stroke:#6b7280,stroke-dasharray:5 3,color:#111827
    class extract_signals,assess_risk,recommend_actions,summarise llm
    class compute_outlook,respond code
    class validate_signals,validate_output guard
```

## Approval Intelligence Agent (`pat/monitor_graph.py`)

One checkpointed thread per PAT (`thread_id = "pat:<id>"`). Endpoints: `POST /pat/{id}/tick`, `POST /pat/{id}/resume`, `POST /pat/monitor/run`.

```mermaid
flowchart TD
    START([START]) --> load_state

    load_state["load_state<br/><i>load PAT, terminate if event passed,<br/>compute key facts</i>"]

    subgraph react ["agent_reason (ReAct loop, max 4 steps)"]
        direction LR
        agent_llm["LLM with bound tools<br/><i>decides which tool to call next</i>"]
        tools["Tools (code)<br/>get_gate_status · predict_remaining_time<br/>get_reminder_history · get_similar_cases<br/>get_group_coverage · get_queue_status<br/>get_intervention_history"]
        agent_llm -->|tool call| tools
        tools -->|tool result| agent_llm
    end

    fallback["fallback<br/><i>tool calling failed: run all tools in code</i>"]
    assess_health["assess_health<br/><i>LLM: health + explanation</i><br/>code: may be 1 level more severe, never less"]
    choose_intervention["choose_intervention<br/><i>LLM: action from allowlist, urgency, reason</i>"]
    policy_check["policy_check<br/><i>on-track: no alerts · critical: at least notify<br/>escalate only if critical · cooldowns ·<br/>coverage hours · urgency floor</i>"]
    confirm_escalation["confirm_escalation<br/><i>interrupt(): ask a person to approve</i>"]
    classify_escalation_reply["classify_escalation_reply<br/><i>LLM: confirmed / rejected / unclear</i>"]
    draft_message["draft_message<br/><i>LLM: title, body, next steps</i><br/>code: grounding + time-left check, else template"]
    emit_notification["emit_notification<br/><i>Notifier (LogNotifier now, Teams later)</i>"]
    record["record<br/><i>store InterventionRecord (audit trail)</i>"]

    load_state -->|PAT closed or terminated| END([END])
    load_state -->|In progress| agent_llm
    agent_llm -->|Final notes| assess_health
    agent_llm -.->|Error or no tool calls| fallback
    fallback --> assess_health
    assess_health --> choose_intervention
    choose_intervention --> policy_check
    policy_check -->|No action or suppressed| record
    policy_check -->|Escalation needs approval| confirm_escalation
    policy_check -->|Notify submitter / nudge group| draft_message
    confirm_escalation -->|Resume with reply| classify_escalation_reply
    classify_escalation_reply -->|Confirmed| draft_message
    classify_escalation_reply -->|Rejected or too many unclear| record
    classify_escalation_reply -->|Unclear: ask again| confirm_escalation
    draft_message --> emit_notification
    emit_notification --> record
    record --> END2([END])

    classDef llm fill:#e9e3ff,stroke:#5b3fd6,stroke-width:3px,color:#1d1240
    classDef code fill:#f4f4f5,stroke:#9ca3af,color:#111827
    classDef guard fill:#f4f4f5,stroke:#6b7280,stroke-dasharray:5 3,color:#111827
    classDef human fill:#ffedd5,stroke:#c2410c,stroke-width:2px,color:#431407
    class agent_llm,assess_health,choose_intervention,classify_escalation_reply,draft_message llm
    class load_state,tools,fallback,emit_notification,record code
    class policy_check guard
    class confirm_escalation human
```

`agent_reason` is one LangGraph node; the subgraph shows the loop inside it.

## LLM call sites

| Graph | Node | LLM call | Output schema | What checks it |
|---|---|---|---|---|
| Advisor | `extract_signals` | `invoke_structured` | `SignalExtraction` | `validate_signals`: names must appear in the request text |
| Advisor | `assess_risk` | `invoke_structured` | `RiskRanking` (ids are a `Literal` of computed drivers) | Ungrounded "why" dropped; skipped drivers appended |
| Advisor | `recommend_actions` | `invoke_structured` | `ActionChoices` (ids are a `Literal` of the catalogue) | Ungrounded reason replaced with catalogue text |
| Advisor | `summarise` | `invoke_structured` | `AdvisorSummary` | `validate_output`: one retry, then template |
| Agent | `agent_reason` | `bind_tools` + `invoke` loop | Tool calls / notes | Unknown tools refused; step limit; fallback runs all tools |
| Agent | `assess_health` | `invoke_structured` | `HealthJudgement` | Severity band from statistics; grounding + time-left check |
| Agent | `choose_intervention` | `invoke_structured` | `InterventionChoice` (action is a `Literal` of the allowlist) | `policy_check` |
| Agent | `classify_escalation_reply` | `invoke_structured` | `EscalationReply` | Failure counts as unclear; limit then cancel |
| Agent | `draft_message` | `invoke_structured` | `DraftMessage` | Grounding + time-left check, else template |

Every call goes through `pat/llm.py`; if the model errors, each node falls back to code.
