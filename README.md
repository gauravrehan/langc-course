# langc-course

## Persistent LangGraph checkpoints

The chat graph uses MongoDB when `MONGODB_URI` is set. Copy `.env.example` to
`.env`, replace the Atlas URI placeholders, and start the API:

```bash
cp .env.example .env
uv sync
uv run uvicorn chat:app --reload
```

The first connection creates the checkpoint collections and indexes. When
`MONGODB_URI` is absent, the app logs a warning and uses an in-memory checkpointer;
those checkpoints do not survive a process restart.

## PAT AI Intelligence prototype

The `pat/` package adds a Submission Advisor and an Approval Intelligence Agent
for PAT (Public Appearance Tool) requests, mounted at `/pat` on the same API.
All PAT data is synthetic. See `docs/pat_prototype_plan.md` for the design.

```bash
ollama serve                         # local LLM (llama3.2) must be running
uv run python -m pat.seed            # load synthetic history + 20 in-flight PATs into MongoDB
uv run uvicorn chat:app --reload
```

Without `MONGODB_URI` the API seeds an in-memory store on first use.
Re-run `uv run python -m pat.seed --in-flight-only` before a demo, because the
in-flight PATs are generated relative to the current time.

Try it:

```bash
curl localhost:8000/pat                                  # in-flight PATs, soonest event first
curl -X POST localhost:8000/pat/<id>/tick -d '{}' -H 'content-type: application/json'
curl -X POST localhost:8000/pat/monitor/run -d '{}' -H 'content-type: application/json'
curl localhost:8000/pat/<id>/health
```

The first three in-flight PATs are fixed demo scenarios: a media event with
Control Group stuck for 25 hours and the event 3.5 hours away (critical), a
client webinar due in 20 hours, and a conference four days out.
