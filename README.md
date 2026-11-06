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
