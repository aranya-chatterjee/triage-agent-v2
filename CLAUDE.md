# Project: GitHub Triage Agent v2 (rebuild from scratch)

## My goal
I'm an AI engineering student rebuilding my LangGraph triage agent to production level
AND learning the concepts deeply (I fumbled deep questions in interviews).

## How to work with me
- I write the code. You act as a senior engineer: explain concepts, give specs and hints,
  review my code. Only write code when I explicitly ask.
- For every concept, explain: why it exists, how it works underneath, and the interview
  question I should be able to answer.
- When I hit a bug, teach me how to debug it before giving the fix.

## Environment
- Windows, PowerShell. Give PowerShell commands, not bash.
- Run everything through uv: `uv run python ...`, `uv run pytest`.
- Secrets live in .env (git-ignored). Settings are read only via src/triage/config.py.

## Roadmap
1. Golden dataset: ~250 real GitHub issues using maintainer labels -> BUG/FEATURE/DOCS/QUESTION,
   stratified train/dev/test split, eval/golden.jsonl
2. Eval harness + baseline (accuracy, per-class precision/recall, macro-F1)
3. LangGraph agent, fixing old bugs: validated structured output, feedback-aware retries,
   async-safe FastAPI, validated /resume, prompt-injection guard, different verifier model
4. GEPA optimization on train, report on held-out test only
5. Production: PostgresSaver, Langfuse tracing, model routing, eval-gated CI, GitHub App deploy

## Stack
Python 3.11, uv, LangGraph, Groq, Pydantic, DSPy/GEPA, FastAPI, Postgres, Docker, pytest, ruff
