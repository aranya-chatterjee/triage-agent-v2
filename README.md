# Triage Agent v2

A GitHub issue triage agent built with LangGraph that labels issues as
`BUG`, `FEATURE`, `DOCS` or `QUESTION`, measured against a golden eval set.

Rebuild of [langgraph-adk-version](https://github.com/aranya-chatterjee/langgraph-adk-version)
with evals, tracing, guardrails and production deployment.

## Setup (Windows PowerShell)

```powershell
uv sync                      # install exact locked dependencies into .venv
Copy-Item .env.example .env  # then fill in your keys
uv run pytest                # should print 3 passed
```

## Project layout

```
src/triage/   the agent
eval/         golden dataset and eval scripts
tests/        unit tests
```

## Status

- [x] Project skeleton
- [ ] Golden dataset
- [ ] Eval harness + baseline
- [ ] LangGraph agent
- [ ] GEPA optimization
- [ ] Production layer (Postgres, tracing, guardrails, CI, deploy)
