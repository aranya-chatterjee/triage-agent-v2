# Evaluation

| File | Purpose |
|---|---|
| `build_dataset.py` | *(you write this)* Fetch labeled issues from source repos into `golden.jsonl` |
| `golden.jsonl` | The golden dataset: one issue per line, with `label` and `split` (train/dev/test) |
| `runs/` | Eval results per run (git-ignored) |

## Labeling spec

Ground truth comes from maintainer-applied labels, mapped to our four classes:

Counts are closed issues per label, checked 2026-10-07 via the GitHub Search API.

| Repo | Repo label | Our label | Closed issues |
|---|---|---|---|
| pandas-dev/pandas | `Bug` | BUG | 8,432 |
| pandas-dev/pandas | `Enhancement` | FEATURE | 3,133 |
| pandas-dev/pandas | `Docs` | DOCS | 2,506 |
| pandas-dev/pandas | `Usage Question` | QUESTION | 1,662 |
| python-poetry/poetry | `kind/bug` | BUG | 3,101 |
| python-poetry/poetry | `kind/feature` | FEATURE | 841 |
| python-poetry/poetry | `area/docs` | DOCS | 286 |
| python-poetry/poetry | `kind/question` | QUESTION | 291 |

Target: ~30 issues per class per repo (~240 total), so neither repo dominates.

Rules:
- Closed issues only, pull requests excluded.
- Exactly one mapped label per issue (an issue labeled both `Bug` and `Docs` is skipped).
- Template prefixes like "[BUG]" or "BUG:" stripped from titles.
- Spot-check 20 random issues by hand and record the agreement rate here: __ / 20.
