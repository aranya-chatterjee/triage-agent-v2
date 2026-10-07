# Evaluation

| File | Purpose |
|---|---|
| `build_dataset.py` | *(you write this)* Fetch labeled issues from source repos into `golden.jsonl` |
| `golden.jsonl` | The golden dataset: one issue per line, with `label` and `split` (train/dev/test) |
| `runs/` | Eval results per run (git-ignored) |

## Labeling spec

Ground truth comes from maintainer-applied labels, mapped to our four classes:

| Repo | Repo label | Our label |
|---|---|---|
| _fill in_ | | BUG |
| | | FEATURE |
| | | DOCS |
| | | QUESTION |

Rules: closed issues only, exactly one mapped label, pull requests excluded,
template prefixes like "[BUG]" stripped from titles.
