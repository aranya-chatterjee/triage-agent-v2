"""Hand-review logic for the golden dataset (used by eval/review.py).

Pure functions only, so they can be unit-tested without a terminal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# One key per decision. "q" is QUESTION, so quitting uses the word "quit".
KEY_TO_LABEL = {"b": "BUG", "f": "FEATURE", "d": "DOCS", "q": "QUESTION"}
DROP_KEY = "x"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, records: list[dict]) -> None:
    """Write atomically: to a temp file first, then swap it in.

    If the program crashes halfway through writing, the original file is untouched,
    so a power cut can never leave you with half a dataset.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def apply_decision(rec: dict, key: str, note: str = "") -> dict:
    """Return a reviewed copy of `rec`.

    key "" (Enter) = agree with the current label
    key b/f/d/q    = change the label (agrees if it is the same label)
    key x          = drop: unusable or ambiguous, excluded from evaluation
    """
    key = key.strip().lower()
    out = {**rec, "verified": True, "note": note.strip()}

    if key == "":
        out["review"] = "agree"
    elif key in KEY_TO_LABEL:
        new_label = KEY_TO_LABEL[key]
        if new_label == rec["label"]:
            out["review"] = "agree"
        else:
            out["review"] = "relabel"
            out["label_before_review"] = rec["label"]
            out["label"] = new_label
    elif key == DROP_KEY:
        out["review"] = "drop"
        out["split"] = "excluded"
        out["split_before_review"] = rec["split"]
    else:
        raise ValueError(f"unknown review key: {key!r}")
    return out


def reset_review(rec: dict) -> dict:
    """Undo a review: restore the original label and split, clear review fields."""
    out = {k: v for k, v in rec.items()
           if k not in ("review", "label_before_review", "split_before_review")}
    out["label"] = rec.get("label_before_review", rec["label"])
    out["split"] = rec.get("split_before_review", rec["split"])
    out["verified"] = False
    out["note"] = ""
    return out


def review_stats(records: list[dict]) -> dict:
    """Agreement between the maintainer labels and your review.

    agreement = agreed / (agreed + relabeled). Dropped issues are reported separately:
    they are unclear, not necessarily mislabeled.
    """
    reviewed = [r for r in records if r.get("review")]
    agree = sum(r["review"] == "agree" for r in reviewed)
    relabel = sum(r["review"] == "relabel" for r in reviewed)
    drop = sum(r["review"] == "drop" for r in reviewed)
    judged = agree + relabel
    return {
        "reviewed": len(reviewed),
        "agree": agree,
        "relabel": relabel,
        "drop": drop,
        "agreement": agree / judged if judged else None,
    }
