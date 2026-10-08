"""Run any classifier over dataset records and compute a full report."""

from __future__ import annotations

import time
from collections.abc import Callable

from triage.metrics import accuracy, confusion_matrix, macro_f1, precision_recall_f1

LABELS = ["BUG", "FEATURE", "DOCS", "QUESTION"]
ERROR = "ERROR"  # prediction recorded when the classifier raises


def evaluate(
    records: list[dict],
    classify: Callable[[str, str], str],
    labels: list[str] = LABELS,
) -> dict:
    """Classify every record and score the predictions.

    A classifier that raises on one issue does not stop the run: that issue is recorded
    as ERROR (counted wrong) with the error message, and the next issue continues.
    """
    if not records:
        raise ValueError("no records to evaluate")

    y_true, y_pred, rows = [], [], []
    start = time.perf_counter()
    for rec in records:
        t0 = time.perf_counter()
        error = ""
        try:
            pred = classify(rec["title"], rec["body"])
        except Exception as e:  # noqa: BLE001 - one bad issue must not kill the eval
            pred, error = ERROR, f"{type(e).__name__}: {e}"
        rows.append({
            "id": rec["id"],
            "title": rec["title"],
            "true": rec["label"],
            "pred": pred,
            "correct": pred == rec["label"],
            "seconds": round(time.perf_counter() - t0, 4),
            "error": error,
        })
        y_true.append(rec["label"])
        y_pred.append(pred)

    per_class = {}
    for label in labels:
        p, r, f1 = precision_recall_f1(y_true, y_pred, label)
        per_class[label] = {
            "precision": p, "recall": r, "f1": f1, "support": y_true.count(label),
        }

    return {
        "n": len(records),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels),
        "errors": sum(1 for r in rows if r["error"]),
        "seconds_total": round(time.perf_counter() - start, 3),
        "predictions": rows,
    }


def format_report(result: dict, labels: list[str] = LABELS) -> str:
    """Human-readable report: headline numbers, per-class table, confusion matrix."""
    lines = [
        f"Issues: {result['n']}   Accuracy: {result['accuracy']:.1%}   "
        f"Macro-F1: {result['macro_f1']:.1%}   Errors: {result['errors']}   "
        f"Time: {result['seconds_total']}s",
        "",
        f"{'label':<10}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>9}",
    ]
    for label in labels:
        c = result["per_class"][label]
        lines.append(
            f"{label:<10}{c['precision']:>11.1%}{c['recall']:>9.1%}{c['f1']:>8.1%}{c['support']:>9}"
        )

    short = {"BUG": "BUG", "FEATURE": "FEAT", "DOCS": "DOCS", "QUESTION": "QST"}
    lines += ["", "Confusion matrix (rows = true, columns = predicted)"]
    lines.append(" " * 10 + "".join(f"{short.get(p, p):>6}" for p in labels))
    for t in labels:
        row = result["confusion"][t]
        lines.append(f"{t:<10}" + "".join(f"{row[p]:>6}" for p in labels))
    return "\n".join(lines)
