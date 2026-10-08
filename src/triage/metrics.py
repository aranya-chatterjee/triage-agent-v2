"""Classification metrics, written by hand (no sklearn) so you know exactly what they mean.

Vocabulary, for one label X:
    true positive  (TP): truth is X, prediction is X       -> correct
    false positive (FP): truth is NOT X, prediction is X   -> wrongly said X
    false negative (FN): truth is X, prediction is NOT X   -> missed an X

    precision = TP / (TP + FP)   "when I say X, how often am I right?"
    recall    = TP / (TP + FN)   "of all real Xs, how many did I find?"
    f1        = 2 * P * R / (P + R)

Run the tests with:  uv run pytest tests/test_metrics.py -v
"""

from __future__ import annotations


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Fraction of predictions that exactly match the truth."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    if not y_true:
        raise ValueError("cannot compute accuracy of zero items")

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    """Return cm[true_label][predicted_label] = count.

    Every label appears as a row and a column, even with count 0. Predictions outside
    `labels` (like "UNKNOWN") are not counted in any column.
    """
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm:
            cm[t][p] += 1
    return cm


def precision_recall_f1(
    y_true: list[str], y_pred: list[str], label: str
) -> tuple[float, float, float]:
    """Precision, recall and F1 for ONE label. Returns (precision, recall, f1)."""
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == label and p == label:
            tp += 1  # said label, and it was the label
        elif p == label:
            fp += 1  # said label, but it was something else
        elif t == label:
            fn += 1  # it was the label, but we said something else

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """Average of the per-label F1 scores. Every label counts equally."""
    scores = [precision_recall_f1(y_true, y_pred, label)[2] for label in labels]
    return sum(scores) / len(scores)