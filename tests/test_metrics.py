"""Tests for src/triage/metrics.py.

The main example is the 10-issue table from the lesson, so you can check every
expected number by hand.
"""

import pytest

from triage.metrics import accuracy, confusion_matrix, macro_f1, precision_recall_f1

LABELS = ["BUG", "FEATURE", "DOCS", "QUESTION"]

#            0      1      2          3           4      5           6       7       8          9
Y_TRUE = ["BUG", "BUG", "BUG",     "QUESTION", "QUESTION", "QUESTION", "DOCS", "DOCS", "FEATURE", "FEATURE"]
Y_PRED = ["BUG", "BUG", "QUESTION", "QUESTION", "BUG",     "QUESTION", "DOCS", "DOCS", "FEATURE", "BUG"]


# --- accuracy ----------------------------------------------------------------

def test_accuracy_lesson_example():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.7)  # 7 of 10 correct


def test_accuracy_rejects_different_lengths():
    with pytest.raises(ValueError):
        accuracy(["BUG", "DOCS"], ["BUG"])


def test_accuracy_rejects_empty_input():
    with pytest.raises(ValueError):
        accuracy([], [])


# --- confusion_matrix --------------------------------------------------------

def test_confusion_matrix_lesson_example():
    cm = confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    # cm[true][predicted] = count
    assert cm["BUG"] == {"BUG": 2, "FEATURE": 0, "DOCS": 0, "QUESTION": 1}
    assert cm["FEATURE"] == {"BUG": 1, "FEATURE": 1, "DOCS": 0, "QUESTION": 0}
    assert cm["DOCS"] == {"BUG": 0, "FEATURE": 0, "DOCS": 2, "QUESTION": 0}
    assert cm["QUESTION"] == {"BUG": 1, "FEATURE": 0, "DOCS": 0, "QUESTION": 2}


# --- precision_recall_f1 -----------------------------------------------------

def test_bug_scores():
    p, r, f1 = precision_recall_f1(Y_TRUE, Y_PRED, "BUG")
    assert p == pytest.approx(2 / 4)   # said BUG 4 times, 2 were right
    assert r == pytest.approx(2 / 3)   # 3 real BUGs, found 2
    assert f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_feature_scores():
    p, r, f1 = precision_recall_f1(Y_TRUE, Y_PRED, "FEATURE")
    assert p == pytest.approx(1.0)     # said FEATURE once, it was right
    assert r == pytest.approx(0.5)     # 2 real FEATUREs, found 1
    assert f1 == pytest.approx(2 / 3)


def test_perfect_class():
    assert precision_recall_f1(Y_TRUE, Y_PRED, "DOCS") == pytest.approx((1.0, 1.0, 1.0))


def test_never_predicted_label_gives_zero_not_crash():
    # Nobody predicted QUESTION here: precision would be 0/0. Return 0.0 instead of crashing.
    p, r, f1 = precision_recall_f1(["QUESTION", "BUG"], ["BUG", "BUG"], "QUESTION")
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_invalid_prediction_counts_as_wrong():
    # An LLM might answer "Bug." or "UNKNOWN". That is simply a miss.
    p, r, _ = precision_recall_f1(["BUG", "BUG"], ["BUG", "UNKNOWN"], "BUG")
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(0.5)


# --- macro_f1 ----------------------------------------------------------------

def test_macro_f1_lesson_example():
    bug_f1 = 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)    # 0.571
    feature_f1 = 2 / 3                             # 0.667
    docs_f1 = 1.0
    question_f1 = 2 / 3                            # P = 2/3, R = 2/3
    expected = (bug_f1 + feature_f1 + docs_f1 + question_f1) / 4
    assert macro_f1(Y_TRUE, Y_PRED, LABELS) == pytest.approx(expected)  # about 0.726
