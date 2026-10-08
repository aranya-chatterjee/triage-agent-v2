"""Tests for the eval harness, using tiny fake classifiers."""

import pytest

from triage.evaluation import ERROR, evaluate, format_report

RECORDS = [
    {"id": "r#1", "title": "a", "body": "", "label": "BUG"},
    {"id": "r#2", "title": "b", "body": "", "label": "BUG"},
    {"id": "r#3", "title": "c", "body": "", "label": "DOCS"},
    {"id": "r#4", "title": "d", "body": "", "label": "QUESTION"},
]


def test_perfect_classifier():
    truth = {r["title"]: r["label"] for r in RECORDS}
    result = evaluate(RECORDS, lambda title, body: truth[title])
    assert result["accuracy"] == 1.0
    assert result["per_class"]["BUG"]["support"] == 2


def test_always_bug_has_ok_accuracy_but_poor_macro_f1():
    result = evaluate(RECORDS, lambda title, body: "BUG")
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] < 0.25  # only BUG gets any F1 (0.667 / 4)


def test_crashing_classifier_is_recorded_not_fatal():
    def flaky(title, body):
        if title == "b":
            raise TimeoutError("LLM timed out")
        return "BUG"

    result = evaluate(RECORDS, flaky)
    assert result["errors"] == 1
    row = next(p for p in result["predictions"] if p["id"] == "r#2")
    assert row["pred"] == ERROR
    assert "TimeoutError" in row["error"]
    assert result["n"] == 4  # the run finished


def test_empty_records_rejected():
    with pytest.raises(ValueError):
        evaluate([], lambda t, b: "BUG")


def test_report_contains_headline_and_matrix():
    text = format_report(evaluate(RECORDS, lambda t, b: "BUG"))
    assert "Macro-F1" in text
    assert "Confusion matrix" in text
