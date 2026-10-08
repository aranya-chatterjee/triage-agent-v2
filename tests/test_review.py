"""Unit tests for the review logic. No terminal needed."""

import pytest

from triage.review import apply_decision, load_jsonl, reset_review, review_stats, save_jsonl

REC = {"id": "r#1", "label": "BUG", "split": "test", "verified": False, "note": ""}


def test_enter_agrees():
    out = apply_decision(REC, "")
    assert out["review"] == "agree"
    assert out["verified"] is True
    assert out["label"] == "BUG"


def test_same_label_key_counts_as_agree():
    assert apply_decision(REC, "b")["review"] == "agree"


def test_relabel_keeps_old_label_and_note():
    out = apply_decision(REC, "d", "docs content is wrong, nothing broken")
    assert out["review"] == "relabel"
    assert out["label"] == "DOCS"
    assert out["label_before_review"] == "BUG"
    assert out["note"] == "docs content is wrong, nothing broken"


def test_drop_excludes_from_split():
    out = apply_decision(REC, "x", "ambiguous")
    assert out["review"] == "drop"
    assert out["split"] == "excluded"
    assert out["split_before_review"] == "test"


def test_original_record_is_not_modified():
    apply_decision(REC, "f")
    assert REC["label"] == "BUG" and "review" not in REC


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        apply_decision(REC, "z")


def test_stats_agreement_ignores_drops():
    recs = [
        apply_decision(REC, ""),
        apply_decision(REC, ""),
        apply_decision(REC, ""),
        apply_decision(REC, "f"),
        apply_decision(REC, "x"),
        REC,  # not reviewed yet
    ]
    s = review_stats(recs)
    assert (s["reviewed"], s["agree"], s["relabel"], s["drop"]) == (5, 3, 1, 1)
    assert s["agreement"] == pytest.approx(0.75)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "data.jsonl"
    recs = [REC, {**REC, "id": "r#2", "title": "naïve café ✓"}]
    save_jsonl(path, recs)
    assert load_jsonl(path) == recs
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_reset_review_restores_original():
    back = reset_review(apply_decision(REC, "x"))
    assert back["split"] == "test" and "review" not in back and back["verified"] is False
    back2 = reset_review(apply_decision(REC, "f"))
    assert back2["label"] == "BUG" and "label_before_review" not in back2
