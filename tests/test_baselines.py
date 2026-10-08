"""Tests for the keyword and constant baselines."""

from triage.baselines import always, keyword_classify, keyword_scores


def test_bug_from_traceback():
    assert keyword_classify("read_csv crashes on empty file", "Traceback ... IndexError") == "BUG"


def test_feature_from_request():
    assert keyword_classify("Add support for dark mode", "It would be nice to have this.") == "FEATURE"


def test_docs_from_typo():
    assert keyword_classify("Fix typo in user guide", "The documentation says acts.") == "DOCS"


def test_question_from_how_do_i():
    assert keyword_classify("How do I list all virtualenvs", "") == "QUESTION"


def test_question_mark_in_title_counts():
    assert keyword_scores("Is pandas thread safe?", "")["QUESTION"] >= 3


def test_title_outweighs_body():
    # title says docs; body mentions an error once
    assert keyword_classify("Docs page for merge is misleading", "I got an error") == "DOCS"


def test_fallback_when_nothing_matches():
    assert keyword_classify("hello", "world") == "BUG"


def test_keywords_match_whole_words_only():
    # "address" contains "add" but must not count as FEATURE
    assert keyword_scores("address field", "")["FEATURE"] == 0


def test_always_ignores_input():
    clf = always("DOCS")
    assert clf("anything", "at all") == "DOCS"
