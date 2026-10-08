"""Unit tests for the dataset logic. No network: every issue here is hand-made."""

from triage.dataset import SOURCES, clean_body, clean_title, stratified_split, to_record

PANDAS = SOURCES["pandas-dev/pandas"]


def issue(number=1, title="Something", labels=("Bug",), body="text", **extra):
    return {"number": number, "title": title, "body": body,
            "labels": [{"name": n} for n in labels], **extra}


# --- clean_title -----------------------------------------------------------

def test_clean_title_strips_pandas_prefixes():
    assert clean_title("BUG: read_csv fails on empty file") == "read_csv fails on empty file"
    assert clean_title("ENH: add foo") == "add foo"
    assert clean_title("DOC: fix typo") == "fix typo"
    assert clean_title("QST: how do I group?") == "how do I group?"


def test_clean_title_strips_brackets_and_stacked_prefixes():
    assert clean_title("[Feature] dark mode") == "dark mode"
    assert clean_title("[BUG] BUG: crash") == "crash"


def test_clean_title_strips_compound_prefixes():
    assert clean_title("BUG/DEPR: avoid Series.values") == "avoid Series.values"
    assert clean_title("ENH/BUG: use _values_for_json") == "use _values_for_json"
    assert clean_title("docs/cli: Typo in export") == "Typo in export"


def test_clean_title_keeps_normal_titles():
    assert clean_title("Debugging memory usage") == "Debugging memory usage"
    assert clean_title("Docs should mention sources") == "Docs should mention sources"
    assert clean_title("Bug-fix release broke install") == "Bug-fix release broke install"


# --- clean_body ------------------------------------------------------------

PANDAS_FEATURE_BODY = """### Feature Type

- [x] Adding new functionality to pandas

- [ ] Changing existing functionality in pandas

### Problem Description

I wish merge could override columns.
"""

POETRY_BODY = """<!-- Please fill this in -->
### Issue Kind

Change in current behaviour

### Description

`poetry shell` was quicker to type.
"""


def test_clean_body_removes_template_headings_and_checkboxes():
    out = clean_body(PANDAS_FEATURE_BODY)
    assert "Feature Type" not in out
    assert "Adding new functionality" not in out
    assert "I wish merge could override columns." in out


def test_clean_body_removes_issue_kind_section_and_comments():
    out = clean_body(POETRY_BODY)
    assert "Change in current behaviour" not in out
    assert "Please fill this in" not in out
    assert "`poetry shell` was quicker to type." in out


def test_clean_body_keeps_python_comments_inside_code():
    body = "### Reproducible Example\n```python\n# shifting wraps int64\nx = 1\n```"
    out = clean_body(body)
    assert "# shifting wraps int64" in out
    assert "Reproducible Example" not in out


# --- to_record -------------------------------------------------------------

def test_to_record_maps_label_and_cleans():
    rec = to_record(issue(title="BUG: crash", labels=("Bug", "IO CSV")), "pandas-dev/pandas", PANDAS)
    assert rec["label"] == "BUG"
    assert rec["source_label"] == "Bug"
    assert rec["title"] == "crash"
    assert rec["id"] == "pandas-dev/pandas#1"
    assert rec["verified"] is False


def test_to_record_skips_pull_requests():
    assert to_record(issue(pull_request={"url": "x"}), "pandas-dev/pandas", PANDAS) is None


def test_to_record_skips_issues_with_two_of_our_labels():
    assert to_record(issue(labels=("Bug", "Docs")), "pandas-dev/pandas", PANDAS) is None


def test_to_record_handles_null_body():
    rec = to_record(issue(body=None), "pandas-dev/pandas", PANDAS)
    assert rec["body"] == ""


# --- stratified_split ------------------------------------------------------

def make_records(per_label=10):
    return [
        {"id": f"r#{label}{i}", "label": label}
        for label in ("BUG", "FEATURE", "DOCS", "QUESTION")
        for i in range(per_label)
    ]


def test_split_is_balanced_per_class():
    out = stratified_split(make_records(10))
    for label in ("BUG", "FEATURE", "DOCS", "QUESTION"):
        splits = [r["split"] for r in out if r["label"] == label]
        assert splits.count("train") == 6
        assert splits.count("dev") == 2
        assert splits.count("test") == 2


def test_split_is_reproducible_and_order_independent():
    recs = make_records(10)
    a = {r["id"]: r["split"] for r in stratified_split(recs, seed=42)}
    b = {r["id"]: r["split"] for r in stratified_split(list(reversed(recs)), seed=42)}
    assert a == b


def test_split_keeps_every_record():
    assert len(stratified_split(make_records(7))) == 28
