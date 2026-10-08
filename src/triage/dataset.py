"""Golden dataset construction: fetch maintainer-labeled issues, clean them, split them.

The pure functions here (to_record, clean_title, stratified_split) do no network I/O,
so they are unit-tested in tests/test_dataset.py. Only fetch_issues talks to GitHub.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

import requests

SEARCH_URL = "https://api.github.com/search/issues"
LABELS = ("BUG", "FEATURE", "DOCS", "QUESTION")

# Labeling spec: repo -> {maintainer label: our label}. Mirrors eval/README.md.
SOURCES: dict[str, dict[str, str]] = {
    "pandas-dev/pandas": {
        "Bug": "BUG",
        "Enhancement": "FEATURE",
        "Docs": "DOCS",
        "Usage Question": "QUESTION",
    },
    "python-poetry/poetry": {
        "kind/bug": "BUG",
        "kind/feature": "FEATURE",
        "area/docs": "DOCS",
        "kind/question": "QUESTION",
    },
}

# Title prefixes that give the answer away, e.g. "BUG: ...", "ENH/BUG: ...", "[Feature] ...".
# pandas issue templates add these automatically, so leaving them in would let a model
# "classify" by string matching instead of understanding the issue.
_TAG = (
    r"(?:BUGS?|ENH|ENHANCEMENT|DOCS?|QST|QUESTION|FEAT|FEATURE|FEATURE REQUEST|"
    r"BUG REPORT|REGR|REGRESSION)"
)
# A tag, optionally followed by "/OTHER" parts ("BUG/DEPR", "DOC/RLS", "docs/cli"),
# then a colon or a spaced dash. "Bug-fix release" is NOT stripped (no space before "-").
_PREFIX_WORD = re.compile(rf"^\s*{_TAG}(?:\s*/\s*[A-Za-z]{{1,12}})*\s*(?::|\s+-)\s*", re.IGNORECASE)
_PREFIX_BRACKET = re.compile(r"^\s*\[[^\]]{1,25}\]\s*")

# Body template scaffolding that gives the answer away. GitHub issue templates add
# headings ("### Feature Type", "### Location of the documentation"), checkboxes
# ("- [x] I have confirmed this bug exists") and, in Poetry, an "### Issue Kind" section
# whose value literally names the class. We keep only what the user actually wrote.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]")
_LEAKY_SECTIONS = {"issue kind"}  # sections whose whole content is a label hint

MAX_BODY_CHARS = 4000


def fetch_issues(
    repo: str, label: str, token: str, n: int = 60, timeout: float = 15.0
) -> list[dict]:
    """Return up to n closed issues from `repo` carrying `label`, newest first.

    One request, because the Search API returns at most 100 items per page and we
    only need a few dozen per label.
    """
    query = f'repo:{repo} is:issue is:closed label:"{label}"'
    resp = requests.get(
        SEARCH_URL,
        params={"q": query, "per_page": min(n, 100), "sort": "created", "order": "desc"},
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["items"][:n]


def clean_title(title: str) -> str:
    """Strip template prefixes that leak the label (repeatedly: "[BUG] BUG: x" -> "x")."""
    previous = None
    while previous != title:
        previous = title
        title = _PREFIX_BRACKET.sub("", title)
        title = _PREFIX_WORD.sub("", title)
    return title.strip()


def clean_body(body: str) -> str:
    """Remove template scaffolding (headings, checkboxes, comments, leaky sections).

    Lines inside ``` code fences are always kept: a "# comment" in Python code is not
    a markdown heading.
    """
    body = _HTML_COMMENT.sub("", body)
    kept: list[str] = []
    in_code = False
    skipping_section = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            if not skipping_section:
                kept.append(line)
            continue
        if in_code:
            if not skipping_section:
                kept.append(line)
            continue
        if _HEADING.match(line):
            title = line.lstrip("# ").strip().lower()
            skipping_section = title in _LEAKY_SECTIONS
            continue
        if skipping_section or _CHECKBOX.match(line):
            continue
        kept.append(line)
    text = "\n".join(kept)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def to_record(raw: dict, repo: str, label_map: dict[str, str]) -> dict | None:
    """Turn one raw GitHub issue into a dataset record, or None if it must be skipped.

    Skipped: pull requests, and issues carrying zero or several of our mapped labels
    (those are ambiguous by definition; see the labeling guideline).
    """
    if "pull_request" in raw:
        return None

    names = [lbl["name"] for lbl in raw.get("labels", [])]
    source = [name for name in names if name in label_map]
    mapped = {label_map[name] for name in source}
    if len(mapped) != 1:
        return None

    title = clean_title(raw.get("title") or "")
    if not title:
        return None

    return {
        "id": f"{repo}#{raw['number']}",
        "repo": repo,
        "number": raw["number"],
        "url": raw.get("html_url", ""),
        "title": title,
        "body": clean_body(raw.get("body") or "")[:MAX_BODY_CHARS],
        "source_label": source[0],
        "label": mapped.pop(),
        "created_at": raw.get("created_at", ""),
        "split": None,
        "verified": False,
        "note": "",
    }


def stratified_split(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> list[dict]:
    """Assign train/dev/test so every class keeps the same proportions in every split.

    Splitting per class (not over the whole list) is what "stratified" means: a random
    split of a small dataset can leave one class almost missing from test.
    A fixed seed makes the split reproducible: same input, same split, every run.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in sorted(records, key=lambda r: r["id"]):  # sort first: input order can't change the split
        by_label[rec["label"]].append(rec)

    out: list[dict] = []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        n_train = round(len(group) * ratios[0])
        n_dev = round(len(group) * ratios[1])
        for i, rec in enumerate(group):
            split = "train" if i < n_train else "dev" if i < n_train + n_dev else "test"
            out.append({**rec, "split": split})
    return out


def summary_table(records: list[dict]) -> str:
    """Counts per class per split, as a printable table."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for rec in records:
        counts[(rec["label"], rec["split"])] += 1
    splits = ("train", "dev", "test")
    lines = [f"{'label':<10}" + "".join(f"{s:>7}" for s in splits) + f"{'total':>7}"]
    for label in LABELS:
        row = [counts[(label, s)] for s in splits]
        lines.append(f"{label:<10}" + "".join(f"{c:>7}" for c in row) + f"{sum(row):>7}")
    totals = [sum(counts[(lbl, s)] for lbl in LABELS) for s in splits]
    lines.append(f"{'total':<10}" + "".join(f"{c:>7}" for c in totals) + f"{sum(totals):>7}")
    return "\n".join(lines)
