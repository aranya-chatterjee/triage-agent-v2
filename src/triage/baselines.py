"""Non-LLM baseline classifiers. Any LLM agent must beat these to be worth its cost.

Every classifier has the same signature:  (title: str, body: str) -> label
so the eval harness can score any of them, including the LLM agent later.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Classifier = Callable[[str, str], str]

# Words and phrases that hint at each class. Matched case-insensitively on word boundaries.
KEYWORDS: dict[str, list[str]] = {
    "BUG": [
        "error", "exception", "traceback", "crash", "crashes", "fails", "failing", "failed",
        "broken", "regression", "incorrect", "wrong", "unexpected", "raises", "bug",
        "doesn't work", "does not work", "not working",
    ],
    "FEATURE": [
        "add", "support", "feature", "proposal", "propose", "allow", "enhancement",
        "would be nice", "it would be", "option to", "new parameter", "request",
    ],
    "DOCS": [
        "docs", "documentation", "docstring", "typo", "readme", "changelog", "guide",
        "tutorial", "example in the", "clarify", "misleading", "user guide",
    ],
    "QUESTION": [
        "how do i", "how to", "how can i", "is it possible", "is there a way", "can i",
        "what is the", "why does", "question", "stackoverflow", "am i missing",
    ],
}

TITLE_WEIGHT = 3       # a hit in the title counts 3x a hit in the body
FALLBACK = "BUG"       # used when nothing matches

_PATTERNS = {
    label: [re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in words]
    for label, words in KEYWORDS.items()
}


def keyword_scores(title: str, body: str) -> dict[str, int]:
    """Score each label by counting keyword hits (title hits weighted higher)."""
    scores = {}
    for label, patterns in _PATTERNS.items():
        title_hits = sum(bool(p.search(title)) for p in patterns)
        body_hits = sum(bool(p.search(body)) for p in patterns)
        scores[label] = TITLE_WEIGHT * title_hits + body_hits
    if title.rstrip().endswith("?"):
        scores["QUESTION"] += TITLE_WEIGHT
    return scores


def keyword_classify(title: str, body: str) -> str:
    """Pick the label with the highest keyword score; FALLBACK if nothing matched."""
    scores = keyword_scores(title, body)
    best = max(scores, key=scores.get)  # ties: the first label in KEYWORDS order wins
    return best if scores[best] > 0 else FALLBACK


def always(label: str) -> Classifier:
    """A classifier that ignores the input and always answers `label`.

    The dumbest possible baseline. Its accuracy equals that label's share of the data,
    which shows why accuracy alone can look good while macro-F1 exposes it.
    """
    def classify(title: str, body: str) -> str:
        return label
    return classify
