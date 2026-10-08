"""Hand-review the test split of eval/golden.jsonl, one issue at a time.

Usage (PowerShell, from the project root):
    uv run python eval/review.py               # review the test split
    uv run python eval/review.py --split dev   # review another split
    uv run python eval/review.py --stats       # only print agreement stats
    uv run python eval/review.py --redo pandas-dev/pandas#61424 python-poetry/poetry#9091
                                               # re-review specific issues

Every answer is saved immediately, so you can stop at any time (type "quit" or press
Ctrl+C) and continue later: already-reviewed issues are skipped.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import webbrowser
from pathlib import Path

from triage.review import (
    KEY_TO_LABEL,
    apply_decision,
    load_jsonl,
    reset_review,
    review_stats,
    save_jsonl,
)

DEFAULT_FILE = Path(__file__).parent / "golden.jsonl"
PREVIEW_CHARS = 1500

HELP = (
    "Enter = agree | b/f/d/q = BUG/FEATURE/DOCS/QUESTION | x = drop | "
    "m = full body | o = open in browser | s = skip | quit"
)
GUIDELINE = (
    "BUG: something is broken (incl. CI/build/tooling)   FEATURE: asks for new behavior\n"
    "DOCS: documentation content wrong/missing          QUESTION: asks how to do something"
)


def show(rec: dict, pos: int, total: int, full: bool = False) -> None:
    body = rec["body"] if full else rec["body"][:PREVIEW_CHARS]
    print("\n" + "=" * 78)
    print(f"[{pos}/{total}]  {rec['id']}    label: {rec['label']}  (maintainer: {rec['source_label']})")
    print(f"Title: {rec['title']}")
    print(f"URL:   {rec['url']}")
    print("-" * 78)
    print(textwrap.indent(body or "(empty body)", "  "))
    if not full and len(rec["body"]) > PREVIEW_CHARS:
        print(f"  ... ({len(rec['body']) - PREVIEW_CHARS} more characters, press m)")
    print("-" * 78)


def print_stats(records: list[dict], split: str) -> None:
    in_split = [r for r in records if r["split"] == split or r.get("split_before_review") == split]
    s = review_stats(in_split)
    print(f"\nReviewed {s['reviewed']}/{len(in_split)} issues in '{split}'")
    print(f"  agree: {s['agree']}   relabel: {s['relabel']}   drop: {s['drop']}")
    if s["agreement"] is not None:
        judged = s["agree"] + s["relabel"]
        print(f"  label agreement: {s['agree']}/{judged} = {s['agreement']:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--split", default="test")
    parser.add_argument("--stats", action="store_true", help="print stats and exit")
    parser.add_argument("--redo", nargs="+", metavar="ID", help="re-review these issue ids")
    args = parser.parse_args()

    # Windows consoles can choke on emoji in issue text; replace what can't be shown.
    sys.stdout.reconfigure(errors="replace")

    records = load_jsonl(args.file)
    if args.stats:
        print_stats(records, args.split)
        return

    if args.redo:
        ids = {i: n for n, i in enumerate(r["id"] for r in records)}
        missing = [i for i in args.redo if i not in ids]
        if missing:
            raise SystemExit(f"Unknown issue id(s): {', '.join(missing)}")
        for issue_id in args.redo:
            records[ids[issue_id]] = reset_review(records[ids[issue_id]])
        save_jsonl(args.file, records)

    todo = [i for i, r in enumerate(records) if r["split"] == args.split and not r.get("review")]
    total = sum(r["split"] == args.split or r.get("split_before_review") == args.split
                for r in records)
    print(f"{len(todo)} issues left to review in '{args.split}'.\n{GUIDELINE}\n{HELP}")

    try:
        for n, idx in enumerate(todo, start=total - len(todo) + 1):
            rec = records[idx]
            full = False
            while True:
                show(rec, n, total, full)
                key = input("> ").strip().lower()
                if key == "quit":
                    raise KeyboardInterrupt
                if key == "m":
                    full = True
                    continue
                if key == "o":
                    webbrowser.open(rec["url"])
                    continue
                if key == "s":
                    break
                if key not in ("", "x", *KEY_TO_LABEL):
                    print(HELP)
                    continue
                note = ""
                if key == "x" or (key in KEY_TO_LABEL and KEY_TO_LABEL[key] != rec["label"]):
                    note = input("Why? (one line, Enter to skip) > ")
                records[idx] = apply_decision(rec, key, note)
                save_jsonl(args.file, records)  # save after every answer
                break
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Progress is saved; run the same command to continue.")

    print_stats(records, args.split)


if __name__ == "__main__":
    main()
