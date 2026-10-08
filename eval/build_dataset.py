"""Build eval/golden.jsonl from maintainer-labeled GitHub issues.

Usage (PowerShell, from the project root):
    uv run python eval/build_dataset.py
    uv run python eval/build_dataset.py --per-class 30 --seed 42
    uv run python eval/build_dataset.py --force     # overwrite an existing golden.jsonl

Safety: refuses to overwrite golden.jsonl by default, because after you hand-verify
the test split, re-running would silently throw that work away.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

from triage.config import get_settings, require
from triage.dataset import SOURCES, fetch_issues, stratified_split, summary_table, to_record

log = logging.getLogger("build_dataset")

DEFAULT_OUT = Path(__file__).parent / "golden.jsonl"
# The Search API allows 30 requests/minute with a token; 2.5s between calls stays under it.
PAUSE_SECONDS = 2.5


def collect(per_class: int, token: str) -> list[dict]:
    records: dict[str, dict] = {}  # keyed by id, so an issue found twice is kept once
    for repo, label_map in SOURCES.items():
        for repo_label, our_label in label_map.items():
            try:
                raw_issues = fetch_issues(repo, repo_label, token, n=per_class * 2)
            except requests.HTTPError as e:
                log.error("%s [%s]: HTTP %s, skipping", repo, repo_label, e.response.status_code)
                continue

            kept = 0
            for raw in raw_issues:
                rec = to_record(raw, repo, label_map)
                if rec is None or rec["id"] in records:
                    continue
                records[rec["id"]] = rec
                kept += 1
                if kept == per_class:
                    break

            level = logging.WARNING if kept < per_class else logging.INFO
            log.log(level, "%-22s %-16s -> %-8s kept %d/%d (fetched %d)",
                    repo, repo_label, our_label, kept, per_class, len(raw_issues))
            time.sleep(PAUSE_SECONDS)
    return list(records.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=30, help="issues per class per repo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists. Re-run with --force to overwrite it.")

    token = require(get_settings().github_token, "GITHUB_TOKEN")
    records = stratified_split(collect(args.per_class, token), seed=args.seed)
    records.sort(key=lambda r: (r["split"], r["label"], r["id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} issues to {args.out}\n")
    print(summary_table(records))


if __name__ == "__main__":
    main()
