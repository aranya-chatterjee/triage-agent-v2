"""Score a classifier on one split of eval/golden.jsonl and save the run.

Usage (PowerShell, from the project root):
    uv run python eval/run_eval.py                             # keyword baseline on dev
    uv run python eval/run_eval.py --classifier always-bug     # dumbest baseline
    uv run python eval/run_eval.py --show-errors 10            # list 10 mistakes
    uv run python eval/run_eval.py --split test                # final numbers only!

Rule: tune on dev. Run test only for numbers you will report.
Each run is saved to eval/runs/ (git-ignored) so you can compare runs over time.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from triage.baselines import always, keyword_classify
from triage.evaluation import evaluate, format_report
from triage.review import load_jsonl

EVAL_DIR = Path(__file__).parent
DEFAULT_FILE = EVAL_DIR / "golden.jsonl"
RUNS_DIR = EVAL_DIR / "runs"

# Name -> classifier. The LLM agent gets added here later.
CLASSIFIERS = {
    "keyword": keyword_classify,
    "always-bug": always("BUG"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classifier", choices=sorted(CLASSIFIERS), default="keyword")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--show-errors", type=int, default=0, metavar="N",
                        help="print up to N misclassified issues")
    parser.add_argument("--no-save", action="store_true", help="don't write to eval/runs/")
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")

    records = [r for r in load_jsonl(args.file) if r["split"] == args.split]
    if args.split == "test":
        print("NOTE: test split. Use these numbers for reporting, not for tuning.\n")

    print(f"Classifier: {args.classifier}   Split: {args.split}\n")
    result = evaluate(records, CLASSIFIERS[args.classifier])
    print(format_report(result))

    if args.show_errors:
        wrong = [p for p in result["predictions"] if not p["correct"]]
        print(f"\nMistakes ({len(wrong)} total, showing {min(len(wrong), args.show_errors)}):")
        for p in wrong[: args.show_errors]:
            print(f"  {p['true']:<9} -> {p['pred']:<9} {p['id']:<28} {p['title'][:60]}")

    if not args.no_save:
        RUNS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = RUNS_DIR / f"{stamp}_{args.classifier}_{args.split}.json"
        run = {"classifier": args.classifier, "split": args.split, "timestamp": stamp, **result}
        out.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved run to {out}")


if __name__ == "__main__":
    main()
