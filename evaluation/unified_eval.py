"""
Unified evaluation entry point for QA tasks.

Usage:
    python -m evaluation.unified_eval \
        --task_type qa \
        --gold_path data/unified/nq_swap.jsonl \
        --pred_path outputs/cocoa_nq_swap.jsonl \
        --metrics em,f1,substring_em
"""

import argparse
import json
import sys
from pathlib import Path

from evaluation.utils import load_gold_data, load_pred_data
from evaluation.metrics.qa_metrics import evaluate_qa


TASK_DEFAULTS = {
    "qa": {"metrics": ["em", "f1", "substring_em"]},
}

TASK_FN = {
    "qa": evaluate_qa,
}


def run_evaluation(task_type: str, gold_path: str, pred_path: str,
                   metrics: list = None, output_path: str = None) -> dict:
    """Run evaluation and return results dict."""
    gold_list = load_gold_data(gold_path)
    pred_dict = load_pred_data(pred_path, task_type=task_type)

    if metrics is None:
        metrics = TASK_DEFAULTS.get(task_type, {}).get("metrics", ["em"])

    print(f"Task type:   {task_type}")
    print(f"Gold data:   {len(gold_list)} samples from {gold_path}")
    print(f"Predictions: {len(pred_dict)} entries from {pred_path}")
    print(f"Metrics:     {metrics}")
    print("-" * 60)

    eval_fn = TASK_FN.get(task_type)
    if eval_fn is None:
        print(f"[ERROR] Unknown task_type: {task_type}")
        sys.exit(1)

    results = eval_fn(gold_list, pred_dict, metrics)

    print("\n=== Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for OptCAD")
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["qa"])
    parser.add_argument("--gold_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated metric names")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    metrics = args.metrics.split(",") if args.metrics else None
    run_evaluation(args.task_type, args.gold_path, args.pred_path,
                   metrics, args.output_path)


if __name__ == "__main__":
    main()
