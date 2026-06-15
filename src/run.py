"""End-to-end experiment runner.

Runs the baseline and the transformer on the same data split, saves all metrics
and plots to results/, and prints a side-by-side comparison plus a short error
analysis. This is the single entry point a reviewer (or you) runs to reproduce
the whole study.

Usage:
    python -m src.run --train-size 8000 --test-size 2000 --epochs 2
"""

from __future__ import annotations

import argparse

from .baseline import run_baseline
from .data import load_review_data
from .error_analysis import run_error_analysis
from .transformer_model import run_transformer

RESULTS_DIR = "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="E-commerce sentiment experiment")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="run only the baseline (fast, for a quick check)",
    )
    args = parser.parse_args()

    print("Loading data...")
    data = load_review_data(
        train_size=args.train_size, test_size=args.test_size, seed=args.seed
    )
    print(f"Train: {len(data.train_texts)}  Test: {len(data.test_texts)}")

    print("\n=== Baseline: TF-IDF + Logistic Regression ===")
    baseline_metrics = run_baseline(data, results_path=RESULTS_DIR)
    print(
        f"accuracy={baseline_metrics['accuracy']}  "
        f"macro_f1={baseline_metrics['macro_f1']}"
    )

    transformer_metrics = None
    if not args.skip_transformer:
        print("\n=== Transformer: DistilBERT fine-tune ===")
        transformer_metrics, model, tokenizer = run_transformer(
            data, epochs=args.epochs, results_path=RESULTS_DIR, return_model=True
        )
        print(
            f"accuracy={transformer_metrics['accuracy']}  "
            f"macro_f1={transformer_metrics['macro_f1']}"
        )

        print("\n=== Error analysis (DistilBERT) ===")
        run_error_analysis(
            data, model=model, tokenizer=tokenizer, results_path=RESULTS_DIR
        )

    print("\n=== Summary ===")
    print(f"Baseline    macro-F1: {baseline_metrics['macro_f1']}")
    if transformer_metrics is not None:
        print(f"DistilBERT  macro-F1: {transformer_metrics['macro_f1']}")
        delta = round(
            transformer_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4
        )
        print(f"Improvement from fine-tuning: {delta:+}")
    print(f"\nAll artifacts saved to ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
