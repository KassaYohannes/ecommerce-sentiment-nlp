"""Data loading and preprocessing for e-commerce review sentiment classification.

Uses the Amazon Polarity dataset (binary sentiment) from the Hugging Face Hub.
The dataset is downloaded at runtime and is NOT redistributed in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset


@dataclass
class SplitData:
    """A simple container for a train/test text-label split."""

    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]


def load_review_data(
    train_size: int = 8000,
    test_size: int = 2000,
    seed: int = 42,
) -> SplitData:
    """Load a subsample of the Amazon Polarity review dataset.

    We subsample deliberately so the project trains in a reasonable time on a
    single GPU / CPU. The full dataset has 3.6M training rows; for a focused
    demonstration a balanced subsample is sufficient and keeps the experiment
    reproducible on modest hardware.

    Args:
        train_size: number of training examples to sample.
        test_size: number of test examples to sample.
        seed: random seed for reproducible sampling.

    Returns:
        SplitData with text and integer labels (0 = negative, 1 = positive).
    """
    rng = np.random.default_rng(seed)

    ds = load_dataset("fancyzhx/amazon_polarity")

    train_full = ds["train"]
    test_full = ds["test"]

    train_idx = rng.choice(len(train_full), size=train_size, replace=False)
    test_idx = rng.choice(len(test_full), size=test_size, replace=False)

    train = train_full.select(train_idx.tolist())
    test = test_full.select(test_idx.tolist())

    # The 'content' field is the review body; 'title' is the review headline.
    # We concatenate title + content to give the model the full signal.
    def join_text(row: dict) -> str:
        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        return f"{title}. {content}".strip()

    train_texts = [join_text(r) for r in train]
    test_texts = [join_text(r) for r in test]

    return SplitData(
        train_texts=train_texts,
        train_labels=list(train["label"]),
        test_texts=test_texts,
        test_labels=list(test["label"]),
    )


if __name__ == "__main__":
    data = load_review_data(train_size=100, test_size=20)
    print(f"Loaded {len(data.train_texts)} train, {len(data.test_texts)} test")
    print("Example:", data.train_texts[0][:160])
    print("Label:", data.train_labels[0])
