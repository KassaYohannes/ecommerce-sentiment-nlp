"""Shared evaluation utilities.

Both the baseline and the transformer report through this module so the numbers
are directly comparable: same metrics, same format, same saved artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABEL_NAMES = ["negative", "positive"]


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    model_name: str,
    results_path: str | None = None,
) -> dict:
    """Compute metrics, optionally save a JSON report and confusion matrix plot."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()

    report = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, output_dict=True
    )

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {
            LABEL_NAMES[i]: round(per_class_f1[i], 4) for i in range(len(LABEL_NAMES))
        },
        "full_report": report,
    }

    if results_path is not None:
        out_dir = Path(results_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / f"{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        _plot_confusion(y_true, y_pred, model_name, out_dir)

    return metrics


def _plot_confusion(
    y_true: list[int], y_pred: list[int], model_name: str, out_dir: Path
) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {model_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_confusion.png", dpi=120)
    plt.close(fig)
