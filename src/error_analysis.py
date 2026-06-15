"""Error analysis.

A results table tells you *how well* a model does. Error analysis tells you
*where and why* it fails — which is what separates an engineering demo from a
piece of research. Here we re-run the transformer, collect its misclassified
reviews, and surface patterns: confident-but-wrong predictions, and the length
distribution of errors versus correct cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import SplitData

MODEL_NAME = "distilbert-base-uncased"


def _has_saved_model(path: Path) -> bool:
    """A directory only counts as a loadable checkpoint if it has a config."""
    return (path / "config.json").exists()


def run_error_analysis(
    data: SplitData,
    model=None,
    tokenizer=None,
    model_dir: str = "./distilbert_out",
    results_path: str = "results",
    top_k: int = 15,
) -> dict:
    """Identify and characterize the transformer's worst errors.

    Pass the in-memory fine-tuned `model` and `tokenizer` from run_transformer
    for a true analysis of the trained model. If they are not provided, the
    function falls back to a saved checkpoint in `model_dir`, and if that does
    not exist either, it loads base weights only to demonstrate the analysis
    structure (the error patterns will not reflect a trained model in that case).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None or tokenizer is None:
        ckpt = Path(model_dir)
        load_from = str(ckpt) if _has_saved_model(ckpt) else MODEL_NAME
        if load_from == MODEL_NAME:
            print(
                "[error_analysis] No saved checkpoint found; loading base "
                "weights. For trained-model analysis, pass the model/tokenizer "
                "from run_transformer directly."
            )
        tokenizer = AutoTokenizer.from_pretrained(load_from)
        model = AutoModelForSequenceClassification.from_pretrained(
            load_from, num_labels=2
        )

    model = model.to(device)
    model.eval()

    texts = data.test_texts
    labels = data.test_labels

    probs_pos: list[float] = []
    preds: list[int] = []

    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_pos.extend(p.tolist())
            preds.extend((p > 0.5).astype(int).tolist())

    errors = []
    for i, (true, pred, p) in enumerate(zip(labels, preds, probs_pos)):
        if true != pred:
            confidence = p if pred == 1 else (1 - p)
            errors.append(
                {
                    "index": i,
                    "true": true,
                    "pred": pred,
                    "confidence": round(float(confidence), 4),
                    "length_words": len(texts[i].split()),
                    "text_preview": texts[i][:200],
                }
            )

    # Sort by confidence: the most confident errors are the most instructive.
    errors.sort(key=lambda e: e["confidence"], reverse=True)

    correct_lengths = [
        len(texts[i].split()) for i in range(len(texts)) if labels[i] == preds[i]
    ]
    error_lengths = [e["length_words"] for e in errors]

    summary = {
        "n_errors": len(errors),
        "error_rate": round(len(errors) / len(texts), 4),
        "mean_length_correct": round(float(np.mean(correct_lengths)), 1)
        if correct_lengths
        else None,
        "mean_length_error": round(float(np.mean(error_lengths)), 1)
        if error_lengths
        else None,
        "most_confident_errors": errors[:top_k],
    }

    out_dir = Path(results_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "error_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Errors: {summary['n_errors']} ({summary['error_rate']:.1%})")
    print(
        f"Mean review length — correct: {summary['mean_length_correct']}, "
        f"error: {summary['mean_length_error']}"
    )
    print(f"Saved {top_k} most-confident errors to {out_dir/'error_analysis.json'}")

    return summary
