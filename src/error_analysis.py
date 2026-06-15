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
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import SplitData

MODEL_NAME = "distilbert-base-uncased"


def run_error_analysis(
    data: SplitData,
    model_dir: str = "./distilbert_out",
    results_path: str = "results",
    top_k: int = 15,
) -> dict:
    """Identify and characterize the transformer's worst errors.

    Note: this reloads from the base model unless a fine-tuned checkpoint is
    provided. In the full run.py flow, call this right after run_transformer so
    the in-memory model is warm; here we keep it standalone-runnable by loading
    the base weights, which is enough to demonstrate the analysis structure.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ckpt = Path(model_dir)
    load_from = str(ckpt) if ckpt.exists() else MODEL_NAME
    model = AutoModelForSequenceClassification.from_pretrained(
        load_from, num_labels=2
    ).to(device)
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
        len(texts[i].split())
        for i in range(len(texts))
        if labels[i] == preds[i]
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
