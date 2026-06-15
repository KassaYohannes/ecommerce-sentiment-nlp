"""Transformer model: fine-tune DistilBERT for binary sentiment classification.

This is the main model. It is compared against the TF-IDF baseline through the
shared evaluate module, so the comparison is apples-to-apples on the same split.

Note on implementation: we deliberately avoid datasets.set_format("torch"),
which on some environments triggers an unrelated torchvision import bug. Instead
we tokenize into a plain dict-backed Dataset and let DataCollatorWithPadding
handle batching and tensor conversion at training time.
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .data import SplitData
from .evaluate import evaluate_predictions

MODEL_NAME = "distilbert-base-uncased"


def _tokenize_dataset(texts: list[str], labels: list[int], tokenizer) -> Dataset:
    ds = Dataset.from_dict({"text": texts, "labels": labels})

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    ds = ds.map(tok, batched=True)
    ds = ds.remove_columns(["text"])
    # No set_format("torch") here on purpose — the data collator handles tensors.
    return ds


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def run_transformer(
    data: SplitData,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    results_path: str | None = None,
    output_dir: str = "./distilbert_out",
) -> dict:
    """Fine-tune DistilBERT and evaluate on the test split."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    train_ds = _tokenize_dataset(data.train_texts, data.train_labels, tokenizer)
    test_ds = _tokenize_dataset(data.test_texts, data.test_labels, tokenizer)

    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=_compute_metrics,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()

    pred_output = trainer.predict(test_ds)
    preds = np.argmax(pred_output.predictions, axis=-1).tolist()

    metrics = evaluate_predictions(
        y_true=data.test_labels,
        y_pred=preds,
        model_name="distilbert",
        results_path=results_path,
    )
    return metrics


if __name__ == "__main__":
    from .data import load_review_data

    d = load_review_data(train_size=500, test_size=200)
    m = run_transformer(d, epochs=1)
    print(m)
