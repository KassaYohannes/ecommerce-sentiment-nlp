# E-commerce Review Sentiment Classification

Binary sentiment classification on product reviews, comparing a classic
TF-IDF + logistic regression baseline against a fine-tuned DistilBERT transformer.

The goal of this project is not just to fine-tune a transformer — it is to do so
the way an empirical study should be run: establish a baseline first, compare on
an identical split, report calibrated metrics (not just accuracy), and analyze
*where the model fails*, not only how often.

## Why this project

Product reviews are a core source of unstructured text in any e-commerce
platform. Turning that text into reliable sentiment signal is a foundational NLP
task with direct business value (ranking, moderation, recommendation, trend
detection). This repo implements that task end-to-end with a reproducible,
research-style methodology.

## Approach

| Stage | What | Why |
|-------|------|-----|
| Baseline | TF-IDF (1–2 grams) + logistic regression | Establishes the performance floor before spending compute on a transformer |
| Main model | Fine-tuned `distilbert-base-uncased` | Measures how much contextual representations actually add |
| Evaluation | Accuracy, macro-F1, per-class F1, confusion matrix | Accuracy alone hides class-imbalance failures; macro-F1 and per-class scores expose them |
| Error analysis | Most-confident misclassifications + length distribution | Surfaces *why* the model fails — the part that turns a demo into research |

Both models report through the same `evaluate` module, so the comparison is
strictly apples-to-apples on one fixed train/test split.

## Dataset

Amazon Polarity (binary sentiment), loaded at runtime from the Hugging Face Hub.
The full set has 3.6M training rows; this project uses a reproducible balanced
subsample (default 8k train / 2k test) so the experiment runs on modest hardware.

**The dataset is not redistributed in this repository.** It is downloaded on
first run via the `datasets` library.

## Project structure

```
src/
  data.py               # load + preprocess a reproducible review subsample
  baseline.py           # TF-IDF + logistic regression
  transformer_model.py  # DistilBERT fine-tuning
  evaluate.py           # shared metrics + confusion matrix (used by both models)
  error_analysis.py     # most-confident errors + length analysis
  run.py                # single entry point: runs the full comparison
results/                # metrics JSON + plots are written here
```

## How to run

```bash
pip install -r requirements.txt

# Full experiment (baseline + DistilBERT + error analysis)
python -m src.run --train-size 8000 --test-size 2000 --epochs 2

# Quick baseline-only check (no GPU needed, runs in under a minute)
python -m src.run --skip-transformer --train-size 4000 --test-size 1000
```

All metrics and plots are written to `results/`.

## Results

Trained on a balanced 8,000-review subsample, evaluated on 2,000 held-out reviews
(seed 42, DistilBERT fine-tuned for 2 epochs).

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 0.8755 | 0.8754 |
| DistilBERT (2 epochs) | 0.9335 | 0.9335 |

Fine-tuning DistilBERT improved macro-F1 by **+0.0581** over the baseline.

### Error analysis

On the test split the fine-tuned model made 133 errors (6.7% error rate).
Misclassified reviews were slightly longer on average than correctly classified
ones (81.0 vs 78.3 words), suggesting longer, more nuanced reviews — where
positive and negative sentiment often coexist — are the harder cases. The 15
most-confident misclassifications are saved in `results/error_analysis.json`
for inspection.

These results match expectations: a strong TF-IDF baseline already captures most
of the signal in binary sentiment, and contextual representations from a
fine-tuned transformer add a meaningful but bounded improvement on the harder,
more ambiguous reviews.

## Methodology notes

- **Fixed seed (42)** for the data subsample and training, so runs are reproducible.
- **Same split for both models** — the only variable is the model, not the data.
- **Macro-F1 as the headline metric** rather than accuracy, because it weights
  both sentiment classes equally and does not flatter a model that simply
  predicts the majority class.
- **Title + body concatenation** — the review headline often carries the
  strongest sentiment signal, so it is prepended to the body rather than discarded.

## Possible extensions

- Multi-class star-rating prediction (1–5) instead of binary sentiment
- Calibration analysis (reliability diagrams) on the transformer's probabilities
- A small ablation on `max_length` and learning rate

## License

MIT — see `LICENSE`.
