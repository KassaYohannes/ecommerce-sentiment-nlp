"""Classic baseline: TF-IDF features + logistic regression.

A baseline matters. Before claiming a transformer is worth its cost, you should
know what a simple, fast, interpretable model achieves on the same data. This is
the same discipline used in serious empirical research: establish the floor first,
then measure how much the heavier method actually buys you.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import SplitData
from .evaluate import evaluate_predictions


def build_baseline() -> Pipeline:
    """Construct a TF-IDF + logistic regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_baseline(data: SplitData, results_path: str | None = None) -> dict:
    """Train the baseline and evaluate it on the test split.

    Returns a metrics dict (accuracy, macro-F1, per-class scores).
    """
    model = build_baseline()
    model.fit(data.train_texts, data.train_labels)
    preds = model.predict(data.test_texts)

    metrics = evaluate_predictions(
        y_true=data.test_labels,
        y_pred=list(preds),
        model_name="tfidf_logreg",
        results_path=results_path,
    )
    return metrics


if __name__ == "__main__":
    from .data import load_review_data

    d = load_review_data(train_size=2000, test_size=500)
    m = run_baseline(d)
    print(m)
