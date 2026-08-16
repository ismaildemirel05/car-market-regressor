"""
Methodology validation tools, run BEFORE committing to a final model.

Answers a different question than evaluate.py: not "how good is my final
trained model", but "can I trust the scores I'm measuring, given my
current dataset size and pipeline design".

Two tools are provided:
    - run_cross_validation(): k-fold cross-validation on the full,
      unfitted pipeline, with both train and test scores per fold, to
      diagnose bias vs. variance.
    - plot_learning_curve(): train/test scores as a function of training
      set size, to decide whether scraping more data is likely to help
      more than tuning the current model.

Both operate on the pipeline BEFORE any .fit() call. cross_validate() and
learning_curve() clone and refit the whole pipeline (imputer -> target
encoder -> one-hot -> model) from scratch on each fold's training portion
only, so the stateful transformers (OutlierMedianImputer,
SmoothedTargetEncoder, FixedCategoryOneHotEncoder) never see their own
test fold at fit time. This is what keeps the estimate free of data
leakage — but it only holds if the pipeline passed in has NOT already
been fit on the full dataset beforehand.

Usage :
    python -m src.model.diagnostics
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.model.train import build_full_pipeline, load_dataset

CV_FOLDS = 5
TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]
RANDOM_STATE = 0

# sklearn scorers are "greater is better" by convention, so error metrics
# are prefixed with neg_ and sign-flipped back when displayed.
SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
    "MAPE": "neg_mean_absolute_percentage_error",
}


def run_cross_validation(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = CV_FOLDS
) -> dict:
    """
    Runs k-fold cross-validation on the full, UNFITTED pipeline.

    Returns the raw dict from cross_validate (one array per metric per
    fold, train and test), with error metrics already flipped back to
    positive values for readability.
    """
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1,
    )

    # Flip every error metric (all but R2) back to positive values.
    for metric_name in SCORING:
        if metric_name == "R2":
            continue
        for split in ("train", "test"):
            key = f"{split}_{metric_name}"
            results[key] = -results[key]

    return results


def print_cv_summary(results: dict) -> None:
    """
    Prints mean +/- std for every metric, train and test side by side,
    so bias (both columns bad) vs. variance (big gap between columns)
    is directly readable.
    """
    print(f"\n--- Cross-validation summary ({len(results['fit_time'])} folds) ---")
    header = f"{'Metric':<8} {'Train (mean +/- std)':<28} {'Test (mean +/- std)':<28}"
    print(header)
    print("-" * len(header))

    for metric_name in SCORING:
        train_scores = results[f"train_{metric_name}"]
        test_scores = results[f"test_{metric_name}"]

        if metric_name == "MAPE":
            train_str = f"{train_scores.mean():.2%} +/- {train_scores.std():.2%}"
            test_str = f"{test_scores.mean():.2%} +/- {test_scores.std():.2%}"
        elif metric_name == "R2":
            train_str = f"{train_scores.mean():.3f} +/- {train_scores.std():.3f}"
            test_str = f"{test_scores.mean():.3f} +/- {test_scores.std():.3f}"
        else:
            train_str = f"{train_scores.mean():,.0f} +/- {train_scores.std():,.0f}"
            test_str = f"{test_scores.mean():,.0f} +/- {test_scores.std():,.0f}"

        print(f"{metric_name:<8} {train_str:<28} {test_str:<28}")


def plot_learning_curve(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes=None,
    cv: int = CV_FOLDS,
):
    """
    Plots train and test MAE as a function of training set size.

    If both curves are still clearly converging at the rightmost point
    (full dataset size), collecting more data is likely to help more
    than tuning the current model. If they have already plateaued well
    before the full dataset size, more data alone probably won't move
    the needle much.
    """
    if train_sizes is None:
        train_sizes = TRAIN_SIZES

    absolute_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # Flip MAE back to positive, then average across folds for each size.
    train_mae_mean = -train_scores.mean(axis=1)
    train_mae_std = train_scores.std(axis=1)
    test_mae_mean = -test_scores.mean(axis=1)
    test_mae_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(absolute_sizes, train_mae_mean, "o-", label="Train MAE", color="tab:blue")
    ax.fill_between(
        absolute_sizes,
        train_mae_mean - train_mae_std,
        train_mae_mean + train_mae_std,
        alpha=0.15,
        color="tab:blue",
    )

    ax.plot(absolute_sizes, test_mae_mean, "o-", label="Test MAE", color="tab:orange")
    ax.fill_between(
        absolute_sizes,
        test_mae_mean - test_mae_std,
        test_mae_mean + test_mae_std,
        alpha=0.15,
        color="tab:orange",
    )

    ax.set_xlabel("Training set size (number of rows)")
    ax.set_ylabel("MAE (EUR)")
    ax.set_title("Learning curve")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()

    return absolute_sizes, train_mae_mean, test_mae_mean


if __name__ == "__main__":
    df = load_dataset()
    print(f"{len(df)} annonces loaded for diagnostics.")

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    X = df.drop(columns=["price", "id"])
    y = df["price"]

    pipeline = build_full_pipeline()

    cv_results = run_cross_validation(pipeline, X, y)
    print_cv_summary(cv_results)

    plot_learning_curve(pipeline, X, y)