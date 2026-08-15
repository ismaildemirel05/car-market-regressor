"""
Prediction entry point.

Turns a dict of manually provided car characteristics into a price
prediction, using the full pipeline (imputation -> target encoding ->
one-hot encoding -> RandomForestRegressor) saved by train.py.

The pipeline only contains the "learned" preprocessing steps (medians,
category lists, target means). The deterministic transformations from
cleaning.py (year -> age) still have to be reapplied here, which is why
prepare_input_row() reuses cleaning.year_to_age() instead of
redefining the same logic — keeping training and prediction consistent.

Usage (from a notebook or a REPL):

    from src.model.predict import predict_from_features

    result = predict_from_features({
        "car_brand": "Peugeot",
        "car_model": "208",
        "mileage": 60000,
        "horse_power": 100,
        "gearbox": "Manuelle",
        "fuel": "Diesel",
        "color": "Grise",
        "doors": 5,
        "seats": 5,
        "vehicle_type": "Citadine",
        "department_num": 38,
        "first_release_year": 2018,
    })

    print(result["predicted_price"])
    print(result["imputed_fields"])
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.features.cleaning import year_to_age

MODEL_PATH = "models/model_v1.pkl"

# Columns expected by the pipeline, in the exact shape produced by
# cleaning.build_clean_dataframe() for training (i.e. "id" and "price"
# already removed, "first_release_year" already converted to "age").
FEATURE_COLUMNS = [
    "car_brand",
    "car_model",
    "mileage",
    "horse_power",
    "gearbox",
    "fuel",
    "color",
    "doors",
    "seats",
    "vehicle_type",
    "department_num",
    "age",
]


def load_pipeline(model_path: str = MODEL_PATH):
    """Loads the full trained pipeline (preprocessing + model) from disk."""
    return joblib.load(model_path)


def prepare_input_row(features: dict) -> dict:
    """
    Converts a user-facing dict of car characteristics into the exact
    row shape the pipeline expects.

    Accepts `first_release_year` (e.g. 2018) instead of `age`, so the
    caller never has to do the date arithmetic itself — it mirrors what
    build_clean_dataframe() does for training data.

    Any field not provided is left as None. This is safe: the pipeline's
    imputer falls back to group/global medians for numeric columns, and
    the encoders fall back to the global mean (target encoding) or an
    all-zero dummy row (one-hot encoding) for categorical columns — so a
    partial dict never raises, it just produces a less informed
    prediction.
    """
    row = {
        column: features.get(column)
        for column in FEATURE_COLUMNS
        if column != "age"
    }
    row["age"] = year_to_age(features.get("first_release_year"))
    return row


def predict_from_features(features: dict, model_path: str = MODEL_PATH) -> dict:
    """
    Predicts a car's price from a dict of characteristics.

    Returns:
        {
            "predicted_price": float,
            "imputed_fields": list[str],  # features missing from `features`,
                                           # filled in by the pipeline instead
                                           # of being real provided values
        }
    """
    row = prepare_input_row(features)
    imputed_fields = [column for column, value in row.items() if value is None]

    X = pd.DataFrame([row])
    pipeline = load_pipeline(model_path)

    # FixedCategoryOneHotEncoder drops each encoded column and appends its
    # dummy columns at the end, so the column order coming out of
    # preprocessing depends on the order columns were given in. Rather
    # than having to mirror the training column order by hand here, we
    # reindex by name against what the model actually recorded at fit
    # time (feature_names_in_) — robust no matter what order the raw
    # input row happens to be built in.
    preprocessed = pipeline[:-1].transform(X)
    model = pipeline.named_steps["model"]
    preprocessed = preprocessed[model.feature_names_in_]

    predicted_price = float(model.predict(preprocessed)[0])

    return {
        "predicted_price": predicted_price,
        "imputed_fields": imputed_fields,
    }


if __name__ == "__main__":
    # Quick manual check
    example = {
        "car_brand": "Ford",
        "car_model": "Focus",
        "mileage": 220000,
        "horse_power": 90,
        "gearbox": "1",
        "fuel": "2",
        "color": "gris",
        "doors": 5,
        "seats": 5,
        "vehicle_type": "berline",
        "department_num": 25,
        "first_release_year": 2013,
    }

    result = predict_from_features(example)
    print(f"Predicted price: {result['predicted_price']:.0f} EUR")
    if result["imputed_fields"]:
        print(f"Imputed / missing fields: {', '.join(result['imputed_fields'])}")