"""
Preprocessing orchestration layer.

Assembles the imputation and encoding transformers into a single
scikit-learn Pipeline, so the whole preprocessing sequence is fitted
and applied with one call — and can later be extended with the model
itself as a final Pipeline step (see train.py).

Order matters: imputation must run before one-hot encoding, since
grouped median imputation needs car_brand in its original text form,
which one-hot encoding would otherwise have already turned into dummy
columns. The target encoder can run in any order relative to the
one-hot step, since they touch different columns.
"""

from sklearn.pipeline import Pipeline

from src.features.encoding import FixedCategoryOneHotEncoder, SmoothedTargetEncoder
from src.features.imputation import OutlierMedianImputer


def build_preprocessing_pipeline() -> Pipeline:
    """
    Builds an unfitted preprocessing Pipeline:
        imputation -> target encoding -> one-hot encoding

    Call .fit(X_train, y_train) once on the training data, then
    .transform(X) on train, test, or any future single row to predict.
    """
    return Pipeline([
        ("imputer", OutlierMedianImputer()),
        ("target_encoder", SmoothedTargetEncoder()),
        ("onehot", FixedCategoryOneHotEncoder()),
    ])


if __name__ == "__main__":
    import pandas as pd

    # Quick manual check with a tiny fake dataset
    fake_df = pd.DataFrame({
        "car_brand": ["Peugeot", "Renault", "Peugeot", "BMW", "BMW"],
        "car_model": ["208", "Clio", "208", "X1", "X3"],
        "fuel": ["Diesel", "Essence", "Diesel", "Essence", "Diesel"],
        "mileage": [50000, None, 60000, None, 20000],
        "horse_power": [100, 90, None, 190, 250],
        "seats": [5, 5, 999999, 5, 4],
        "doors": [5, 999999, 5, 3, 3],
    })
    fake_price = pd.Series([10000, 12000, 9000, 15000, 20000])

    pipeline = build_preprocessing_pipeline()
    result = pipeline.fit_transform(fake_df, fake_price)
    print(result)