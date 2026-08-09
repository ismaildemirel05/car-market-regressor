"""
Feature encoding layer.

Turns categorical columns into numeric ones, packaged as two
scikit-learn compatible transformers:
    - FixedCategoryOneHotEncoder : one-hot encoding for low-cardinality
      columns (car_brand, fuel, gearbox, vehicle_type, region), with the
      set of categories learned at fit time so train/test/future data
      always produce the exact same columns.
    - SmoothedTargetEncoder : target encoding for high-cardinality columns
      (car_model, department_num), replacing each category with a
      smoothed average of the target (price), learned on the training
      set only to avoid leakage.

Both follow the scikit-learn estimator API (BaseEstimator + TransformerMixin)
so they can be chained inside a Pipeline alongside the imputation
transformer and the model itself.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- Columns handled by simple one-hot encoding (low cardinality) ---
ONEHOT_COLUMNS = [
    "car_brand",
    "fuel",
    "gearbox",
    "vehicle_type",
    "color",
]

# --- Columns handled by target encoding (high cardinality) ---
TARGET_ENCODED_COLUMNS = [
    "car_model",
    "department_num",
]

# Smoothing strength for target encoding: higher m means rare categories
# are pulled closer to the global mean. See SmoothedTargetEncoder docstring.
TARGET_ENCODING_SMOOTHING = 10


class FixedCategoryOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encodes a fixed list of low-cardinality columns.

    fit() records the categories seen in the training data for each
    column; transform() forces every DataFrame to use that exact same
    set of categories before calling get_dummies, so train, test, and
    future single-row predictions always end up with identical columns
    — even if a category is missing, or a never-seen category appears.

    Missing values (NaN) get their own explicit dummy column
    (e.g. color_nan) rather than silently ending up as all-zeros across
    every category column — this lets the model learn from the fact that
    a value was missing, instead of treating it as indistinguishable
    from "none of the known categories".
    """

    def __init__(self, columns=None):
        self.columns = columns if columns is not None else list(ONEHOT_COLUMNS)

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {
            column: sorted(X[column].dropna().unique().tolist())
            for column in self.columns
            if column in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, known_categories in self.categories_.items():
            if column not in X.columns:
                continue

            X[column] = pd.Categorical(X[column], categories=known_categories)
            dummies = pd.get_dummies(X[column], prefix=column, dtype=int, dummy_na=True)
            X = pd.concat([X.drop(columns=[column]), dummies], axis=1)

        return X


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target-encodes a fixed list of high-cardinality columns, using a
    smoothed average of the target rather than the raw category mean:

        encoded = (n * category_mean + smoothing * global_mean) / (n + smoothing)

    where n is the number of training rows for that category. A category
    seen only once or twice ends up close to the global mean instead of
    taking its own (unreliable) average at face value.

    Unlike FixedCategoryOneHotEncoder, fit() requires y (the target,
    e.g. price) since the whole point is to learn the relationship
    between each category and the target — this is standard for a
    supervised transformer in scikit-learn.
    """

    def __init__(self, columns=None, smoothing: float = TARGET_ENCODING_SMOOTHING):
        self.columns = columns if columns is not None else list(TARGET_ENCODED_COLUMNS)
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.global_mean_ = y.mean()
        self.target_maps_ = {}

        for column in self.columns:
            if column not in X.columns:
                continue

            stats = y.groupby(X[column]).agg(["mean", "count"])
            smoothed = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            self.target_maps_[column] = smoothed.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, encoding_map in self.target_maps_.items():
            if column not in X.columns:
                continue
            # Categories unseen at fit time fall back to the global mean
            # instead of producing NaN.
            X[column] = X[column].map(encoding_map).fillna(self.global_mean_)

        return X


if __name__ == "__main__":
    # Quick manual check with a tiny fake dataset
    fake_df = pd.DataFrame({
        "car_brand": ["Peugeot", "Renault", "Peugeot", "BMW", "BMW"],
        "car_model": ["208", "Clio", "208", "X1", "X3"],
        "fuel": ["Diesel", "Essence", "Diesel", "Essence", "Diesel"],
    })
    fake_price = pd.Series([10000, 12000, 9000, 15000, 20000])

    onehot = FixedCategoryOneHotEncoder()
    target_enc = SmoothedTargetEncoder()

    result = onehot.fit_transform(fake_df)
    result = target_enc.fit_transform(result, fake_price)
    print(result)