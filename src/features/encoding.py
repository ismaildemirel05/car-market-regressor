"""
Feature encoding layer.

Turns the clean, typed DataFrame produced by `cleaning.py` into a fully
numeric DataFrame ready for model training.

This module deliberately separates two steps:
    - fit_encoders(df_train, ...)  -> learns encoding parameters from the
      training data ONLY (category lists, target means, etc.)
    - apply_encoders(df, encoders) -> applies the previously learned
      parameters to any DataFrame (train, test, or a single new ad)

This fit/transform split avoids data leakage (e.g. computing a category's
average price using rows that are also used for evaluation) and guarantees
that train and test end up with the exact same columns, even if some rare
category only appears in one of the two sets.
"""

from dataclasses import dataclass, field

import pandas as pd

# --- Columns handled by simple one-hot encoding (low cardinality) ---
ONEHOT_COLUMNS = [
    "car_brand",
    "fuel",
    "gearbox",
    "vehicle_type",
    "color",
    "vehicle_emissions",
    "vehicle_damage",
    # "region",
]

# --- Columns handled by target encoding (high cardinality) ---
TARGET_ENCODED_COLUMNS = [
    "car_model",
    "department_num",
]

# Smoothing strength for target encoding: higher m means rare categories
# are pulled closer to the global mean. See docstring of target_encode_fit.
TARGET_ENCODING_SMOOTHING = 10


@dataclass
class Encoders:
    """
    Holds every parameter learned from the training set.
    Must be produced by fit_encoders() and passed unchanged to
    apply_encoders() for train, test, and any future prediction.
    """
    onehot_categories: dict = field(default_factory=dict)   # column -> list of known categories
    target_maps: dict = field(default_factory=dict)         # column -> {category: encoded_value}
    global_mean: float = 0.0                                # fallback for unseen categories


def target_encode_fit(
    series: pd.Series,
    target: pd.Series,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
) -> tuple[dict, float]:
    """
    Learns a smoothed target encoding map for one categorical column.

    For each category, the encoded value is a weighted average between:
        - the category's own mean target value
        - the global mean target value

    The weighting depends on how many rows the category has (n):
        encoded = (n * category_mean + smoothing * global_mean) / (n + smoothing)

    A category seen only once or twice ends up very close to the global
    mean instead of taking its own (unreliable) average at face value.
    """
    global_mean = target.mean()

    stats = target.groupby(series).agg(["mean", "count"])
    smoothed = (
        stats["count"] * stats["mean"] + smoothing * global_mean
    ) / (stats["count"] + smoothing)

    return smoothed.to_dict(), global_mean


def target_encode_apply(
    series: pd.Series,
    encoding_map: dict,
    global_mean: float,
) -> pd.Series:
    """
    Applies a previously learned target encoding map to a column.
    Categories not present in the map (unseen at fit time) fall back
    to the global mean instead of raising an error or producing NaN.
    """
    return series.map(encoding_map).fillna(global_mean)


def fit_encoders(df_train: pd.DataFrame, target_column: str = "price") -> Encoders:
    """
    Learns every encoding parameter from the training set only.
    Must be called once, on df_train, before any call to apply_encoders.
    """
    encoders = Encoders()
    target = df_train[target_column]

    for column in ONEHOT_COLUMNS:
        if column in df_train.columns:
            encoders.onehot_categories[column] = sorted(
                df_train[column].dropna().unique().tolist()
            )

    for column in TARGET_ENCODED_COLUMNS:
        if column in df_train.columns:
            encoding_map, global_mean = target_encode_fit(df_train[column], target)
            encoders.target_maps[column] = encoding_map
            encoders.global_mean = global_mean

    return encoders


def apply_encoders(df: pd.DataFrame, encoders: Encoders) -> pd.DataFrame:
    """
    Applies encoding parameters learned by fit_encoders() to any
    DataFrame (train, test, or a single new ad to predict).

    Returns a new DataFrame; does not mutate the input.
    """
    df = df.copy()

    for column, known_categories in encoders.onehot_categories.items():
        if column not in df.columns:
            continue

        # Force the column's categories to exactly match what was seen
        # at fit time, so get_dummies always produces the same columns,
        # regardless of which categories are present in df.
        df[column] = pd.Categorical(df[column], categories=known_categories)
        dummies = pd.get_dummies(df[column], prefix=column, dtype=int)
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)

    for column, encoding_map in encoders.target_maps.items():
        if column not in df.columns:
            continue
        df[column] = target_encode_apply(df[column], encoding_map, encoders.global_mean)

    return df


if __name__ == "__main__":
    # Quick manual check with a tiny fake dataset
    fake_df = pd.DataFrame({
        "price": [10000, 12000, 9000, 15000, 20000],
        "car_brand": ["Peugeot", "Renault", "Peugeot", "BMW", "BMW"],
        "car_model": ["208", "Clio", "208", "X1", "X3"],
        "fuel": ["Diesel", "Essence", "Diesel", "Essence", "Diesel"],
    })

    encoders = fit_encoders(fake_df)
    result = apply_encoders(fake_df, encoders)
    print(result)