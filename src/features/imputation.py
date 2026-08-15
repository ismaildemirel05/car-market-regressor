"""
Imputation layer.

Handles two related problems on numeric columns, packaged as a single
scikit-learn compatible transformer:
    1. Outlier detection: some values are not missing, but are clearly
       impossible (e.g. seats=999999, doors=999999) — likely placeholder
       values used by the source website or a parsing artifact. These are
       first turned into NaN using fixed plausibility ranges, so they get
       treated the same way as genuinely missing values.
    2. Median imputation: NaN values (whether originally missing or just
       flagged as outliers above) are filled using the median of their
       group (e.g. median mileage for the row's car_brand), learned on
       the training set only, with a fallback to the global median.

Implemented as a scikit-learn transformer (BaseEstimator + TransformerMixin)
so it can be chained inside a Pipeline alongside the encoding transformers
and the model itself.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- Plausibility ranges used to flag outliers as missing values ---
# (column -> (min_valid, max_valid), inclusive). Values outside this range
# are treated as NaN before imputation, since they are not real values
# (typically default/placeholder codes from the source website).
OUTLIER_RANGES = {
    "seats": (1, 9),
    "doors": (1, 6),
    "mileage": (0, 500_000),
    "horse_power": (1, 2000),
    "age": (0, 50),
}

# --- Numeric columns imputed via median, grouped by a reference column ---
IMPUTE_COLUMNS = [
    "mileage",
    "horse_power",
    "seats",
    "doors",
]
IMPUTE_GROUP_COLUMN = "car_brand"


class OutlierMedianImputer(BaseEstimator, TransformerMixin):
    """
    Marks implausible numeric values as NaN, then fills every NaN with
    the median of the row's group (e.g. median mileage per car_brand),
    falling back to the global median when the group is unknown or has
    no valid value at all.

    Follows the scikit-learn estimator API: fit() learns group_medians_
    and global_medians_ from the training data only; transform() applies
    them to any DataFrame (train, test, or a single new ad to predict).
    """

    def __init__(
        self,
        columns=None,
        group_column=IMPUTE_GROUP_COLUMN,
        outlier_ranges=None,
    ):
        # Defaults are assigned in __init__ (not as mutable class defaults)
        # per scikit-learn's convention for estimator parameters.
        self.columns = columns if columns is not None else list(IMPUTE_COLUMNS)
        self.group_column = group_column
        self.outlier_ranges = outlier_ranges if outlier_ranges is not None else dict(OUTLIER_RANGES)

    def _mark_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, (min_valid, max_valid) in self.outlier_ranges.items():
            if column not in X.columns:
                continue
            out_of_range = ~X[column].between(min_valid, max_valid) & X[column].notna()
            X.loc[out_of_range, column] = None
        return X

    def fit(self, X: pd.DataFrame, y=None):
        X = self._mark_outliers(X)

        self.group_medians_ = {}
        self.global_medians_ = {}

        for column in self.columns:
            if column in X.columns and self.group_column in X.columns:
                self.group_medians_[column] = X.groupby(self.group_column)[column].median().to_dict()
                self.global_medians_[column] = X[column].median()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._mark_outliers(X)

        for column, group_medians in self.group_medians_.items():
            if column not in X.columns or self.group_column not in X.columns:
                continue

            missing = X[column].isna()
            fallback_values = X.loc[missing, self.group_column].map(group_medians)
            X.loc[missing, column] = fallback_values.fillna(self.global_medians_[column])

        return X


if __name__ == "__main__":
    # Quick manual check with a tiny fake dataset
    fake_df = pd.DataFrame({
        "car_brand": ["Peugeot", "Renault", "Peugeot", "BMW", "BMW"],
        "mileage": [50000, None, 60000, None, 20000],
        "horse_power": [100, 90, None, 190, 250],
        "seats": [5, 5, 999999, 5, 4],
        "doors": [5, 999999, 5, 3, 3],
    })

    imputer = OutlierMedianImputer()
    result = imputer.fit_transform(fake_df)
    print(result)