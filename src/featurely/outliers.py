from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def impute_outliers_with_knn(
    df: pd.DataFrame,
    features: list[str],
    n_neighbors: int = 7,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Replace IQR outliers with NaN, then impute with KNN."""
    result = df.copy()

    for col in features:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask = (result[col] < lower) | (result[col] > upper)
        result.loc[outlier_mask, col] = np.nan
        print(f"{col}: {outlier_mask.sum():>4} outliers replaced with NaN")

    print(f"\nTotal NaN values introduced: {result[features].isna().sum().sum()}")

    imputer = KNNImputer(n_neighbors=n_neighbors)
    result[features] = imputer.fit_transform(result[features])

    print(f"NaN values remaining after imputation: {result.isna().sum().sum()}")
    return result


def clip_outliers(df: pd.DataFrame, features: list[str], threshold: float = 1.5) -> pd.DataFrame:
    """Clip feature values to IQR bounds."""
    result = df.copy()

    for col in features:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        result[col] = result[col].clip(lower=lower, upper=upper)
        print(f"{col}: Outliers clipped to [{lower:.2f}, {upper:.2f}]")

    return result


def transform_outliers(df: pd.DataFrame, features: list[str], threshold: float = 1.5) -> pd.DataFrame:
    """Log-transform features that contain IQR outliers and are non-negative."""
    result = df.copy()

    for col in features:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        n_outliers = ((result[col] < lower) | (result[col] > upper)).sum()

        if n_outliers > 0:
            if result[col].min() >= 0:
                result[col] = np.log1p(result[col])
                print(f"{col}: {n_outliers:>4} outliers -> log-transformed")
            else:
                print(f"{col}: {n_outliers:>4} outliers -> skipped (contains negative values)")
        else:
            print(f"{col}: no outliers")

    return result
