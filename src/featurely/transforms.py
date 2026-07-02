from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler


def apply_standard_scale(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a copy with standard scaling applied to selected feature columns.

    Args:
        df: Input frame; not modified.
        feature_cols: Columns to scale to zero mean and unit variance.

    Returns:
        A copy of ``df`` with the selected columns scaled.
    """

    result = df.copy()
    result[feature_cols] = StandardScaler().fit_transform(result[feature_cols])

    return result


def apply_log1p(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a copy with log1p transform (with non-negative shift) then scaling.

    Args:
        df: Input frame; not modified.
        feature_cols: Columns to transform; columns with negative minimums are
            shifted to zero before ``log1p``.

    Returns:
        A copy of ``df`` with the selected columns transformed and scaled.
    """

    result = df.copy()
    x = result[feature_cols].copy()

    for col in feature_cols:
        col_min = x[col].min()
        shift = -col_min if col_min < 0 else 0
        result[col] = np.log1p(x[col] + shift)

    result[feature_cols] = StandardScaler().fit_transform(result[feature_cols])

    return result


def apply_sqrt(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a copy with square-root transform (with non-negative shift) then scaling.

    Args:
        df: Input frame; not modified.
        feature_cols: Columns to transform; columns with negative minimums are
            shifted to zero before the square root.

    Returns:
        A copy of ``df`` with the selected columns transformed and scaled.
    """

    result = df.copy()
    x = result[feature_cols].copy()

    for col in feature_cols:
        col_min = x[col].min()
        shift = -col_min if col_min < 0 else 0
        result[col] = np.sqrt(x[col] + shift)

    result[feature_cols] = StandardScaler().fit_transform(result[feature_cols])

    return result


def apply_yeo_johnson(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a copy with Yeo-Johnson transform applied to selected columns.

    Yeo-Johnson fits a Box-Cox-style power parameter per column and handles
    negative values natively.

    Args:
        df: Input frame; not modified.
        feature_cols: Columns to transform.

    Returns:
        A copy of ``df`` with the selected columns transformed.
    """

    result = df.copy()
    result[feature_cols] = PowerTransformer(method="yeo-johnson").fit_transform(result[feature_cols])

    return result


def apply_quantile_normal(df: pd.DataFrame, feature_cols: list[str], random_state: int = 315) -> pd.DataFrame:
    """Return a copy with quantile-to-normal transform applied to selected columns.

    Maps each column's empirical CDF onto a standard normal distribution,
    which erases outliers and skew but distorts within-column spacing.

    Args:
        df: Input frame; not modified.
        feature_cols: Columns to transform.
        random_state: Seed for the quantile transformer's subsampling.

    Returns:
        A copy of ``df`` with the selected columns transformed.
    """

    result = df.copy()

    result[feature_cols] = QuantileTransformer(output_distribution="normal", random_state=random_state).fit_transform(
        result[feature_cols]
    )

    return result
