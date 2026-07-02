from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute variance inflation factors for the selected columns.

    VIF measures how much a coefficient's variance is inflated by collinearity
    with the other columns; values above roughly 10 signal problematic
    redundancy. Columns whose VIF cannot be computed are reported as infinity.

    Args:
        df: Input frame. NaNs are filled with 0 before computation.
        cols: Columns to evaluate.

    Returns:
        A frame with ``feature`` and ``VIF`` columns, sorted descending by VIF.
    """
    x = df[cols].fillna(0).values.astype(float)
    vifs: list[float] = []

    for i in range(len(cols)):
        try:
            value = variance_inflation_factor(x, i)
        except Exception:
            value = np.inf
        vifs.append(value)

    return pd.DataFrame({"feature": cols, "VIF": vifs}).sort_values("VIF", ascending=False).reset_index(drop=True)
