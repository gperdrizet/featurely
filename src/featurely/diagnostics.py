from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute VIF values for selected columns, returning a descending summary."""
    x = df[cols].fillna(0).values.astype(float)
    vifs: list[float] = []

    for i in range(len(cols)):
        try:
            value = variance_inflation_factor(x, i)
        except Exception:
            value = np.inf
        vifs.append(value)

    return (
        pd.DataFrame({"feature": cols, "VIF": vifs})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )
