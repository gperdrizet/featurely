"""Shared test fixtures.

The synthetic frame mimics the California housing schema so tests exercise
the same call patterns as the example notebooks, without shipping data.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(315)


@pytest.fixture()
def housing_df(rng: np.random.Generator) -> pd.DataFrame:
    """Small synthetic frame with a planted linear signal plus noise."""

    n = 400

    df = pd.DataFrame(
        {
            "MedInc": rng.gamma(2.0, 2.0, n),
            "HouseAge": rng.uniform(1, 52, n),
            "AveRooms": rng.gamma(3.0, 1.5, n),
            "AveOccup": rng.gamma(2.0, 1.2, n),
            "Population": rng.gamma(2.0, 700.0, n),
            "Latitude": rng.uniform(32.5, 42.0, n),
            "Longitude": rng.uniform(-124.4, -114.1, n),
        }
    )

    df["MedHouseVal"] = 0.5 * df["MedInc"] + 0.01 * df["HouseAge"] + rng.normal(0, 0.5, n)
    return df
