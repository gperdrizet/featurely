"""Location-based feature encodings for latitude and longitude.

These helpers turn raw coordinates into representations a linear model can
use: distances to fixed anchor points, discrete spatial cells, and rotated
axes. Raw latitude and longitude only let a linear model fit a single plane
over the map; these encodings expose distance decay and neighborhood
structure that the plane cannot capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EARTH_RADIUS_KM = 6371.0

# Geohash uses a 32-character alphabet that omits a, i, l, and o to avoid
# ambiguity with digits when hashes are read by humans.
_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def haversine_distance(
    lat1: float | np.ndarray | pd.Series,
    lon1: float | np.ndarray | pd.Series,
    lat2: float | np.ndarray | pd.Series,
    lon2: float | np.ndarray | pd.Series,
) -> float | np.ndarray:
    """Great-circle distance in kilometers between coordinate pairs.

    The haversine formula treats Earth as a sphere, which is accurate to
    roughly 0.5 percent; plenty for feature engineering distances.

    Args:
        lat1: Latitude of the first point; scalar or array-like, degrees.
        lon1: Longitude of the first point; scalar or array-like, degrees.
        lat2: Latitude of the second point; scalar or array-like, degrees.
        lon2: Longitude of the second point; scalar or array-like, degrees.

    Returns:
        Distance in kilometers, matching the broadcast shape of the inputs.
    """

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def compute_city_distances(
    df: pd.DataFrame,
    cities: dict[str, tuple[float, float]],
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.DataFrame:
    """Return distance-to-anchor candidate features in kilometers.

    One column per anchor point plus dist_nearest_city, which collapses the
    set into a single proximity measure. Many spatial outcomes decay with
    distance from activity centers, a pattern raw coordinates cannot express
    linearly.

    Args:
        df: Input frame with coordinate columns.
        cities: Mapping of anchor name to (latitude, longitude). Column
            names follow the pattern ``dist_{name}``.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        A frame of distance columns plus ``dist_nearest_city``.
    """

    out = {}

    for name, (city_lat, city_lon) in cities.items():
        out[f"dist_{name}"] = haversine_distance(df[lat_col], df[lon_col], city_lat, city_lon)

    frame = pd.DataFrame(out, index=df.index)
    frame["dist_nearest_city"] = frame.min(axis=1)

    return frame


def encode_geohash(lat: float, lon: float, precision: int = 4) -> str:
    """Encode one coordinate pair as a geohash string.

    Geohashing interleaves bits from successive binary subdivisions of the
    longitude and latitude ranges, then packs each group of 5 bits into a
    base32 character. Nearby points usually share a prefix, so shorter
    hashes give coarser spatial cells: precision 4 cells are roughly
    39 km by 19.5 km.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        precision: Number of base32 characters in the hash.

    Returns:
        The geohash string.
    """

    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    bits: list[int] = []
    use_lon = True

    while len(bits) < precision * 5:
        rng = lon_range if use_lon else lat_range
        value = lon if use_lon else lat
        mid = (rng[0] + rng[1]) / 2

        if value >= mid:
            bits.append(1)
            rng[0] = mid

        else:
            bits.append(0)
            rng[1] = mid

        use_lon = not use_lon

    chars = []

    for i in range(0, len(bits), 5):
        idx = 0

        for bit in bits[i : i + 5]:
            idx = (idx << 1) | bit

        chars.append(_GEOHASH_BASE32[idx])

    return "".join(chars)


def compute_geohash_cells(
    df: pd.DataFrame,
    precision: int = 4,
    min_cell_count: int = 100,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.DataFrame:
    """Return one-hot geohash cell membership candidates.

    Cells with fewer than min_cell_count rows are pooled into a shared
    "other" bucket so the linear model does not fit dummy coefficients to
    nearly empty cells. Membership indicators are target-free, so there is
    no leakage risk from this encoding.

    Args:
        df: Input frame with coordinate columns.
        precision: Geohash length; higher values give smaller cells.
        min_cell_count: Minimum rows per cell before pooling into "other".
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        A frame of one-hot indicator columns named ``gh{precision}_{cell}``.
    """

    hashes = [encode_geohash(lat, lon, precision) for lat, lon in zip(df[lat_col], df[lon_col], strict=False)]
    cells = pd.Series(hashes, index=df.index)

    counts = cells.value_counts()
    keep = counts[counts >= min_cell_count].index
    pooled = cells.where(cells.isin(keep), "other")

    return pd.get_dummies(pooled, prefix=f"gh{precision}", dtype=float)


def compute_rotated_coordinates(
    df: pd.DataFrame,
    angle_deg: float,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.DataFrame:
    """Return coordinate axes rotated by an arbitrary angle.

    Raw latitude and longitude only let a linear model fit gradients that
    run north-south or east-west. Rotating the frame exposes gradients that
    run diagonally across the map, for example along a coastline or a
    mountain range. A 45 degree rotation reproduces the classic
    sum-and-difference encoding up to scale.

    Args:
        df: Input frame with coordinate columns.
        angle_deg: Rotation angle in degrees, counterclockwise. Column names
            embed the angle, e.g. ``rot45_x`` and ``rot45_y``.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        A frame with the two rotated coordinate columns.
    """

    theta = np.radians(angle_deg)
    x = df[lon_col]
    y = df[lat_col]

    # Sanitize the angle for CSV-safe column names: rot-30.5 -> rotm30p5.
    label = f"{angle_deg:g}".replace("-", "m").replace(".", "p")

    return pd.DataFrame(
        {
            f"rot{label}_x": x * np.cos(theta) + y * np.sin(theta),
            f"rot{label}_y": -x * np.sin(theta) + y * np.cos(theta),
        },
        index=df.index,
    )
