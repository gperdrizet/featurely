"""Shared notebook configuration values.

Import these symbols in split notebooks to keep pipeline settings consistent.
"""

from pathlib import Path

# Data locations
DATA_URL = "https://media.githubusercontent.com/media/gperdrizet/fullstack-2605/refs/heads/main/data/california_housing.csv"
DATA_DIR = Path("data")

# Ensure output directory exists for read/write pipeline steps.
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Feature engineering configuration
OUTLIER_METHOD = "clip"  # Options: 'impute', 'clip', 'transform', 'ignore'
OUTLIER_THRESHOLD = 2.25
LOG_FEATURES = ["AveRooms", "AveOccup"]
