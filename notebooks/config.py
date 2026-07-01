"""Shared notebook configuration values.

Import these symbols in split notebooks to keep pipeline settings consistent.
"""

from pathlib import Path

# Data locations
DATA_URL = "https://media.githubusercontent.com/media/gperdrizet/fullstack-2605/refs/heads/main/data/california_housing.csv"
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Ensure output directory exists for read/write pipeline steps.
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Shared stage-metrics store used by notebooks 02-05.
PIPELINE_RESULTS_PATH = DATA_DIR / "pipeline-results.pkl"

# Feature engineering configuration
OUTLIER_METHOD = "clip"  # Options: 'impute', 'clip', 'transform', 'ignore'
OUTLIER_THRESHOLD = 2.25
LOG_FEATURES = ["AveRooms", "AveOccup"]

PIPELINE_COLORS = {
    'raw':            '#aaaaaa',
    '+ p_censored':   '#ffc000',
    '+ cleaned':      '#5b9bd5',
    '+ transforms':   '#70ad47',
    '+ interactions': '#7030a0',
}