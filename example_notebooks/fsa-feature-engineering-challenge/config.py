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

# Location encoding (notebook 06)
GEOHASH_PRECISION = 4
GEOHASH_MIN_CELL_COUNT = 100

# Bin aggregates (notebook 07)
AGG_BIN_FEATURES = ["Latitude", "MedInc"]
AGG_N_BINS = 10
AGG_STATS = ("mean",)

# Clustering (notebook 08)
CLUSTER_GEO_FEATURES = ["Latitude", "Longitude"]
CLUSTER_GEO_K = 8
CLUSTER_SOCIO_FEATURES = ["MedInc", "AveRooms", "AveOccup"]
CLUSTER_SOCIO_K = 5

# Spatial smoothing (notebook 09)
SMOOTH_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveOccup", "Population"]
SMOOTH_N_NEIGHBORS = 50

# Polynomial expansion and PCA (notebook 10)
POLY_DEGREE = 2

PIPELINE_COLORS = {
    'raw':            '#aaaaaa',
    '+ p_censored':   '#ffc000',
    '+ cleaned':      '#5b9bd5',
    '+ transforms':   '#70ad47',
    '+ interactions': '#7030a0',
    '+ location':     '#ed7d31',
    '+ aggregates':   '#255e91',
    '+ clusters':     '#9e480e',
    '+ smoothed':     '#43682b',
    '+ poly_pca':     '#c00000',
}