# Getting started

## Installation

```bash
pip install featurely
```

featurely requires Python 3.10 or newer and depends on numpy, pandas, matplotlib, scipy, statsmodels, and scikit-learn.

## Core workflow

featurely is built around a screen-then-commit loop: build candidate features, test whether they explain variance your current model misses, and keep only the winners.

### 1. Establish a baseline

```python
import pandas as pd
import featurely as fl

df = pd.read_csv("my_data.csv")
target = "price"
features = [c for c in df.columns if c != target]

results = fl.add_pipeline_step(
    None, "raw", df[features], df[target],
    results_path="pipeline-results.pkl",  # persisted; reruns upsert by stage
)
```

### 2. Build candidate features

Every builder returns a new DataFrame of candidates and leaves the input untouched:

```python
# Distance to anchor points (any dict of name -> (lat, lon))
distances = fl.compute_city_distances(df, cities={"downtown": (40.71, -74.01)})

# Per-bin summary statistics of other features
aggregates = fl.compute_bin_aggregates(df, "latitude", ["income"], n_bins=10)

# Cluster membership and centroid distance
clusters = fl.compute_kmeans_features(df, ["latitude", "longitude"], k=6, prefix="geo")

# Kernel-weighted neighborhood averages
smoothed = fl.compute_spatial_smoothed(df, ["income"], lat_col="latitude", lon_col="longitude")
```

### 3. Screen candidates statistically

The candidate scan correlates each candidate against the residuals of a baseline linear model, then applies Benjamini-Hochberg false discovery rate correction:

```python
candidates = pd.concat([distances, aggregates], axis=1)

scan = fl.run_candidate_scan(df, candidates, target=target)
significant = fl.plot_candidate_scan(scan, title="Candidate scan")

keep = [name for name, is_sig in significant.items() if is_sig]
df = pd.concat([df, candidates[keep]], axis=1)
```

For grouped feature sets that act jointly (one-hot encodings, cluster memberships), compare whole sets with cross-validation and paired t-tests instead of per-column correlations.

### 4. Track progress

```python
results = fl.add_pipeline_step(
    results, "+ location", df.drop(columns=target), df[target],
    results_path="pipeline-results.pkl",
)
fl.plot_pipeline_steps(results, results_path="pipeline-results.pkl")
```

Each stage prints mean cross-validated R² with its standard deviation and percent improvement over the raw baseline.

## Complete worked example

The [fsa-feature-engineering-challenge notebooks](https://github.com/gperdrizet/featurely/tree/main/example_notebooks/fsa-feature-engineering-challenge) demonstrate the full loop across ten stages: outlier cleaning, monotonic transforms, interaction features, a censoring probability feature, location encodings, bin aggregates, clustering, spatial smoothing, and polynomial expansion with PCA component selection.
