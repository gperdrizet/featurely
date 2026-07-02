# featurely

Solution repository for the lesson 16 feature engineering challenge, now refactored into a simple Python package layout.


## Project lineage

This project grew out of the instructor solution for the Fullstack Academy feature engineering challenge activity. The full solution is included in this repository as an example use case for Featurely under `example_notebooks/fsa-feature-engineering-challenge`. The original activity is located in the [cohort 2605 materials repo](https://gperdrizet.github.io/fullstack-2605). See the unit 2, lesson 16 activity notebook. 

## Package layout

```text
featurely/
├── .devcontainers/          # Configuration files for devcontainer development environments
├── .github/                 # GitHub workflow files for CI/CD
├── example-notebooks/       # Notebooks demoing the use of Featurely 
├── src/
│   └── featurely/           # The featurely package source tree
│       ├── __init__.py
│       ├── pipeline.py
│       ├── eda.py
│       ├── outliers.py
│       ├── transforms.py
│       ├── scans.py
│       ├── geo.py
│       ├── aggregate.py
│       ├── cluster.py
│       ├── smoothing.py
│       ├── decomposition.py  
│       └── diagnostics.py
├── tests/                    # Unit tests
├── AGENTS.md                 # Onboarding/orientation for AI agents 
├── LICENSE                   # MIT license file
├── README.md                 # README document
├── pyproject.toml            # Python package metadata for PyPI
├── requirements-dev.txt      # Package build/test requirements
└── requirements.txt          # Requirements for local dev & example notebooks
```

## Example notebooks

The example notebooks in the `fsa-feature-engineering-challenge` use the Featurely library to progressively clean the classic California housing data and add engineered features. To reproduce the final dataset, run them in order:

| Step | Notebook | Purpose | Output |
|---|---|---|---|
| 1 | `01-EDA.ipynb` | EDA and baseline profiling | `data/01-EDA.csv` |
| 2 | `02-outlier-cleaning.ipynb` | Outlier strategy evaluation and cleaning | `data/02-outlier-cleaning.csv` |
| 3 | `03-feature-transformations.ipynb` | Per-feature transform scans and apply selected transforms | `data/03-feature-transformations.csv` |
| 4 | `04-interaction-features.ipynb` | Interaction scans and feature adds | `data/04-interaction-features.csv` |
| 5 | `05-p_censoring.ipynb` | OOF censoring probability feature | `data/05-p_censoring.csv` |
| 6 | `06-location-feature-encoding.ipynb` | City distances, geohash cells, rotated coordinates | `data/06-location-feature-encoding.csv` |
| 7 | `07-aggregate-features.ipynb` | Quantile bin summary statistics | `data/07-aggregate-features.csv` |
| 8 | `08-clustering.ipynb` | K-means membership and centroid distance features | `data/08-clustering.csv` |
| 9 | `09-smoothing.ipynb` | Spatial kernel smoothing of features | `data/09-smoothing.csv` |
| 10 | `10-polyfeatures-pca.ipynb` | Polynomial expansion and PCA component selection | `data/10-polyfeatures.csv`, `data/final.csv` |

`original-assignment.ipynb` preserves the baseline assignment flow, and `lesson-16-activity-solution.ipynb` is the final, distilled solution.


## Install locally

```bash
pip install -e .
```


## Run tests

```bash
PYTHONPATH=src pytest
```


## Import in notebook

```python
import featurely as fl

fl.add_pipeline_step(...)
fl.plot_pipeline_steps(...)
fl.plot_feature_distributions(...)
fl.get_feature_correlations(...)
fl.plot_feature_correlations(...)
fl.plot_features_vs_label(...)
fl.run_per_feature_scan(...)
fl.plot_combined_per_feature_scan(...)
fl.plot_significant_transform_scatters(...)
fl.run_pairwise_scan(...)
fl.plot_combined_pairwise_scan(...)
fl.plot_significant_pairwise_scatters(...)
```
