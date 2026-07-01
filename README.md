# featurely

Solution repository for the lesson 16 feature engineering challenge, now refactored into a simple Python package layout.


## Project lineage

- Current repository: https://github.com/gperdrizet/featurely
- Origin: Fullstack Academy lesson 16 feature engineering assignment from https://github.com/gperdrizet/fullstack-2605


## Package layout

```text
featurely/
├── data/
├── notebooks/
│   ├── original-assignment.ipynb
│   ├── 01-EDA.ipynb
│   ├── 02-outlier-cleaning.ipynb
│   ├── 03-feature-transformations.ipynb
│   ├── 04-interaction-features.ipynb
│   ├── 05-p_censoring.ipynb
│   ├── complete-solution.ipynb
│   ├── config.py
│   └── data/
├── src/
│   └── featurely/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── eda.py
│       ├── outliers.py
│       ├── transforms.py
│       ├── scans.py
│       └── diagnostics.py
├── tests/
│   └── test_imports.py
├── AGENTS.md
├── README.md
└── pyproject.toml
```

## Notebook run order

Run the staged notebooks in this order when you want to reproduce the step-by-step feature engineering workflow:

| Step | Notebook | Purpose | Output |
|---|---|---|---|
| 1 | `01-EDA.ipynb` | EDA and baseline profiling | `data/01-EDA.csv` |
| 2 | `02-outlier-cleaning.ipynb` | Outlier strategy evaluation and cleaning | `data/02-outlier-cleaning.csv` |
| 3 | `03-feature-transformations.ipynb` | Per-feature transform scans and apply selected transforms | `data/03-feature-transformations.csv` |
| 4 | `04-interaction-features.ipynb` | Interaction scans and feature adds | `data/04-interaction-features.csv` |
| 5 | `05-p_censoring.ipynb` | OOF censoring probability feature | `data/x-final.csv` |

`original-assignment.ipynb` preserves the baseline assignment flow, and `complete-solution.ipynb` is the end-to-end combined walkthrough.


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
fl.impute_outliers_with_knn(...)
fl.clip_outliers(...)
fl.transform_outliers(...)
fl.apply_standard_scale(...)
fl.apply_log1p(...)
fl.apply_sqrt(...)
fl.apply_yeo_johnson(...)
fl.apply_quantile_normal(...)
fl.run_per_feature_scan(...)
fl.plot_combined_per_feature_scan(...)
fl.plot_significant_transform_scatters(...)
fl.run_pairwise_scan(...)
fl.plot_combined_pairwise_scan(...)
fl.plot_significant_pairwise_scatters(...)
fl.compute_vif(...)
```
