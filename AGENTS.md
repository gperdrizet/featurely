# featurely AGENTS.md

## Purpose

`featurely` started as a solution to Fullstack Academy AM/ML unit 2, lesson 16: improve a linear regression model on the California housing dataset with feature engineering. The repository now serves two roles:

1. A general-purpose Python package of reusable feature-engineering utilities, published to PyPI.
2. A notebook-driven demo and experiment log that shows the progression from EDA to the final solution.

## Repository Structure

- `src/featurely/`: importable package code.
- `tests/`: per-module unit tests plus shared fixtures in `tests/conftest.py`.
- `example_notebooks/fsa-feature-engineering-challenge/`: the original assignment, staged working notebooks, and the distilled solution notebook.
- `data/fsa-feature-engineering-challenge/`: local CSV outputs and intermediate artifacts created by the notebooks (gitignored).
- `docs/` and `mkdocs.yml`: MkDocs Material documentation site, deployed to GitHub Pages.
- `.github/workflows/`: CI (`test.yml`), release (`publish.yml`), and docs deployment (`docs.yml`).
- `.devcontainer/`: CPU-only data science container used for development.

## Package Overview

The package is intentionally small and function-oriented. `src/featurely/__init__.py` re-exports the public API so notebooks can import from `featurely` directly.

- `pipeline.py`: cross-validation helpers for comparing pipeline stages.
- `eda.py`: distribution plots, correlation analysis, and target-vs-feature visualization.
- `outliers.py`: IQR-based outlier clipping, log-style transformation, and KNN imputation.
- `transforms.py`: scaling and monotonic transforms used during feature engineering.
- `scans.py`: per-feature, pairwise, and generic candidate scan utilities for residual analysis and significance testing.
- `geo.py`: haversine distances to anchor cities, hand-rolled geohash encoding, and rotated coordinates.
- `aggregate.py`: quantile bin summary statistic features.
- `cluster.py`: k-means selection diagnostics and cluster membership features.
- `smoothing.py`: Gaussian kernel spatial smoothing of features.
- `decomposition.py`: polynomial expansion, PCA variance plots, and CV-based component selection.
- `diagnostics.py`: variance inflation factor calculation.

Implementation pattern:

- Most functions accept a `pandas.DataFrame`, copy it, and return a transformed copy instead of mutating input.
- Functions are dataset-agnostic: target and coordinate column names are explicit parameters, never hardcoded.
- Plotting helpers use `matplotlib` and are meant for notebook use.
- Scan helpers compare transformed features against linear-model residuals, then apply multiple-testing correction where appropriate.
- Public functions carry Google-style docstrings; the documentation site renders them via mkdocstrings, and `mkdocs build --strict` fails on missing parameter annotations.

## Notebook Workflow

The notebook sequence is the main narrative for the project:

1. `original-assignment.ipynb`: the baseline assignment context.
2. `01-EDA.ipynb`: exploratory analysis and feature inspection.
3. `02-outlier-cleaning.ipynb`: outlier handling experiments.
4. `03-feature-transformations.ipynb`: monotonic and distribution-shaping transforms.
5. `04-interaction-features.ipynb`: pairwise interaction exploration.
6. `05-p_censoring.ipynb`: target-censoring probability feature.
7. `06-location-feature-encoding.ipynb`: city distances, geohash cells, and rotated coordinates.
8. `07-aggregate-features.ipynb`: quantile bin summary statistics.
9. `08-clustering.ipynb`: k-means membership and centroid distance features.
10. `09-smoothing.ipynb`: spatial kernel smoothing of features.
11. `10-polyfeatures-pca.ipynb`: polynomial expansion and PCA component selection; produces the final dataset.

The staged notebooks live in `example_notebooks/fsa-feature-engineering-challenge/`. Shared constants such as the data URL, output directory, outlier threshold, and selected log-transform features live in `config.py` in that directory. Prefer changing shared settings there rather than duplicating values across notebooks.

## Working Rules For Agents

- Prefer editing the package modules in `src/featurely/` when the logic should be reusable or testable.
- Prefer editing notebooks when the task is presentation, exploration, or updating the staged demo flow.
- Keep new public helpers exported from `src/featurely/__init__.py` if notebooks or tests need them.
- Preserve the notebook narrative order and the existing stage naming when adding or revising demo content.
- Use the existing CSV artifacts only as notebook outputs; regenerate them from the notebooks when the analysis changes.
- Keep changes lightweight and consistent with the current function-based style.

## Style

Write without symbols or emojis unless they are required for code or data. Avoid em dashes; use commas, colons, or semicolons instead. Use American English spellings, for example visualize instead of visualise. Write at a technical level that fits technical professionals or college graduates in a data science and AI or ML bootcamp, without boilerplate phrasing or buzzwords. Documentation and code comments should teach the reader as well as describe implementation details and functionality.

## Local Checks

- Tests: `pytest`
- Lint: `ruff check src/ tests/`
- Format: `ruff format --check src/ tests/`
- Docs build: `mkdocs build --strict` (requires `pip install -e ".[docs]"`)
- Editable install: `pip install -e ".[dev]"`

## CI and Releases

- Pull requests against `main` run lint, format check, and the test matrix (Python 3.10 to 3.13) via `.github/workflows/test.yml`. Development happens on the `dev` branch.
- Releases are manually triggered from `main` via `.github/workflows/publish.yml` with a version input. TestPyPI rehearsal runs use a dev-suffixed version and make no repository changes; real runs publish to PyPI first, then commit the version bump, tag, and GitHub release.
- Documentation deploys to GitHub Pages on push to `main` via `.github/workflows/docs.yml`.

If you change notebook code that writes artifacts, make sure the generated outputs still match the updated analysis and data flow.