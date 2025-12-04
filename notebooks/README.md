# Example Notebooks

This directory contains Jupyter notebooks demonstrating how to use `geopdp` for spatial machine learning interpretability.

## Notebooks

- **`spatial_pdp_demo.ipynb`**: Complete example showing how to train a model on geospatial data and analyze it using spatial PDPs. Includes synthetic data with realistic geographic patterns.

## Running the Notebooks

Install Jupyter and required dependencies:

```bash
uv sync --extra dev
uv add --dev jupyter
```

Then run:

```bash
uv run jupyter notebook notebooks/
```
