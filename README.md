# geopdp

**Spatial Partial Dependence Profiles made easy.**

`geopdp` is a Python library designed to bridge the gap between machine learning interpretability and geospatial analysis. It allows you to compute and visualize Partial Dependence Profiles (PDPs) specifically for geographic regions, helping you understand how your model's predictions vary across space.

`geopdp` is built on top of `scikit-learn` and `plotly`, and extends their capabilities to handle geospatial data. Contributions are welcome and encouraged!

## Why use geopdp?

Standard PDP tools (like `scikit-learn` or `shap`) operate on tabular data and don't natively understand "regions" or coupled coordinates. Manually calculating spatial PDPs requires writing complex loops to shift data points across regions and aggregate predictions.

`geopdp` automates this workflow:
1.  **Automated Sweeping**: `compute_geopdp` handles the logic of moving your dataset to different regions and computing average predictions.
2.  **Instant Visualization**: `plot_geopdp` merges your results with GeoJSON geometries and creates interactive Plotly choropleth maps in one line.
3.  **Geometry Optimization**: Built-in tools to simplify complex GeoJSONs, making your visualizations faster and lighter.

## Installation

### From GitHub

```bash
pip install git+https://github.com/adevol/geopdp.git
```

### Local Development

Clone the repository and install in editable mode:

```bash
git clone https://github.com/adevol/geopdp.git
cd geopdp
pip install -e .
```

Or using `uv` (recommended):

```bash
git clone https://github.com/adevol/geopdp.git
cd geopdp
uv sync
```

## Quickstart

### 1. Spatial PDP Analysis

```python
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from geopdp import compute_geopdp, plot_geopdp
from geopdp.data import TANZANIA_GEOJSON

# 1. Load your data and trained pipeline
pipe: Pipeline = ...
X: pd.DataFrame = ...

# 2. Compute PDPs for each region in the GeoJSON
results = compute_geopdp(
    X,
    pipe,
    col_index_to_predict=1,
    geojson_path=TANZANIA_GEOJSON,
)

# 3. Visualize the results
fig = plot_geopdp(
    results,
    geojson=TANZANIA_GEOJSON,
    region_col="region",
    title="Model Predictions by Region",
)
fig.show()
```

### 2. Geometry Optimization

Large GeoJSON files can slow down rendering. Use `geopdp` to simplify them:

```python
from geopdp.geometry import simplify_geojson
from geopdp.visualization import compare_geojson_geometry
from geopdp.data import TANZANIA_GEOJSON

# Reduce vertices and coordinate precision
simplified_gdf = simplify_geojson(
    TANZANIA_GEOJSON,
    tolerance=0.01,  # Higher = more simplification
    precision=0.001  # Grid size for coordinate snapping
)

# Visually compare the original vs simplified version
fig = compare_geojson_geometry(TANZANIA_GEOJSON, simplified_gdf)
fig.show()
```

## Demo Notebook

Check out the **[spatial PDP demo notebook](notebooks/spatial_pdp_demo.ipynb)** for a complete example with synthetic data showing:
- Training a model on geospatial health data
- Computing spatial PDPs across Tanzanian regions
- Visualizing geographic patterns in model predictions
- Geometry simplification for faster rendering

## Running Tests

To run the test suite:

```bash
uv run pytest
```

To run the linter:

```bash
uv run ruff check .
```

## Sample Data

- `data/tanzania.geojson`: High-resolution Tanzanian regional boundaries.
- `data/sample_regions.geojson`: Toy dataset for testing.
