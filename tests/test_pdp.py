import numpy as np
import pandas as pd
import pytest

from geopdp import (
    compute_geopdp,
    make_copy_of_dataset_with_midpoints,
)


def test_make_copy_of_dataset_with_midpoints_updates_columns_without_touching_source():
    original = pd.DataFrame(
        {"longitude": [1.0, 2.0], "latitude": [-1.0, -2.0], "region": ["A", "B"]}
    )
    updated = make_copy_of_dataset_with_midpoints(
        original,
        "New",
        (42.0, -10.0),
        region_col="region",
        lon_col="longitude",
        lat_col="latitude",
    )

    assert (original["longitude"] == [1.0, 2.0]).all()
    assert (original["region"] == ["A", "B"]).all()
    assert (updated["longitude"] == 42.0).all()
    assert (updated["latitude"] == -10.0).all()
    assert (updated["region"] == "New").all()


def test_make_copy_of_dataset_with_midpoints_validates_coordinate_length():
    df = pd.DataFrame({"longitude": [0.0], "latitude": [0.0], "region": ["Start"]})
    with pytest.raises(ValueError):
        make_copy_of_dataset_with_midpoints(
            df,
            "Broken",
            (10.0,),
            region_col="region",
            lon_col="longitude",
            lat_col="latitude",
        )


def test_make_copy_of_dataset_with_midpoints_handles_missing_region_column():
    original = pd.DataFrame({"longitude": [1.0, 2.0], "latitude": [-1.0, -2.0]})
    updated = make_copy_of_dataset_with_midpoints(
        original,
        "New",
        (42.0, -10.0),
        region_col="region",
        lon_col="longitude",
        lat_col="latitude",
    )

    assert "region" not in updated.columns
    assert (updated["longitude"] == 42.0).all()
    assert (updated["latitude"] == -10.0).all()


def test_make_copy_of_dataset_with_midpoints_uses_custom_columns():
    original = pd.DataFrame({"x": [1.0, 2.0], "y": [-1.0, -2.0], "region": ["A", "B"]})
    updated = make_copy_of_dataset_with_midpoints(
        original,
        "New",
        (42.0, -10.0),
        region_col="region",
        lon_col="x",
        lat_col="y",
    )

    assert (updated["x"] == 42.0).all()
    assert (updated["y"] == -10.0).all()


def test_compute_geopdp_computes_average_probabilities(monkeypatch):
    X = pd.DataFrame(
        {
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 1.0],
            "region": ["seed", "seed"],
        }
    )

    def fake_midpoints(*args, **kwargs):
        return {"North": (10.0, -2.0), "South": (20.0, -5.0)}

    monkeypatch.setattr("geopdp.pdp.define_midpoint_of_regions", fake_midpoints)

    class DummyPipeline:
        def predict_proba(self, df):
            # Check if region is present (it should be for this test)
            if "region" in df.columns:
                # Vectorized prediction: 0.2 for North, 0.8 for South
                # We use numpy where to generate predictions for all rows at once
                base = np.where(df["region"] == "North", 0.2, 0.8)
                # Return shape (n_samples, 2)
                return np.column_stack([base, 1 - base])
            else:
                # Fallback if region is not passed
                return np.tile([0.5, 0.5], (len(df), 1))

    df_results = compute_geopdp(
        X,
        DummyPipeline(),
        col_index_to_predict=0,
        geojson_path="dummy.geojson",
        region_col="region",
        geojson_region_property="NAME_1",
        lon_col="longitude",
        lat_col="latitude",
    )
    df_results = df_results.sort_values("region").reset_index(drop=True)

    assert list(df_results["region"]) == ["North", "South"]
    assert df_results.loc[0, "prediction"] == pytest.approx(0.2)
    assert df_results.loc[1, "prediction"] == pytest.approx(0.8)


def test_compute_geopdp_passes_custom_columns(monkeypatch):
    X = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "region": ["seed", "seed"],
        }
    )

    def fake_midpoints(*args, **kwargs):
        return {"North": (10.0, -2.0)}

    monkeypatch.setattr("geopdp.pdp.define_midpoint_of_regions", fake_midpoints)

    class DummyPipeline:
        def predict_proba(self, df):
            # Verify that x and y were updated
            assert (df["x"] == 10.0).all()
            assert (df["y"] == -2.0).all()
            return np.tile([0.1, 0.9], (len(df), 1))

    compute_geopdp(
        X,
        DummyPipeline(),
        col_index_to_predict=0,
        geojson_path="dummy.geojson",
        region_col="region",
        geojson_region_property="NAME_1",
        lon_col="x",
        lat_col="y",
    )


def test_compute_geopdp_defaults_to_index_0_for_binary_class(monkeypatch):
    X = pd.DataFrame(
        {
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 1.0],
            "region": ["seed", "seed"],
        }
    )

    def fake_midpoints(*args, **kwargs):
        return {"North": (10.0, -2.0)}

    monkeypatch.setattr("geopdp.pdp.define_midpoint_of_regions", fake_midpoints)

    class BinaryPipeline:
        def predict_proba(self, df):
            # Returns 2 columns (binary classification)
            # Col 0: 0.1, Col 1: 0.9
            return np.tile([0.1, 0.9], (len(df), 1))

    # Should not raise, and should use index 0 by default (0.1)
    df_results = compute_geopdp(
        X,
        BinaryPipeline(),
        geojson_path="dummy.geojson",
        region_col="region",
        geojson_region_property="NAME_1",
        lon_col="longitude",
        lat_col="latitude",
        # col_index_to_predict is omitted
    )

    assert df_results.loc[0, "prediction"] == pytest.approx(0.1)


def test_compute_geopdp_raises_for_multiclass_without_index(monkeypatch):
    X = pd.DataFrame(
        {
            "longitude": [0.0],
            "latitude": [0.0],
            "region": ["seed"],
        }
    )

    def fake_midpoints(*args, **kwargs):
        return {"North": (10.0, -2.0)}

    monkeypatch.setattr("geopdp.pdp.define_midpoint_of_regions", fake_midpoints)

    class MultiClassPipeline:
        def predict_proba(self, df):
            # Returns 3 columns
            return np.tile([0.1, 0.2, 0.7], (len(df), 1))

    with pytest.raises(ValueError, match="col_index_to_predict must be specified"):
        compute_geopdp(
            X,
            MultiClassPipeline(),
            geojson_path="dummy.geojson",
            region_col="region",
            geojson_region_property="NAME_1",
            lon_col="longitude",
            lat_col="latitude",
            # col_index_to_predict is omitted
        )


def test_plot_geopdp_smoke_test(monkeypatch):
    from geopdp import plot_geopdp

    df = pd.DataFrame({"region": ["A", "B"], "prediction": [0.1, 0.2]})

    # Mock choropleth_basic to avoid actual plotting and dependency issues during test
    def fake_choropleth(*args, **kwargs):
        return "Figure"

    monkeypatch.setattr("geopdp.pdp.choropleth_basic", fake_choropleth)

    fig = plot_geopdp(
        df,
        geojson="dummy.geojson",
        region_col="region",
        geojson_region_property="NAME_1",
        color_scale="Viridis",
    )
    assert fig == "Figure"


def test_compute_geopdp_works_with_regressor(monkeypatch):
    X = pd.DataFrame(
        {
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 1.0],
            "region": ["seed", "seed"],
        }
    )

    def fake_midpoints(*args, **kwargs):
        return {"North": (10.0, -2.0), "South": (20.0, -5.0)}

    monkeypatch.setattr("geopdp.pdp.define_midpoint_of_regions", fake_midpoints)

    class DummyRegressor:
        """Regressor with only predict method (no predict_proba)."""

        def predict(self, df):
            # Return different values based on region
            if "region" in df.columns:
                return np.where(df["region"] == "North", 100.0, 200.0)
            return np.full(len(df), 150.0)

    df_results = compute_geopdp(
        X,
        DummyRegressor(),
        geojson_path="dummy.geojson",
        region_col="region",
        geojson_region_property="NAME_1",
        lon_col="longitude",
        lat_col="latitude",
    )
    df_results = df_results.sort_values("region").reset_index(drop=True)

    assert list(df_results["region"]) == ["North", "South"]
    assert df_results.loc[0, "prediction"] == pytest.approx(100.0)
    assert df_results.loc[1, "prediction"] == pytest.approx(200.0)


def test_compute_geopdp_ignores_col_index_for_regressor(monkeypatch):
    X = pd.DataFrame(
        {
            "longitude": [0.0],
            "latitude": [0.0],
            "region": ["seed"],
        }
    )

    def fake_midpoints(*args, **kwargs):
        return {"North": (10.0, -2.0)}

    monkeypatch.setattr("geopdp.pdp.define_midpoint_of_regions", fake_midpoints)

    class DummyRegressor:
        """Regressor with only predict method."""

        def predict(self, df):
            return np.full(len(df), 42.0)

    # col_index_to_predict should be ignored for regressors
    df_results = compute_geopdp(
        X,
        DummyRegressor(),
        col_index_to_predict=5,  # This would fail for classifiers
        geojson_path="dummy.geojson",
        region_col="region",
        geojson_region_property="NAME_1",
        lon_col="longitude",
        lat_col="latitude",
    )

    assert df_results.loc[0, "prediction"] == pytest.approx(42.0)


def test_compute_geopdp_handles_multi_output_regressor(monkeypatch):
    """Multi-output regressor returns 2D array; should use first column by default."""
    X = pd.DataFrame(
        {
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 1.0],
            "region": ["seed", "seed"],
        }
    )

    def fake_midpoints(*args, **kwargs):
        return {"North": (10.0, -2.0)}

    monkeypatch.setattr("geopdp.pdp.define_midpoint_of_regions", fake_midpoints)

    class MultiOutputRegressor:
        """Regressor returning multiple outputs (2D array)."""

        def predict(self, df):
            # Returns 2 outputs: first=10.0, second=99.0
            return np.column_stack(
                [
                    np.full(len(df), 10.0),
                    np.full(len(df), 99.0),
                ]
            )

    df_results = compute_geopdp(
        X,
        MultiOutputRegressor(),
        geojson_path="dummy.geojson",
        region_col="region",
        geojson_region_property="NAME_1",
        lon_col="longitude",
        lat_col="latitude",
    )

    # Should use first column (10.0), not second (99.0)
    assert df_results.loc[0, "prediction"] == pytest.approx(10.0)
