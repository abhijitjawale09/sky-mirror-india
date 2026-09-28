"""Unit and integration tests for the multi-model climate prediction framework."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import numpy as np
import pytest

from climate_twin import create_app
from climate_twin.models import (
    list_available_models,
    create_model,
    load_model,
    TARGET_COLUMNS,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture
def client():
    app = create_app()
    with app.test_client() as client:
        yield client


def test_model_registry_contains_all_four_models():
    models = list_available_models()
    expected = ["hist_gradient_boosting", "lstm", "random_forest", "xgboost"]
    for m in expected:
        assert m in models, f"Model {m} missing from registry: {models}"


def test_model_instantiation():
    for name in [
        "random_forest",
        "xgboost",
        "hist_gradient_boosting",
        "hurdle_lightgbm",
        "temporal_transformer",
        "st_gnn",
        "stacking_ensemble",
    ]:
        model = create_model(name)
        assert model.name == name
        assert model.display_name is not None
        assert not model.is_fitted


def test_saved_models_can_load_and_predict():
    dummy_features = np.ones((2, 16), dtype=np.float64)

    for model_name in [
        "random_forest",
        "xgboost",
        "hist_gradient_boosting",
        "hurdle_lightgbm",
        "temporal_transformer",
        "st_gnn",
        "stacking_ensemble",
    ]:
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            continue
        model = load_model(model_name, model_dir)
        assert model.is_fitted
        preds = model.predict(dummy_features)
        assert preds.shape == (2, 3), f"Expected shape (2, 3), got {preds.shape} for {model_name}"


def test_api_model_comparison_endpoint(client):
    response = client.get("/api/model-comparison")
    assert response.status_code == 200
    data = response.get_json()
    assert "model_comparison" in data
    assert "best_models" in data
    assert "trained_model_names" in data
    assert len(data["trained_model_names"]) >= 3


def test_dashboard_contains_model_scorecard(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Multi-Model Comparison" in html or "view-scorecard" in html
    assert "Best Model Per Target Variable" in html


def test_calculate_metrics_accuracy():
    from climate_twin.training.model_comparison import calculate_metrics
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [12.0, 18.0, 33.0, 39.0]
    # errors: +2, -2, +3, -1
    # absolute errors: 2, 2, 3, 1 -> mean = 2.0
    # squared errors: 4, 4, 9, 1 -> mean = 4.5 -> sqrt(4.5) = 2.1213
    # mean(y_true) = 25.0
    # sst = (10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2 = 225 + 25 + 25 + 225 = 500
    # sse = 4 + 4 + 9 + 1 = 18
    # r2 = 1 - 18/500 = 0.964
    res = calculate_metrics(y_true, y_pred)
    assert "mae" in res
    assert "rmse" in res
    assert "r2" in res
    assert res["mae"] == 2.0
    assert abs(res["rmse"] - 2.1213) < 1e-3
    assert abs(res["r2"] - 0.964) < 1e-3


def test_model_comparison_detailed_includes_mae_rmse_r2(client):
    response = client.get("/api/model-comparison")
    assert response.status_code == 200
    data = response.get_json()
    assert "model_comparison_detailed" in data
    detailed = data["model_comparison_detailed"]

    # Verify overall ranking has avg_mae, avg_rmse, avg_r2
    assert "overall_ranking" in detailed
    if detailed["overall_ranking"]:
        first = detailed["overall_ranking"][0]
        assert "avg_mae" in first
        assert "avg_rmse" in first
        assert "avg_r2" in first

    # Verify chart_data has mae, rmse, and r2
    assert "chart_data" in detailed
    cd = detailed["chart_data"]
    assert "rainfall_mae" in cd
    assert "rainfall_rmse" in cd
    assert "tmax_mae" in cd
    assert "tmax_rmse" in cd
    assert "tmin_mae" in cd
    assert "tmin_rmse" in cd
    assert "r2_scores" in cd
    assert "rmse_scores" in cd
    assert "mae_scores" in cd

