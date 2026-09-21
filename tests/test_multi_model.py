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
    for name in ["random_forest", "xgboost", "hist_gradient_boosting"]:
        model = create_model(name)
        assert model.name == name
        assert model.display_name is not None
        assert not model.is_fitted


def test_saved_models_can_load_and_predict():
    dummy_features = np.ones((2, 16), dtype=np.float64)

    for model_name in ["random_forest", "xgboost", "hist_gradient_boosting"]:
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
