#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from climate_twin.data.loader import load_training_observations, save_processed_dataset
from climate_twin.services.forecasting import ClimateForecaster


def main() -> int:
    project_root = PROJECT_ROOT
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    observations, data_source = load_training_observations()
    save_processed_dataset(observations)

    forecaster = ClimateForecaster()
    metrics = forecaster.fit(observations)

    with open(models_dir / "random_forest_model.pkl", "wb") as handle:
        pickle.dump(forecaster, handle)

    with open(models_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"data_source": data_source, **metrics}, handle, indent=2)

    print(f"Saved trained Random Forest model to {models_dir / 'random_forest_model.pkl'}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
