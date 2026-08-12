from pathlib import Path

import h5py
import pandas as pd

from climate_twin import create_app
from climate_twin.data.insat_merge import merge_insat_with_imd


def test_health_endpoint():
    app = create_app()
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_dashboard_loads():
    app = create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"AI-Powered Digital Twin" in response.data


def test_insat_merge_adds_satellite_values(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "region": "Kerala Coast",
                "latitude": 9.5,
                "longitude": 76.5,
                "rainfall_mm": 12.1,
                "tmax_c": 31.4,
                "tmin_c": 24.9,
                "humidity_pct": 82.0,
            }
        ]
    )

    h5_path = tmp_path / "3RIMG_L2B_LST_20240101.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("LST", data=[[22.0, 23.0], [24.0, 25.0]])
        handle.create_dataset("lat", data=[9.5, 10.5])
        handle.create_dataset("lon", data=[76.5, 77.5])

    merged = merge_insat_with_imd(frame, tmp_path, value_name_hint="LST")

    assert merged.shape[0] == frame.shape[0]
    assert "insat_lst_c" in merged.columns
    assert merged["insat_lst_c"].notna().all()
