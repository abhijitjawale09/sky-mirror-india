from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from .services.digital_twin import TwinScenario


main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def dashboard() -> str:
    engine = current_app.extensions["digital_twin_engine"]
    region = request.args.get("region", current_app.config["PILOT_REGION"])
    state = engine.get_dashboard_state(region)
    return render_template("dashboard.html", state=state, region=region)


@main_bp.post("/api/predict")
def predict() -> tuple[object, int]:
    payload = request.get_json(force=True, silent=True) or {}
    engine = current_app.extensions["digital_twin_engine"]

    scenario = TwinScenario(
        rainfall_delta_pct=float(payload.get("rainfall_delta_pct", 0.0)),
        temp_delta_c=float(payload.get("temp_delta_c", 0.0)),
        horizon_days=max(3, min(int(payload.get("horizon_days", current_app.config["FORECAST_HORIZON_DAYS"])), 30)),
    )
    region = str(payload.get("region", current_app.config["PILOT_REGION"]))
    return jsonify(engine.get_dashboard_state(region, scenario=scenario)), 200


@main_bp.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200
