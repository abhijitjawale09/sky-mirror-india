from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from .services.digital_twin import DashboardMode
from .services.twin_simulation import TwinScenario


main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def dashboard() -> str:
    engine = current_app.extensions["digital_twin_engine"]
    region = request.args.get("region", current_app.config.get("PILOT_REGION", "Kerala Coast"))
    state = engine.get_dashboard_state(region)
    return render_template("dashboard.html", state=state, region=region)


@main_bp.post("/api/predict")
def predict() -> tuple[object, int]:
    payload = request.get_json(force=True, silent=True) or {}
    engine = current_app.extensions["digital_twin_engine"]

    scenario = TwinScenario(
        rainfall_delta_pct=float(payload.get("rainfall_delta_pct", 0.0)),
        temp_delta_c=float(payload.get("temp_delta_c", 0.0)),
        horizon_days=max(3, min(int(payload.get("horizon_days", current_app.config.get("FORECAST_HORIZON_DAYS", 14))), 30)),
    )
    region = str(payload.get("region", current_app.config.get("PILOT_REGION", "Kerala Coast")))
    mode = DashboardMode(
        name=str(payload.get("mode", "live")),
        replay_start=payload.get("replay_start"),
        replay_end=payload.get("replay_end"),
    )
    return jsonify(engine.get_dashboard_state(region, scenario=scenario, mode=mode)), 200


@main_bp.post("/api/live-sync")
def live_sync() -> tuple[object, int]:
    """Trigger an on-demand real-time weather stream sync for today."""
    engine = current_app.extensions["digital_twin_engine"]
    sync_result = engine.sync_live_feed()
    region = request.args.get("region", current_app.config["PILOT_REGION"])
    state = engine.get_dashboard_state(region)
    return jsonify({
        "status": sync_result.status,
        "synced_at": sync_result.synced_at,
        "latest_date": sync_result.latest_date,
        "records_count": sync_result.records_count,
        "latency_ms": sync_result.latency_ms,
        "state": state,
    }), 200


@main_bp.post("/api/replay")
def replay() -> tuple[object, int]:
    payload = request.get_json(force=True, silent=True) or {}
    engine = current_app.extensions["digital_twin_engine"]

    region = str(payload.get("region", current_app.config["PILOT_REGION"]))
    start_date = payload.get("start_date", "2024-06-01")
    end_date = payload.get("end_date", "2024-06-30")

    mode = DashboardMode(
        name="replay",
        replay_start=start_date,
        replay_end=end_date,
    )
    state = engine.get_dashboard_state(region, mode=mode)
    return jsonify({"region": region, "replay": state.get("replay")}), 200


@main_bp.get("/api/model-comparison")
def model_comparison() -> tuple[object, int]:
    """Return multi-model comparison data for the dashboard."""
    engine = current_app.extensions["digital_twin_engine"]
    forecaster = engine.forecaster
    return jsonify({
        "model_comparison": forecaster.get_model_comparison(),
        "model_comparison_detailed": forecaster.get_model_comparison_detailed(),
        "best_models": forecaster.get_best_models(),
        "trained_model_names": forecaster.get_model_names(),
        "all_feature_importances": forecaster.get_all_feature_importances(),
        "active_model": forecaster.active_model_name,
    }), 200


@main_bp.get("/api/forecast-7day")
def forecast_7day() -> tuple[object, int]:
    """Return a structured 7-day climate forecast for a region or custom coordinates."""
    engine = current_app.extensions["digital_twin_engine"]

    region = request.args.get("region")
    lat_str = request.args.get("lat")
    lon_str = request.args.get("lon")

    try:
        lat = float(lat_str) if lat_str else None
        lon = float(lon_str) if lon_str else None
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid latitude or longitude values."}), 400

    if lat is not None and lon is not None:
        # Validate coordinate ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"error": "Coordinates out of range."}), 400
        result = engine.get_seven_day_forecast(latitude=lat, longitude=lon, region_name=region)
    elif region:
        result = engine.get_seven_day_forecast(region_name=region)
    else:
        # Default to the currently configured pilot region
        default_region = current_app.config.get("PILOT_REGION", "Kerala Coast")
        result = engine.get_seven_day_forecast(region_name=default_region)

    if result.get("error"):
        return jsonify(result), 502

    return jsonify(result), 200


@main_bp.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200
