from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..data.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    prepare_inference_features,
    prepare_training_features,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ForecastBundle:
    rainfall: float
    tmax: float
    tmin: float
    rainfall_lower: float
    rainfall_upper: float
    tmax_lower: float
    tmax_upper: float
    tmin_lower: float
    tmin_upper: float


class ClimateForecaster:
    """Multi-model climate forecaster with backward-compatible Random Forest default.

    If multi-model results exist (from train_all_models.py), loads the best model
    per target and exposes full comparison data. Otherwise, falls back to the
    original inline Random Forest training for backward compatibility.
    """

    def __init__(self) -> None:
        self.model: RandomForestRegressor | None = None
        self.region_code_map: dict[str, int] = {}
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.feature_importance: dict[str, float] = {}
        self.baseline_metrics: dict[str, dict[str, float]] = {}
        self.deterministic_metrics: dict[str, dict[str, float]] = {}

        # Multi-model support
        self._active_model_name: str = "random_forest"
        self._all_model_metrics: list[dict] = []
        self._best_models: dict[str, dict] = {}
        self._multi_model_loaded: bool = False
        self._loaded_models: dict[str, Any] = {}
        self._feature_importances: dict[str, dict[str, float]] = {}

        # Try to load multi-model results
        self._try_load_multi_model_results()

    @property
    def active_model_name(self) -> str:
        return self._active_model_name

    def _try_load_multi_model_results(self) -> None:
        """Attempt to load pre-trained multi-model comparison results."""
        metrics_path = RESULTS_DIR / "model_metrics.json"
        best_path = RESULTS_DIR / "best_models.json"
        fi_path = RESULTS_DIR / "feature_importance.json"

        if metrics_path.exists() and best_path.exists():
            try:
                with open(metrics_path) as f:
                    self._all_model_metrics = json.load(f)
                with open(best_path) as f:
                    self._best_models = json.load(f)
                if fi_path.exists():
                    with open(fi_path) as f:
                        self._feature_importances = json.load(f)
                self._multi_model_loaded = True
                logger.info(
                    f"Loaded multi-model results: {len(self._all_model_metrics)} metrics, "
                    f"best models: {list(self._best_models.keys())}"
                )
            except Exception as exc:
                logger.warning(f"Failed to load multi-model results: {exc}")
                self._multi_model_loaded = False

    def prepare_features(self, observations: pd.DataFrame) -> pd.DataFrame:
        feature_frame, self.region_code_map = prepare_training_features(observations)
        return feature_frame

    def fit(self, observations: pd.DataFrame) -> dict[str, float]:
        """Train the inline Random Forest (backward compatibility).

        This method preserves the original training flow. The multi-model
        comparison runs separately via train_all_models.py.
        """
        frame = self.prepare_features(observations)

        # Temporal split: use last 15% as test set (approximates last year)
        split_index = max(30, int(len(frame) * 0.85))
        train_frame = frame.iloc[:split_index]
        test_frame = frame.iloc[split_index:]

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(train_frame[self.feature_columns], train_frame[TARGET_COLUMNS])

        # Extract feature importance
        importances = self.model.feature_importances_
        self.feature_importance = {
            col: round(float(imp) * 100, 1)
            for col, imp in sorted(
                zip(self.feature_columns, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        metrics: dict[str, float] = {}
        if not test_frame.empty:
            predictions = self.model.predict(test_frame[self.feature_columns])
            y_test = test_frame[TARGET_COLUMNS].values

            # Compute baseline metrics (persistence model = yesterday's value)
            persistence = test_frame[["rainfall_lag_1", "tmax_lag_1", "tmin_lag_1"]].values

            target_names = ["rainfall_mm", "tmax_c", "tmin_c"]
            persistence_cols = ["rainfall_lag_1", "tmax_lag_1", "tmin_lag_1"]

            for i, (name, pcol) in enumerate(zip(target_names, persistence_cols)):
                # RF metrics
                mae = float(mean_absolute_error(y_test[:, i], predictions[:, i]))
                rmse = float(np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i])))
                r2 = float(r2_score(y_test[:, i], predictions[:, i]))
                metrics[f"mae_{name}"] = mae

                self.deterministic_metrics[name] = {
                    "mae": round(mae, 3),
                    "rmse": round(rmse, 3),
                    "r2": round(r2, 3),
                }

                # Persistence baseline metrics
                p_vals = test_frame[pcol].values
                p_mae = float(mean_absolute_error(y_test[:, i], p_vals))
                p_rmse = float(np.sqrt(mean_squared_error(y_test[:, i], p_vals)))
                p_r2 = float(r2_score(y_test[:, i], p_vals))
                self.baseline_metrics[name] = {
                    "mae": round(p_mae, 3),
                    "rmse": round(p_rmse, 3),
                    "r2": round(p_r2, 3),
                }

        return metrics

    def predict_with_uncertainty(
        self,
        features: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict using all trees individually and return mean, lower, upper (80% interval)."""
        if self.model is None:
            raise RuntimeError("Forecast model has not been trained")

        # Get predictions from each individual tree
        tree_predictions = np.array([
            tree.predict(features[self.feature_columns].values)
            for tree in self.model.estimators_
        ])  # shape: (n_trees, n_samples, 3)

        mean_pred = np.mean(tree_predictions, axis=0)
        lower = np.percentile(tree_predictions, 10, axis=0)
        upper = np.percentile(tree_predictions, 90, axis=0)

        return mean_pred, lower, upper

    def predict_next(
        self,
        history: pd.DataFrame,
        horizon_days: int = 14,
        rainfall_delta_pct: float = 0.0,
        temp_delta_c: float = 0.0,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Forecast model has not been trained")

        region = str(history.iloc[-1]["region"])
        forecast_rows: list[dict[str, float | str | pd.Timestamp]] = []
        working_history = history.copy().sort_values("date").reset_index(drop=True)

        for step in range(horizon_days):
            features = self._make_feature_row(working_history, step + 1)
            mean_pred, lower, upper = self.predict_with_uncertainty(features)

            rainfall = max(0.0, float(mean_pred[0][0]) * (1.0 + rainfall_delta_pct / 100.0))
            tmax = float(mean_pred[0][1]) + temp_delta_c
            tmin = float(mean_pred[0][2]) + temp_delta_c * 0.82

            # Uncertainty bounds with scenario adjustments
            rain_lower = max(0.0, float(lower[0][0]) * (1.0 + rainfall_delta_pct / 100.0))
            rain_upper = max(0.0, float(upper[0][0]) * (1.0 + rainfall_delta_pct / 100.0))
            tmax_lower = float(lower[0][1]) + temp_delta_c
            tmax_upper = float(upper[0][1]) + temp_delta_c
            tmin_lower = float(lower[0][2]) + temp_delta_c * 0.82
            tmin_upper = float(upper[0][2]) + temp_delta_c * 0.82

            forecast_date = pd.to_datetime(working_history.iloc[-1]["date"]) + pd.Timedelta(days=1)
            next_row = {
                "date": forecast_date,
                "region": region,
                "rainfall_mm": round(rainfall, 2),
                "tmax_c": round(tmax, 2),
                "tmin_c": round(tmin, 2),
                "rainfall_lower": round(rain_lower, 2),
                "rainfall_upper": round(rain_upper, 2),
                "tmax_lower": round(tmax_lower, 2),
                "tmax_upper": round(tmax_upper, 2),
                "tmin_lower": round(tmin_lower, 2),
                "tmin_upper": round(tmin_upper, 2),
                "latitude": float(working_history.iloc[-1]["latitude"]),
                "longitude": float(working_history.iloc[-1]["longitude"]),
                "humidity_pct": max(35.0, min(98.0, float(working_history.iloc[-1]["humidity_pct"]) + rainfall_delta_pct * 0.12 - temp_delta_c * 0.8)),
                "diurnal_range_c": max(1.0, tmax - tmin),
            }
            forecast_rows.append(next_row)

            working_history = pd.concat([working_history, pd.DataFrame([next_row])], ignore_index=True)

        return pd.DataFrame(forecast_rows)

    def get_model_comparison(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return the full model comparison table.

        If multi-model results are loaded, returns all models' metrics.
        Otherwise, falls back to persistence vs RF (original behavior).
        """
        if self._multi_model_loaded and self._all_model_metrics:
            comparison: dict[str, dict[str, dict[str, float]]] = {}

            # Always include persistence baseline
            comparison["persistence"] = self.baseline_metrics

            # Add all trained models
            for m in self._all_model_metrics:
                model_name = m["model_name"]
                target = m["target"]
                if model_name not in comparison:
                    comparison[model_name] = {}
                comparison[model_name][target] = {
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                }

            return comparison

        # Fallback: original 2-model comparison
        return {
            "persistence": self.baseline_metrics,
            "random_forest": self.deterministic_metrics,
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances as percentages, sorted descending."""
        return self.feature_importance

    def get_all_feature_importances(self) -> dict[str, dict[str, float]]:
        """Return feature importances for all models that support it."""
        result = {"random_forest": self.feature_importance}
        if self._feature_importances:
            result.update(self._feature_importances)
        return result

    def set_active_model(self, model_name: str) -> None:
        """Set the active model for forecasting."""
        from ..models import list_available_models
        if model_name in list_available_models():
            self._active_model_name = model_name

    def get_best_models(self) -> dict[str, dict]:
        """Return best model selection per target variable."""
        return self._best_models

    def get_model_names(self) -> list[str]:
        """Return list of all trained model names."""
        if self._multi_model_loaded:
            return sorted(set(m["model_name"] for m in self._all_model_metrics))
        return ["random_forest"]

    def get_model_comparison_detailed(self) -> dict[str, Any]:
        """Compute comprehensive multi-model comparison with accuracy scores, deltas, and rankings."""
        model_display_names = {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "hist_gradient_boosting": "HistGradientBoosting",
            "lstm": "LSTM Neural Net",
            "hurdle_lightgbm": "Hurdle LightGBM",
            "temporal_transformer": "Temporal Transformer",
            "st_gnn": "Spatio-Temporal GNN",
            "stacking_ensemble": "Physics Stacking Ensemble",
            "persistence": "Persistence Baseline",
        }

        model_colors = {
            "random_forest": "#00f5d4",
            "hist_gradient_boosting": "#2ecc71",
            "xgboost": "#ff9f43",
            "lstm": "#a55eea",
            "hurdle_lightgbm": "#e056fd",
            "temporal_transformer": "#0abde3",
            "st_gnn": "#feca57",
            "stacking_ensemble": "#ff6b6b",
        }

        model_icons = {
            "random_forest": "🌲",
            "hist_gradient_boosting": "📊",
            "xgboost": "⚡",
            "lstm": "🧠",
            "hurdle_lightgbm": "🎯",
            "temporal_transformer": "🔮",
            "st_gnn": "🌐",
            "stacking_ensemble": "🏗️",
        }

        model_arch_descriptions = {
            "random_forest": "300 bootstrap-aggregated decision trees average out high-variance predictions, preventing outlier over-estimation on zero-inflated monsoon rainfall events.",
            "hist_gradient_boosting": "Discrete 255-bin histogram split evaluation captures subtle overnight radiative cooling gradients, with native NaN support for missing satellite channels.",
            "xgboost": "L1/L2 regularized gradient boosting with second-order Taylor approximation prevents overfitting on seasonal peaks while maintaining strong temperature predictions.",
            "lstm": "14-day lookback recurrent network learns temporal momentum patterns directly from sequences, but is sample-constrained on 2-year daily records compared to tree ensembles.",
            "hurdle_lightgbm": "Two-stage architecture: binary rain occurrence classifier gates a Tweedie intensity regressor, eliminating false rain predictions on dry days and improving temperature via dedicated gradient-boosted trees.",
            "temporal_transformer": "Multi-head self-attention with sinusoidal positional encodings captures long-range seasonal teleconnections across 14-day windows, with learned temporal pooling for multi-target regression.",
            "st_gnn": "Graph Convolutional Network propagates atmospheric state signals across India's 7 pilot regions using geodesic distance and monsoon track adjacency weights for spatially-aware predictions.",
            "stacking_ensemble": "Non-negative Ridge meta-learner blends cross-validated predictions from RF, HistGB, XGBoost, and LightGBM with physics constraints: Tmax ≥ Tmin + 0.5°C and Rain ≥ 0.",
        }

        targets_detailed: dict[str, list[dict[str, Any]]] = {}

        # Dynamically discover all model keys from metrics
        all_model_keys: list[str] = []
        if self._multi_model_loaded and self._all_model_metrics:
            all_model_keys = sorted(set(m["model_name"] for m in self._all_model_metrics))

        chart_data: dict[str, Any] = {
            "labels": [model_display_names.get(k, k) for k in all_model_keys],
            "model_keys": all_model_keys,
            "colors": [model_colors.get(k, "#888") for k in all_model_keys],
            "rainfall_mae": [],
            "tmax_mae": [],
            "tmin_mae": [],
            "r2_scores": {"rainfall": [], "tmax": [], "tmin": []},
            "error_gap_pct": {"rainfall": [], "tmax": [], "tmin": []},
            "training_times": [],
        }

        if self._multi_model_loaded and self._all_model_metrics:
            for target in ["rainfall_mm", "tmax_c", "tmin_c"]:
                t_metrics = [m for m in self._all_model_metrics if m["target"] == target]
                if not t_metrics:
                    continue
                t_metrics.sort(key=lambda x: x["mae"])
                best_mae = t_metrics[0]["mae"]
                best_rmse = t_metrics[0]["rmse"]
                best_r2 = t_metrics[0]["r2"]

                rows = []
                for rank, m in enumerate(t_metrics, 1):
                    mae = m["mae"]
                    rmse = m["rmse"]
                    r2 = m["r2"]
                    diff_mae = round(mae - best_mae, 4)
                    diff_mae_pct = round(((mae - best_mae) / best_mae) * 100, 1) if best_mae > 0 else 0.0
                    diff_rmse = round(rmse - best_rmse, 4)
                    diff_rmse_pct = round(((rmse - best_rmse) / best_rmse) * 100, 1) if best_rmse > 0 else 0.0
                    diff_r2 = round(r2 - best_r2, 4)
                    accuracy_score = round(max(0.0, r2) * 100, 1)

                    rows.append({
                        "rank": rank,
                        "model": m["model_name"],
                        "display_name": model_display_names.get(m["model_name"], m["model_name"]),
                        "mae": round(mae, 4),
                        "diff_mae": diff_mae,
                        "diff_mae_pct": diff_mae_pct,
                        "rmse": round(rmse, 4),
                        "diff_rmse": diff_rmse,
                        "diff_rmse_pct": diff_rmse_pct,
                        "r2": round(r2, 4),
                        "diff_r2": diff_r2,
                        "accuracy_score": accuracy_score,
                        "training_time": round(m.get("training_time_s", 0.0), 2),
                        "is_best": rank == 1,
                        "verdict": "🏆 1st (Best Model)" if rank == 1 else f"+{diff_mae_pct}% error (+{diff_mae})",
                        "verdict_badge": "best" if rank == 1 else ("close" if diff_mae_pct < 10 else "gap"),
                    })
                targets_detailed[target] = rows

            # Build chart data for all models
            for m_key in all_model_keys:
                for t, key in [("rainfall_mm", "rainfall_mae"), ("tmax_c", "tmax_mae"), ("tmin_c", "tmin_mae")]:
                    found = [m for m in self._all_model_metrics if m["model_name"] == m_key and m["target"] == t]
                    short_t = "rainfall" if "rainfall" in t else ("tmax" if "tmax" in t else "tmin")
                    if found:
                        chart_data[key].append(round(found[0]["mae"], 3))
                        chart_data["r2_scores"][short_t].append(round(max(0.0, found[0]["r2"]) * 100, 1))
                        t_rows = targets_detailed.get(t, [])
                        m_row = [r for r in t_rows if r["model"] == m_key]
                        chart_data["error_gap_pct"][short_t].append(m_row[0]["diff_mae_pct"] if m_row else 0.0)
                    else:
                        chart_data[key].append(0.0)
                        chart_data["r2_scores"][short_t].append(0.0)
                        chart_data["error_gap_pct"][short_t].append(0.0)

                m_rows = [m for m in self._all_model_metrics if m["model_name"] == m_key]
                chart_data["training_times"].append(round(m_rows[0].get("training_time_s", 0.0), 2) if m_rows else 0.0)

        # Dynamically compute overall ranking from actual metrics
        overall_ranking = []
        if self._multi_model_loaded and self._all_model_metrics:
            model_aggregates: dict[str, dict[str, Any]] = {}
            for m_key in all_model_keys:
                m_all = [m for m in self._all_model_metrics if m["model_name"] == m_key]
                if not m_all:
                    continue
                avg_mae = round(sum(m["mae"] for m in m_all) / len(m_all), 3)
                avg_r2 = round(sum(max(0.0, m["r2"]) for m in m_all) / len(m_all), 3)
                training_time = round(m_all[0].get("training_time_s", 0.0), 2)

                # Count target wins
                wins = 0
                win_targets_list: list[str] = []
                target_labels = {"rainfall_mm": "Rainfall", "tmax_c": "Max Temp", "tmin_c": "Min Temp"}
                for target in ["rainfall_mm", "tmax_c", "tmin_c"]:
                    t_metrics = [m for m in self._all_model_metrics if m["target"] == target]
                    if t_metrics:
                        t_metrics.sort(key=lambda x: x["mae"])
                        if t_metrics[0]["model_name"] == m_key:
                            wins += 1
                            win_targets_list.append(target_labels[target])

                model_aggregates[m_key] = {
                    "avg_mae": avg_mae,
                    "avg_r2": avg_r2,
                    "overall_accuracy": round(avg_r2 * 100, 1),
                    "training_time": training_time,
                    "wins": wins,
                    "win_targets": " & ".join(win_targets_list) if win_targets_list else "—",
                }

            # Sort by: most target wins first, then lowest avg_mae
            sorted_models = sorted(
                model_aggregates.items(),
                key=lambda x: (-x[1]["wins"], x[1]["avg_mae"]),
            )

            rank_badges = ["🥇 OVERALL WINNER", "🥈 RUNNER-UP", "🥉 3RD PLACE"]
            for rank, (m_key, agg) in enumerate(sorted_models, 1):
                badge = rank_badges[rank - 1] if rank <= 3 else f"{rank}TH PLACE"
                icon = model_icons.get(m_key, "📦")
                desc = model_arch_descriptions.get(m_key, "Machine learning model for climate prediction.")

                overall_ranking.append({
                    "rank": rank,
                    "model": m_key,
                    "display_name": model_display_names.get(m_key, m_key),
                    "icon": icon,
                    "wins": agg["wins"],
                    "win_targets": agg["win_targets"],
                    "avg_mae": agg["avg_mae"],
                    "avg_r2": agg["avg_r2"],
                    "overall_accuracy": agg["overall_accuracy"],
                    "training_time": agg["training_time"],
                    "badge": badge,
                    "verdict": desc,
                    "strengths": desc,
                    "color": model_colors.get(m_key, "#888"),
                })

        return {
            "targets": targets_detailed,
            "overall_ranking": overall_ranking,
            "chart_data": chart_data,
            "best_models": self._best_models,
            "model_display_names": model_display_names,
            "model_colors": model_colors,
            "model_icons": model_icons,
            "model_arch_descriptions": model_arch_descriptions,
        }

    def get_multi_model_forecasts(
        self,
        history: pd.DataFrame,
        horizon_days: int = 14,
        rainfall_delta_pct: float = 0.0,
        temp_delta_c: float = 0.0,
    ) -> dict[str, Any]:
        """Generate 14-day forecasts from all trained models simultaneously for comparison."""
        from ..models import list_available_models, load_model

        region = str(history.iloc[-1]["region"])
        dates = [
            (pd.to_datetime(history.iloc[-1]["date"]) + pd.Timedelta(days=i + 1)).strftime("%d %b")
            for i in range(horizon_days)
        ]

        result: dict[str, Any] = {
            "labels": dates,
            "models": {},
        }

        for model_name in list_available_models():
            model_dir = MODELS_DIR / model_name
            if not model_dir.exists():
                continue
            if model_name not in self._loaded_models:
                try:
                    self._loaded_models[model_name] = load_model(model_name, model_dir)
                except Exception as exc:
                    logger.warning(f"Failed to load {model_name}: {exc}")
                    continue

            m = self._loaded_models.get(model_name)
            if not m:
                continue

            working = history.copy().sort_values("date").reset_index(drop=True)
            rain_list, tmax_list, tmin_list = [], [], []

            try:
                for step in range(horizon_days):
                    feat = self._make_feature_row(working, step + 1)
                    pred = m.predict(feat[self.feature_columns].values)

                    r = max(0.0, float(pred[0][0]) * (1.0 + rainfall_delta_pct / 100.0))
                    tx = float(pred[0][1]) + temp_delta_c
                    tn = float(pred[0][2]) + temp_delta_c * 0.82

                    rain_list.append(round(r, 2))
                    tmax_list.append(round(tx, 2))
                    tmin_list.append(round(tn, 2))

                    next_row = {
                        "date": pd.to_datetime(working.iloc[-1]["date"]) + pd.Timedelta(days=1),
                        "region": region,
                        "rainfall_mm": r,
                        "tmax_c": tx,
                        "tmin_c": tn,
                        "latitude": float(working.iloc[-1]["latitude"]),
                        "longitude": float(working.iloc[-1]["longitude"]),
                        "humidity_pct": 80.0,
                        "diurnal_range_c": max(1.0, tx - tn),
                    }
                    working = pd.concat([working, pd.DataFrame([next_row])], ignore_index=True)

                result["models"][model_name] = {
                    "rainfall": rain_list,
                    "tmax": tmax_list,
                    "tmin": tmin_list,
                }
            except Exception as exc:
                logger.warning(f"Error forecasting with {model_name}: {exc}")

        return result

    def _make_feature_row(self, history: pd.DataFrame, step_ahead: int) -> pd.DataFrame:
        target_date = pd.to_datetime(history.iloc[-1]["date"]) + pd.Timedelta(days=step_ahead)
        return prepare_inference_features(
            history=history,
            region_code_map=self.region_code_map,
            target_date=target_date,
            step_ahead=step_ahead,
        )
