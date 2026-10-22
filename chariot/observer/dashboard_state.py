from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import load_config
from observer.analytics import campaign_summary
from observer.loader import load_brine_csv
from shadow.decision_log import load_decision_log
from shadow.predictions import load_prediction_log, validate_prediction_log
from shadow.status import live_pipeline_status
from shadow.view import build_shadow_view


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if pd.notna(x) else None


def build_dashboard_snapshot(
    campaign_file: str | Path,
    prediction_file: str | Path,
    decision_log_file: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the complete live dashboard state from disk.

    No state is cached here. Every call reflects the files as they exist at
    that moment, matching the live Gemini observer design.
    """
    cfg = load_config(config_path)
    campaign_path = Path(campaign_file)
    prediction_path = Path(prediction_file)
    decision_path = Path(decision_log_file)

    campaign = load_brine_csv(campaign_path, cfg)
    predictions = load_prediction_log(prediction_path)
    decisions = load_decision_log(decision_path)
    shadow_view = build_shadow_view(campaign, decisions)

    summary = campaign_summary(campaign, shadow_view, cfg)
    status = live_pipeline_status(
        campaign_path,
        prediction_path,
        decision_path,
        cfg,
    )

    best = summary.get("best_by_conductivity") or {}
    latest = summary.get("latest") or {}
    chariot = summary.get("chariot_shadow") or {}

    latest_decision = None
    if not decisions.empty:
        order = pd.to_numeric(decisions["trial_index"], errors="coerce")
        if order.notna().any():
            latest_decision = decisions.loc[order.idxmax()].to_dict()

    latest_prediction = None
    if not predictions.empty and "trial_index" in predictions.columns:
        order = pd.to_numeric(predictions["trial_index"], errors="coerce")
        if order.notna().any():
            latest_prediction = predictions.loc[order.idxmax()].to_dict()

    target = float(cfg["campaign"]["conductivity_target_S_per_m"])
    conductivity_stats = (
        summary.get("outputs", {}).get("ionic_conductivity_S_per_m", {})
    )

    return {
        "cfg": cfg,
        "campaign": campaign,
        "predictions": predictions,
        "decisions": decisions,
        "shadow_view": shadow_view,
        "summary": summary,
        "status": status,
        "target_S_per_m": target,
        "best_trial": best,
        "latest_trial": latest,
        "latest_decision": latest_decision,
        "latest_prediction": latest_prediction,
        "best_conductivity_S_per_m": _safe_float(best.get("conductivity")),
        "latest_conductivity_S_per_m": _safe_float(latest.get("conductivity")),
        "mean_conductivity_S_per_m": _safe_float(conductivity_stats.get("mean")),
        "run_lsv": int(chariot.get("run_lsv", 0) or 0),
        "review_lsv": int(chariot.get("review_lsv", 0) or 0),
        "skip_lsv": int(chariot.get("skip_lsv", 0) or 0),
        "review_breakdown": chariot.get("review_breakdown", {}),
        "model_ready_counts": chariot.get("model_ready_counts", {}),
        "prediction_integrity_errors": validate_prediction_log(predictions),
    }
