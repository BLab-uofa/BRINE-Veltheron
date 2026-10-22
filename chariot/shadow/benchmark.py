from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import json
import math
import pandas as pd

from shadow.calibration import calibration_state_from_log
from shadow.policy import decide_shadow, ShadowAction, ShadowDecision


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def apply_policy_to_prediction_history(prediction_history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reapply CHARIOT causally to fixed, already-causal pre-trial predictions.

    This is used for sensitivity analysis. It does not refit the replay GP and
    therefore isolates policy/calibration sensitivity from surrogate variation.
    """
    target = float(cfg["campaign"]["conductivity_target_S_per_m"])
    tri = cfg.get("triage", {})
    history = prediction_history.sort_values("trial_index", kind="stable").reset_index(drop=True)
    records: list[dict] = []

    for _, row in history.iterrows():
        prior = pd.DataFrame(records)
        cal = calibration_state_from_log(prior, cfg)
        y = row.get("measured_eis_conductivity_S_per_m")
        mu = row.get("twin_predicted_conductivity_S_per_m")
        unc = row.get("twin_raw_uncertainty_S_per_m")

        d = decide_shadow(
            y,
            mu,
            unc,
            calibration=cal,
            target=target,
            borderline_band=float(tri.get("borderline_band_S_per_m", 0.5)),
            disagreement_score_threshold=float(tri.get("disagreement_score_threshold", 1.0)),
            uncertainty_floor=float(tri.get("uncertainty_floor_S_per_m", 0.05)),
            require_prediction_upper_below_target_for_skip=bool(tri.get("require_prediction_upper_below_target_for_skip", True)),
        )
        if d.action in {ShadowAction.WAIT_FOR_TWIN, ShadowAction.WAIT_FOR_EIS}:
            d = ShadowDecision(
                ShadowAction.REVIEW_LSV,
                "Decision insufficient during causal replay; conservative REVIEW.",
                "MODEL_NOT_READY",
                True,
                target,
                d.calibrated_lower_S_per_m,
                d.calibrated_upper_S_per_m,
                d.raw_disagreement_ratio,
                d.normalized_disagreement_score,
            )

        trial = int(row["trial_index"])
        fixed_action = "run_lsv" if (_finite(y) and float(y) >= target) else "skip_lsv"
        records.append({
            "trial_index": trial,
            "measured_eis_conductivity_S_per_m": float(y) if _finite(y) else None,
            "twin_predicted_conductivity_S_per_m": float(mu) if _finite(mu) else None,
            "twin_raw_uncertainty_S_per_m": float(unc) if _finite(unc) else None,
            "chariot_shadow_action": d.action.value,
            "chariot_flag_code": d.flag_code,
            "decision_insufficient": d.decision_insufficient,
            "calibration_points_used": cal.n_points,
            "calibration_factor": cal.factor,
            "model_ready": cal.model_ready,
            "calibrated_prediction_lower_S_per_m": d.calibrated_lower_S_per_m,
            "calibrated_prediction_upper_S_per_m": d.calibrated_upper_S_per_m,
            "raw_disagreement_ratio": d.raw_disagreement_ratio,
            "normalized_disagreement_score": d.normalized_disagreement_score,
            "fixed_gate_action": fixed_action,
            "fixed_gate_would_skip": fixed_action == "skip_lsv",
            "review_due_to_disagreement": d.flag_code == "TWIN_EIS_DISAGREEMENT",
        })
    return pd.DataFrame(records)


def benchmark_summary(decisions: pd.DataFrame, target: float) -> dict:
    if decisions.empty:
        return {"n_trials": 0}

    d = decisions.copy()
    y = pd.to_numeric(d["measured_eis_conductivity_S_per_m"], errors="coerce")
    n = int(y.notna().sum())
    fixed_skip = y < float(target)
    fixed_run = y >= float(target)

    actions = d["chariot_shadow_action"].astype(str)
    ch_skip = actions.eq("skip_lsv") & y.notna()
    ch_review = actions.eq("review_lsv") & y.notna()
    ch_run = actions.eq("run_lsv") & y.notna()
    ch_lsv = ch_run | ch_review

    n_fixed_skips = int(fixed_skip.sum())
    n_ch_skips = int(ch_skip.sum())
    n_qualified = int(fixed_run.sum())
    qualified_skipped = int((fixed_run & ch_skip).sum())

    below_review = fixed_skip & ch_review
    disagreement_review = below_review & d.get("review_due_to_disagreement", pd.Series(False, index=d.index)).fillna(False).astype(bool)

    pred = pd.to_numeric(d.get("twin_predicted_conductivity_S_per_m"), errors="coerce")
    err_mask = pred.notna() & y.notna()
    mae = float((pred[err_mask] - y[err_mask]).abs().mean()) if err_mask.any() else None

    lower = pd.to_numeric(d.get("calibrated_prediction_lower_S_per_m"), errors="coerce")
    upper = pd.to_numeric(d.get("calibrated_prediction_upper_S_per_m"), errors="coerce")
    band_mask = lower.notna() & upper.notna() & y.notna()
    band_coverage = float(((y[band_mask] >= lower[band_mask]) & (y[band_mask] <= upper[band_mask])).mean()) if band_mask.any() else None

    return {
        "n_trials": n,
        "full_protocol": {
            "lsv_performed": n,
            "lsv_avoided": 0,
            "lsv_reduction_vs_full": 0.0 if n else None,
        },
        "fixed_20_S_per_m_gate": {
            "lsv_performed": int(fixed_run.sum()),
            "lsv_avoided": n_fixed_skips,
            "lsv_reduction_vs_full": (n_fixed_skips / n) if n else None,
            "note": "Maximum-savings baseline. It blindly skips every below-target EIS result and has no model-disagreement review state.",
        },
        "chariot": {
            "run_lsv": int(ch_run.sum()),
            "review_lsv": int(ch_review.sum()),
            "skip_lsv": n_ch_skips,
            "effective_lsv_performed_run_plus_review": int(ch_lsv.sum()),
            "lsv_avoided": n_ch_skips,
            "lsv_reduction_vs_full": (n_ch_skips / n) if n else None,
            "fixed_gate_savings_retained_fraction": (n_ch_skips / n_fixed_skips) if n_fixed_skips else None,
            "below_target_cases_escalated_to_review": int(below_review.sum()),
            "below_target_disagreement_cases_escalated_to_review": int(disagreement_review.sum()),
            "conductivity_qualified_candidates": n_qualified,
            "conductivity_qualified_candidates_auto_skipped": qualified_skipped,
            "conductivity_qualified_retention_fraction": ((n_qualified - qualified_skipped) / n_qualified) if n_qualified else None,
            "mean_absolute_twin_error_S_per_m": mae,
            "empirical_calibrated_band_coverage": band_coverage,
            "headline_note": "CHARIOT is not designed to beat the blind fixed gate on raw savings. Its value is retaining much of that savings while escalating uncertain/model-disagreement cases instead of automatically discarding them.",
        },
    }


def sensitivity_table(prediction_history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    grid = cfg.get("benchmark", {}).get("sensitivity", {})
    mins = grid.get("min_calibration_points", [8, 10, 12])
    coverages = grid.get("calibration_coverage_target", [0.85, 0.90, 0.95])
    disagreements = grid.get("disagreement_score_threshold", [0.8, 1.0, 1.2])
    target = float(cfg["campaign"]["conductivity_target_S_per_m"])

    rows = []
    for min_pts, cov, dis in product(mins, coverages, disagreements):
        local = deepcopy(cfg)
        local.setdefault("triage", {})
        local["triage"]["min_calibration_points"] = int(min_pts)
        local["triage"]["calibration_coverage_target"] = float(cov)
        local["triage"]["disagreement_score_threshold"] = float(dis)
        decisions = apply_policy_to_prediction_history(prediction_history, local)
        summary = benchmark_summary(decisions, target)
        ch = summary.get("chariot", {})
        rows.append({
            "min_calibration_points": int(min_pts),
            "calibration_coverage_target": float(cov),
            "disagreement_score_threshold": float(dis),
            "run_lsv": ch.get("run_lsv"),
            "review_lsv": ch.get("review_lsv"),
            "skip_lsv": ch.get("skip_lsv"),
            "lsv_reduction_vs_full": ch.get("lsv_reduction_vs_full"),
            "fixed_gate_savings_retained_fraction": ch.get("fixed_gate_savings_retained_fraction"),
            "below_target_cases_escalated_to_review": ch.get("below_target_cases_escalated_to_review"),
            "below_target_disagreement_cases_escalated_to_review": ch.get("below_target_disagreement_cases_escalated_to_review"),
            "conductivity_qualified_retention_fraction": ch.get("conductivity_qualified_retention_fraction"),
        })
    return pd.DataFrame(rows)


def save_benchmark(summary: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
