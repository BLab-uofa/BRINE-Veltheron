from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math

import pandas as pd

from observer.loader import load_brine_csv
from shadow.calibration import calibration_state_from_log
from shadow.decision_log import append_frozen_decision, load_decision_log
from shadow.policy import decide_shadow, ShadowAction
from shadow.predictions import (
    load_prediction_log,
    prediction_for_trial,
    validate_prediction_log,
)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _composition_check(
    campaign_row: pd.Series,
    prediction: dict,
    tolerance_uL: float,
) -> tuple[bool | None, str | None]:
    """Verify the prediction refers to the formulation BRINE actually ran.

    Returns (None, None) when Reza's file omitted composition.  That is allowed
    for compatibility, but the decision log records that composition was not
    independently verified.
    """
    pred_values = [prediction.get(k) for k in ("x1", "x2", "x3", "x4")]
    if all(pd.isna(v) for v in pred_values):
        return None, None

    mismatches: list[str] = []
    for key in ("x1", "x2", "x3", "x4"):
        pv = prediction.get(key)
        cv = campaign_row.get(key)
        if pd.isna(pv) or pd.isna(cv):
            mismatches.append(f"{key}: missing value")
            continue
        if abs(float(pv) - float(cv)) > tolerance_uL:
            mismatches.append(f"{key}: twin={float(pv):.6g} uL, BRINE={float(cv):.6g} uL")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, None


def process_new_trials(
    campaign_csv: str | Path,
    prediction_log_csv: str | Path,
    decision_log_csv: str | Path,
    cfg: dict,
    *,
    start_trial: int | None = None,
) -> dict:
    """Append new causal CHARIOT shadow decisions from live files.

    BRINE and the digital twin are READ ONLY.  A decision is created only when:
      1) the BRINE row contains measured EIS conductivity,
      2) exactly one valid frozen pre-trial twin prediction exists,
      3) the causal leakage guard passes, and
      4) if prediction composition is present, it matches BRINE.

    Old decisions are never recomputed when BRINE later fills HER/OER columns.
    """
    df = load_brine_csv(campaign_csv, cfg).sort_values("trial_index", kind="stable")
    predictions = load_prediction_log(prediction_log_csv)
    decisions = load_decision_log(decision_log_csv)

    integrity_errors = validate_prediction_log(predictions)

    campaign_trials = pd.to_numeric(df.get("trial_index"), errors="coerce").dropna().astype(int)
    dup_campaign = sorted(campaign_trials[campaign_trials.duplicated(keep=False)].unique().tolist())
    if dup_campaign:
        integrity_errors.append(
            f"BRINE CSV contains duplicate trial_index values {dup_campaign}; "
            "CHARIOT will not freeze decisions for ambiguous rows."
        )

    decided = set(
        pd.to_numeric(
            decisions.get("trial_index", pd.Series(dtype=float)), errors="coerce"
        ).dropna().astype(int).tolist()
    )

    target = float(cfg["campaign"]["conductivity_target_S_per_m"])
    tri = cfg.get("triage", {})
    live_cfg = cfg.get("live", {})
    tolerance_uL = float(live_cfg.get("composition_match_tolerance_uL", 0.01))
    policy_version = cfg["shadow"].get("policy_version", "uncertainty_triage_v2")

    appended: list[dict] = []
    waiting_eis: list[int] = []
    waiting_twin: list[int] = []
    waiting_integrity: list[int] = []

    for _, row in df.iterrows():
        if pd.isna(row.get("trial_index")):
            continue
        trial = int(row["trial_index"])
        if start_trial is not None and trial < int(start_trial):
            continue
        if trial in dup_campaign:
            waiting_integrity.append(trial)
            continue
        if trial in decided:
            continue
        if pd.isna(row.get("conductivity")):
            waiting_eis.append(trial)
            continue

        try:
            pred = prediction_for_trial(predictions, trial)
        except ValueError as exc:
            integrity_errors.append(str(exc))
            waiting_integrity.append(trial)
            continue

        if pred is None:
            waiting_twin.append(trial)
            continue

        composition_verified, composition_problem = _composition_check(
            row, pred, tolerance_uL=tolerance_uL
        )
        if composition_verified is False:
            integrity_errors.append(
                f"Trial {trial}: frozen twin prediction composition does not match BRINE: "
                f"{composition_problem}. Decision not frozen."
            )
            waiting_integrity.append(trial)
            continue

        # Calibration uses only decisions frozen for PRIOR trials.
        prior = (
            decisions[
                pd.to_numeric(decisions["trial_index"], errors="coerce") < trial
            ].copy()
            if not decisions.empty
            else decisions
        )
        cal = calibration_state_from_log(prior, cfg)

        y = float(row["conductivity"])
        mu = float(pred["twin_predicted_conductivity_S_per_m"])
        raw_u = float(pred["twin_raw_uncertainty_S_per_m"])
        decision = decide_shadow(
            y,
            mu,
            raw_u,
            calibration=cal,
            target=target,
            borderline_band=float(tri.get("borderline_band_S_per_m", 0.5)),
            disagreement_score_threshold=float(
                tri.get("disagreement_score_threshold", 1.0)
            ),
            uncertainty_floor=float(tri.get("uncertainty_floor_S_per_m", 0.05)),
            require_prediction_upper_below_target_for_skip=bool(
                tri.get("require_prediction_upper_below_target_for_skip", True)
            ),
        )

        if decision.action in {ShadowAction.WAIT_FOR_EIS, ShadowAction.WAIT_FOR_TWIN}:
            if decision.action == ShadowAction.WAIT_FOR_EIS:
                waiting_eis.append(trial)
            else:
                waiting_twin.append(trial)
            continue

        fixed_action = "run_lsv" if y >= target else "skip_lsv"
        record = {
            "trial_index": trial,
            "decision_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "policy_version": policy_version,
            "shadow_only_no_control": True,
            "chariot_shadow_action": decision.action.value,
            "chariot_flag_code": decision.flag_code,
            "chariot_reason": decision.reason,
            "decision_insufficient": decision.decision_insufficient,
            "x1": row.get("x1"),
            "x2": row.get("x2"),
            "x3": row.get("x3"),
            "x4": row.get("x4"),
            "conductivity_target_S_per_m": target,
            "measured_eis_conductivity_S_per_m": y,
            "twin_predicted_conductivity_S_per_m": mu,
            "twin_raw_uncertainty_S_per_m": raw_u,
            "prediction_timestamp_utc": pred.get("prediction_timestamp_utc"),
            "training_data_last_trial": int(float(pred["training_data_last_trial"])),
            "prediction_source": pred.get("prediction_source", "external_twin"),
            "prediction_composition_verified": composition_verified,
            "calibration_points_used": cal.n_points,
            "calibration_coverage_target": cal.coverage_target,
            "calibration_factor": cal.factor,
            "raw_band_coverage_prior": cal.raw_band_coverage,
            "calibrated_band_coverage_prior": cal.calibrated_band_coverage,
            "model_ready": cal.model_ready,
            "model_ready_reason": cal.ready_reason,
            "calibrated_prediction_lower_S_per_m": decision.calibrated_lower_S_per_m,
            "calibrated_prediction_upper_S_per_m": decision.calibrated_upper_S_per_m,
            "raw_disagreement_ratio": decision.raw_disagreement_ratio,
            "normalized_disagreement_score": decision.normalized_disagreement_score,
            "fixed_gate_action": fixed_action,
            "fixed_gate_would_skip": bool(y < target),
            "review_due_to_disagreement": decision.flag_code == "TWIN_EIS_DISAGREEMENT",
            "causal_absolute_prediction_error_S_per_m": abs(y - mu),
        }
        decisions = append_frozen_decision(decision_log_csv, record)
        decided.add(trial)
        appended.append(record)

    latest_campaign = int(campaign_trials.max()) if len(campaign_trials) else None
    pred_trials = pd.to_numeric(predictions.get("trial_index"), errors="coerce").dropna()
    latest_prediction = int(pred_trials.max()) if len(pred_trials) else None
    decision_trials = pd.to_numeric(decisions.get("trial_index"), errors="coerce").dropna()
    latest_decision = int(decision_trials.max()) if len(decision_trials) else None

    return {
        "appended": appended,
        "waiting_for_eis": sorted(set(waiting_eis)),
        "waiting_for_twin": sorted(set(waiting_twin)),
        "waiting_for_integrity": sorted(set(waiting_integrity)),
        "integrity_errors": list(dict.fromkeys(integrity_errors)),
        "decision_count": int(len(decisions)),
        "prediction_count": int(len(predictions)),
        "campaign_trial_count": int(len(df)),
        "latest_campaign_trial": latest_campaign,
        "latest_prediction_trial": latest_prediction,
        "latest_decision_trial": latest_decision,
    }
