from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

from observer.loader import load_brine_csv
from shadow.calibration import calibration_state_from_log
from shadow.policy import decide_shadow, ShadowAction
from shadow.replay_model import ReplayPrediction, predict_with_causal_gp


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _decision_record(
    row: pd.Series,
    pred: ReplayPrediction,
    cal,
    decision,
    cfg: dict,
    training_data_last_trial: int | None,
) -> dict:
    target = float(cfg["campaign"]["conductivity_target_S_per_m"])
    y = float(row["conductivity"]) if pd.notna(row.get("conductivity")) else None
    trial = int(row["trial_index"])
    fixed_action = "run_lsv" if (y is not None and y >= target) else "skip_lsv"

    causal_abs_error = None
    if y is not None and _finite(pred.mean):
        causal_abs_error = abs(float(y) - float(pred.mean))

    return {
        "trial_index": trial,
        "decision_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy_version": cfg["shadow"].get("policy_version", "uncertainty_triage_v2"),
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
        "twin_predicted_conductivity_S_per_m": pred.mean,
        "twin_raw_uncertainty_S_per_m": pred.raw_uncertainty,
        "prediction_timestamp_utc": None,
        "training_data_last_trial": training_data_last_trial,
        "prediction_source": pred.source,
        "replay_model_training_points": pred.training_points,
        "replay_uncertainty_prior_error_points": pred.uncertainty_points,
        "causal_absolute_prediction_error_S_per_m": causal_abs_error,
        "replay_model_error": pred.error,
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
        "fixed_gate_would_skip": bool(y is not None and y < target),
        "review_due_to_disagreement": decision.flag_code == "TWIN_EIS_DISAGREEMENT",
        "simulated_retrospective": True,
    }


def replay_campaign(
    input_csv: str | Path,
    output_csv: str | Path,
    cfg: dict,
) -> pd.DataFrame:
    """Leakage-free sequential replay of CHARIOT on historical/demo data.

    Causal order for trial n:

        1. Train the replay mean model only on trials 1..n-1.
        2. Build replay uncertainty only from PRIOR one-step-ahead errors.
        3. Predict trial n before revealing its EIS conductivity.
        4. Reveal trial n EIS and make the shadow CHARIOT decision.
        5. Freeze the decision.
        6. Only then add trial n to the future model/error history.

    The replay GP's native standard deviation is intentionally not used. The
    historical replay uncertainty is a causal empirical error scale. Live
    CHARIOT remains unchanged and should consume Reza's actual frozen pre-trial
    prediction plus raw uncertainty.
    """
    df = load_brine_csv(input_csv, cfg).copy()

    if df.empty:
        out = pd.DataFrame()
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        return out

    df = df.sort_values("trial_index", kind="stable").reset_index(drop=True)

    target = float(cfg["campaign"]["conductivity_target_S_per_m"])
    tri = cfg.get("triage", {})

    X_hist: list[list[float]] = []
    y_hist: list[float] = []
    prior_trial_indices: list[int] = []

    # These are strictly prior, out-of-sample one-step-ahead prediction errors.
    prior_abs_prediction_errors: list[float] = []

    records: list[dict] = []

    for _, row in df.iterrows():
        x = np.asarray(
            [
                row.get("x1"),
                row.get("x2"),
                row.get("x3"),
                row.get("x4"),
            ],
            dtype=float,
        )

        y = row.get("conductivity")

        # Calibration is always built from already-frozen PRIOR decisions only.
        prior_log = pd.DataFrame(records)
        cal = calibration_state_from_log(prior_log, cfg)

        if np.isfinite(x).all() and len(y_hist) > 0:
            pred = predict_with_causal_gp(
                np.asarray(X_hist),
                np.asarray(y_hist),
                x,
                cfg,
                prior_abs_errors=prior_abs_prediction_errors,
            )
        else:
            pred = ReplayPrediction(
                None,
                None,
                "causal_replay_gp_mean_empirical_residual_uncertainty",
                len(y_hist),
                len(prior_abs_prediction_errors),
                "No valid prior training set.",
            )

        decision = decide_shadow(
            y,
            pred.mean,
            pred.raw_uncertainty,
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

        # Missing causal model/uncertainty is conservative REVIEW in replay.
        # REVIEW counts as performing LSV, so this cannot inflate savings.
        if decision.action == ShadowAction.WAIT_FOR_TWIN:
            from shadow.policy import ShadowDecision

            decision = ShadowDecision(
                ShadowAction.REVIEW_LSV,
                (
                    "Decision insufficient: a causal replay mean and empirical "
                    "uncertainty estimate were not both available yet. "
                    "Conservative REVIEW during cold start."
                ),
                "MODEL_NOT_READY",
                True,
                target,
                None,
                None,
                None,
                None,
            )

        training_last = max(prior_trial_indices) if prior_trial_indices else None

        records.append(
            _decision_record(
                row,
                pred,
                cal,
                decision,
                cfg,
                training_last,
            )
        )

        # -------------------------------------------------------------
        # ONLY NOW is the current trial allowed to affect future trials.
        # -------------------------------------------------------------
        if np.isfinite(x).all() and pd.notna(y) and _finite(y):
            y_float = float(y)

            # Current error becomes available only after the current decision.
            if _finite(pred.mean):
                prior_abs_prediction_errors.append(
                    abs(y_float - float(pred.mean))
                )

            X_hist.append(x.tolist())
            y_hist.append(y_float)
            prior_trial_indices.append(int(row["trial_index"]))

    out = pd.DataFrame(records)
    p = Path(output_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return out
