from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


DECISION_COLUMNS = [
    "trial_index",
    "decision_timestamp_utc",
    "policy_version",
    "shadow_only_no_control",
    "chariot_shadow_action",
    "chariot_flag_code",
    "chariot_reason",
    "decision_insufficient",
    "x1",
    "x2",
    "x3",
    "x4",
    "conductivity_target_S_per_m",
    "measured_eis_conductivity_S_per_m",
    "twin_predicted_conductivity_S_per_m",
    "twin_raw_uncertainty_S_per_m",
    "prediction_timestamp_utc",
    "training_data_last_trial",
    "prediction_source",
    "calibration_points_used",
    "calibration_coverage_target",
    "calibration_factor",
    "raw_band_coverage_prior",
    "calibrated_band_coverage_prior",
    "model_ready",
    "model_ready_reason",
    "calibrated_prediction_lower_S_per_m",
    "calibrated_prediction_upper_S_per_m",
    "raw_disagreement_ratio",
    "normalized_disagreement_score",
    "fixed_gate_action",
    "fixed_gate_would_skip",
    "review_due_to_disagreement",
    "prediction_composition_verified",
    "causal_absolute_prediction_error_S_per_m",
]


def load_decision_log(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    df = pd.read_csv(p)
    for col in DECISION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[DECISION_COLUMNS + [c for c in df.columns if c not in DECISION_COLUMNS]]


def append_frozen_decision(path: str | Path, record: dict) -> pd.DataFrame:
    """Append a decision once; never revise a prior trial's decision.

    The file is atomically replaced after appending so readers never observe a
    half-written file. Existing decision rows are copied byte-logically, not
    re-decided or edited.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_decision_log(p)
    trial = int(record["trial_index"])
    if not existing.empty and (pd.to_numeric(existing["trial_index"], errors="coerce") == trial).any():
        raise ValueError(
            f"Decision for trial {trial} already exists in {p}. Frozen decisions are never overwritten."
        )

    row = {col: record.get(col, pd.NA) for col in DECISION_COLUMNS}
    extras = {k: v for k, v in record.items() if k not in row}
    row.update(extras)
    out = pd.DataFrame([row]) if existing.empty else pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    out = out.sort_values("trial_index", kind="stable").reset_index(drop=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, p)
    return out
