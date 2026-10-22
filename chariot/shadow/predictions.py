from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import os
import time

import pandas as pd


PREDICTION_COLUMNS = [
    "trial_index",
    "prediction_timestamp_utc",
    "training_data_last_trial",
    "x1",
    "x2",
    "x3",
    "x4",
    "twin_predicted_conductivity_S_per_m",
    "twin_raw_uncertainty_S_per_m",
    "prediction_source",
]

# Reza may use the shorter, more natural column names we agreed on.  The
# loader normalizes them so the rest of CHARIOT has one internal schema.
ALIASES = {
    "predicted_conductivity_S_per_m": "twin_predicted_conductivity_S_per_m",
    "raw_uncertainty_S_per_m": "twin_raw_uncertainty_S_per_m",
    "predicted_conductivity": "twin_predicted_conductivity_S_per_m",
    "raw_uncertainty": "twin_raw_uncertainty_S_per_m",
    "source": "prediction_source",
}


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _read_csv_with_retry(path: Path, attempts: int = 3, delay_s: float = 0.08) -> pd.DataFrame:
    """Read a CSV that another process may be appending to.

    BRINE/twin files are external and may briefly be between writes on Windows.
    A tiny retry avoids treating a transient half-write as a scientific error.
    """
    last_exc: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            return pd.read_csv(path)
        except (pd.errors.ParserError, PermissionError, OSError) as exc:
            last_exc = exc
            if i < attempts - 1:
                time.sleep(delay_s)
    assert last_exc is not None
    raise last_exc


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for source, target in ALIASES.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    for col in PREDICTION_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[PREDICTION_COLUMNS + [c for c in out.columns if c not in PREDICTION_COLUMNS]]


def load_prediction_log(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return normalize_prediction_columns(_read_csv_with_retry(p))


def validate_prediction_log(df: pd.DataFrame) -> list[str]:
    """Return human-readable integrity problems without modifying the file."""
    problems: list[str] = []
    if df is None or df.empty:
        return problems

    trial_num = pd.to_numeric(df["trial_index"], errors="coerce")
    if trial_num.isna().any():
        bad = df.index[trial_num.isna()].tolist()
        problems.append(f"Prediction rows have missing/non-numeric trial_index at row indices {bad}.")

    duplicates = trial_num.dropna().astype(int)
    duplicates = sorted(duplicates[duplicates.duplicated(keep=False)].unique().tolist())
    if duplicates:
        problems.append(
            "Duplicate frozen twin predictions exist for trial(s) "
            f"{duplicates}. Live CHARIOT requires exactly one pre-trial prediction per trial."
        )

    for idx, row in df.iterrows():
        if not _finite(row.get("trial_index")):
            continue
        trial = int(float(row["trial_index"]))
        training_last = row.get("training_data_last_trial")
        if not _finite(training_last):
            problems.append(
                f"Trial {trial}: training_data_last_trial is missing/non-numeric; causality cannot be verified."
            )
        elif int(float(training_last)) >= trial:
            problems.append(
                f"Trial {trial}: causal leakage guard failed because "
                f"training_data_last_trial={int(float(training_last))} is not < trial_index={trial}."
            )

        mean = row.get("twin_predicted_conductivity_S_per_m")
        unc = row.get("twin_raw_uncertainty_S_per_m")
        if not _finite(mean):
            problems.append(f"Trial {trial}: predicted conductivity is missing/non-finite.")
        if not _finite(unc) or float(unc) <= 0:
            problems.append(
                f"Trial {trial}: raw uncertainty must be a positive absolute conductivity width in S/m."
            )

    return problems


def append_pretrial_prediction(
    path: str | Path,
    *,
    trial_index: int,
    twin_mean: float,
    twin_raw_uncertainty: float,
    training_data_last_trial: int,
    x1: float | None = None,
    x2: float | None = None,
    x3: float | None = None,
    x4: float | None = None,
    source: str = "external_twin",
    timestamp_utc: str | None = None,
) -> pd.DataFrame:
    """Append one immutable PRE-TRIAL twin prediction.

    This helper is optional in production: Reza may write the CSV himself.
    The live loader accepts the agreed external column names and validates them.
    """
    trial_index = int(trial_index)
    training_data_last_trial = int(training_data_last_trial)
    if training_data_last_trial >= trial_index:
        raise ValueError(
            f"Causal leakage guard: training_data_last_trial={training_data_last_trial} "
            f"must be < trial_index={trial_index}."
        )
    if not _finite(twin_mean):
        raise ValueError("twin_mean must be finite.")
    if not _finite(twin_raw_uncertainty) or float(twin_raw_uncertainty) <= 0:
        raise ValueError(
            "twin_raw_uncertainty must be a positive absolute width in S/m. "
            "Do not pass an unlabeled percentage."
        )

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_prediction_log(p)
    if not existing.empty and (
        pd.to_numeric(existing["trial_index"], errors="coerce") == trial_index
    ).any():
        raise ValueError(
            f"A pre-trial prediction for trial {trial_index} is already frozen in {p}. "
            "Predictions are append-only and are never overwritten."
        )

    row = {
        "trial_index": trial_index,
        "prediction_timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "training_data_last_trial": training_data_last_trial,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "twin_predicted_conductivity_S_per_m": float(twin_mean),
        "twin_raw_uncertainty_S_per_m": float(twin_raw_uncertainty),
        "prediction_source": source,
    }
    out = (
        pd.DataFrame([row])
        if existing.empty
        else pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    )

    tmp = p.with_suffix(p.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, p)
    return out


def prediction_for_trial(predictions: pd.DataFrame, trial_index: int) -> dict | None:
    if predictions is None or predictions.empty:
        return None

    idx = pd.to_numeric(predictions["trial_index"], errors="coerce")
    rows = predictions[idx == int(trial_index)]
    if rows.empty:
        return None
    if len(rows) > 1:
        raise ValueError(
            f"Duplicate frozen predictions found for trial {trial_index}. "
            "CHARIOT will not choose one silently."
        )

    row = rows.iloc[0].to_dict()
    training_last = row.get("training_data_last_trial")
    if not _finite(training_last):
        raise ValueError(
            f"Causality cannot be verified for trial {trial_index}: "
            "training_data_last_trial is missing/non-numeric."
        )
    if int(float(training_last)) >= int(trial_index):
        raise ValueError(
            f"Causal leakage detected in pre-trial prediction for trial {trial_index}: "
            f"training_data_last_trial={training_last}."
        )

    mean = row.get("twin_predicted_conductivity_S_per_m")
    unc = row.get("twin_raw_uncertainty_S_per_m")
    if not _finite(mean):
        raise ValueError(f"Trial {trial_index}: predicted conductivity is missing/non-finite.")
    if not _finite(unc) or float(unc) <= 0:
        raise ValueError(
            f"Trial {trial_index}: raw uncertainty must be a positive absolute width in S/m."
        )

    return row
