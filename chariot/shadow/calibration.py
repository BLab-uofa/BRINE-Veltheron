from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationState:
    n_points: int
    coverage_target: float
    factor: float | None
    raw_band_coverage: float | None
    calibrated_band_coverage: float | None
    model_ready: bool
    ready_reason: str


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def calibration_state_from_log(decision_log: pd.DataFrame | None, cfg: dict) -> CalibrationState:
    """Build a causal calibration state from already-frozen PRIOR decisions only.

    The twin's uncertainty is treated as a raw width, not a Gaussian sigma.
    We empirically scale that width using past |EIS-prediction| / raw_uncertainty
    ratios. No current-trial information is used by this function when the caller
    passes only the existing decision log.
    """
    tri = cfg.get("triage", {})
    min_points = int(tri.get("min_calibration_points", 10))
    coverage_target = float(tri.get("calibration_coverage_target", 0.90))
    min_factor = float(tri.get("min_calibration_factor", 1.0))
    floor = float(tri.get("uncertainty_floor_S_per_m", 0.05))

    if decision_log is None or decision_log.empty:
        return CalibrationState(0, coverage_target, None, None, None, False, "No prior prediction/error pairs are available.")

    ratios: list[float] = []
    for _, row in decision_log.iterrows():
        mu = row.get("twin_predicted_conductivity_S_per_m")
        unc = row.get("twin_raw_uncertainty_S_per_m")
        y = row.get("measured_eis_conductivity_S_per_m")
        if not (_finite(mu) and _finite(unc) and _finite(y)):
            continue
        unc = max(abs(float(unc)), floor)
        ratios.append(abs(float(y) - float(mu)) / unc)

    n = len(ratios)
    if n == 0:
        return CalibrationState(0, coverage_target, None, None, None, False, "No valid prior prediction/error pairs are available.")

    arr = np.asarray(ratios, dtype=float)
    raw_coverage = float(np.mean(arr <= 1.0))
    # Conservative: never shrink the twin's native uncertainty band.
    q = float(np.quantile(arr, coverage_target, method="higher"))
    factor = max(min_factor, q)
    calibrated_coverage = float(np.mean(arr <= factor))

    if n < min_points:
        return CalibrationState(
            n, coverage_target, factor, raw_coverage, calibrated_coverage, False,
            f"Cold start: {n} prior calibration points; {min_points} required before automatic SKIP is allowed.",
        )

    return CalibrationState(
        n, coverage_target, factor, raw_coverage, calibrated_coverage, True,
        f"Calibration history is sufficient: {n} prior prediction/error pairs. The raw twin uncertainty is empirically scaled by {factor:.2f}; a large factor automatically widens the prediction band rather than being hidden as false confidence.",
    )
