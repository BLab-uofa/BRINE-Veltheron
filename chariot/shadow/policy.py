from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from shadow.calibration import CalibrationState


class ShadowAction(str, Enum):
    WAIT_FOR_EIS = "wait_for_eis"
    WAIT_FOR_TWIN = "wait_for_twin"
    RUN_LSV = "run_lsv"
    REVIEW_LSV = "review_lsv"
    SKIP_LSV = "skip_lsv"


@dataclass(frozen=True)
class ShadowDecision:
    action: ShadowAction
    reason: str
    flag_code: str
    decision_insufficient: bool
    target_S_per_m: float
    calibrated_lower_S_per_m: float | None
    calibrated_upper_S_per_m: float | None
    raw_disagreement_ratio: float | None
    normalized_disagreement_score: float | None


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def decide_shadow(
    conductivity: float | None,
    twin_mean: float | None = None,
    twin_raw_uncertainty: float | None = None,
    calibration: CalibrationState | None = None,
    target: float = 20.0,
    borderline_band: float = 0.5,
    disagreement_score_threshold: float = 1.0,
    uncertainty_floor: float = 0.05,
    require_prediction_upper_below_target_for_skip: bool = True,
) -> ShadowDecision:
    """Uncertainty-aware, read-only CHARIOT characterization triage.

    The twin predicts CONDUCTIVITY ONLY. CHARIOT does not predict HER, OER,
    stability window, or LSV outcome. It decides whether the conductivity
    evidence is strong enough to justify spending the destructive LSV.

    The twin uncertainty is deliberately NOT treated as a Gaussian sigma.
    `normalized_disagreement_score` is a dimensionless disagreement score,
    not a z-score.
    """
    if not _finite(conductivity):
        return ShadowDecision(
            ShadowAction.WAIT_FOR_EIS,
            "No processed EIS conductivity is available yet.",
            "WAITING_FOR_EIS",
            True,
            target,
            None,
            None,
            None,
            None,
        )

    y = float(conductivity)

    if not (_finite(twin_mean) and _finite(twin_raw_uncertainty)):
        return ShadowDecision(
            ShadowAction.WAIT_FOR_TWIN,
            "A frozen pre-trial conductivity prediction and raw uncertainty are required before CHARIOT can make a causal shadow decision.",
            "WAITING_FOR_PRETRIAL_TWIN",
            True,
            target,
            None,
            None,
            None,
            None,
        )

    mu = float(twin_mean)
    raw_u = max(abs(float(twin_raw_uncertainty)), float(uncertainty_floor))
    cal = calibration or CalibrationState(0, 0.90, None, None, None, False, "Calibration state unavailable.")

    factor = cal.factor if cal.factor is not None else 1.0
    half_width = max(raw_u * float(factor), float(uncertainty_floor))
    lower = mu - half_width
    upper = mu + half_width
    raw_disagreement = abs(y - mu) / raw_u
    normalized_disagreement = abs(y - mu) / half_width

    if not cal.model_ready:
        return ShadowDecision(
            ShadowAction.REVIEW_LSV,
            f"Decision insufficient: {cal.ready_reason} Measured EIS is {y:.2f} S/m; twin predicted {mu:.2f} S/m with raw uncertainty {raw_u:.2f} S/m.",
            "MODEL_NOT_READY",
            True,
            target,
            lower,
            upper,
            raw_disagreement,
            normalized_disagreement,
        )

    if normalized_disagreement > float(disagreement_score_threshold):
        return ShadowDecision(
            ShadowAction.REVIEW_LSV,
            f"Decision insufficient: measured EIS ({y:.2f} S/m) disagrees with the twin prediction ({mu:.2f} S/m) beyond the empirically calibrated prediction band. Review instead of automatically discarding or accepting the candidate.",
            "TWIN_EIS_DISAGREEMENT",
            True,
            target,
            lower,
            upper,
            raw_disagreement,
            normalized_disagreement,
        )

    # Safety invariant: a conductivity-qualified formulation is never auto-skipped.
    if y >= target:
        return ShadowDecision(
            ShadowAction.RUN_LSV,
            f"Measured EIS conductivity {y:.2f} S/m meets the {target:.2f} S/m qualification target and is consistent with the calibrated twin expectation. The candidate has earned destructive LSV characterization; no LSV outcome is predicted here.",
            "EIS_QUALIFIED",
            False,
            target,
            lower,
            upper,
            raw_disagreement,
            normalized_disagreement,
        )

    if y >= target - float(borderline_band):
        return ShadowDecision(
            ShadowAction.REVIEW_LSV,
            f"Decision insufficient: measured EIS conductivity {y:.2f} S/m is within the configured {borderline_band:.2f} S/m borderline band below the {target:.2f} S/m target.",
            "EIS_BORDERLINE",
            True,
            target,
            lower,
            upper,
            raw_disagreement,
            normalized_disagreement,
        )

    if require_prediction_upper_below_target_for_skip and upper >= target:
        return ShadowDecision(
            ShadowAction.REVIEW_LSV,
            f"Decision insufficient: EIS is below target ({y:.2f} S/m), but the empirically calibrated twin band still reaches the {target:.2f} S/m qualification threshold. The blind fixed gate would skip; CHARIOT escalates this case for review.",
            "PREDICTION_BAND_CROSSES_TARGET",
            True,
            target,
            lower,
            upper,
            raw_disagreement,
            normalized_disagreement,
        )

    return ShadowDecision(
        ShadowAction.SKIP_LSV,
        f"Measured EIS conductivity {y:.2f} S/m is below the {target:.2f} S/m target, the calibrated twin prediction is consistent with that result, and its upper prediction band remains below target. CHARIOT therefore flags LSV as potentially avoidable in shadow mode.",
        "CONFIDENT_BELOW_TARGET",
        False,
        target,
        lower,
        upper,
        raw_disagreement,
        normalized_disagreement,
    )
