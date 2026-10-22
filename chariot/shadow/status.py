from __future__ import annotations

from pathlib import Path
import pandas as pd

from observer.loader import load_brine_csv
from shadow.decision_log import load_decision_log
from shadow.predictions import load_prediction_log, validate_prediction_log


def live_pipeline_status(
    campaign_csv: str | Path,
    prediction_log_csv: str | Path,
    decision_log_csv: str | Path,
    cfg: dict,
) -> dict:
    """Build deterministic current-state counts for the scientist/Gemini."""
    campaign = load_brine_csv(campaign_csv, cfg)
    predictions = load_prediction_log(prediction_log_csv)
    decisions = load_decision_log(decision_log_csv)

    campaign_trials_all = pd.to_numeric(campaign["trial_index"], errors="coerce")
    campaign_trial_set = set(campaign_trials_all.dropna().astype(int).tolist())
    eis_trial_set = set(
        pd.to_numeric(
            campaign.loc[campaign["conductivity"].notna(), "trial_index"], errors="coerce"
        ).dropna().astype(int).tolist()
    )
    pred_trial_set = set(
        pd.to_numeric(predictions.get("trial_index"), errors="coerce")
        .dropna().astype(int).tolist()
    )
    decision_trial_set = set(
        pd.to_numeric(decisions.get("trial_index"), errors="coerce")
        .dropna().astype(int).tolist()
    )

    waiting_for_twin = sorted(eis_trial_set - pred_trial_set - decision_trial_set)
    waiting_for_eis = sorted(pred_trial_set - eis_trial_set - decision_trial_set)
    ready_but_not_decided = sorted(
        (eis_trial_set & pred_trial_set) - decision_trial_set
    )

    action_counts = {}
    if not decisions.empty and "chariot_shadow_action" in decisions.columns:
        action_counts = {
            str(k): int(v)
            for k, v in decisions["chariot_shadow_action"].value_counts().to_dict().items()
        }

    def _max_or_none(values: set[int]):
        return max(values) if values else None

    return {
        "campaign_rows": int(len(campaign)),
        "campaign_trials_with_eis": int(len(eis_trial_set)),
        "prediction_rows": int(len(predictions)),
        "frozen_chariot_decisions": int(len(decisions)),
        "latest_campaign_trial": _max_or_none(campaign_trial_set),
        "latest_eis_trial": _max_or_none(eis_trial_set),
        "latest_prediction_trial": _max_or_none(pred_trial_set),
        "latest_decision_trial": _max_or_none(decision_trial_set),
        "waiting_for_twin_prediction": waiting_for_twin,
        "waiting_for_eis": waiting_for_eis,
        "prediction_and_eis_ready_but_not_frozen": ready_but_not_decided,
        "decision_action_counts": action_counts,
        "prediction_integrity_errors": validate_prediction_log(predictions),
        "shadow_only": True,
        "note": (
            "This status is rebuilt from disk on demand. The digital twin predicts conductivity only; "
            "CHARIOT decisions are shadow recommendations and never control BRINE."
        ),
    }
