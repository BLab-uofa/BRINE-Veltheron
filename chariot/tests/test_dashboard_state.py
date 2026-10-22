from pathlib import Path

import pandas as pd

from observer.dashboard_state import build_dashboard_snapshot


def test_dashboard_snapshot_rebuilds_current_files(tmp_path: Path):
    campaign = tmp_path / "brine.csv"
    predictions = tmp_path / "predictions.csv"
    decisions = tmp_path / "decisions.csv"

    pd.DataFrame(
        [
            {
                "trial_index": 1,
                "x1": 100,
                "x2": 200,
                "x3": 300,
                "x4": 400,
                "y1": 21.0,
                "y3": 0.1,
                "y4": 2.0,
            }
        ]
    ).to_csv(campaign, index=False)

    pd.DataFrame(
        [
            {
                "trial_index": 1,
                "prediction_timestamp_utc": "2026-10-20T00:00:00Z",
                "training_data_last_trial": 0,
                "x1": 100,
                "x2": 200,
                "x3": 300,
                "x4": 400,
                "predicted_conductivity_S_per_m": 20.5,
                "raw_uncertainty_S_per_m": 1.0,
            }
        ]
    ).to_csv(predictions, index=False)

    pd.DataFrame(
        [
            {
                "trial_index": 1,
                "chariot_shadow_action": "review_lsv",
                "chariot_flag_code": "MODEL_NOT_READY",
                "decision_insufficient": True,
                "model_ready": False,
            }
        ]
    ).to_csv(decisions, index=False)

    snap = build_dashboard_snapshot(campaign, predictions, decisions)
    assert snap["status"]["campaign_rows"] == 1
    assert snap["status"]["prediction_rows"] == 1
    assert snap["status"]["frozen_chariot_decisions"] == 1
    assert snap["review_lsv"] == 1
    assert snap["best_conductivity_S_per_m"] == 21.0
