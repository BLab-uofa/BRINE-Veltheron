from pathlib import Path
import pandas as pd
import pytest

from shadow.predictions import append_pretrial_prediction, load_prediction_log
from shadow.decision_log import append_frozen_decision, load_decision_log


def test_prediction_log_rejects_leakage_and_overwrite(tmp_path):
    p = tmp_path / "pred.csv"
    append_pretrial_prediction(
        p,
        trial_index=10,
        twin_mean=21.0,
        twin_raw_uncertainty=1.0,
        training_data_last_trial=9,
    )
    assert len(load_prediction_log(p)) == 1
    with pytest.raises(ValueError):
        append_pretrial_prediction(
            p,
            trial_index=10,
            twin_mean=22.0,
            twin_raw_uncertainty=1.0,
            training_data_last_trial=9,
        )
    with pytest.raises(ValueError):
        append_pretrial_prediction(
            tmp_path / "bad.csv",
            trial_index=10,
            twin_mean=22.0,
            twin_raw_uncertainty=1.0,
            training_data_last_trial=10,
        )


def test_decision_log_is_frozen(tmp_path):
    p = tmp_path / "decisions.csv"
    append_frozen_decision(p, {
        "trial_index": 5,
        "chariot_shadow_action": "review_lsv",
        "chariot_flag_code": "MODEL_NOT_READY",
        "chariot_reason": "cold start",
    })
    assert len(load_decision_log(p)) == 1
    with pytest.raises(ValueError):
        append_frozen_decision(p, {
            "trial_index": 5,
            "chariot_shadow_action": "skip_lsv",
            "chariot_flag_code": "CONFIDENT_BELOW_TARGET",
            "chariot_reason": "should not overwrite",
        })


def test_live_processor_freezes_once(tmp_path):
    from config import load_config
    from shadow.live import process_new_trials

    cfg = load_config(Path(__file__).resolve().parents[1] / "config/system.yaml")
    campaign = tmp_path / "campaign.csv"
    pd.DataFrame([{
        "trial_index": 5,
        "x1": 100.0, "x2": 200.0, "x3": 300.0, "x4": 100.0,
        "y1": 18.5, "y3": 0.1, "y4": 2.1,
    }]).to_csv(campaign, index=False)

    pred = tmp_path / "pred.csv"
    append_pretrial_prediction(
        pred,
        trial_index=5,
        twin_mean=18.7,
        twin_raw_uncertainty=1.0,
        training_data_last_trial=4,
    )
    decisions = tmp_path / "decisions.csv"
    first = process_new_trials(campaign, pred, decisions, cfg)
    second = process_new_trials(campaign, pred, decisions, cfg)
    assert len(first["appended"]) == 1
    assert len(second["appended"]) == 0
    assert len(load_decision_log(decisions)) == 1
