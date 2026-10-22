from pathlib import Path

import pandas as pd

from config import load_config
from shadow.decision_log import load_decision_log
from shadow.live import process_new_trials
from shadow.predictions import load_prediction_log, validate_prediction_log
from shadow.status import live_pipeline_status


ROOT = Path(__file__).resolve().parents[1]


def test_external_twin_schema_aliases_are_normalized(tmp_path):
    p = tmp_path / "twin.csv"
    pd.DataFrame([
        {
            "trial_index": 2,
            "prediction_timestamp_utc": "2026-10-21T10:00:00Z",
            "training_data_last_trial": 1,
            "x1": 100,
            "x2": 200,
            "x3": 300,
            "x4": 400,
            "predicted_conductivity_S_per_m": 21.5,
            "raw_uncertainty_S_per_m": 1.2,
        }
    ]).to_csv(p, index=False)

    d = load_prediction_log(p)
    assert d.loc[0, "twin_predicted_conductivity_S_per_m"] == 21.5
    assert d.loc[0, "twin_raw_uncertainty_S_per_m"] == 1.2
    assert validate_prediction_log(d) == []


def test_live_waits_for_eis_then_freezes_once_and_lsv_update_does_not_rewrite(tmp_path):
    cfg = load_config(ROOT / "config/system.yaml")
    campaign = tmp_path / "campaign.csv"
    predictions = tmp_path / "twin.csv"
    decisions = tmp_path / "decisions.csv"

    pd.DataFrame([
        {"trial_index": 1, "x1": 100, "x2": 200, "x3": 300, "x4": 400, "y1": None, "y3": None, "y4": None}
    ]).to_csv(campaign, index=False)
    pd.DataFrame([
        {
            "trial_index": 1,
            "prediction_timestamp_utc": "2026-10-21T10:00:00Z",
            "training_data_last_trial": 0,
            "x1": 100,
            "x2": 200,
            "x3": 300,
            "x4": 400,
            "predicted_conductivity_S_per_m": 21.0,
            "raw_uncertainty_S_per_m": 1.0,
        }
    ]).to_csv(predictions, index=False)

    s1 = process_new_trials(campaign, predictions, decisions, cfg)
    assert s1["appended"] == []
    assert s1["waiting_for_eis"] == [1]

    pd.DataFrame([
        {"trial_index": 1, "x1": 100, "x2": 200, "x3": 300, "x4": 400, "y1": 21.2, "y3": None, "y4": None}
    ]).to_csv(campaign, index=False)
    s2 = process_new_trials(campaign, predictions, decisions, cfg)
    assert len(s2["appended"]) == 1
    frozen = load_decision_log(decisions).copy()
    assert len(frozen) == 1

    # BRINE later fills destructive LSV outputs in the SAME row.
    pd.DataFrame([
        {"trial_index": 1, "x1": 100, "x2": 200, "x3": 300, "x4": 400, "y1": 21.2, "y3": -0.4, "y4": 1.8}
    ]).to_csv(campaign, index=False)
    s3 = process_new_trials(campaign, predictions, decisions, cfg)
    assert s3["appended"] == []
    after = load_decision_log(decisions)
    assert len(after) == 1
    assert after.loc[0, "decision_timestamp_utc"] == frozen.loc[0, "decision_timestamp_utc"]


def test_composition_mismatch_blocks_decision(tmp_path):
    cfg = load_config(ROOT / "config/system.yaml")
    campaign = tmp_path / "campaign.csv"
    predictions = tmp_path / "twin.csv"
    decisions = tmp_path / "decisions.csv"

    pd.DataFrame([
        {"trial_index": 5, "x1": 100, "x2": 200, "x3": 300, "x4": 400, "y1": 18.0, "y3": None, "y4": None}
    ]).to_csv(campaign, index=False)
    pd.DataFrame([
        {
            "trial_index": 5,
            "prediction_timestamp_utc": "2026-10-21T10:00:00Z",
            "training_data_last_trial": 4,
            "x1": 100,
            "x2": 999,  # wrong candidate
            "x3": 300,
            "x4": 400,
            "predicted_conductivity_S_per_m": 18.2,
            "raw_uncertainty_S_per_m": 1.0,
        }
    ]).to_csv(predictions, index=False)

    status = process_new_trials(campaign, predictions, decisions, cfg)
    assert status["appended"] == []
    assert status["waiting_for_integrity"] == [5]
    assert any("does not match BRINE" in x for x in status["integrity_errors"])


def test_live_status_tracks_pending_handoff(tmp_path):
    cfg = load_config(ROOT / "config/system.yaml")
    campaign = tmp_path / "campaign.csv"
    predictions = tmp_path / "twin.csv"
    decisions = tmp_path / "decisions.csv"

    pd.DataFrame([
        {"trial_index": 1, "x1": 1, "x2": 2, "x3": 3, "x4": 4, "y1": 22.0, "y3": None, "y4": None},
        {"trial_index": 2, "x1": 1, "x2": 2, "x3": 3, "x4": 4, "y1": None, "y3": None, "y4": None},
    ]).to_csv(campaign, index=False)
    pd.DataFrame([
        {
            "trial_index": 2,
            "prediction_timestamp_utc": "2026-10-21T10:00:00Z",
            "training_data_last_trial": 1,
            "x1": 1,
            "x2": 2,
            "x3": 3,
            "x4": 4,
            "predicted_conductivity_S_per_m": 20.0,
            "raw_uncertainty_S_per_m": 1.0,
        }
    ]).to_csv(predictions, index=False)

    status = live_pipeline_status(campaign, predictions, decisions, cfg)
    assert status["waiting_for_twin_prediction"] == [1]
    assert status["waiting_for_eis"] == [2]
    assert status["frozen_chariot_decisions"] == 0
