from pathlib import Path
import pandas as pd

from config import load_config
from shadow.calibration import CalibrationState
from shadow.policy import decide_shadow, ShadowAction
from shadow.replay import replay_campaign
from shadow.benchmark import benchmark_summary

ROOT = Path(__file__).resolve().parents[1]


def ready_calibration():
    return CalibrationState(
        n_points=12,
        coverage_target=0.90,
        factor=1.2,
        raw_band_coverage=0.80,
        calibrated_band_coverage=0.92,
        model_ready=True,
        ready_reason="ready",
    )


def test_policy_three_states():
    cal = ready_calibration()
    run = decide_shadow(21.0, 21.1, 0.5, cal, target=20.0)
    assert run.action == ShadowAction.RUN_LSV

    skip = decide_shadow(18.0, 18.1, 0.4, cal, target=20.0)
    assert skip.action == ShadowAction.SKIP_LSV

    review = decide_shadow(18.5, 22.0, 0.5, cal, target=20.0)
    assert review.action == ShadowAction.REVIEW_LSV
    assert review.flag_code == "TWIN_EIS_DISAGREEMENT"


def test_qualified_candidate_is_never_skipped():
    cal = ready_calibration()
    d = decide_shadow(20.1, 17.0, 0.5, cal, target=20.0)
    assert d.action != ShadowAction.SKIP_LSV


def test_replay_is_causal_and_benchmark_ordering(tmp_path):
    cfg = load_config(ROOT / "config/system.yaml")
    out = tmp_path / "replay.csv"
    replay = replay_campaign(ROOT / "data/demo_candidates.csv", out, cfg)
    assert out.exists()
    assert len(replay) == 45

    for _, row in replay.dropna(subset=["training_data_last_trial"]).iterrows():
        assert int(row["training_data_last_trial"]) < int(row["trial_index"])

    summary = benchmark_summary(replay, float(cfg["campaign"]["conductivity_target_S_per_m"]))
    full = summary["full_protocol"]["lsv_performed"]
    fixed = summary["fixed_20_S_per_m_gate"]["lsv_performed"]
    chariot = summary["chariot"]["effective_lsv_performed_run_plus_review"]
    assert fixed <= chariot <= full
    assert summary["chariot"]["conductivity_qualified_candidates_auto_skipped"] == 0
