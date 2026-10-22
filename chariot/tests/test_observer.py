from pathlib import Path

from config import load_config
from observer.analytics import campaign_summary
from observer.flags import generate_flags
from observer.loader import load_brine_csv
from shadow.replay import replay_campaign
from shadow.view import build_shadow_view

ROOT = Path(__file__).resolve().parents[1]


def test_summary_and_flags_from_frozen_replay(tmp_path):
    cfg = load_config(ROOT / "config/system.yaml")
    df = load_brine_csv(ROOT / "data/demo_candidates.csv", cfg)
    log = replay_campaign(ROOT / "data/demo_candidates.csv", tmp_path / "replay.csv", cfg)
    view = build_shadow_view(df, log)
    s = campaign_summary(df, view, cfg)
    assert s["n_trials"] == 45
    assert s["outputs"]["ionic_conductivity_S_per_m"]["max"] > 25
    assert "chariot_shadow" in s
    flags = generate_flags(df, view, cfg)
    assert isinstance(flags, list)
