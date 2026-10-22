from pathlib import Path
from config import load_config
from observer.loader import load_brine_csv

ROOT = Path(__file__).resolve().parents[1]


def test_demo_mapping_and_units():
    cfg = load_config(ROOT / "config/system.yaml")
    df = load_brine_csv(ROOT / "data/demo_candidates.csv", cfg)
    assert len(df) == 45
    assert df["conductivity"].notna().all()
    assert df["her"].notna().all()
    assert df["oer"].notna().all()
    assert abs(df["conductivity"].mean() - 20.7326786667) < 1e-6


def test_stock_concentration_null_is_safe():
    cfg = load_config(ROOT / "config/system.yaml")
    df = load_brine_csv(ROOT / "data/demo_candidates.csv", cfg)
    assert df["x1_M"].isna().all()
