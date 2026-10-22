from __future__ import annotations

from pathlib import Path
import pandas as pd

from shadow.replay import replay_campaign
from shadow.benchmark import benchmark_summary


def generate_shadow(input_csv: str | Path, output_csv: str | Path, cfg: dict) -> pd.DataFrame:
    """Backward-compatible wrapper: now performs leakage-free causal replay.

    For live operation use shadow_monitor.py + frozen pre-trial predictions.
    """
    return replay_campaign(input_csv, output_csv, cfg)


def shadow_summary(shadow_df: pd.DataFrame, target: float = 20.0) -> dict:
    return benchmark_summary(shadow_df, target)
