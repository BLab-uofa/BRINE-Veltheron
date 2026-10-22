from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config
from observer.analytics import campaign_summary
from observer.flags import generate_flags
from observer.loader import load_brine_csv
from shadow.decision_log import load_decision_log
from shadow.view import build_shadow_view


def _default_log(cfg: dict) -> Path:
    live = Path(cfg["shadow"]["decision_log_csv"])
    replay = Path(cfg["shadow"]["replay_output_csv"])
    return live if live.exists() else replay


def main():
    p = argparse.ArgumentParser(description="Deterministic BRINE + causal CHARIOT report; no LLM required")
    p.add_argument("--file", default="data/demo_candidates.csv")
    p.add_argument("--decision-log", "--shadow", dest="decision_log", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    log_path = Path(args.decision_log) if args.decision_log else _default_log(cfg)
    df = load_brine_csv(args.file, cfg)
    decisions = load_decision_log(log_path)
    view = build_shadow_view(df, decisions)
    print(json.dumps(campaign_summary(df, view, cfg), indent=2, default=str))
    flags = [f.to_dict() for f in generate_flags(df, view, cfg)]
    print("\nFLAGS")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
