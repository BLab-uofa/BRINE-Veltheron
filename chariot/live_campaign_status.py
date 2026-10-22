from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config
from shadow.status import live_pipeline_status


def main():
    p = argparse.ArgumentParser(description="Print current live BRINE/twin/CHARIOT handoff status.")
    p.add_argument("--file", required=True, help="Live BRINE CSV")
    p.add_argument("--predictions", default=None, help="Twin prediction CSV")
    p.add_argument("--decision-log", default=None, help="Frozen CHARIOT decision log")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    predictions = Path(args.predictions or cfg["shadow"]["pretrial_prediction_log_csv"])
    decisions = Path(args.decision_log or cfg["shadow"]["decision_log_csv"])
    status = live_pipeline_status(args.file, predictions, decisions, cfg)
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()
