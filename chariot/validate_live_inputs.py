from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from config import load_config
from observer.loader import load_brine_csv
from shadow.predictions import load_prediction_log, validate_prediction_log


def main():
    p = argparse.ArgumentParser(description="Validate BRINE + twin CSV handoff before the live campaign.")
    p.add_argument("--file", required=True, help="BRINE CSV")
    p.add_argument("--predictions", required=True, help="Twin prediction CSV")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    campaign_path = Path(args.file)
    pred_path = Path(args.predictions)

    print("LIVE INPUT VALIDATION")
    print("---------------------")

    if not campaign_path.exists():
        print(f"BRINE CSV not found yet: {campaign_path.resolve()}")
    else:
        campaign = load_brine_csv(campaign_path, cfg)
        print(f"BRINE CSV readable: {campaign_path.resolve()}")
        print(f"BRINE rows: {len(campaign)}")
        print(f"Rows with EIS conductivity: {int(campaign['conductivity'].notna().sum())}")

        trials = pd.to_numeric(campaign["trial_index"], errors="coerce").dropna().astype(int)
        duplicates = sorted(trials[trials.duplicated(keep=False)].unique().tolist())
        if duplicates:
            print(f"WARNING: duplicate BRINE trial_index values: {duplicates}")
        else:
            print("BRINE trial_index uniqueness: OK")

    if not pred_path.exists():
        print(f"Twin prediction CSV not found yet: {pred_path.resolve()}")
        print("This is normal before Reza writes the first pre-trial prediction.")
        return

    predictions = load_prediction_log(pred_path)
    print(f"Twin prediction CSV readable: {pred_path.resolve()}")
    print(f"Prediction rows: {len(predictions)}")
    problems = validate_prediction_log(predictions)
    if problems:
        print("Prediction integrity: FAILED")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("Prediction integrity: OK")
        print("Causal leakage guard: OK for every prediction row")
        print("Raw uncertainty units expected: absolute S/m (not an unlabeled percentage)")


if __name__ == "__main__":
    main()
