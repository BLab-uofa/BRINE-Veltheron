from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config
from shadow.predictions import append_pretrial_prediction


def main():
    p = argparse.ArgumentParser(
        description="Freeze one PRE-TRIAL conductivity-twin prediction for causal CHARIOT shadow mode."
    )
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--mean", type=float, required=True, help="Predicted conductivity in S/m")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--uncertainty", type=float, help="Twin raw uncertainty in S/m")
    group.add_argument("--uncertainty-pct", type=float, help="Twin raw uncertainty as percent of predicted mean, e.g. 8 for 8%%")
    p.add_argument("--training-last-trial", type=int, required=True, help="Last trial used to train the twin; must be < --trial")
    p.add_argument("--x1", type=float, default=None)
    p.add_argument("--x2", type=float, default=None)
    p.add_argument("--x3", type=float, default=None)
    p.add_argument("--x4", type=float, default=None)
    p.add_argument("--source", default="external_twin")
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or cfg["shadow"]["pretrial_prediction_log_csv"])
    uncertainty = args.uncertainty
    if uncertainty is None:
        uncertainty = abs(args.mean) * float(args.uncertainty_pct) / 100.0

    append_pretrial_prediction(
        out,
        trial_index=args.trial,
        twin_mean=args.mean,
        twin_raw_uncertainty=uncertainty,
        training_data_last_trial=args.training_last_trial,
        x1=args.x1, x2=args.x2, x3=args.x3, x4=args.x4,
        source=args.source,
    )
    print(f"Frozen pre-trial prediction appended to: {out.resolve()}")
    print(f"trial={args.trial} | mean={args.mean:.4f} S/m | raw uncertainty={uncertainty:.4f} S/m")
    print("Causal guard passed: training_data_last_trial < trial_index")


if __name__ == "__main__":
    main()
