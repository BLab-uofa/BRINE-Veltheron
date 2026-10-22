from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config
from shadow.replay import replay_campaign
from shadow.benchmark import benchmark_summary, sensitivity_table, save_benchmark


def main():
    p = argparse.ArgumentParser(
        description="Leakage-free causal retrospective CHARIOT replay. Uses only prior trials to predict the next trial."
    )
    p.add_argument("--file", default="data/demo_candidates.csv")
    p.add_argument("--out", default=None)
    p.add_argument("--benchmark-json", default="outputs/chariot_benchmark.json")
    p.add_argument("--sensitivity-csv", default="outputs/chariot_sensitivity.csv")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or cfg["shadow"]["replay_output_csv"])
    replay = replay_campaign(args.file, out, cfg)
    summary = benchmark_summary(replay, float(cfg["campaign"]["conductivity_target_S_per_m"]))
    save_benchmark(summary, args.benchmark_json)
    sensitivity = sensitivity_table(replay, cfg)
    Path(args.sensitivity_csv).parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(args.sensitivity_csv, index=False)

    print(f"Causal replay written to: {out.resolve()}")
    print(f"Benchmark written to:     {Path(args.benchmark_json).resolve()}")
    print(f"Sensitivity written to:   {Path(args.sensitivity_csv).resolve()}")
    print("\nIMPORTANT: this is a simulated retrospective on the demonstration/historical campaign unless the source data are real historical BRINE runs.")
    print("CHARIOT is not expected to beat the fixed gate on raw LSV savings. REVIEW counts as performing LSV.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
