from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config
from shadow.generator import generate_shadow, shadow_summary


def main():
    p = argparse.ArgumentParser(description="Generate leakage-free retrospective CHARIOT shadow replay from an existing BRINE CSV")
    p.add_argument("--file", default="data/demo_candidates.csv")
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out or cfg["shadow"]["replay_output_csv"])
    shadow = generate_shadow(args.file, out, cfg)
    print(f"Causal shadow replay written to: {out.resolve()}")
    print("This is simulated retrospective performance unless the source file is real historical BRINE data.")
    print(json.dumps(shadow_summary(shadow, float(cfg["campaign"]["conductivity_target_S_per_m"])), indent=2))


if __name__ == "__main__":
    main()
