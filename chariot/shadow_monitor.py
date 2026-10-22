from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import load_config
from shadow.live import process_new_trials


def main():
    p = argparse.ArgumentParser(
        description=(
            "Continuously append causal CHARIOT shadow decisions as the BRINE CSV "
            "and Reza's frozen pre-trial conductivity predictions arrive."
        )
    )
    p.add_argument("--file", required=True, help="Live BRINE CSV; CHARIOT reads it only")
    p.add_argument(
        "--predictions",
        default=None,
        help="Reza's append-only twin prediction CSV",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Frozen CHARIOT decision log",
    )
    p.add_argument(
        "--start-trial",
        type=int,
        default=None,
        help=(
            "Optional first trial CHARIOT should observe. Useful if the BRINE CSV already "
            "contains older trials that do not have frozen twin predictions."
        ),
    )
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    src = Path(args.file)
    pred = Path(args.predictions or cfg["shadow"]["pretrial_prediction_log_csv"])
    out = Path(args.out or cfg["shadow"]["decision_log_csv"])

    print("\nCHARIOT LIVE CAUSAL SHADOW MONITOR")
    print("----------------------------------")
    print("READ-ONLY with respect to BRINE, BO, the twin, robot, EIS and LSV execution.")
    print("The digital twin predicts CONDUCTIVITY ONLY; CHARIOT never predicts LSV outcome.")
    print(f"BRINE CSV:             {src.resolve()}")
    print(f"Twin prediction CSV:   {pred.resolve()}")
    print(f"Frozen decision log:   {out.resolve()}")
    if args.start_trial is not None:
        print(f"First monitored trial: {args.start_trial}")
    print("A decision is frozen only after both the causal pre-trial twin output and EIS are available.")
    print("If BRINE later fills HER/OER in the same row, the old CHARIOT decision is NOT rewritten.")
    print("Press Ctrl+C to stop.\n")

    last_signature = None
    last_waiting = None
    try:
        while True:
            try:
                sig = (
                    src.stat().st_mtime_ns if src.exists() else None,
                    pred.stat().st_mtime_ns if pred.exists() else None,
                )

                if sig != last_signature:
                    if not src.exists():
                        print(f"Waiting for BRINE CSV: {src}")
                        last_signature = sig
                        time.sleep(args.interval)
                        continue

                    status = process_new_trials(
                        src,
                        pred,
                        out,
                        cfg,
                        start_trial=args.start_trial,
                    )

                    for record in status["appended"]:
                        print(
                            f"Trial {record['trial_index']}: {record['chariot_shadow_action'].upper()} | "
                            f"{record['chariot_flag_code']} | frozen causal shadow decision"
                        )

                    waiting_signature = (
                        tuple(status["waiting_for_twin"]),
                        tuple(status["waiting_for_eis"]),
                        tuple(status["waiting_for_integrity"]),
                        tuple(status["integrity_errors"]),
                    )
                    if waiting_signature != last_waiting:
                        if status["waiting_for_twin"]:
                            print(
                                "Waiting for frozen pre-trial twin prediction: "
                                f"{status['waiting_for_twin']}"
                            )
                        if status["waiting_for_eis"]:
                            print(f"Waiting for processed EIS: {status['waiting_for_eis']}")
                        if status["waiting_for_integrity"]:
                            print(
                                "Integrity check blocked decision for trial(s): "
                                f"{status['waiting_for_integrity']}"
                            )
                        for problem in status["integrity_errors"]:
                            print(f"INTEGRITY WARNING: {problem}")
                        last_waiting = waiting_signature

                    print(
                        "State: "
                        f"campaign={status['campaign_trial_count']} rows | "
                        f"predictions={status['prediction_count']} | "
                        f"frozen decisions={status['decision_count']}"
                    )
                    last_signature = sig

            except FileNotFoundError:
                print(f"Waiting for BRINE CSV: {src}")
            except Exception as exc:
                # Never touch/stop BRINE. Keep the observer alive and retry.
                print(f"Monitor warning (BRINE unaffected): {exc}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nCHARIOT live shadow monitor stopped. BRINE was not modified.")


if __name__ == "__main__":
    main()
