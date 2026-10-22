from __future__ import annotations

import argparse
from pathlib import Path

from agent.gemini_observer import GeminiObserver
from config import load_config
from observer.loader import load_brine_csv
from observer.plots import create_plot
from shadow.decision_log import load_decision_log
from shadow.view import build_shadow_view


def explicit_plot_request(question: str) -> bool:
    q = question.lower()
    terms = [
        "plot",
        "graph",
        "chart",
        "figure",
        "visualize",
        "visualise",
        "visualization",
        "visualisation",
        "show me a plot",
        "show me a graph",
    ]
    return any(term in q for term in terms)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Read-only Gemini scientific observer for a continuously updating "
            "BRINE + causal CHARIOT shadow campaign"
        )
    )
    p.add_argument("--file", required=True, help="Live BRINE CSV to observe")
    p.add_argument(
        "--decision-log",
        default=None,
        help="Frozen live CHARIOT decision log",
    )
    p.add_argument(
        "--predictions",
        default=None,
        help="Reza's frozen pre-trial twin prediction CSV",
    )
    p.add_argument(
        "--bond",
        action="store_true",
        help="Allow Valency Bond only when the scientist explicitly requests literature",
    )
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    campaign = Path(args.file)
    decision_log = Path(args.decision_log or cfg["shadow"]["decision_log_csv"])
    predictions = Path(args.predictions or cfg["shadow"]["pretrial_prediction_log_csv"])
    obs = GeminiObserver(config_path=args.config)

    print("\nBRINE Scientific Observer is running.")
    print("Gemini is READ-ONLY: it cannot modify BRINE, BO, the twin, or CHARIOT.")
    print("The twin predicts conductivity only; CHARIOT never predicts LSV outcome.")
    print("Campaign, twin handoff, and CHARIOT state are reloaded before EVERY answer.")
    print("Valency Bond is called ONLY when you explicitly ask for literature/papers/Bond.")
    print("Conversation topic is remembered for follow-ups such as 'compare it with literature'.")
    print(f"Campaign file:   {campaign.resolve()}")
    print(f"Twin predictions:{predictions.resolve()}")
    print(f"Decision log:    {decision_log.resolve()}")
    if not decision_log.exists():
        print("Decision log does not exist yet. Campaign questions still work while CHARIOT waits for live decisions.")
    print("Type 'quit' to exit.\n")

    while True:
        q = input("You > ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break

        try:
            response = obs.analyze(
                q,
                campaign,
                decision_log,
                use_bond=args.bond,
                prediction_log_file=predictions,
            )

            print("\nGemini > " + response.answer + "\n")

            if response.plot_request and explicit_plot_request(q):
                df = load_brine_csv(campaign, cfg)
                decisions = load_decision_log(decision_log)
                shadow_view = build_shadow_view(df, decisions)

                safe_name = response.plot_request.plot_type
                if response.plot_request.input_name:
                    safe_name += "_" + response.plot_request.input_name

                out = Path("outputs/plots") / f"{safe_name}.png"
                create_plot(
                    df,
                    shadow_view,
                    response.plot_request.plot_type,
                    out,
                    cfg,
                    input_name=response.plot_request.input_name,
                )
                print(f"Plot saved to: {out.resolve()}\n")

        except Exception as exc:
            print(f"\nObserver error: {exc}\n")


if __name__ == "__main__":
    main()
