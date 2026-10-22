from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ALLOWED_PLOTS = {
    "conductivity_over_trials",
    "her_oer_over_trials",
    "stability_window_over_trials",
    "input_vs_conductivity",
    "concentration_vs_conductivity",
    "conductivity_vs_stability_window",
    "shadow_actions_over_trials",
}


def create_plot(df: pd.DataFrame, shadow_df: pd.DataFrame | None, plot_type: str,
                output_path: str | Path, cfg: dict, input_name: str | None = None) -> Path:
    if plot_type not in ALLOWED_PLOTS:
        raise ValueError(f"Unsupported plot type {plot_type!r}. Allowed: {sorted(ALLOWED_PLOTS)}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = float(cfg["campaign"]["conductivity_target_S_per_m"])

    fig, ax = plt.subplots(figsize=(8, 5))
    if plot_type == "conductivity_over_trials":
        ax.plot(df["trial_index"], df["conductivity"], marker="o")
        ax.axhline(target, linestyle="--", label=f"Target = {target:g} S/m")
        ax.set_xlabel("Trial index")
        ax.set_ylabel("Ionic conductivity (S/m)")
        ax.set_title("Ionic conductivity across campaign")
        ax.legend()
    elif plot_type == "her_oer_over_trials":
        ax.plot(df["trial_index"], df["her"], marker="o", label="HER")
        ax.plot(df["trial_index"], df["oer"], marker="o", label="OER")
        ax.set_xlabel("Trial index")
        ax.set_ylabel("Potential (V)")
        ax.set_title("HER and OER across campaign")
        ax.legend()
    elif plot_type == "stability_window_over_trials":
        ax.plot(df["trial_index"], df["stability_window_V"], marker="o")
        ax.set_xlabel("Trial index")
        ax.set_ylabel("OER - HER (V)")
        ax.set_title("Electrochemical window across campaign")
    elif plot_type in {"input_vs_conductivity", "concentration_vs_conductivity"}:
        input_name = input_name or "x3"
        if input_name not in {"x1", "x2", "x3", "x4"}:
            raise ValueError("input_name must be one of x1, x2, x3, x4")
        if plot_type == "input_vs_conductivity":
            xcol, xlabel = input_name, f"{input_name} solution amount (uL)"
        else:
            xcol, xlabel = f"{input_name}_M", f"{input_name} concentration (M)"
            if df[xcol].notna().sum() == 0:
                raise ValueError(f"Stock concentration for {input_name} is not configured yet.")
        d = df.dropna(subset=[xcol, "conductivity"])
        ax.scatter(d[xcol], d["conductivity"])
        ax.axhline(target, linestyle="--", label=f"Target = {target:g} S/m")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Ionic conductivity (S/m)")
        ax.set_title(f"{input_name} vs ionic conductivity")
        ax.legend()
    elif plot_type == "conductivity_vs_stability_window":
        d = df.dropna(subset=["conductivity", "stability_window_V"])
        ax.scatter(d["conductivity"], d["stability_window_V"])
        ax.axvline(target, linestyle="--", label=f"Conductivity target = {target:g} S/m")
        ax.set_xlabel("Ionic conductivity (S/m)")
        ax.set_ylabel("OER - HER (V)")
        ax.set_title("Conductivity vs electrochemical window")
        ax.legend()
    elif plot_type == "shadow_actions_over_trials":
        if shadow_df is None or shadow_df.empty:
            raise ValueError("No shadow CSV is available.")
        mapping = {"skip_lsv": 0, "review_lsv": 1, "run_lsv": 2, "wait_for_eis": -1, "wait_for_twin": -2}
        vals = shadow_df["chariot_shadow_action"].map(mapping)
        ax.scatter(shadow_df["trial_index"], vals)
        ax.set_yticks([-2, -1, 0, 1, 2], ["wait twin", "wait EIS", "skip", "review", "run"])
        ax.set_xlabel("Trial index")
        ax.set_ylabel("Shadow CHARIOT recommendation")
        ax.set_title("Shadow CHARIOT decisions")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
