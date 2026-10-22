from __future__ import annotations

from typing import Any
import pandas as pd


def _stats(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
    }


def campaign_summary(df: pd.DataFrame, shadow_df: pd.DataFrame | None, cfg: dict) -> dict[str, Any]:
    target = float(cfg["campaign"]["conductivity_target_S_per_m"])
    if df.empty:
        return {"n_trials": 0, "conductivity_target_S_per_m": target}

    y1 = pd.to_numeric(df["conductivity"], errors="coerce")
    her = pd.to_numeric(df["her"], errors="coerce")
    oer = pd.to_numeric(df["oer"], errors="coerce")
    win = pd.to_numeric(df["stability_window_V"], errors="coerce")
    above = int((y1 >= target).sum())
    best_idx = y1.idxmax() if y1.notna().any() else None
    worst_idx = y1.idxmin() if y1.notna().any() else None

    stocks = {}
    for x in ("x1", "x2", "x3", "x4"):
        stock = cfg["stocks"][x]
        stocks[x] = {
            "solution_name": stock.get("name", x),
            "stock_concentration_M": stock.get("stock_concentration_M"),
            "volume_uL": _stats(df[x]),
            "calculated_concentration_M": _stats(df[f"{x}_M"]),
        }

    summary = {
        "n_trials": int(len(df)),
        "conductivity_target_S_per_m": target,
        "n_conductivity_at_or_above_target": above,
        "fraction_conductivity_at_or_above_target": float(above / y1.notna().sum()) if y1.notna().sum() else None,
        "outputs": {
            "ionic_conductivity_S_per_m": _stats(y1),
            "HER_V": _stats(her),
            "OER_V": _stats(oer),
            "OER_minus_HER_window_V": _stats(win),
        },
        "inputs": stocks,
        "best_by_conductivity": df.loc[best_idx].to_dict() if best_idx is not None else None,
        "worst_by_conductivity": df.loc[worst_idx].to_dict() if worst_idx is not None else None,
        "latest": df.iloc[-1].to_dict(),
    }

    if shadow_df is not None and not shadow_df.empty and "chariot_shadow_action" in shadow_df.columns:
        decided = shadow_df[shadow_df["chariot_shadow_action"].notna()].copy()
        actions = decided["chariot_shadow_action"].value_counts().to_dict()
        n_skip = int(decided["chariot_shadow_action"].eq("skip_lsv").sum())
        n_review = int(decided["chariot_shadow_action"].eq("review_lsv").sum())
        n_run = int(decided["chariot_shadow_action"].eq("run_lsv").sum())
        n_avoidable_observed = int(decided.get("potentially_avoidable_lsv", pd.Series(False, index=decided.index)).fillna(False).sum())
        disagreement = int(decided.get("review_due_to_disagreement", pd.Series(False, index=decided.index)).fillna(False).astype(bool).sum())
        insufficient = int(decided.get("decision_insufficient", pd.Series(False, index=decided.index)).fillna(False).astype(bool).sum())
        fixed_skip = int((pd.to_numeric(decided["conductivity"], errors="coerce") < target).sum()) if "conductivity" in decided.columns else None

        # Deterministic flag/review counts. Gemini should explain these values,
        # never infer or recount them from recent rows.
        if "chariot_flag_code" in decided.columns:
            flag_counts_raw = decided["chariot_flag_code"].fillna("UNSPECIFIED").value_counts().to_dict()
            flag_counts = {str(k): int(v) for k, v in flag_counts_raw.items()}
        else:
            flag_counts = {}

        review_rows = decided[decided["chariot_shadow_action"].eq("review_lsv")].copy()
        model_not_ready_reviews = int(
            review_rows.get("chariot_flag_code", pd.Series(index=review_rows.index, dtype=object))
            .eq("MODEL_NOT_READY")
            .sum()
        )
        disagreement_reviews = int(
            review_rows.get("chariot_flag_code", pd.Series(index=review_rows.index, dtype=object))
            .eq("TWIN_EIS_DISAGREEMENT")
            .sum()
        )
        other_reviews = max(0, n_review - model_not_ready_reviews - disagreement_reviews)

        if "model_ready" in decided.columns:
            model_ready_series = decided["model_ready"].fillna(False).astype(bool)
            model_ready_true = int(model_ready_series.sum())
            model_ready_false = int((~model_ready_series).sum())
        else:
            model_ready_true = None
            model_ready_false = None

        summary["chariot_shadow"] = {
            "decisions_frozen": int(len(decided)),
            "action_counts": {str(k): int(v) for k, v in actions.items()},
            "run_lsv": n_run,
            "review_lsv": n_review,
            "skip_lsv": n_skip,
            "effective_lsv_if_connected_run_plus_review": n_run + n_review,
            "flag_counts": flag_counts,
            "review_breakdown": {
                "model_not_ready": model_not_ready_reviews,
                "twin_eis_disagreement": disagreement_reviews,
                "other_review": other_reviews,
                "total_review": n_review,
            },
            "model_ready_counts": {
                "true": model_ready_true,
                "false": model_ready_false,
            },
            "decision_insufficient_count": insufficient,
            "twin_eis_disagreement_review_count": disagreement,
            "potentially_avoidable_lsv_observed_count": n_avoidable_observed,
            "lsv_reduction_vs_full_if_connected": (n_skip / len(decided)) if len(decided) else None,
            "fixed_gate_savings_retained_fraction": (n_skip / fixed_skip) if fixed_skip else None,
            "counting_rule": (
                "For CHARIOT action/review counts, use these deterministic Python fields exactly. "
                "Do not infer counts from recent rows or from whether prediction fields are missing."
            ),
            "note": (
                "Shadow/counterfactual only. CHARIOT did not control BRINE. REVIEW counts as performing LSV in benchmark comparisons; "
                "CHARIOT is not designed to beat the blind fixed threshold on raw savings."
            ),
        }
    return summary


def recent_trials(df: pd.DataFrame, shadow_df: pd.DataFrame | None = None, n: int = 10) -> list[dict]:
    d = df.tail(n).copy()
    if shadow_df is not None and not shadow_df.empty and "chariot_shadow_action" in shadow_df.columns:
        keep = [
            "trial_index", "chariot_shadow_action", "chariot_flag_code", "chariot_reason",
            "decision_insufficient", "twin_predicted_conductivity_S_per_m",
            "twin_raw_uncertainty_S_per_m", "calibration_factor",
            "normalized_disagreement_score", "model_ready", "potentially_avoidable_lsv",
            "review_due_to_disagreement",
        ]
        keep = [c for c in keep if c in shadow_df.columns]
        s = shadow_df[keep].drop_duplicates("trial_index", keep="first")
        d = d.merge(s, on="trial_index", how="left")
    return d.where(pd.notnull(d), None).to_dict(orient="records")


def top_trials(df: pd.DataFrame, metric: str = "conductivity", n: int = 5, highest: bool = True) -> list[dict]:
    aliases = {
        "conductivity": "conductivity", "y1": "conductivity", "her": "her", "oer": "oer",
        "window": "stability_window_V", "stability_window": "stability_window_V",
    }
    col = aliases.get(metric.lower(), metric)
    if col not in df.columns:
        raise ValueError(f"Unknown metric {metric!r}")
    d = df.dropna(subset=[col]).sort_values(col, ascending=not highest).head(n)
    return d.where(pd.notnull(d), None).to_dict(orient="records")
