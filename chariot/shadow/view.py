from __future__ import annotations

import pandas as pd


def build_shadow_view(campaign_df: pd.DataFrame, decision_log: pd.DataFrame | None) -> pd.DataFrame:
    """Merge frozen CHARIOT decisions with the CURRENT BRINE state for display.

    This view may change as HER/OER later arrive, but the frozen CHARIOT decision
    columns come only from the append-only decision log and are never recomputed.
    """
    if decision_log is None or decision_log.empty:
        base = campaign_df.copy()
        base["chariot_shadow_action"] = pd.NA
        base["chariot_flag_code"] = pd.NA
        base["chariot_reason"] = pd.NA
        base["decision_insufficient"] = pd.NA
        base["lsv_result_present"] = base["her"].notna() | base["oer"].notna()
        base["complete_her_oer_present"] = base["her"].notna() & base["oer"].notna()
        base["potentially_avoidable_lsv"] = False
        return base

    d = decision_log.drop_duplicates("trial_index", keep="first").copy()
    merged = campaign_df.merge(d, on="trial_index", how="left", suffixes=("", "_decision"))
    merged["lsv_result_present"] = merged["her"].notna() | merged["oer"].notna()
    merged["complete_her_oer_present"] = merged["her"].notna() & merged["oer"].notna()
    merged["potentially_avoidable_lsv"] = (
        merged["chariot_shadow_action"].eq("skip_lsv") & merged["lsv_result_present"]
    )
    return merged
