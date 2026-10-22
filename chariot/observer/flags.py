from __future__ import annotations

from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ScientificFlag:
    severity: str
    trial_index: int | None
    code: str
    message: str

    def to_dict(self):
        return asdict(self)


def generate_flags(df: pd.DataFrame, shadow_df: pd.DataFrame | None, cfg: dict) -> list[ScientificFlag]:
    flags: list[ScientificFlag] = []
    target = float(cfg["campaign"]["conductivity_target_S_per_m"])
    total = float(cfg["campaign"]["total_formulation_volume_uL"])
    if df.empty:
        return flags

    for _, row in df.iterrows():
        t = int(row["trial_index"]) if pd.notna(row.get("trial_index")) else None
        sigma, her, oer, vol = row.get("conductivity"), row.get("her"), row.get("oer"), row.get("total_solution_volume_uL")
        if pd.notna(sigma) and float(sigma) < target:
            flags.append(ScientificFlag("info", t, "CONDUCTIVITY_BELOW_TARGET", f"Ionic conductivity {float(sigma):.2f} S/m is below the {target:.2f} S/m campaign target."))
        if pd.notna(her) and pd.notna(oer) and float(oer) <= float(her):
            flags.append(ScientificFlag("warning", t, "HER_OER_ORDER_INVALID", f"OER ({float(oer):.3f} V) is not above HER ({float(her):.3f} V); verify column mapping or measurement."))
        if pd.notna(vol) and float(vol) > total + 1e-6:
            flags.append(ScientificFlag("warning", t, "INPUT_VOLUME_EXCEEDS_TOTAL", f"x1+x2+x3+x4 = {float(vol):.1f} uL exceeds configured total formulation volume {total:.1f} uL."))

    y1 = pd.to_numeric(df["conductivity"], errors="coerce").dropna()
    if len(y1) >= 5 and y1.std(ddof=1) > 0:
        mu, sd = y1.mean(), y1.std(ddof=1)
        threshold = float(cfg.get("observer", {}).get("outlier_z_threshold", 2.5))
        for idx, val in y1.items():
            z = abs(val - mu) / sd
            if z >= threshold:
                t = int(df.loc[idx, "trial_index"]) if pd.notna(df.loc[idx, "trial_index"]) else None
                flags.append(ScientificFlag("info", t, "CONDUCTIVITY_CAMPAIGN_OUTLIER", f"Conductivity {float(val):.2f} S/m is {z:.1f} campaign standard deviations from the current mean."))

    if shadow_df is not None and not shadow_df.empty and "chariot_shadow_action" in shadow_df.columns:
        for _, row in shadow_df[shadow_df["chariot_shadow_action"].notna()].iterrows():
            t = int(row["trial_index"]) if pd.notna(row.get("trial_index")) else None
            code = row.get("chariot_flag_code")
            if code == "TWIN_EIS_DISAGREEMENT":
                flags.append(ScientificFlag("notice", t, "CHARIOT_TWIN_EIS_DISAGREEMENT", "Twin and measured EIS disagreed beyond the empirically calibrated prediction band. Shadow CHARIOT escalated the case to REVIEW; this does not predict the LSV outcome."))
            elif bool(row.get("decision_insufficient", False)):
                flags.append(ScientificFlag("info", t, "CHARIOT_DECISION_INSUFFICIENT", str(row.get("chariot_reason", "Evidence was insufficient for automatic characterization triage."))))

            if bool(row.get("potentially_avoidable_lsv", False)):
                flags.append(ScientificFlag("notice", t, "POTENTIAL_OVERCHARACTERIZATION", "LSV data were collected even though causal shadow CHARIOT would have recommended SKIP after conductivity qualification. This is a counterfactual efficiency flag, not evidence that the real experiment was wrong."))
    return flags
