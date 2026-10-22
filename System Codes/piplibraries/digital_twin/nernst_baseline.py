import math


class NernstBaselineBuilder:
    """
    Calculates thermodynamic HER/OER water-window limits.

    Returns potentials vs SHE.
    """

    R = 8.314462618
    F = 96485.33212

    E0_HER_V = 0.0
    E0_OER_V = 1.229

    def __init__(self, config):
        self.config = config

    def build(self, df):
        df = df.copy()

        targets = set(self.config.objective_targets)
        if not targets & {"her", "oer"}:
            return df
        active_columns = []
        if "her" in targets:
            active_columns.append("dt_E_HER_nernst_vs_SHE_V")
        if "oer" in targets:
            active_columns.append("dt_E_OER_nernst_vs_SHE_V")
        if {"her", "oer"} <= targets:
            active_columns.append("dt_ESW_nernst_V")
        for col in active_columns:
            df[col] = math.nan
        df["dt_nernst_ok"] = False
        df["dt_nernst_error"] = ""

        temperature_K = float(self.config.temp_C) + 273.15
        rt_over_f = self.R * temperature_K / self.F
        ph_slope = 2.303 * rt_over_f

        for idx, row in df.iterrows():
            try:
                pH = float(row["dt_pH"])
                if not bool(row.get("dt_phreeqc_ok", False)) or not math.isfinite(pH):
                    raise ValueError("Valid PHREEQC pH required for Nernst baselines.")

                water_activity = row.get("dt_water_activity", self.config.water_activity_default)
                if water_activity is None or str(water_activity).strip() == "":
                    water_activity = self.config.water_activity_default

                water_activity = float(water_activity)

                if not math.isfinite(water_activity) or water_activity <= 0:
                    raise ValueError("water activity must be positive.")

                p_H2 = float(self.config.nernst_p_H2)
                p_O2 = float(self.config.nernst_p_O2)

                e_her = (
                    self.E0_HER_V
                    - ph_slope * pH
                    - (rt_over_f / 2.0) * math.log(p_H2)
                )

                e_oer = (
                    self.E0_OER_V
                    - ph_slope * pH
                    - (rt_over_f / 4.0) * math.log(water_activity ** 2)
                    + (rt_over_f / 4.0) * math.log(p_O2)
                )

                df.at[idx, "dt_nernst_ok"] = True
                df.at[idx, "dt_nernst_error"] = ""

                if "her" in targets:
                    df.at[idx, "dt_E_HER_nernst_vs_SHE_V"] = e_her
                if "oer" in targets:
                    df.at[idx, "dt_E_OER_nernst_vs_SHE_V"] = e_oer
                if {"her", "oer"} <= targets:
                    df.at[idx, "dt_ESW_nernst_V"] = e_oer - e_her

            except Exception as e:
                df.at[idx, "dt_nernst_ok"] = False
                df.at[idx, "dt_nernst_error"] = str(e)

                for col in active_columns:
                    df.at[idx, col] = math.nan

        return df
