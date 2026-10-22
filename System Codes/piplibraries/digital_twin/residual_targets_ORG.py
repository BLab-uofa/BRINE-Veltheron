import pandas as pd


class ResidualTargetBuilder:
    """
    Builds hybrid physics + ML residual targets.

    measured = physics_baseline + ML_residual

    Objective convention:
        first objective  = conductivity [S/m]
        second objective = HER limit [V vs SHE]
        third objective  = OER limit [V vs SHE]
    """

    def __init__(self, config):
        self.config = config

    def build(self, df):
        df = df.copy()

        if self.config.num_objectives < 3:
            raise ValueError(
                "Residual DT requires at least 3 objectives:\n"
                "1 = conductivity [S/m]\n"
                "2 = HER limit [V vs SHE]\n"
                "3 = OER limit [V vs SHE]"
            )

        objective_cols = list(
            df.columns[
                self.config.num_inputs:
                self.config.num_inputs + self.config.num_objectives
            ]
        )

        sigma_col = objective_cols[0]
        her_col = objective_cols[1]
        oer_col = objective_cols[2]

        sigma_measured = pd.to_numeric(df[sigma_col], errors="coerce")
        her_measured = pd.to_numeric(df[her_col], errors="coerce")
        oer_measured = pd.to_numeric(df[oer_col], errors="coerce")

        df["dt_residual_sigma_S_m"] = (
            sigma_measured
            - pd.to_numeric(df["dt_sigma_theory_S_m"], errors="coerce")
        )

        df["dt_residual_HER_V"] = (
            her_measured
            - pd.to_numeric(
                df["dt_E_HER_nernst_vs_SHE_V"],
                errors="coerce",
            )
        )

        df["dt_residual_OER_V"] = (
            oer_measured
            - pd.to_numeric(
                df["dt_E_OER_nernst_vs_SHE_V"],
                errors="coerce",
            )
        )

        return df