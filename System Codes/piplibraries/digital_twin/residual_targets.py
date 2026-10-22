import pandas as pd


class ResidualTargetBuilder:

    TARGETS = {
        "sigma": {
            "baseline": "dt_sigma_theory_S_m",
            "residual": "dt_residual_sigma_S_m",
        },

        "her": {
            "baseline": "dt_E_HER_nernst_vs_SHE_V",
            "residual": "dt_residual_HER_V",
        },

        "oer": {
            "baseline": "dt_E_OER_nernst_vs_SHE_V",
            "residual": "dt_residual_OER_V",
        },
    }

    def __init__(self, config):
        self.config = config

    def build(self, df):

        df = df.copy()

        for target in self.config.objective_targets:

            info = self.TARGETS[target]

            measured = pd.to_numeric(
                df[self.config.objective_column(target)],
                errors="coerce",
            )

            baseline = pd.to_numeric(
                df[info["baseline"]],
                errors="coerce",
            )

            if target == "sigma":
                measured = measured * self.config.conductivity_scale_to_S_m
                df["dt_sigma_measured_S_m"] = measured
            elif target == "her":
                if self.config.her_bo_negated:
                    measured = -measured
                df["dt_HER_measured_vs_SHE_V"] = measured
            elif target == "oer":
                df["dt_OER_measured_vs_SHE_V"] = measured

            df[info["residual"]] = measured - baseline

        return df
