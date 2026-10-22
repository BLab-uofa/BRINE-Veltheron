class FeatureSchema:
    """
    Defines exactly what each residual ML model is allowed to see.

    No experimental outputs are ever included as features.
    """

    def __init__(self, config):
        self.config = config

    def ensure_static_features(self, df):
        df = df.copy()

        # If temperature is not already row-specific,
        # use the temperature supplied from main.py.
        if "dt_temperature_C" not in df.columns:
            df["dt_temperature_C"] = float(self.config.temp_C)

        return df

    def _concentration_columns(self):
        return [
            f"dt_C_{self._safe_token(formula)}_final_M"
            for formula in self.config.salt_formulas
        ]

    def common_features(self):
        features = (
            self._concentration_columns()
            + [
                "dt_temperature_C",
                "dt_pH",
                "dt_ionic_strength",
                "dt_free_zn_fraction",
                "dt_neutral_complex_fraction",
                "dt_weighted_mobility_charge_scaled",
            ]
        )
        if self.config.laliberte_enabled:
            features.append("dt_viscosity_cP")
        return features

    def conductivity_features(self):
        return self.common_features() + [
            "dt_sigma_theory_S_m",
        ]

    def her_features(self):
        return self.common_features() + [
            "dt_E_HER_nernst_vs_SHE_V",
        ]

    def oer_features(self):
        return self.common_features() + [
            "dt_E_OER_nernst_vs_SHE_V",
        ]

    def for_target(self, target):
        target = target.lower()

        if target == "sigma":
            return self.conductivity_features()

        if target == "her":
            return self.her_features()

        if target == "oer":
            return self.oer_features()

        raise ValueError(f"Unknown DT target: {target}")


    def validate(self, df):
        required = set()
        for target in self.config.objective_targets:
            required.update(self.for_target(target))

        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                "DT ML feature columns are missing:\n"
                + "\n".join(missing))

    @staticmethod
    def _safe_token(text):
        text = str(text).strip()

        replacements = {
            " ": "",
            "-": "_",
            "+": "plus",
            "(": "",
            ")": "",
            "[": "",
            "]": "",
            "{": "",
            "}": "",
            "/": "_",
            "\\": "_",
            ",": "_",
            ".": "p",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text
