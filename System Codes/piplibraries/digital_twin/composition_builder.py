import pandas as pd


class CompositionBuilder:
    """
    Converts candidate input volumes into final stock-equivalent salt concentrations.

    Assumption:
        x_i = volume of stock solution i added to the cell, in µL

    Calculation:
        C_final_i = volume_i_uL * C_stock_i / cell_volume_uL
    """

    def __init__(self, config):
        self.config = config

    def build(self, df, input_cols):
        df = df.copy()

        input_numeric = df[input_cols].apply(pd.to_numeric, errors="coerce")

        df["dt_cell_volume_uL"] = float(self.config.cell_volume_uL)
        df["dt_total_input_volume_uL"] = input_numeric.sum(axis=1)
        df["dt_water_volume_uL"] = (
            float(self.config.cell_volume_uL) - df["dt_total_input_volume_uL"]
        )
        df["dt_water_fraction"] = (
            df["dt_water_volume_uL"] / float(self.config.cell_volume_uL)
        )

        # Ignore floating-point round-off at the cell-volume boundary.
        df["dt_volume_valid"] = True
        df.loc[df["dt_water_volume_uL"] < -1e-6, "dt_volume_valid"] = False
        df.loc[input_numeric.isna().any(axis=1), "dt_volume_valid"] = False
        df.loc[(input_numeric < 0).any(axis=1), "dt_volume_valid"] = False

        for col, formula, stock_molarity in zip(
            input_cols,
            self.config.salt_formulas,
            self.config.stock_molarities,
        ):
            token = self._safe_column_token(formula)
            out_col = f"dt_C_{token}_final_M"

            df[out_col] = (
                pd.to_numeric(df[col], errors="coerce")
                * float(stock_molarity)
                / float(self.config.cell_volume_uL)
            )

        return df

    @staticmethod
    def _safe_column_token(text):
        """
        Turns chemistry strings into CSV-safe column tokens.

        Examples:
            ZnCl2         -> ZnCl2
            Zn(CH3COO)2   -> ZnCH3COO2
            SO4-2         -> SO4_2
        """
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
