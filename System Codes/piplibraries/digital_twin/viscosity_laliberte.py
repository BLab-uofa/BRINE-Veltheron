import math
import pandas as pd
from scipy.optimize import brentq
from thermo.electrochem import Laliberte_density, Laliberte_viscosity


class LaliberteViscosityBuilder:
    """
    Calculates Laliberte density and viscosity from final analytical salt molarities.

    Input comes from composition_builder columns:
        dt_C_<salt>_final_M

    Output:
        dt_density_kg_m3
        dt_density_g_mL
        dt_viscosity_Pa_s
        dt_viscosity_cP
    """

    CAS = {
        "NaCl": "7647-14-5",
        "KCl": "7447-40-7",
        "ZnCl2": "7646-85-7",
        "ZnSO4": "7733-02-0",
        "Zn(CH3COO)2": "557-34-6",
        "ZnAcetate": "557-34-6",
        "Zn(OAc)2": "557-34-6",
    }

    MW = {
        "NaCl": 58.44,
        "KCl": 74.5513,
        "ZnCl2": 136.315,
        "ZnSO4": 161.44,
        "Zn(CH3COO)2": 183.48,
        "ZnAcetate": 183.48,
        "Zn(OAc)2": 183.48,
    }

    def __init__(self, config):
        self.config = config

    def build(self, df, input_cols):
        df = df.copy()

        numeric_outputs = ["dt_density_kg_m3", "dt_density_g_mL", "dt_water_mass_kg_per_L",
                           "dt_water_mass_fraction", "dt_viscosity_Pa_s", "dt_viscosity_cP"]
        for col in numeric_outputs:
            df[col] = math.nan
        df["dt_viscosity_ok"] = False
        df["dt_viscosity_error"] = ""

        if not bool(self.config.laliberte_enabled):
            df["dt_viscosity_ok"] = False
            df["dt_viscosity_error"] = "Laliberte disabled."
            return df

        T_K = float(self.config.temp_C) + 273.15

        for idx, row in df.iterrows():
            try:
                if not row.get("dt_volume_valid", True):
                    raise ValueError("Invalid volume composition.")
                salts_M = self._row_to_salts_M(row, input_cols)
                result = self.viscosity_from_final_molarity(salts_M, T=T_K)

                df.at[idx, "dt_viscosity_ok"] = True
                df.at[idx, "dt_viscosity_error"] = ""

                df.at[idx, "dt_density_kg_m3"] = result["density_kg_m3"]
                df.at[idx, "dt_density_g_mL"] = result["density_g_mL"]
                df.at[idx, "dt_water_mass_kg_per_L"] = result["water_mass_kg_per_L"]
                df.at[idx, "dt_water_mass_fraction"] = 1.0 - sum(result["solute_weight_fractions"])
                df.at[idx, "dt_viscosity_Pa_s"] = result["viscosity_Pa_s"]
                df.at[idx, "dt_viscosity_cP"] = result["viscosity_cP"]

            except Exception as e:
                df.at[idx, "dt_viscosity_ok"] = False
                df.at[idx, "dt_viscosity_error"] = str(e)

                df.at[idx, "dt_density_kg_m3"] = math.nan
                df.at[idx, "dt_density_g_mL"] = math.nan
                df.at[idx, "dt_viscosity_Pa_s"] = math.nan
                df.at[idx, "dt_viscosity_cP"] = math.nan

        return df

    def _row_to_salts_M(self, row, input_cols):
        salts_M = {}

        for formula in self.config.salt_formulas:
            token = self._safe_column_token(formula)
            c_col = f"dt_C_{token}_final_M"

            c = row.get(c_col, 0.0)
            if pd.isna(c):
                c = 0.0
            c = float(c)

            if c > 0:
                if formula not in self.CAS or formula not in self.MW:
                    raise ValueError(f"No Laliberte mapping for {formula}.")
                salts_M[formula] = c

        return salts_M

    def density_from_final_molarity(self, salts_M, T):
        """
        salts_M:
            {"NaCl": 3.0, "ZnSO4": 2.0}
            final analytical salt molarities, mol/L
        """

        mass_g_per_L = {
            salt: float(M) * self.MW[salt]
            for salt, M in salts_M.items()
        }

        cas = [self.CAS[salt] for salt in salts_M]
        masses = [mass_g_per_L[salt] for salt in salts_M]

        def residual(rho_kg_m3):
            total_solution_mass_g_per_L = rho_kg_m3
            ws = [m / total_solution_mass_g_per_L for m in masses]
            rho_model = Laliberte_density(T, ws, cas)
            return rho_model - rho_kg_m3

        # A trial density must exceed the solute mass per litre, so water mass stays positive.
        lower = max(900.0, sum(masses) * (1.0 + 1e-8))
        rho = brentq(residual, lower, max(2500.0, 1.25 * lower))
        ws = [m / rho for m in masses]

        return rho, ws, cas

    def viscosity_from_final_molarity(self, salts_M, T):
        rho, ws, cas = self.density_from_final_molarity(salts_M, T)

        mu_Pa_s = Laliberte_viscosity(T, ws, cas)
        if not math.isfinite(mu_Pa_s) or mu_Pa_s <= 0 or not 0 <= sum(ws) < 1:
            raise ValueError("Laliberte returned non-physical viscosity or mass fractions.")

        return {
            "density_kg_m3": rho,
            "density_g_mL": rho / 1000.0,
            "solute_weight_fractions": ws,
            "water_mass_kg_per_L": rho * (1.0 - sum(ws)) / 1000.0,
            "viscosity_Pa_s": mu_Pa_s,
            "viscosity_cP": mu_Pa_s * 1000.0,
        }

    @staticmethod
    def _safe_column_token(text):
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
