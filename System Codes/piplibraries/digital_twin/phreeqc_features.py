import math
import pandas as pd
import json
import re

from .phreeqc_engine import PhreeqcEngine


class PhreeqcFeatureBuilder:
    """
    Converts final salt concentrations into PHREEQC totals,
    runs PHREEQC, and writes compact DT features.
    """

    def __init__(self, config):
        self.config = config
        self.engine = PhreeqcEngine(config)

    def build(self, df, input_cols):
        df = df.copy()

        df["dt_phreeqc_ok"] = False
        df["dt_phreeqc_error"] = ""
        df["dt_species_json"] = "{}"
        df["dt_species_concentration_unit"] = "mol/kgw"
        numeric_outputs = ["dt_pH", "dt_ionic_strength", "dt_free_zn_fraction", "dt_complexed_zn_fraction",
                           "dt_neutral_complex_fraction", "dt_charged_complex_fraction", "dt_major_species_count",
                           "dt_total_species_count", "dt_phreeqc_molality_to_molarity_factor",
                           "dt_ionic_strength_analytical_M"]
        for col in numeric_outputs:
            df[col] = math.nan

        for idx, row in df.iterrows():
            if not bool(row.get("dt_volume_valid", True)):
                df.at[idx, "dt_phreeqc_ok"] = False
                df.at[idx, "dt_phreeqc_error"] = "Invalid volume composition."
                continue

            try:
                density = float(row["dt_density_g_mL"])
                factor = float(row["dt_water_mass_kg_per_L"])
                if not math.isfinite(density) or density <= 0 or not math.isfinite(factor) or factor <= 0:
                    raise ValueError("Mixture density/water mass unavailable for PHREEQC molarity input.")
                totals = self._row_to_phreeqc_totals(row, input_cols)
                charges = {"Zn": 2, "S(6)": -2, "Na": 1, "K": 1, "Cl": -1, "Acetate": -1, "Triflate": -1}
                df.at[idx, "dt_ionic_strength_analytical_M"] = 0.5 * sum(
                    value * charges[element] ** 2 for element, value in totals.items())
                case_name = f"cycle_{int(self.config.Psycho):04d}_row_{int(idx):05d}"

                result = self.engine.run(
                    totals=totals,
                    case_name=case_name,
                    density_kg_L=density,
                )

                features = self._compact_features(result)
                
                df.at[idx, "dt_species_json"] = self._species_to_json(result["species"])
                df.at[idx, "dt_phreeqc_ok"] = True
                df.at[idx, "dt_phreeqc_error"] = ""
                df.at[idx, "dt_pH"] = features["pH"]
                df.at[idx, "dt_ionic_strength"] = features["ionic_strength"]
                # kg water / L solution converts every mol/kgw species to mol/L.
                df.at[idx, "dt_phreeqc_molality_to_molarity_factor"] = factor

                df.at[idx, "dt_free_zn_fraction"] = features["free_zn_fraction"]
                df.at[idx, "dt_complexed_zn_fraction"] = features["complexed_zn_fraction"]
                df.at[idx, "dt_neutral_complex_fraction"] = features["neutral_complex_fraction"]
                df.at[idx, "dt_charged_complex_fraction"] = features["charged_complex_fraction"]

                df.at[idx, "dt_major_species_count"] = features["major_species_count"]
                df.at[idx, "dt_total_species_count"] = features["total_species_count"]

                # Save only major species as columns to avoid feature explosion.
                for sp, conc in features["major_species"].items():
                    col = f"dt_sp_{self._safe_column_token(sp)}_mol_kgw"
                    df.at[idx, col] = conc

            except Exception as e:
                df.at[idx, "dt_phreeqc_ok"] = False
                df.at[idx, "dt_phreeqc_error"] = str(e)

        return df

    def _row_to_phreeqc_totals(self, row, input_cols):
        """
        Uses salt_formulas and final salt-equivalent molarity columns
        produced by composition_builder.py.

        For now, this salt registry is the only chemistry mapping.
        When you add new salts, add one entry here.
        """
        totals = {}

        for col, formula in zip(input_cols, self.config.salt_formulas):
            token = self._safe_column_token(formula)
            c_col = f"dt_C_{token}_final_M"

            c = row.get(c_col, 0.0)
            if pd.isna(c):
                c = 0.0
            c = float(c)

            if c <= 0:
                continue

            contribution = self._salt_to_phreeqc_totals(formula, c)

            for key, value in contribution.items():
                totals[key] = totals.get(key, 0.0) + float(value)

        return totals

    @staticmethod
    def _species_to_json(species):
        return json.dumps({str(k): float(v) for k,v in species.items()}, sort_keys=True,)
    
    
    
    @staticmethod
    def _salt_to_phreeqc_totals(formula, c):
        """
        Convert a salt-equivalent concentration into PHREEQC master totals.

        This is intentionally explicit, because databases differ in naming.
        Your modified LLNL database already uses Acetate and may use custom species.

        Current first campaign:
            ZnCl2
            ZnSO4
            Zn(CH3COO)2
            KCl
            NaCl
        """
        f = str(formula).strip()

        if f == "ZnCl2":
            return {
                "Zn": c,
                "Cl": 2.0 * c,
            }

        if f == "ZnSO4":
            return {
                "Zn": c,
                "S(6)": c,
            }

        if f in ("Zn(CH3COO)2", "ZnAcetate", "Zn(OAc)2"):
            return {
                "Zn": c,
                "Acetate": 2.0 * c,
            }

        if f == "KCl":
            return {
                "K": c,
                "Cl": c,
            }

        if f == "NaCl":
            return {
                "Na": c,
                "Cl": c,
            }

        if f in ("Zn(OTf)2", "ZnTriflate"):
            return {
                "Zn": c,
                "Triflate": 2.0 * c,
            }

        raise ValueError(
            f"No PHREEQC total mapping defined for salt formula: {formula}. "
            "Add it inside _salt_to_phreeqc_totals()."
        )

    def _compact_features(self, result):
        species = result["species"]

        free_zn = self._get_first_existing_species(
            species,
            ["Zn+2", "Zn(2)", "Zn++"],
            default=0.0,
        )

        zn_species = {
            sp: c for sp, c in species.items()
            if self._is_zn_aqueous_species(sp)
        }

        neutral_zn_complex = 0.0
        charged_zn_complex = 0.0

        for sp, c in zn_species.items():
            if sp in ("Zn+2", "Zn(2)", "Zn++"):
                continue

            # Zn2... contains two Zn atoms per aqueous complex.
            zn_amount = float(c) * self._zn_stoichiometry(sp)

            charge = self._infer_charge(sp)

            if charge == 0:
                neutral_zn_complex += zn_amount
            else:
                charged_zn_complex += zn_amount

        complexed_zn = neutral_zn_complex + charged_zn_complex
        total_zn = free_zn + complexed_zn

        if total_zn > 0:
            free_zn_fraction = free_zn / total_zn
            complexed_zn_fraction = complexed_zn / total_zn
            neutral_complex_fraction = neutral_zn_complex / total_zn
            charged_complex_fraction = charged_zn_complex / total_zn
        else:
            # No Zn (e.g. a pure KCl candidate): Zn descriptors are absent, not failed.
            free_zn_fraction = 0.0
            complexed_zn_fraction = 0.0
            neutral_complex_fraction = 0.0
            charged_complex_fraction = 0.0

        threshold = float(self.config.phreeqc_species_column_threshold_M)

        major_species = {
            sp: c for sp, c in species.items()
            if float(c) >= threshold
        }

        return {
            "pH": float(result["pH"]),
            "ionic_strength": float(result["ionic_strength"]),
            "free_zn_fraction": free_zn_fraction,
            "complexed_zn_fraction": complexed_zn_fraction,
            "neutral_complex_fraction": neutral_complex_fraction,
            "charged_complex_fraction": charged_complex_fraction,
            "major_species_count": len(major_species),
            "total_species_count": len(species),
            "major_species": major_species,
        }

    @staticmethod
    def _get_first_existing_species(species, candidates, default=0.0):
        for c in candidates:
            if c in species:
                return float(species[c])
        return float(default)

    @staticmethod
    def _is_zn_aqueous_species(sp):
        """Only aqueous rows reach this point; recognise Zn as an element token."""
        return bool(re.search(r"Zn(?![a-z])", sp))

    @staticmethod
    def _zn_stoichiometry(sp):
        return sum(int(count) if count else 1 for count in re.findall(r"Zn(\d*)", sp))

    @staticmethod
    def _infer_charge(sp):
        """
        Rough charge parser for PHREEQC species names.

        Examples:
            Zn+2       -> +2
            ZnCl4-2    -> -2
            ZnCl+      -> +1
            Acetate-   -> -1
            ZnCl2      -> 0
            Cl(-1)     -> -1
        """
        s = str(sp).strip()

        ending = re.search(r"([+-])(\d*)$", s)
        if ending:
            magnitude = int(ending.group(2)) if ending.group(2) else 1
            return magnitude if ending.group(1) == "+" else -magnitude

        # Pattern like Cl(-1)
        if "(" in s and ")" in s:
            inside = s[s.rfind("(") + 1:s.rfind(")")]
            try:
                return int(inside)
            except Exception:
                pass

        # Ending like +2 or -2
        if len(s) >= 2 and s[-2] in "+-" and s[-1].isdigit():
            sign = 1 if s[-2] == "+" else -1
            return sign * int(s[-1])

        # Ending like + or -
        if s.endswith("+"):
            return 1

        if s.endswith("-"):
            return -1

        return 0

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
