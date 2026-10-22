import os
import math
import pandas as pd
from .phreeqc_features import PhreeqcFeatureBuilder
from .transport_parameters import extend_transport_parameters


class MobilityFeatureBuilder:
    """
    Calculates weighted mobility descriptors from PHREEQC species concentrations.

    Uses dt_species_json, on a common mol/kg-water basis.

    Stores:
        dt_weighted_mobility_conc
        dt_weighted_mobility_charge
        scaled versions
    """

    CHARGE = {
        "H+": 1,
        "OH-": -1,
        "Na+": 1,
        "K+": 1,
        "Zn+2": 2,
        "Cl-": -1,
        "SO4-2": -2,
        "Acetate": -1,
        "Acetate-": -1,
        "Triflate-": -1,
    }

    SPECIES_ALIASES = {
        "H+": ["H+", "Hplus"],
        "OH-": ["OH-"],
        "Na+": ["Na+", "Naplus"],
        "K+": ["K+", "Kplus"],
        "Zn+2": ["Zn+2", "Znplus2", "Zn(2)", "Zn++"],
        "Cl-": ["Cl-", "Cl_"],
        "SO4-2": ["SO4-2", "SO4_2"],
        "Acetate": ["Acetate", "Acetate-"],
        "Triflate-": ["Triflate-"],
    }

    def __init__(self, config):
        self.config = config
        self.mobility = self._load_mobility_file(config.mobility_file)
        self.mobility, self.diameters, self.mobility_sources = extend_transport_parameters(
            self.mobility, config.conductivity_ion_diameters_m)

    def build(self, df):
        df = df.copy()

        for idx, row in df.iterrows():
            try:
                if not bool(row.get("dt_phreeqc_ok", False)):
                    raise ValueError("PHREEQC speciation unavailable for this row.")
                species_conc = self._extract_supported_species(row)
                if not species_conc:
                    raise ValueError("No charged species with mobility data were available.")

                conc_avg = self._weighted_average(
                    species_conc=species_conc,
                    charge_weighted=False,
                )

                charge_avg = self._weighted_average(
                    species_conc=species_conc,
                    charge_weighted=True,
                )

                scale = float(self.config.mobility_scale_factor)

                df.at[idx, "dt_mobility_ok"] = True
                df.at[idx, "dt_mobility_error"] = ""

                df.at[idx, "dt_weighted_mobility_conc"] = conc_avg
                df.at[idx, "dt_weighted_mobility_charge"] = charge_avg

                df.at[idx, "dt_weighted_mobility_conc_scaled"] = (
                    conc_avg * scale if conc_avg is not None else math.nan
                )
                df.at[idx, "dt_weighted_mobility_charge_scaled"] = (
                    charge_avg * scale if charge_avg is not None else math.nan
                )

                df.at[idx, "dt_mobility_species_count"] = len(species_conc)

            except Exception as e:
                df.at[idx, "dt_mobility_ok"] = False
                df.at[idx, "dt_mobility_error"] = str(e)

                df.at[idx, "dt_weighted_mobility_conc"] = math.nan
                df.at[idx, "dt_weighted_mobility_charge"] = math.nan
                df.at[idx, "dt_weighted_mobility_conc_scaled"] = math.nan
                df.at[idx, "dt_weighted_mobility_charge_scaled"] = math.nan
                df.at[idx, "dt_mobility_species_count"] = 0

        return df

    def _load_mobility_file(self, mobility_file):
        if not os.path.isfile(mobility_file):
            raise FileNotFoundError(f"Mobility file not found: {mobility_file}")

        df = pd.read_csv(mobility_file)

        if len(df) == 0:
            raise ValueError("Mobility file is empty.")

        row = df.iloc[0].to_dict()

        mobility = {}
        for k, v in row.items():
            if pd.isna(v):
                continue
            value = float(v)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Mobility must be a positive SI magnitude for {k}.")
            mobility[str(k).strip()] = value

        return mobility
    

    def _extract_supported_species(self, row):
        import json

        raw = row.get("dt_species_json", "{}")

        if pd.isna(raw) or str(raw).strip() == "":
            species = {}
        else:
            species = json.loads(raw)

        species_conc = {}

        for name, conc in species.items():
            canonical_sp = self.canonical_species(name)
            if self.species_charge(canonical_sp) == 0 or self.mobility_for(canonical_sp) is None:
                continue
            if float(conc) > 0:
                species_conc[canonical_sp] = species_conc.get(canonical_sp, 0.0) + float(conc)

        return species_conc

    @classmethod
    def canonical_species(cls, name):
        for canonical, aliases in cls.SPECIES_ALIASES.items():
            if name in aliases:
                return canonical
        return name

    @classmethod
    def species_charge(cls, name):
        return cls.CHARGE.get(name, PhreeqcFeatureBuilder._infer_charge(name))

    def mobility_for(self, name):
        # Explicit file values take precedence over the documented paper estimates.
        for key in [name] + self.SPECIES_ALIASES.get(name, []):
            if key in self.mobility:
                return self.mobility[key]
        return None

    def _weighted_average(self, species_conc, charge_weighted):
        num = 0.0
        den = 0.0

        for sp, c in species_conc.items():
            mu = self.mobility_for(sp)
            z = self.species_charge(sp)

            if mu is None or z is None:
                continue

            if charge_weighted:
                w = float(c) * float(z) ** 2
            else:
                w = float(c)

            num += w * float(mu)
            den += w

        if den <= 0:
            return None

        return num / den

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
