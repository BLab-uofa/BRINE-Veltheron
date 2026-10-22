import os
import re
import subprocess
import pandas as pd
from typing import Dict


class PhreeqcEngine:
    def __init__(self, config):
        self.config = config

    def make_input(
        self,
        totals: Dict[str, float],
        case_name: str,
        selected_file: str,
        density_kg_L: float,
    ) -> str:
        lines = []
        for el, conc in totals.items():
            if conc is None:
                continue
            conc = float(conc)
            if conc <= 0:
                continue
            lines.append(f"    {el} {conc}")

        ph_line = (
            f"    pH {self.config.phreeqc_ph_guess} charge"
            if self.config.phreeqc_charge_balance_pH
            else f"    pH {self.config.phreeqc_ph_guess}"
        )

        return f"""
SOLUTION 1 {case_name}
    temp {float(self.config.temp_C)}
    units {self.config.phreeqc_units}
    density {float(density_kg_L)}
{ph_line}
{chr(10).join(lines)}
    water 1.0

SELECTED_OUTPUT 1
    -file {selected_file}
    -reset false
    -high_precision true
    -pH true
    -ionic_strength true

END
"""

    @staticmethod
    def _is_float(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    def parse_species_from_output(self, output_text: str) -> Dict[str, float]:
        min_conc = float(self.config.phreeqc_parse_min_species_M)

        species = {}
        inside = False

        stop_words = (
            "Saturation indices",
            "Phase assemblage",
            "Exchange assemblage",
            "Surface assemblage",
            "Gas phase",
            "Solid solutions",
            "Beginning of batch-reaction",
            "End of simulation",
        )

        for line in output_text.splitlines():
            s = line.strip()

            if "Distribution of species" in s:
                inside = True
                continue

            if not inside:
                continue

            if any(w in s for w in stop_words):
                break

            if not s or s.startswith(("Species", "Log", "-")):
                continue

            parts = s.split()

            # Element totals have two fields; species rows have molality, activity
            # and three log fields. Do not admit totals or saturation-index rows.
            if len(parts) >= 6 and all(self._is_float(value) for value in parts[1:6]):
                name = parts[0]
                conc = float(parts[1])

                if name != "H2O" and conc > min_conc:
                    species[name] = conc

        return species

    def run(self, totals: Dict[str, float], case_name: str, density_kg_L: float) -> Dict[str, object]:
        exe = os.path.abspath(self.config.phreeqc_exe_path)
        db = os.path.abspath(self.config.phreeqc_database_path)

        run_dir = self.config.phreeqc_run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(
                self.config.output_folder,
                self.config.dt_subfolder,
                run_dir,
            )
        run_dir = os.path.abspath(run_dir)

        if not os.path.isfile(exe):
            raise FileNotFoundError(f"PHREEQC executable not found:\n{exe}")

        if not os.path.isfile(db):
            raise FileNotFoundError(f"PHREEQC database not found:\n{db}")

        os.makedirs(run_dir, exist_ok=True)

        inp = f"{case_name}.phr"
        out = f"{case_name}.out"
        screen = f"{case_name}_screen.out"
        selected = f"{case_name}_selected.out"

        paths = {
            "inp": os.path.join(run_dir, inp),
            "out": os.path.join(run_dir, out),
            "screen": os.path.join(run_dir, screen),
            "selected": os.path.join(run_dir, selected),
        }

        for p in paths.values():
            if os.path.isfile(p):
                os.remove(p)

        input_text = self.make_input(
            totals=totals,
            case_name=case_name,
            selected_file=selected,
            density_kg_L=density_kg_L,
        )

        with open(paths["inp"], "w", encoding="utf-8") as f:
            f.write(input_text)

        result = subprocess.run(
            [exe,inp,out,db,screen],
            cwd=run_dir,
            capture_output=True,
            text=True,
            shell=False,)

        output_text = ""
        screen_text = ""

        if os.path.isfile(paths["out"]):
            with open(paths["out"], "r", encoding="utf-8", errors="replace") as f:
                output_text = f.read()

        if os.path.isfile(paths["screen"]):
            with open(paths["screen"], "r", encoding="utf-8", errors="replace") as f:
                screen_text = f.read()

        if result.returncode != 0:
            raise RuntimeError(
                "PHREEQC failed.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}\n"
                f"SCREEN:\n{screen_text}\n"
                f"OUTPUT:\n{output_text}"
            )

        if not os.path.isfile(paths["selected"]):
            raise RuntimeError(f"PHREEQC selected output was not created: {paths['selected']}")

        selected_df = pd.read_csv(paths["selected"], sep=r"\s+", engine="python")
        row = selected_df.iloc[0].to_dict()

        species = self.parse_species_from_output(output_text)
        if not species:
            raise ValueError("No aqueous species parsed from PHREEQC output.")

        return {
            "pH": float(row["pH"]),
            "ionic_strength": float(row["mu"]),
            "species": species,
            "totals": totals,
            "case_name": case_name,
        }
