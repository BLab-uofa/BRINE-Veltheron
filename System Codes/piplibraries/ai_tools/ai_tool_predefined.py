# ai_tool_predefined.py

import os
import pandas as pd


class Predefined:
    """
    Reads predefined experiment conditions from a hardcoded CSV,
    picks the row matching self.psycho, and appends/writes it to self.input_file.

    Expected columns:
      x1..xN, y1..yM, trial_index

    The predefined file may also contain one of:
      iteration, psycho, Psycho, trial_index

    If no matching column exists, it falls back to CSV row number.
    """
    ## HARDCODES
    PREDEFINED_FILE = r"C:/Users/Blab2/Desktop/BRINE Veltheron/candidates_ref.csv"
    # CHANGE LINE: return df.iloc[self.psycho ... which shows the starting point of the candid ref file

    def __init__(
        self,
        input_file,
        output_file,
        output_folder,
        num_objectives,
        num_proposals,
        num_inputs,
        psycho,
        force_start = False,
    ):
        self.input_file = input_file
        self.output_file = output_file          # kept for interface consistency
        self.output_folder = output_folder      # kept for interface consistency

        self.num_objectives = int(num_objectives)
        self.num_proposals = int(num_proposals)
        self.num_inputs = int(num_inputs)
        self.psycho = int(psycho)
        self.force_start = bool(force_start)

        self._ensure_output_file_exists()

    def select(self):
        print("Start of Predefined!")
        self.run_predefined()
        print("Selection of predefined proposal ended!")
        return True

    def run_predefined(self):
        predefined_df = pd.read_csv(self.PREDEFINED_FILE)

        input_cols, output_cols = self._expected_columns()

        missing_inputs = [c for c in input_cols if c not in predefined_df.columns]
        if missing_inputs:
            raise ValueError(
                f"Predefined file is missing input columns: {missing_inputs}"
            )

        selected = self._select_row_for_iteration(predefined_df)

        output_df = pd.read_csv(self.input_file)

        # Build one proposal row: x values from predefined file, y values empty.
        row = {c: selected[c] for c in input_cols}

        for y_col in output_cols:
            row[y_col] = None

        # Keep trial_index consistent with the iteration number unless the
        # predefined file explicitly supplies trial_index.
        if "trial_index" in predefined_df.columns and pd.notna(selected.get("trial_index")):
            row["trial_index"] = int(selected["trial_index"])
        else:
            row["trial_index"] = self.psycho

        output_df.loc[len(output_df)] = row
        output_df.to_csv(self.input_file, index=False)

        print(f"Proposed predefined row for iteration {self.psycho}:")
        print({c: row[c] for c in input_cols})
        print(f"-> Added to {self.input_file}")

    def _select_row_for_iteration(self, df):
        """
        Prefer matching by explicit iteration-like column.
        Otherwise fall back to row position.
        """
        if self.force_start:
            return df.iloc[self.psycho] # force it with [self.psycho - XXX] and XXX is current psycho Pos

        possible_iteration_cols = [
            "iteration",
            "Iteration",
            "psycho",
            "Psycho",
            "trial_index",
        ]

        for col in possible_iteration_cols:
            if col in df.columns:
                matches = df[pd.to_numeric(df[col], errors="coerce") == self.psycho]
                if not matches.empty:
                    return matches.iloc[0]

        # Fallback: choose by row position.
        # Assumes self.psycho is zero-based.
        if self.psycho < 0 or self.psycho >= len(df):
            raise IndexError(
                f"self.psycho={self.psycho} is outside predefined file range "
                f"0..{len(df) - 1}"
            )

        return df.iloc[self.psycho]

    def _expected_columns(self):
        input_cols = [f"x{i + 1}" for i in range(self.num_inputs)]
        output_cols = [f"y{i + 1}" for i in range(self.num_objectives)]
        return input_cols, output_cols

    def _ensure_output_file_exists(self):
        if os.path.exists(self.input_file):
            return

        input_cols, output_cols = self._expected_columns()
        cols = input_cols + output_cols + ["trial_index"]

        pd.DataFrame(columns=cols).to_csv(self.input_file, index=False)
        print(f"Created output/input file '{self.input_file}' with headers: {cols}")