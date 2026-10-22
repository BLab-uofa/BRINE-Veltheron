import os
import pandas as pd


class CandidateTable:
    """
    Handles the candidates.csv structure.

    First num_inputs columns are inputs; selected objectives are explicitly named.
    """

    def __init__(self, input_file, num_inputs, objective_columns):
        self.input_file = input_file
        self.num_inputs = int(num_inputs)
        self.selected_columns = list(objective_columns)

    def load(self):
        if not os.path.isfile(self.input_file):
            raise FileNotFoundError(f"Candidate file not found: {self.input_file}")

        df = pd.read_csv(self.input_file)

        missing = [col for col in self.selected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Configured objective columns are missing: {missing}")
        if len(df.columns) < self.num_inputs or set(self.input_columns(df)) & set(self.selected_columns):
            raise ValueError("Input columns and objective columns must be separate.")

        return df

    def save(self, df):
        df.to_csv(self.input_file, index=False)

    def input_columns(self, df):
        return list(df.columns[:self.num_inputs])

    def objective_columns(self, df):
        return self.selected_columns.copy()

    def metadata_columns(self, df):
        used = set(self.input_columns(df) + self.selected_columns)
        return [col for col in df.columns if col not in used]

    def pending_mask(self, df):
        obj_cols = self.objective_columns(df)

        obj_block = df[obj_cols].apply(pd.to_numeric, errors="coerce")
        return obj_block.isna().any(axis=1)

    def completed_mask(self, df):
        obj_cols = self.objective_columns(df)

        obj_block = df[obj_cols].apply(pd.to_numeric, errors="coerce")
        return obj_block.notna().all(axis=1)

    def pending_rows(self, df):
        return df[self.pending_mask(df)].copy()

    def completed_rows(self, df):
        return df[self.completed_mask(df)].copy()
