import time
import sys
import pandas as pd
import os


class Standard():
    """Class of Standard

    This class can perform analysis of outputs from robot.
    """

    def __init__(self, input_file, num_objectives, num_proposals, output_folder):
        """Constructor

        Args:
            input_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            num_proposals (int): unused here
            output_folder (str): the folder where the output files are stored by robot
        """

        self.input_file = input_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        self.output_folder = output_folder

    def perform(self, Psycho):
        """Perform analysis of output from robots."""

        global psycho
        psycho = Psycho

        print("Start analysis output!")

        # res = self.recieve_exit_message(self.output_folder)
        # if res is False:
        #     print("ErrorCode: error in recieve_exit_message function")
        #     sys.exit()

        res, i_List, rows = self.load_data(self.input_file)
        if res is False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res, r_List = self.extract_objectives(self.output_folder)
        if res is False:
            print("ErrorCode: error in extract_objectives function")
            sys.exit()

        res = self.update_candidate_file(self.input_file, rows, r_List, i_List)
        if res is False:
            print("ErrorCode: error in update_candidate_file function")
            sys.exit()

        print("Finish analysis output!")
        return "True"

    def load_data(self, input_file):
        """
        Find the last row with any pending objective among the final
        self.num_objectives columns. Return that row and the full CSV.
        """
        try:
            df = pd.read_csv(input_file)

            if self.num_objectives <= 0:
                raise ValueError("num_objectives must be >= 1.")
            if self.num_objectives > len(df.columns):
                raise ValueError("num_objectives exceeds column count in input_file.")

            # obj_cols = list(df.columns[-self.num_objectives:]) Im not doing last_col based anymore
            obj_cols = self._get_objective_columns(df) # but y_i based now
            
            obj_block = df[obj_cols].replace(r"^\s*$", pd.NA, regex=True)
            pending_mask = obj_block.isna().any(axis=1)

            if not pending_mask.any():
                raise ValueError("No pending experiment row found.")

            target_idx = pending_mask[::-1].idxmax()
            target_row = df.iloc[target_idx].tolist()
            rows = df.values.tolist()
            self._target_idx = target_idx

            return True, target_row, rows

        except Exception as e:
            print("load_data error:", e)
            return False, None, None

    def recieve_exit_message(self, output_folder):
        """Recieving exit message from machine

        This function DEPENDS on robot.
        """
        try:
            filepath = os.path.join(output_folder, "outputend.txt")

            while not os.path.isfile(filepath):
                time.sleep(10)

            os.remove(filepath)
            return True

        except Exception:
            return False

    def extract_objectives(self, output_folder):
        """
        Read the last non-empty row from results.csv and return exactly
        self.num_objectives values from left to right.
        """
        try:
            filepath = os.path.join(output_folder, "results.csv")
            df = pd.read_csv(filepath, header=None)
            df = df.dropna(how="all")

            if df.empty:
                raise ValueError("results.csv is empty.")

            last = df.iloc[-1].tolist()
            if len(last) < self.num_objectives:
                raise ValueError("results.csv does not contain enough objective values.")

            vals = []
            for v in last[:self.num_objectives]:
                if pd.isna(v):
                    raise ValueError("results.csv contains a missing objective value.")
                s = str(v).strip()
                if s == "":
                    raise ValueError("results.csv contains a blank objective value.")
                vals.append(float(s))

            return True, vals

        except Exception as e:
            print(f"extract_objectives error: {e}")
            return False, None


    def _get_objective_columns(self, df):
        cols = list(df.columns)

        plain = [f"y{i+1}" for i in range(self.num_objectives)]
        underscored = [f"y_{i+1}" for i in range(self.num_objectives)]

        if all(c in cols for c in plain):
            return plain

        if all(c in cols for c in underscored):
            return underscored

        raise ValueError(f"Could not find objective columns. Expected {plain} or {underscored}, found {cols}")



    def update_candidate_file(self, input_file, rows_unused, r_List, i_List_unused):
        """
        Update the last pending row by writing objective values directly into
        the final self.num_objectives columns in order.
        """
        try:
            df = pd.read_csv(input_file)

            if self.num_objectives <= 0:
                raise ValueError("num_objectives must be >= 1.")
            if self.num_objectives > len(df.columns):
                raise ValueError("num_objectives exceeds column count in input_file.")
            if len(r_List) != self.num_objectives:
                raise ValueError("Number of extracted objectives does not match num_objectives.")

            obj_cols = self._get_objective_columns(df)
            obj_block = df[obj_cols].replace(r"^\s*$", pd.NA, regex=True)
            pending_mask = obj_block.isna().any(axis=1)

            if not pending_mask.any():
                raise ValueError("No pending experiment row found to update.")

            target_idx = pending_mask[::-1].idxmax()

            for col, value in zip(obj_cols, r_List):
                df.at[target_idx, col] = value

            df.to_csv(input_file, index=False)
            print(
                f"Updated row {target_idx + 1}: " + ", ".join(
                    [f"{col}={value}" for col, value in zip(obj_cols, r_List)]
                )
            )
            return True

        except Exception as e:
            print(f"Error updating candidates: {e}")
            return False
