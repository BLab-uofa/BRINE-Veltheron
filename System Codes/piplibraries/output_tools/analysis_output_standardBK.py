import csv
import time
import sys
import math
import csv
import pandas as pd
import os


class Standard():
    """Class of Standard

    This class can perform analysis of outputs from robot.

    """

    def __init__(self, input_file, num_objectives, num_proposals, output_folder):
        """Constructor
        
        This function do not depend on robot.

        Args:
            DELETED (output_file (str): the file for proposals from MI algorithm)
            input_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the output files are stored by robot

        """

        self.input_file = input_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        self.output_folder = output_folder


    def perform(self, Psycho):
        """perfroming analysis of output from robots

        This function do not depend on robot.
    
        Returns:
            res (str): True for success, False otherwise.

        """

        global psycho
        psycho = Psycho
        
        print("Start analysis output!")

        # res = self.recieve_exit_message(self.output_folder)

        # if res == False:
        #     print("ErrorCode: error in recieve_exit_message function")
        #     sys.exit()
        
        res, i_List, rows = self.load_data(self.input_file)

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res, r_List = self.extract_objectives(self.output_folder)

        if res == False:
            print("ErrorCode: error in extract_objectives function")
            sys.exit()

        res = self.update_candidate_file(self.input_file, rows, r_List, i_List)

        if res == False:
            print("ErrorCode: error in update_candidate_file function")
            sys.exit()

        # # self.log_psycho()
        
        print("Finish analysis output!")
        return "True"



    def load_data(self, input_file):
        """
        Locate the last row whose objective column is empty/NaN
        and return that row so it can be updated with the real result.
        Returns
        -------
        res        : bool
        target_row : list[str]       – the row to overwrite
        rows       : list[list[str]] – full CSV content
        """
        try:
            import pandas as pd
            df = pd.read_csv(input_file, dtype=str)   # keep cells as strings
            obj_col = df.columns[-1]                  # objective = last column
            # boolean mask for rows whose objective cell is blank or NaN
            mask_empty = df[obj_col].isna() | (df[obj_col].str.strip() == "")
            if not mask_empty.any():
                raise ValueError("No pending experiment row (objective NaN) found.")
            # index of the last such row
            target_idx = mask_empty[::-1].idxmax()
            target_row = df.iloc[target_idx].tolist()
            rows = df.values.tolist()                 # list‑of‑lists for updater
            res  = True
            # store index for updater (optional: as an attribute)
            self._target_idx = target_idx
            
        except Exception as e:
            print("load_data error:", e)
            res, target_row, rows = False, None, None
            
        return res, target_row, rows



    def extract_objectives(self, output_folder):    
        """Extracting objective values from output files by robot

        This function DEPENDS on robot.

        Args:
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the results by machine are stored
            p_List (list[float]): the list of proposals

        Returns:
            res (bool): True for success, False otherwise.
            r_List (list[float]): the list of objectives

        """
        try:
                import pandas as pd, os
                filepath = os.path.join(output_folder, "results.csv")
                df = pd.read_csv(filepath, header=None)  # no header
                
                if df.empty:
                    print("extract_objectives warning: results.csv is empty.")
                    return True, []
                
                # take the last num_proposals rows (bottom of file = latest)
                need = self.num_proposals
                got = min(len(df), need)
                
                if got < need:
                    print(f"extract_objectives warning: requested {need} proposals, only found {got} rows in results.csv.")
                df = df.tail(got).reset_index(drop=True)
                m = self.num_objectives
                r_List = []
                
                for idx, row in df.iterrows():
                    vals = [str(x) for x in row.tolist()]
                    if len(vals) < m:
                        print(f"extract_objectives warning: row {idx} has {len(vals)} < num_objectives={m}; padding with blanks.")
                        vals = vals + [""] * (m - len(vals))
                    elif len(vals) > m:
                        print(f"extract_objectives warning: row {idx} has {len(vals)} > num_objectives={m}; truncating extras.")
                        vals = vals[:m]
                    r_List.append(vals)
                return True, r_List
            
        except Exception as e:
            print("extract_objectives error:", e)
            return False, None


    def update_candidate_file(self, input_file, rows_unused, r_List, i_List_unused):
        """Updating candidates
        This function does not depend on robot.
        Args:
            num_objectives (int): the number of objectives
            DELETED (output_file)
            r_List (list[float]): the list of objectives
        Returns:
            res (bool): True for success, False otherwise.
        """
        global psycho
        
        try:
            import pandas as pd
            df = pd.read_csv(input_file, dtype=str)
            m = self.num_objectives
            obj_cols = list(df.columns[-m:])  # use class config, not detection
            
            def _blank(x):
                return (pd.isna(x)) or (str(x).strip() == "")
            
            pending_mask = df[obj_cols].applymap(_blank).all(axis=1)
            pending_idxs = list(df[pending_mask].index)
            
            if not pending_idxs:
                raise ValueError("No placeholder rows (all objective columns empty) found.")
            need = min(self.num_proposals, len(r_List))
            
            if need < self.num_proposals:
                print(f"update_candidate_file warning: requested {self.num_proposals} proposals but only {need} result rows provided.")
            fill_n = min(need, len(pending_idxs))
            
            if fill_n < need:
                print(f"update_candidate_file warning: only {fill_n} pending rows available for {need} results.")
                
            # newest pending rows (bottom) get the latest results (end of r_List)
            targets = pending_idxs[-fill_n:]
            payload = r_List[-fill_n:]
            
            for cand_idx, vals in zip(targets, payload):
                # vals already coerced to length m in extract_objectives
                for col, v in zip(obj_cols, vals):
                    df.at[cand_idx, col] = v
                    
            df.to_csv(input_file, index=False)
            print(f"Saved {fill_n} result row(s) into {input_file} (columns: {obj_cols})")
            return True
        
        except Exception as e:
            print(f"Error updating candidates: {e}")
            return False

