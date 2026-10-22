import csv
import sys
import json
import os
#import subprocess
#import time
#import pathlib
# sys.path.append('C:/Users/Blab2/Desktop/NIMSOS_base')
# from globals import cycle_no

# print('CycleNo in input-class: ', cycle_no)
# print(f"py lib path: {sys.path}")


class Standard():
    """Class of Standard

    This class can create input file for robot experiments and start the robot experiments.

    """

    def __init__(self, input_file, input_folder, output_folder, Real_Inputs, EC_Tests, Salt_Tank, Water_Tank, cell_volume_uL, stock_molarities, Electrode_Config=None):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm
            input_folder (str): the folder where input files for robot are stored

        """

        self.input_file = input_file
        self.inputFolder = input_folder
        self.outputFolder = output_folder
        self.Real_Inputs = Real_Inputs
        self.EC_Tests = EC_Tests
        self.Salt_Tank = Salt_Tank
        self.Water_Tank = Water_Tank
        self.cell_volume_uL = cell_volume_uL
        self.stock_molarities = stock_molarities
        self.Electrode_Config = Electrode_Config


    def perform(self, Psycho):
        """perfroming preparation input and starting robot experiments 

        This function do not depend on robot.
    
        Returns:
            res (str): True for success, False otherwise.

        """
        global psycho
        psycho = Psycho
        
        print("Start preparation input!")

        res, p_List = self.load_data(self.input_file)

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res = self.make_machine_file(p_List, self.inputFolder, self.Real_Inputs, self.EC_Tests, self.Salt_Tank, self.Water_Tank, self.cell_volume_uL, self.stock_molarities, self.Electrode_Config)

        if res == False:
            print("ErrorCode: error in make_machine_file function")
            sys.exit()

        res = self.send_message_machine(self.inputFolder, self.outputFolder)

        if res == False:
            print("ErrorCode: error in send_message_machine function")
            sys.exit()


        print("Finish preparation input!")

        return "True"



    def load_data(self, input_file):
        """Loading proposals

        This function do not depend on robot.
    
        Args:
            input_file (str): the file for proposals from AI algorithm

        Returns:
            res (bool): True for success, False otherwise.
            p_List (list[float]): the list of proposals

        """

        p_List = []

        try:
            with open(input_file) as inf:

                reader = csv.reader(inf)
                rows = list(reader)
                p_List = rows
                
                
            res = True

        except:
            res = False

        return res, p_List 



    def make_machine_file(self, p_List, inputFolder, Real_Inputs, EC_Tests, Salt_Tank, Water_Tank, cell_volume_uL, stock_molarities, Electrode_Config):
        """Making input files for robot

        This function DEPEND on robot.

        Args:
            p_List (list[float]): the list of proposals 
            inputFolder (str): the folder where the input files for robot are stored

        Returns:
            res (bool): True for success, False otherwise.

        """

        res = False

        try:
            
            global psycho
            
            
            suggest_ratios = p_List[-1][0:self.Real_Inputs]  # Since the last one is NaN
            print('Make machine file in progress...')
            
            # Let's now save the sum of inputs to keep track of the consumed whiskey, beer, etc
            header = p_List[0]
            data_rows = p_List[1:]
            col_sums = []
            for i in range(Real_Inputs):
                vals = []
                for row in data_rows:
                    try:
                        vals.append(float(row[i]))
                    except:
                        raise ValueError
                col_sums.append(sum(vals))


            # UPDATES.CONF Formation
            conf_path = inputFolder + "/UPDATES.json"

            old_conf = {}
            if os.path.exists(conf_path):
                try:
                    with open(conf_path, "r", encoding="utf-8") as f:
                        old_conf = json.load(f)
                except Exception:
                    old_conf = {}

            conf = {
                "suggest_ratios": suggest_ratios,
                "psycho": psycho,
                "real_salt_no": Real_Inputs,
                "input_column_sums": col_sums,
                "tests": EC_Tests,
                "cell_volume_uL": cell_volume_uL,
                "stock_molarities": stock_molarities,
                "Salt_Tank": Salt_Tank,
                "Water_Tank": Water_Tank,
                "electrode_config": Electrode_Config,}
            
            
            if Electrode_Config is not None:

                if psycho == 1 or "electrode_state" not in old_conf:
                    conf["electrode_state"] = {
                        "vm1": Electrode_Config["vm1"]["initial"]}

                else:
                    conf["electrode_state"] = old_conf["electrode_state"]

            # Preserve the shared liquid ledger across iterations.
            for key in [
                "warning_margin_uL",
                "protocol_refill_margin_uL",
                "physical_tube_capacity_uL",
                "refill_request",
                "liquid_state",
            ]:
                if key in old_conf:
                    conf[key] = old_conf[key]

            # Backward-compatible defaults if this is the first run.
            conf.setdefault("warning_margin_uL", 3000)
            conf.setdefault("protocol_refill_margin_uL", 1000)
            conf.setdefault("physical_tube_capacity_uL", 50000)

            conf.setdefault("refill_request", {
                "enabled": False,
                "salt_uL": [None for _ in range(Real_Inputs)],
                "water_uL": None,
            })

            conf.setdefault("liquid_state", {
                "salt_fill_uL": [float(x) for x in Salt_Tank[:Real_Inputs]],
                "salt_refill_at_usage_uL": [0.0 for _ in range(Real_Inputs)],
                "water_fill_uL": float(Water_Tank),
                "water_refill_at_usage_uL": 0.0,
            })

            with open(conf_path, "w", encoding="utf-8") as conf_file:
                json.dump(conf, conf_file, indent=2)
                
            res = True  

        except Exception as e:
            print(f"Error: {e}")
            res = False

        return res


    def send_message_machine(self, inputFolder, outputFolder):
        
        """Sending a message to start the robot
    
        This function DEPENDS on robot.
    
        Args:
            inputFolder (str): the folder where the input files for robot are stored
    
        Returns:
            res (bool): True for success, False otherwise.
    
        """
        res = False
        print('Send machine file in progress, wait for execution log...')
    
        try:
            # Path to the WebSocket protocol script
            protocol_exec_path = inputFolder + "/RoboticProcess.py"   #OT2_Executer_WS_Verification.py
            
            exec_globals = {
                "__name__":"__main__",
                "__file__":protocol_exec_path,
                "inputFolder": inputFolder,
                "outputFolder": outputFolder,
            }
            with open(protocol_exec_path, 'r') as file:
                code = file.read()
                
            exec(code, exec_globals)   

            res = True
            
    
        except Exception as e:
            res = False
            print(f"Error: {e}")
    
        return res


