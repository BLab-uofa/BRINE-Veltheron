import os
print(os.getcwd())
os.chdir(r"C:\Users\Blab2\Desktop\BRINE VeltheronDT")
import nimsos

candidates_file = "./candidates.csv"
output_folder = "./EXPOutput"

Inputs = 4
ObjectivesNum = 3
Objective_Map = {"y1": "conductivity"}

ProposalsNum = 1

# objective_targets = ("her", "oer")

cell_volume_uL = 1105.0
stock_molarities = [5,3,4,4]
salt_formulas = ["ZnCl2", "ZnSO4", "NaCl", "KCl",]

PHREEQC_EXE_PATH = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\bin\ClrRelease\phreeqc.exe"
PHREEQC_DATABASE_PATH = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\database\llnl_VTH.dat"

mobility_file = "./mobility.csv"
temp_C = 25.0

nimsos.digital_twin(
    input_file=candidates_file,
    output_folder=output_folder,
    
    num_inputs=Inputs,
    num_objectives=ObjectivesNum,
    num_proposals=ProposalsNum,
    
    objective_map = Objective_Map,
    
    Psycho=1,
    cell_volume_uL=cell_volume_uL,
    stock_molarities=stock_molarities,
    salt_formulas=salt_formulas,
    phreeqc_exe_path=PHREEQC_EXE_PATH,
    phreeqc_database_path=PHREEQC_DATABASE_PATH,
    temp_C=temp_C,
    mobility_file=mobility_file, 
)