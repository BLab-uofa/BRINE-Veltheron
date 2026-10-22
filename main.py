import nimsos
import os
print(os.getcwd())
os.chdir(r"C:\Users\Blab2\Desktop\BRINE Veltheron")


ObjectivesNum = 3
ProposalsNum = 1
CyclesNum = 150
Inputs = 4
Real_Inputs = Inputs # we do 1 for VERIFICATION (predefined no AI)


cell_vol = 1105.0 # 1105 is better for constraint violation 


salts = ["ZnCl2", "ZnSO4", "NaCl", "KCl"]
salt_tank = [50000, 50000, 50000, 40000]; water_tank = 50000 # salt_tank = [50000, 50000, 50000, 50000, 50000]; water_tank = 50000
stock_M = [5, 3, 4, 4] #[8,3,3,4,4]
tempC = 25
PHREEQC_EXE_PATH =  r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\bin\phreeqc.bat"
PHREEQC_DATABASE_PATH = r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\database\llnl_VTH.dat"


tests = ['EIS' ,'LSV'] #, 'EIS', 'LSV', "CV"
lsv_order = ["HER", "OER"]

vm1_electrodes = [21, 22]
vm1_initial = 22

test_electrodes = {
    "EIS": 22,
    "CV": 69, # :))))))))))))))))
    "LSV": {"HER": 22, "OER": 21}}

electrode_config = {"vm1":
    {"available": vm1_electrodes, "initial": vm1_initial, "required_by_test": test_electrodes, "lsv_order": lsv_order}}


tactical_mode = False

candidates_file = "./candidates.csv"
proposals_file = "./proposals.csv"

input_folder = os.path.abspath("./EXPInput")
output_folder = os.path.abspath("./EXPOutput")

mobility_file = input_folder + "/mobility.csv"


BO_type = "Ax" #OR: Ax

for K in range(44, CyclesNum):

    print("Initialisation in progress...", K+1)
    nimsos.selection(method = BO_type,
                    input_file = candidates_file,
                    output_file = proposals_file,
                    output_folder = output_folder,
                    num_objectives = ObjectivesNum,
                    num_proposals = ProposalsNum,
                    num_inputs = Inputs,
                    Psycho = K+1)
    
    # nimsos.digital_twin(input_file = candidates_file,
    #                  output_folder = output_folder,
    #                  num_inputs = Inputs,
    #                  num_objectives = ObjectivesNum,
    #                  num_proposals = ProposalsNum,
    #                  EC_Tests = tests,
    #                  Psycho = K+1,
    #                  cell_volume_uL = cell_vol, stock_molarities = stock_M, salt_formulas= salts, Salt_Tank=salt_tank, Water_Tank=water_tank, tempC = tempC,
    #                  phreeqc_exe_path = PHREEQC_EXE_PATH,
    #                  phreeqc_database_path = PHREEQC_DATABASE_PATH,
    #                  mobility_file = mobility_file,)


    nimsos.feedback(input_file = candidates_file,
                    input_folder = input_folder,
                    num_inputs = Inputs,
                    num_proposals = ProposalsNum,
                    Psycho = K+1,
                    Salt_Tank = salt_tank,
                    Water_Tank = water_tank,
                    tg_bot_token = TG_BOT_TOKEN,
                    tg_chat_id = TG_CHAT_ID)
    

    nimsos.preparation_input(machine = "STAN",
                             input_file = candidates_file,
                             input_folder = input_folder,
                             output_folder = output_folder,
                             Real_Inputs = Real_Inputs,
                             EC_Tests = tests,
                             Electrode_Config = electrode_config,
                             Salt_Tank = salt_tank, Water_Tank = water_tank, cell_volume_uL = cell_vol, stock_molarities = stock_M,
                             Psycho = K+1)
    

    nimsos.analysis_output(machine = "STAN",
                           input_file = candidates_file,
                           num_objectives = ObjectivesNum,
                           num_proposals = ProposalsNum,
                           output_folder = output_folder,
                           Psycho = K+1)
    
    if tactical_mode:
        nimsos.tactical_pareto(input_file= candidates_file, num_inputs=Real_Inputs,
                               num_objectives=ObjectivesNum, Psycho= K+1,
                               update_every=3, panels= 4,
                               save_png=False, save_html=False, show=True)
    
    
    # if K + 1 == 20 or K + 1 == 40: #CyclesNum / 2
    #     while True:
    #         usr_in = input("Type 'done' to continue with new electrode: ").strip().lower()
    #         if usr_in == "done":
    #             break    
