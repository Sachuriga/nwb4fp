import sys
sys.path.append(r"Q:/sachuriga/Sachuriga_Python/quattrocolo-nwb4fp/src/")
                
from nwb4fp.main.main_create_nwb import run_qmnwb,test_qmnwb
from pathlib import Path
import pandas as pd

def main():
    import os
    import sys
    base_path = Path("Q:/Sachuriga/Sachuriga_Python")
    base_data_folder = Path("S:/Sachuriga/")
    sex = "F"
    #animals = ["66537", "66538", "66539"] 
    animals = ["63383", "63385", "65091", "65165", "65283", "65588", "66537", "66538", "66539", "66922", "65622"] 
    #animals = ["66539", "66538", "66537"]
    
    age = "P45+"
    project_name = "CR_CA1"
    species = "Mus musculus"
    vedio_search_directory = base_data_folder/fr"Ephys_Vedio/CR_CA1/pytorch_model"
    path_save = base_data_folder/fr"nwb"
    temp_folder = Path(r'C:/temp_waveform/')
    save_path_test=(r"S:\Sachuriga/Ephys_Recording/4nwb_check.csv")
    file_suffix = "phy_k"
    post_fix_dlc = "shuffle1_snapshot_350_sk_filtered.h5"

    idun_vedio_path=r"P:/Overlap_project/data/CR_implant_add_new"
    # test_qmnwb(animals,
    #            base_data_folder,
    #            project_name,
    #            file_suffix,
    #            temp_folder,
    #            save_path_test,
    #            vedio_search_directory,
    #            idun_vedio_path=idun_vedio_path,
    #            post_fix_dlc = post_fix_dlc)

    run_qmnwb(animals,
              base_data_folder,
              project_name,
              file_suffix,
              sex,
              age,
              species,
              vedio_search_directory,
              path_save,
              temp_folder,
              skip_qmr=False,
              skip_lfp=False,
              post_fix_dlc = post_fix_dlc)
    
if __name__ == "__main__":
    main()