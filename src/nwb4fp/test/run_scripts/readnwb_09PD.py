from nwb4fp.main.main_create_nwb import run_qmnwb,test_qmnwb
from pathlib import Path
import pandas as pd

def main():
    import os
    import sys
    base_path = Path("Q:/Sachuriga/Sachuriga_Python")
    base_data_folder = Path("S:/Sachuriga/")
    sex = "F"
    animals = ["65165", "65091", "65283"] 
    age = "P45+"
    project_name = "CR_CA1"
    species = "Mus musculus"
    vedio_search_directory = base_data_folder/fr"Ephys_Vedio/CR_CA1/"
    path_save = base_data_folder/fr"nwb"
    temp_folder = Path(r'C:/temp_waveform09006/')
    save_path_test=(r"S:\Sachuriga/Ephys_Recording/4nwb_check.csv")
    file_suffix = "phy_k"
    idun_vedio_path=r"P:/Overlap_project/data/CR_implant_add_new"
    #run_qmnwb(animals,base_data_folder,project_name,file_suffix,sex,age,species,vedio_search_directory,path_save,temp_folder)
    # test_qmnwb(animals,
    #            base_data_folder,
            #    project_name,
            #    file_suffix,
    #            temp_folder,
    #            save_path_test,
    #            vedio_search_directory,
    #            idun_vedio_path=idun_vedio_path)

    run_qmnwb(animals,
              base_data_folder,
              project_name,
              file_suffix,
              sex,
              age,
              species,
              vedio_search_directory,
              path_save,
              temp_folder)
    
if __name__ == "__main__":
    main()