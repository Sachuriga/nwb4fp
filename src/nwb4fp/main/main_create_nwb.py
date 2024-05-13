from pathlib import Path
import string
import sys
from nwb4fp.preprocess.load_data import load_data
from nwb4fp.postprocess.quality_metrix import (qualitymetrix, test_clusterInfo)
from nwb4fp.postprocess.nwbPHYnOPHYS import nwbPHYnOPHYS
from nwb4fp.postprocess.add_wfcor import add_wf_cor
from nwb4fp.postprocess.extract_wf import wf4unim


def main():
    from pathlib import Path
    import string
    import sys
    from nwb4fp.preprocess.load_data import load_data
    from nwb4fp.postprocess.quality_metrix import qualitymetrix,test_clusterInfo
    from nwb4fp.postprocess.nwbPHYnOPHYS import nwbPHYnOPHYS
    from nwb4fp.postprocess.add_wfcor import add_wf_cor

    base_path = Path("Q:/Sachuriga/Sachuriga_Python")
    base_data_folder = Path("S:/Sachuriga/")
    project_path = fr"{base_path}/nwb4fprobe/src/nwb4fp/"
    sys.path.append(fr"{project_path}")

    # change to project root
    sys.path.append(project_path)

    from nwb4fp.preprocess.load_data import load_data
    from nwb4fp.postprocess.quality_metrix import qualitymetrix
    from nwb4fp.postprocess.nwbPHYnOPHYS import nwbPHYnOPHYS
    from nwb4fp.postprocess.add_wfcor import add_wf_cor
    # set params for nwb
    sex = "F"
    animals = ["65165","65091","65283"] 
    age = "P45+"
    species = "Mus musculus"
    vedio_search_directory = base_data_folder/fr"Ephys_Vedio/CR_CA1/"
    path_save = base_data_folder/fr"nwb"
    temp_folder = Path(r'C:/temp_waveform/')
    run_qmnwb(animals,base_data_folder,sex,age,species,vedio_search_directory,path_save,temp_folder)

def test_qmnwb(animals,base_data_folder,project_name, file_suffix, temp_folder,save_path_test,vedio_search_directory,idun_vedio_path):
    import pandas as pd
    df = pd.DataFrame(columns=['File', 'competability','dlc_model', 'video_name','video_file'])
    df.to_csv(save_path_test, index=False)
    for indvi in animals:
        ID = indvi
        counter = 0
        #getting sorted files02
        folder_path = fr"{str(base_data_folder)}/Ephys_Recording/{project_name}/{ID}/"
        ##for quality metrix
        sorted_files = load_data(folder_path, file_suffix=fr"{file_suffix}")

        for file in sorted_files:
            test_clusterInfo(file,
                             temp_folder,
                             save_path_test,
                             vedio_search_directory,
                             idun_vedio_path)


def run_qmnwb(animals,base_data_folder,project_name,file_suffix, sex,age,species,vedio_search_directory,path_save,temp_folder):
    for indvi in animals:
        ID = indvi
        counter = 0
        #getting sorted files02
        folder_path = fr"{str(base_data_folder)}/Ephys_Recording/{project_name}/{ID}/"
        ##for quality metrix
        sorted_files = load_data(folder_path, file_suffix=fr"{file_suffix}")

        for file in sorted_files:
            print(file)
            qualitymetrix(file,temp_folder)
            add_wf_cor(fr"{file}_manual")
            nwbPHYnOPHYS(fr"{file}_manual",
                        sex,
                        age,
                        species,
                        vedio_search_directory,
                        path_to_save_nwbfile = path_save)
            counter += 1
            percent = counter/len(sorted_files)
            #wf4unim(fr"{file}_manual")
            print(f"{percent} % completet!!!!{file}\ncreated new phy folder {file}_manual \ncreated nwb file at {path_save}for {ID} {age} {species}\n\n\n\n")

if __name__== "__main__":
    main()