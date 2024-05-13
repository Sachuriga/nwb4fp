# Neuroscience Data to NWB Conversion Script

This repository contains a Python package `nwb4fp`, designed to convert neuroscience data into the Neurodata Without Borders (NWB) format. It is specifically tailored for processing electrophysiology data from the open ephys system and behavioral tracking data analyzed with DeepLabCut.

## Introduction

The `test_qmnwb` function checks whether the manually curated sorting files and the DLC file meet the requirements for the next step. This function creates a `4nwb_check.csv` file, allowing you to verify if the files meet the necessary criteria. The `run_qmnwb` function facilitates the conversion of neuroscience data into the NWB format, a standardized format for neurophysiology data sharing and storage. This script is particularly useful for researchers working with Mus musculus, focusing on electrophysiology and behavioral data. The `run_qmnwb` reads all phy outputs ending with the `{phy suffix}` folder under each individual animal and selects the curated `good` units to calculate the quality metrics using the `spikeinterface` package's built-in function. It then creates a new phy output folder ending with `{phy suffix}_manual` to prepare for conversion to the `.nwb` file.

## Features

- **Data Conversion**: Efficiently converts electrophysiology and behavioral data into the NWB format.
- **Species and Demographic Specificity**: Required by 'nwbpy'.
- **Video File Handling**: Automatically searches for and processes video files from specified directories, integrating them with the NWB dataset.
- **Data Verification**: Generates a CSV file post-conversion to allow users to verify the integrity and completeness of the processed data.

## Installation

To use this script, you need to clone this repository and install the required Python package, `nwb4fp`.

### Cloning the Repository

Clone the repository to your local machine using the following command:

```bash
git clone <https://github.com/sachuriga283/QuattrocoloLab2nwb-nwb4fp.git>
cd ./QuattrocoloLab2nwb-nwb4fp
pip install requirements.txt
```

### Install with pipy
```bash
pip install nwb4fp
```
or create a condda env and install (recommended)
```bash
conda create -n nwb4fp -y python
conda activate nwb4fp
pip install nwb4fp
```

## Required Folder structure and the files for nwb4fp
### -base_data_folder Structure
The `-base_data_folder` is organized into two main subdirectories, each containing specific project-related files (recordings with phy output and videos with deeplabcut results). Here's the structure and description of the contained files:

- `-base_data_folder`
  - `Ephys_Video`
    - `project_name`
      - `fr"{video_name}{dlc_model_name}"_filtered.h5`: deeplabcut results for corresponding video file.
      - `fr"{video_name}".avi`: Original video file.
  - `Ephys_recording`
    - `project_name`
      - `individuals`
        - `recording/recording nodes`
          - `.continues`
            - `sample_index.npy`: index of each ephys sampling point.
            - `time_stemps.npy`: time stemps of each ephys sampling point (computer_based).
          - `.events`
            - `sample_index.npy`: index for each sampling point of TTL signal(which we used for aliging the time stemps).
            - `time_stemps.npy`: time stemps of each TTL sampling point (Is camera base).
            - `states.npy`: States signal of TTL signal, which is high low signal(-6 6 for 50Hz aquiring) refers to the start of the TTL and end of the TTL.
        - `phy_output`
          - `spike_times.npy`: this is `int` number which refers to the sampling index to - `.continues`  - `sample_index.npy`. it's contains all the spikes deteced by the sorting algorism.
          - `recording.dat`: raw binary data of the recording.
          - `spike_clusters.npy`: same length vector as - `spike_times.npy`, which labes the cluster name to each spikes.
          - `cluster_info.tsv`: sunmmary info of the sorting.

Please replace `project_name`, `video_name`, `dlc_model_name`, etc., with your specific project's details.


## Usage (example custom python file for running the nwb4fp)
```bash

from nwb4fp.main.main_create_nwb import run_qmnwb,test_qmnwb
from pathlib import Path
import pandas as pd

def main():
    import os
    import sys

    base_data_folder = Path("base folder")
    project_name = "Your_project"
    vedio_search_directory = base_data_folder/fr"Ephys_Vedio/{project_name}/"
    path_save = base_data_folder/fr"nwb"

    #temp folder to save temporally created waveform folder from spikeinterface
    temp_folder = Path(r'C:/temp_waveform/')
    save_path_test=(r"Your prefered saving path/4nwb_check.csv")

    ## The function will copy the videos to the deeplabcut video folder, which were analyzed by older Deeplabcut models
    idun_vedio_path=r"dlc_video_folder"
    sex = "F" # or "M"

    ## animals name for now only support 5 numbers str, for example here listed 6 animals
    animals = ["33331", "33332", "33333", "33334", "33335", "33336"]

    ## animals ages for first recording day
    age = "P45+"
    species = "Mus musculus"
    ## file suffix for phy output folder for example "phy_k"
    file_suffix = "phy_k"

    test_qmnwb(animals,
               base_data_folder,
               project_name,
               file_suffix,
               temp_folder,
               save_path_test,
               vedio_search_directory,
               idun_vedio_path=idun_vedio_path)

    ## check the 4nwb_check.csv file whether all the files (phy output and dlc .h5 file) is there and whether the file is competble to process quality metrix or not
    while True:
    user_input = input("Press 'c' to continue or 'q' to quit: ").strip().lower()
    if user_input == 'c':
        print("Continuing...")
        continue  # This will continue the loop
    elif user_input == 'q':
        print("Quitting...")
        break  # This will break out of the loop
    else:
        print("Invalid input. Please press 'c' to continue or 'q' to quit.")

    ## conversionning the data to nwb format
    run_qmnwb(animals,
              base_data_folder,
              project_name,
              file_suffix,
              sex,age,
              species,
              vedio_search_directory,
              path_save,
              temp_folder)

if __name__ == "__main__":
    main()
```

# Support
For any questions or issues, please open an issue on this repository, or contact the maintainers directly via GitHub Issues.

# Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes. For more information, see the contributing guide.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
