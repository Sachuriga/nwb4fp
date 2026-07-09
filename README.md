[![DOI](https://zenodo.org/badge/799944513.svg)](https://doi.org/10.5281/zenodo.16760325)

# nwb4fp — Neuroscience data to NWB conversion

`nwb4fp` is a Python package for converting neuroscience data into the
[Neurodata Without Borders (NWB)](https://www.nwb.org/) format. It is tailored for
electrophysiology recorded with Open Ephys and behavioural tracking analysed with
DeepLabCut, and also bundles the analysis code for the CR–CA1 project (see
[Paper analyses](#paper-analyses) below).

## Introduction

Two functions drive the conversion pipeline:

- **`test_qmnwb`** checks whether the manually curated spike-sorting output and the
  DeepLabCut files meet the requirements for conversion. It writes a `4nwb_check.csv`
  file so you can confirm everything is in place before continuing.
- **`run_qmnwb`** performs the conversion. It reads every phy output folder ending in
  `{phy_suffix}` for each animal, keeps the units curated as `good`, computes quality
  metrics with [SpikeInterface](https://github.com/SpikeInterface/spikeinterface), and
  writes a new `{phy_suffix}_manual` folder that is then packaged into a `.nwb` file.

The pipeline targets *Mus musculus* electrophysiology and behavioural data.

## Features

- **Data conversion**: turns Open Ephys electrophysiology and DeepLabCut behaviour into NWB.
- **Species / demographic metadata**: collected as required by `pynwb`.
- **Video handling**: finds and links the relevant video files into the NWB dataset.
- **Verification**: writes a CSV after conversion so you can check the data is complete.

## Installation

### From PyPI (recommended: in a fresh conda env)

```bash
conda create -n nwb4fp -y python
conda activate nwb4fp
pip install nwb4fp
```

### From source

```bash
git clone https://github.com/Sachuriga/nwb4fp.git
cd nwb4fp
pip install -e .            # runtime install
pip install -e ".[dev]"     # + test/lint/build tools
```

Optional extras: `.[phy]` (interactive phy GUI), `.[dlc]` (DeepLabCut kinematics),
or `.[all]` for both.

### SpikeInterface

Most functions depend on SpikeInterface and track its latest release. See
<https://github.com/SpikeInterface/spikeinterface> for details.

## Expected folder structure

`run_qmnwb` expects a `base_data_folder` with two subdirectories — the recordings
(with phy output) and the videos (with DeepLabCut results):

- `base_data_folder/`
  - `Ephys_Video/`
    - `project_name/`
      - `{video_name}{dlc_model_name}_filtered.h5` — DeepLabCut results for the video
      - `{video_name}.avi` — original video
  - `Ephys_recording/`
    - `project_name/`
      - `individuals/`
        - `recording nodes/`
          - `.continuous/`
            - `sample_index.npy` — index of each ephys sample
            - `timestamps.npy` — timestamp of each ephys sample (computer clock)
          - `.events/`
            - `sample_index.npy` — sample index of each TTL event (used to align time)
            - `timestamps.npy` — timestamp of each TTL event (camera clock)
            - `states.npy` — TTL state (high/low; ±6 for 50 Hz), marking TTL on/off
        - `phy_output/`
          - `spike_times.npy` — sample index of every detected spike
          - `recording.dat` — raw binary recording
          - `spike_clusters.npy` — cluster label for each spike
          - `cluster_info.tsv` — sorting summary

Replace `project_name`, `video_name`, `dlc_model_name`, etc. with your own details.

## Usage

A minimal script that runs the check and then the conversion:

```python
from nwb4fp.main.main_create_nwb import run_qmnwb, test_qmnwb
from pathlib import Path


def main():
    base_data_folder = Path("base folder")
    project_name = "Your_project"
    vedio_search_directory = base_data_folder / f"Ephys_Vedio/{project_name}/"
    path_save = base_data_folder / "nwb"

    # temp folder for the waveform folder SpikeInterface creates
    temp_folder = Path(r"C:/temp_waveform/")
    save_path_test = r"Your preferred saving path/4nwb_check.csv"

    # videos are copied to the DeepLabCut video folder (analysed by older DLC models)
    idun_vedio_path = r"dlc_video_folder"
    sex = "F"  # or "M"

    # animal names; currently only 5-character strings are supported
    animals = ["33331", "33332", "33333", "33334", "33335", "33336"]

    age = "P45+"               # age at the first recording day
    species = "Mus musculus"
    file_suffix = "phy_k"      # phy output folder suffix, e.g. "phy_k"

    test_qmnwb(animals,
               base_data_folder,
               project_name,
               file_suffix,
               temp_folder,
               save_path_test,
               vedio_search_directory,
               idun_vedio_path=idun_vedio_path)

    # inspect 4nwb_check.csv: are all phy outputs + DLC .h5 files present and usable?
    while True:
        user_input = input("Press 'c' to continue or 'q' to quit: ").strip().lower()
        if user_input == "c":
            print("Continuing...")
            break
        elif user_input == "q":
            print("Quitting...")
            return
        else:
            print("Invalid input. Please press 'c' to continue or 'q' to quit.")

    # convert the data to NWB
    run_qmnwb(animals,
              base_data_folder,
              project_name,
              file_suffix,
              sex, age,
              species,
              vedio_search_directory,
              path_save,
              temp_folder)


if __name__ == "__main__":
    main()
```

## Repository layout

```
src/nwb4fp/     installable library (main / preprocess / postprocess / analyses / data)
paper/          CR–CA1 paper reproducibility code — NOT part of the installed package
examples/       demo notebooks and stand-alone run-scripts
```

Because the package uses a `src/` layout, everything outside `src/` (`paper/`,
`examples/`) is excluded from the built wheel/sdist — `pip install nwb4fp` gives you the
library only.

## Paper analyses

The analysis code for the CR–CA1 study lives under [`paper/CR_CA1_paper/`](paper/),
including the main- and supplementary-figure notebooks and the spatial-coding and LFP
analyses. See [`paper/README.md`](paper/README.md) for how to reproduce the figures and
for the data-availability statement.

## Support

For questions or problems, please open an issue on this repository.

## Contributing

Contributions are welcome — please fork the repository and open a pull request.

## License

This project is licensed under the MIT License; see the [LICENSE](LICENSE) file.
