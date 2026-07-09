# CR–CA1 paper analyses

This directory holds the analysis code that reproduces the figures for the CR–CA1 study.
It is **not** part of the installable `nwb4fp` package (it lives outside `src/` and is
excluded from the wheel/sdist); it depends on `nwb4fp` plus additional analysis
libraries.

## Contents

`CR_CA1_paper/` is organised by analysis theme, e.g.:

- `script4figures/`, `script4PlosBiology/` — main- and supplementary-figure notebooks
- `Cell_types/`, `Functional_properties/`, `Locomotions/` — functional characterisation
- `Local_field_potential/`, `Phase_precess/`, `Remapping/` — spatial-coding / LFP analyses
- `Unit_match/`, `Neurons_location/`, `CRs_counting/` — unit matching and anatomy

## Reproducing a figure

1. Install the library and the extra analysis dependencies used by the notebooks:

   ```bash
   pip install -e ".[all]"
   # some notebooks additionally use: neurochat, pynapple, elephant, UnitMatchPy
   ```

2. Point the notebook at your local copy of the data (see below), then run it. Notebook
   outputs have been cleared to keep the repository small — re-running a notebook against
   the data regenerates its figure.

Notebooks that import the local `unit_match_files` helper expect to be run from their own
directory (Jupyter's default working directory).

## Data availability

The raw and processed neural data (NWB files, spike-sorted unit tables, position
tracking, LFP, rate maps) are **not** included in this repository. They are available
from the authors on reasonable request. Paths inside the notebooks point at the original
acquisition machines and must be updated to your local data locations.
