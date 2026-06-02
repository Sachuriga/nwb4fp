"""
Compute within-field vs outside-field firing rate per cell and append as new
columns to the functional-properties table.

Reuses the existing pipeline:
    load_speed_fromNWB -> pos2speed (speed filter) -> Speed_filtered_spikes
    -> SpatialMap.rate_map (smoothed map for field detection)
    -> separate_fields_by_laplace (field mask)
    -> raw (unsmoothed) spike/occupancy maps for true Hz.

Position (x, y over time) is read from the NWB files; spike_times are taken
directly from the table (one row = one unit), so units never need re-matching.

New columns added:
    in_field_rate     spikes inside field(s)  / time inside field(s)      [Hz]
    out_field_rate    spikes outside field(s) / time outside (visited)    [Hz]
    in_out_ratio      in_field_rate / out_field_rate  (spatial S/N)
    log_in_out_ratio  log(in_out_ratio)
    field_peak_rate   peak of the smoothed rate map                       [Hz]
    n_fields_laplace  number of detected fields
    field_fraction    field bins / visited bins

Run on a machine that can reach the NWB drive (here mounted at /Volumes/ntnu...).
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- make nwb4fp importable ---
SRC_DIR = "/Users/sachuriga/Desktop/code/nwb4fp/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pynapple as nap
import nwb4fp.analyses.maps as mapp
from nwb4fp.analyses.data import pos2speed, load_speed_fromNWB, Speed_filtered_spikes
from nwb4fp.analyses.fields import separate_fields_by_laplace

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TABLE_IN = "/Users/sachuriga/Desktop/Projects/CR_CA1_paper/tables/functional_properties_with_python_measurements_pycirc.pkl"
# 默认写到新文件以防误覆盖;要直接覆盖原表把 TABLE_OUT 改成 TABLE_IN
TABLE_OUT = "/Users/sachuriga/Desktop/Projects/CR_CA1_paper/tables/functional_properties_with_python_measurements_pycirc_infield.pkl"
NWB_DIR = "/Volumes/ntnu/mh/kin/quattrocolo/crhip/Sachuriga/nwb"
POS_KEY = "XY_mid_brain"

# spatial / field parameters (与 place-cell 分类口径一致)
BOX_SIZE = [1.0, 1.0]
BIN_SIZE = 0.05          # 1x1 box -> 20x20 bins
SMOOTHING = 0.05         # 平滑图(检测 field 用)
FIELD_THRESHOLD = 0.1    # separate_fields_by_laplace laplacian 阈值
MIN_FIELD_BINS = 4       # 小于此 bin 数的 field 丢弃
MIN_SPEED = 0.05         # pos2speed 速度过滤阈值
USE_PRIMARY_ONLY = False  # True: 仅用最大 field(label==1);False: 所有 field 合并

NEW_COLS = ["in_field_rate", "out_field_rate", "in_out_ratio", "log_in_out_ratio",
            "field_peak_rate", "n_fields_laplace", "field_fraction"]


def infield_outfield_rates(x, y, t, spike_times):
    """Return dict of in/out-field metrics for one unit."""
    sm = mapp.SpatialMap(box_size=BOX_SIZE, bin_size=BIN_SIZE, smoothing=SMOOTHING)
    rate_map = sm.rate_map(x, y, t, spike_times)
    labels = separate_fields_by_laplace(rate_map, threshold=FIELD_THRESHOLD,
                                        minimum_field_area=MIN_FIELD_BINS)
    n_fields = int(labels.max())
    peak_rate = float(np.nanmax(rate_map))

    # raw (unsmoothed) counts/time -> true Hz
    raw = mapp.SpatialMap(box_size=BOX_SIZE, bin_size=BIN_SIZE, smoothing=0)
    spk = raw.spike_map(x, y, t, spike_times, mask_zero_occupancy=False)
    occ = raw.occupancy_map(x, y, t, mask_zero_occupancy=False)

    visited = occ > 0
    infield = (labels == 1) if USE_PRIMARY_ONLY else (labels >= 1)
    outfield = visited & (~infield)

    out = dict(in_field_rate=np.nan, out_field_rate=np.nan, in_out_ratio=np.nan,
               log_in_out_ratio=np.nan, field_peak_rate=peak_rate,
               n_fields_laplace=n_fields, field_fraction=0.0)
    if infield.sum() == 0 or occ[infield].sum() == 0:
        return out

    in_rate = spk[infield].sum() / occ[infield].sum()
    out_rate = (spk[outfield].sum() / occ[outfield].sum()
                if occ[outfield].sum() > 0 else np.nan)
    ratio = in_rate / out_rate if (out_rate and out_rate > 0) else np.nan
    out.update(in_field_rate=in_rate, out_field_rate=out_rate, in_out_ratio=ratio,
               log_in_out_ratio=(np.log(ratio) if ratio and ratio > 0 else np.nan),
               field_fraction=infield.sum() / visited.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-in", default=TABLE_IN)
    ap.add_argument("--table-out", default=TABLE_OUT)
    ap.add_argument("--nwb-dir", default=NWB_DIR)
    ap.add_argument("--overwrite", action="store_true",
                    help="write back to --table-in instead of the *_infield.pkl copy")
    args = ap.parse_args()
    out_path = args.table_in if args.overwrite else args.table_out

    df = pd.read_pickle(args.table_in)
    print(f"loaded {len(df)} cells, {df['session_id'].nunique()} sessions")

    # init new columns
    for c in NEW_COLS:
        df[c] = np.nan

    sessions = list(pd.unique(df["session_id"]))
    n_ok, n_skip_session, n_skip_cell = 0, 0, 0

    for si, sid in enumerate(sessions, 1):
        nwb_path = os.path.join(args.nwb_dir, str(sid))
        try:
            npdata = nap.load_file(nwb_path)
            pos = load_speed_fromNWB(npdata[POS_KEY])
            _, comb, mask, *_ = pos2speed(pos[:, 0], pos[:, 1], pos[:, 2],
                                          filter_speed=True, min_speed=MIN_SPEED)
        except Exception as e:
            n_skip_session += 1
            print(f"[{si}/{len(sessions)}] SKIP session {sid}: {e}")
            continue

        idxs = df.index[df["session_id"] == sid]
        for idx in idxs:
            try:
                st = Speed_filtered_spikes(np.asarray(df.at[idx, "spike_times"]),
                                           pos[:, 0], mask)
                res = infield_outfield_rates(comb[:, 1], comb[:, 2], comb[:, 0], st)
                for c in NEW_COLS:
                    df.at[idx, c] = res[c]
                n_ok += 1
            except Exception as e:
                n_skip_cell += 1
                print(f"    cell idx={idx} failed: {e}")
        print(f"[{si}/{len(sessions)}] {sid}: {len(idxs)} cells done")

    df.to_pickle(out_path)
    print(f"\nDONE. ok={n_ok}  session_skipped={n_skip_session}  cell_failed={n_skip_cell}")
    print(f"saved -> {out_path}")
    print(df[NEW_COLS].describe())


if __name__ == "__main__":
    main()
