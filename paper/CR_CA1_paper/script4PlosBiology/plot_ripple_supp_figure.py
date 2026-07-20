"""
Figure for the RESPONSE TO REVIEWERS only (Reviewer #2.3) -- not for publication,
not part of the Supplementary Material.

It makes one point: ripples ARE detectable and are not a spike artefact, but the
event yield in this continuous-foraging paradigm is far too low for the per-cell
deep/superficial comparison the reviewer asks for.

Panels (no text inside the figure; write the legend separately)
  A  all raw wideband traces of the shank at the strongest event, +/-250 ms
  B  ripple amplitude vs depth (all sites of the shank, single line)
  C  spike-contamination control: envelope-vs-MUA correlation by band
The channel with the largest ripple amplitude is highlighted in red.

Data live on the lab volume. Run from this directory.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType -> text stays editable in Illustrator
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ripple_yield as R                       # noqa: E402
import speed_coding_stats as S                 # noqa: E402

FS = R.LFP_FS
SR = 30000.0
PRB = "/Users/sachuriga/Desktop/code/nwb4fp/src/nwb4fp/data/ASSY-236-F.prb"
OUT = "/Users/sachuriga/Desktop/Projects/CR_CA1_paper/Manuscript/FigR1_ripple_validation.png"
OUT_PDF = OUT.replace(".png", ".pdf")
OUT_SVG = OUT.replace(".png", ".svg")


def pick_session(units):
    """Session with the most pyramidal cells, and within it the shank with the most.

    Returns (session_id_without_nwb, shank, ripple_ch). The ripple channel is the
    lab's own `ripple_ch_3std` for that shank, not a channel we chose ourselves.
    """
    py = units[units.buzaki_py_cell_type == "pyramidal"]
    sid = py.groupby("session_id").size().sort_values(ascending=False).index[0]
    s = py[py.session_id == sid]
    sh = s.groupby("shank_group").size().sort_values(ascending=False).index[0]
    rch = int(s[s.shank_group == sh].ripple_ch_3std.iloc[0])
    print(f"session {sid} ({len(s)} pyramidal cells); "
          f"shank {sh} has the most ({(s.shank_group == sh).sum()}), ripple_ch_3std = {rch}")
    return sid.replace(".nwb", ""), int(sh), rch


def envelope(sig, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return gaussian_filter1d(np.abs(hilbert(filtfilt(b, a, sig))), sigma=12.5 * FS / 1000)


def bandpass(sig, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, sig)


def main():
    lv = {"np": np}
    exec(open(PRB).read(), lv)
    CG = lv["channel_groups"]

    units = S.load(cell_types=None)
    SESSION, SHANK, RCH = pick_session(units)
    folder = os.path.join(R.REC_ROOT, SESSION.split("_")[0], SESSION)
    lfp = np.load(f"{folder}/lfp_zscore.npy", mmap_mode="r")
    lt = np.load(f"{folder}/lfp_times.npy")
    bad = set(np.load(f"{folder}/bad_channels.npy", allow_pickle=True).tolist())
    spikes = np.load(f"{folder}/spike_times.npy").ravel() / SR + lt[0]

    sid = SESSION + ".nwb"
    v, vt = R.session_speed(sid, units)
    sp = np.interp(lt, vt, v, left=np.nan, right=np.nan)
    imm = np.nan_to_num(sp, nan=1e9) < R.IMMOBILE_CMS

    # event detection on the lab's ripple channel (used only to find the event)
    det = np.asarray(lfp[:, RCH], dtype=float)
    env_det = envelope(det, 130, 200)
    m_i, s_i = env_det[imm].mean(), env_det[imm].std()
    events = R._events_from_envelope(env_det, imm, m_i, s_i)
    peak = max(events, key=lambda e: env_det[e[2]])[2]

    half = int(0.25 * FS)
    t = (np.arange(-half, half) / FS) * 1000
    geo = CG[SHANK]["geometry"]
    chs = [c for c in CG[SHANK]["channels"] if c not in bad and c < lfp.shape[1]]
    chs.sort(key=lambda c: geo[c][1])
    ys = [geo[c][1] for c in chs]
    xs = [geo[c][0] for c in chs]

    seg = np.asarray(lfp[peak - half:peak + half, :], dtype=float)[:, chs]
    w = int(0.025 * FS)
    amp = np.array([envelope(seg[:, j], 130, 200)[half - w:half + w].max()
                    for j in range(seg.shape[1])])
    jmax = int(np.argmax(amp))                      # channel with the largest ripple amplitude
    print(f"{len(events)} events; strongest @ {peak/FS:.1f}s; "
          f"max-amplitude channel = {chs[jmax]} at {ys[jmax]:.0f} um")

    mua = gaussian_filter1d(
        np.histogram(spikes, bins=np.concatenate([lt, [lt[-1] + 1 / FS]]))[0].astype(float),
        sigma=12.5 * FS / 1000)

    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.8, 0.9], wspace=0.40)

    # ---- A: every raw trace on the shank
    axA = fig.add_subplot(gs[0, 0])
    step = np.nanmax(np.abs(seg)) * 1.5
    for j in range(seg.shape[1]):
        axA.plot(t, seg[:, j] - j * step,
                 color="crimson" if j == jmax else "0.35",
                 lw=0.9 if j == jmax else 0.6, zorder=3 if j == jmax else 2)
    axA.set_xlim(-250, 250)
    axA.set_yticks([-j * step for j in range(seg.shape[1])])
    axA.set_yticklabels([f"{int(y)}" for y in ys], fontsize=7)
    axA.set_xlabel("time from event peak (ms)")
    axA.set_ylabel("depth on shank (\u00b5m)", labelpad=2)
    axA.set_title("A", fontsize=11, loc="left", weight="bold")

    # ---- B: depth profile
    axB = fig.add_subplot(gs[0, 1])
    # single line: all sites of the shank ordered by depth (chs is already sorted by y)
    axB.plot(amp, ys, "o-", ms=4, lw=1.2, color="0.25")
    axB.plot([amp[jmax]], [ys[jmax]], "o", color="crimson", ms=10, zorder=5)
    axB.invert_yaxis()
    axB.set_xlabel("ripple amplitude"); axB.set_ylabel("depth on shank (\u00b5m)")
    axB.set_title("B", fontsize=11, loc="left", weight="bold")

    # ---- C: spike-contamination control
    axC = fig.add_subplot(gs[0, 2])
    sig = np.asarray(lfp[:, chs[jmax]], dtype=float)
    bands = [("4-12", 4, 12), ("20-40", 20, 40), ("60-100", 60, 100),
             ("130-200", 130, 200), ("250-450", 250, 450)]
    rs = [pearsonr(envelope(sig, lo, hi)[imm], mua[imm])[0] for _, lo, hi in bands]
    cols = ["crimson" if n == "130-200" else ("0.45" if n == "250-450" else "0.78")
            for n, _, _ in bands]
    axC.bar(range(len(bands)), rs, color=cols)
    axC.set_xticks(range(len(bands)))
    axC.set_xticklabels([b[0] for b in bands], fontsize=7, rotation=30)
    axC.set_xlabel("frequency band (Hz)")
    axC.set_ylabel("r (envelope vs MUA)")
    axC.axhline(0, color="k", lw=0.8)
    axC.set_title("C", fontsize=11, loc="left", weight="bold")

    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")          # vector, editable lines + text
    fig.savefig(OUT_SVG, bbox_inches="tight")          # in case you prefer Inkscape
    print("wrote", OUT)
    print("wrote", OUT_PDF)
    print("wrote", OUT_SVG)
    print(f"panel C r: " + ", ".join(f"{n}={r:.3f}" for (n, _, _), r in zip(bands, rs)))


if __name__ == "__main__":
    main()
