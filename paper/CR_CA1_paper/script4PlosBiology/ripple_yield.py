"""
How many sharp-wave ripples are actually available in these recordings?

Context: Reviewer #2 asks whether CR ablation alters ripple activity and deep/superficial
participation in ripples. Our sessions are continuous open-field foraging, so animals are
immobile for only a small fraction of the session, and SWRs occur predominantly during
immobility. This script quantifies the resulting event yield so the limitation can be
stated with numbers rather than asserted.

Detector: `detect_swrs` is taken unchanged from
`Local_field_potential/Ripple_detections_non_move.ipynb` (70-250 Hz band, Hilbert envelope,
Gaussian smoothing, duration limits), with the amplitude threshold set to 3 SD.

Channel choice: the pyramidal layer is identified as the channel with the largest
ripple-band power, which is the same criterion the manuscript uses to locate the centre of
the CA1 pyramidal layer.

Events are then classified by whether their peak falls in an immobile period, since only
those are usable for a ripple analysis.

Data live on the lab volume, not in the repo.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks, hilbert

REC_ROOT = "/Volumes/quattrocolo/crhip/Sachuriga/Ephys_Recording/CR_CA1"
LFP_FS = 1250.0
IMMOBILE_CMS = 2.0          # immobility criterion for SWR analysis (cm/s)
AMP_SD = 3                  # amplitude threshold, in SD of the smoothed envelope


def detect_swrs(lfp_data, electrode_idx, sampling_rate, ripple_band=(70, 250),
                amplitude_threshold=AMP_SD, min_duration=0.05, max_duration=2.0,
                gaussian_std_ms=12.5):
    """Detect SWRs on one electrode. Unchanged from the lab notebook except the
    default amplitude threshold. Returns list of (start_idx, end_idx, peak_idx)."""
    lfp = lfp_data[:, electrode_idx] if lfp_data.ndim == 2 else lfp_data

    nyquist = sampling_rate / 2
    lowcut, highcut = ripple_band
    b, a = butter(4, [lowcut / nyquist, highcut / nyquist], btype="band")
    filtered_lfp = filtfilt(b, a, lfp)

    amplitude_envelope = np.abs(hilbert(filtered_lfp))
    gaussian_std_samples = gaussian_std_ms * sampling_rate / 1000
    smoothed_envelope = gaussian_filter1d(amplitude_envelope, sigma=gaussian_std_samples)

    mean_envelope = np.mean(smoothed_envelope)
    std_envelope = np.std(smoothed_envelope)
    threshold = mean_envelope + amplitude_threshold * std_envelope

    peak_indices = find_peaks(smoothed_envelope, height=threshold)[0]

    swr_events = []
    min_samples = int(min_duration * sampling_rate)
    max_samples = int(max_duration * sampling_rate)
    n = len(smoothed_envelope)
    for peak_idx in peak_indices:
        start_idx = peak_idx
        while start_idx > 0 and smoothed_envelope[start_idx] > mean_envelope:
            start_idx -= 1
        end_idx = peak_idx
        while end_idx < n - 1 and smoothed_envelope[end_idx] > mean_envelope:
            end_idx += 1
        if min_samples <= (end_idx - start_idx) <= max_samples:
            swr_events.append((start_idx, end_idx, peak_idx))
    return swr_events


def pick_ripple_channel(lfp_mm, fs=LFP_FS, probe_s=180):
    """Channel with the largest ripple-band power -- the pyramidal-layer criterion."""
    n = lfp_mm.shape[0]
    lo = max(0, n // 2 - int(probe_s * fs / 2))
    chunk = np.asarray(lfp_mm[lo:lo + int(probe_s * fs), :], dtype=np.float64)
    b, a = butter(4, [70 / (fs / 2), 250 / (fs / 2)], btype="band")
    power = [np.var(filtfilt(b, a, chunk[:, c])) for c in range(chunk.shape[1])]
    return int(np.argmax(power)), np.asarray(power)


def session_speed(sess_id, speed_df):
    """Speed trace (cm/s) + its timestamps for one session, from the units table."""
    g = speed_df[speed_df.session_id == sess_id]
    if not len(g):
        return None, None
    r = g.iloc[0]
    v = np.asarray(r["speed_collect"], dtype=float) * 100.0   # m/s -> cm/s
    t = np.asarray(r["speed_t"], dtype=float)
    return v, t


def _envelope(sig, fs=LFP_FS, ripple_band=(70, 250), gaussian_std_ms=12.5):
    """Ripple-band amplitude envelope, smoothed. Filtering is done on the FULL trace so
    there are no filter edge artefacts from cutting the signal into epochs."""
    b, a = butter(4, [ripple_band[0] / (fs / 2), ripple_band[1] / (fs / 2)], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, sig)))
    return gaussian_filter1d(env, sigma=gaussian_std_ms * fs / 1000)


def _events_from_envelope(env, mask, mean_e, std_e, fs=LFP_FS,
                          amplitude_threshold=AMP_SD, min_duration=0.05, max_duration=2.0):
    """Peaks above mean+k*SD whose peak sample lies inside `mask`."""
    thr = mean_e + amplitude_threshold * std_e
    peaks = find_peaks(env, height=thr)[0]
    lo, hi = int(min_duration * fs), int(max_duration * fs)
    n = len(env)
    out = []
    for p in peaks:
        if not mask[p]:
            continue
        s = p
        while s > 0 and env[s] > mean_e:
            s -= 1
        e = p
        while e < n - 1 and env[e] > mean_e:
            e += 1
        if lo <= (e - s) <= hi:
            out.append((s, e, p))
    return out


def analyse_session(folder, sess_id, speed_df, verbose=True):
    zp = os.path.join(folder, "lfp_zscore.npy")
    tp = os.path.join(folder, "lfp_times.npy")
    if not (os.path.exists(zp) and os.path.exists(tp)):
        return None
    t0 = time.time()
    lfp = np.load(zp, mmap_mode="r")
    lt = np.load(tp)
    ch, _ = pick_ripple_channel(lfp)
    sig = np.asarray(lfp[:, ch], dtype=np.float64)
    env = _envelope(sig)

    v, vt = session_speed(sess_id, speed_df)
    dur = len(lt) / LFP_FS
    out = dict(session=sess_id, animal=sess_id.split("_")[0], channel=ch, dur_s=round(dur, 1))

    # (a) threshold estimated over the WHOLE session (running-dominated)
    n_all = _events_from_envelope(env, np.ones(len(env), bool), env.mean(), env.std())
    out["n_events_session_thr"] = len(n_all)

    if v is not None:
        # immobility mask on the LFP clock
        sp_lfp = np.interp(lt, vt, v, left=np.nan, right=np.nan)
        imm = np.nan_to_num(sp_lfp, nan=1e9) < IMMOBILE_CMS
        out["immobile_s"] = round(float(imm.sum() / LFP_FS), 1)
        out["immobile_pct"] = round(100 * imm.mean(), 2)

        # events found with the session threshold that fall in immobility
        out["n_session_thr_in_immobility"] = int(sum(imm[e[2]] for e in n_all))

        # (b) CORRECT: threshold estimated from immobility periods only
        if imm.sum() > LFP_FS:            # need >1 s of immobility to estimate SD
            m_i, s_i = env[imm].mean(), env[imm].std()
            ev_i = _events_from_envelope(env, imm, m_i, s_i)
            out["n_events_immobility_thr"] = len(ev_i)
            out["rate_hz_immobility"] = round(len(ev_i) / (imm.sum() / LFP_FS), 3)
        else:
            out["n_events_immobility_thr"] = np.nan
            out["rate_hz_immobility"] = np.nan
    out["sec"] = round(time.time() - t0, 1)

    if verbose:
        print(f"  {sess_id[:30]:30s} ch={ch:2d} dur={dur:5.0f}s "
              f"immob={out.get('immobile_s', np.nan):6.1f}s ({out.get('immobile_pct', np.nan):5.2f}%) "
              f"| sessThr={out['n_events_session_thr']:4d} "
              f"(in-immob {out.get('n_session_thr_in_immobility', np.nan)}) "
              f"| immobThr={out.get('n_events_immobility_thr', np.nan)} "
              f"({out.get('rate_hz_immobility', np.nan)} Hz)  [{out['sec']}s]")
    return out


def main(limit=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import speed_coding_stats as S
    units = S.load()

    rows = []
    sess_ids = sorted(units.session_id.unique())
    if limit:
        sess_ids = sess_ids[:limit]
    print(f"analysing {len(sess_ids)} sessions (threshold = {AMP_SD} SD, "
          f"immobility < {IMMOBILE_CMS} cm/s)\n")
    for sid in sess_ids:
        folder = os.path.join(REC_ROOT, sid.split("_")[0], sid.replace(".nwb", ""))
        try:
            r = analyse_session(folder, sid, units)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  {sid[:34]:34s} FAILED: {type(e).__name__}: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    d = main(lim)
    if len(d):
        d.to_csv("/tmp/ripple_yield.csv", index=False)
        print(f"\n=== {len(d)} sessions, {d.animal.nunique()} animals ===")
        print(f"immobility per session      : median {d.immobile_s.median():.1f} s "
              f"({d.immobile_pct.median():.2f}% of session)")
        print(f"SWRs with immobility-derived threshold, occurring during immobility:")
        print(f"    median {d.n_events_immobility_thr.median():.0f} events/session  "
              f"range {d.n_events_immobility_thr.min():.0f}-{d.n_events_immobility_thr.max():.0f}")
        print(f"    rate  median {d.rate_hz_immobility.median():.3f} Hz")
        print(f"(for reference, session-wide threshold flags "
              f"{d.n_events_session_thr.median():.0f} events/session, of which "
              f"{d.n_session_thr_in_immobility.median():.0f} fall in immobility)")
        print("\nwrote /tmp/ripple_yield.csv")
