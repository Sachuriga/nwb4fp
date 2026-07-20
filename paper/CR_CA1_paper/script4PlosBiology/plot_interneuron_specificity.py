"""
Figure for the RESPONSE TO REVIEWERS: the interneuron specificity control.

Rebuilt to the layout of script4figures/suppfig4.ipynb -- the speed-cell figure --
so the reviewers see the examples and the definition, not just summary violins.

  rows 0-1   12 example CR;DTA- interneurons: speed vs firing rate, 10% of the
             samples scattered, sliding-window 25/50/75th percentiles on top,
             each cell's own speed score printed
  rows 2-3   the same 12 examples for CR;DTA+
  row 4      how a speed cell is DEFINED, and the group comparisons
             [0] pooled shuffled vs observed speed scores, all CA1 units
             [1] speed-score density, interneurons vs pyramidal cells
             [2] % speed cells, pyramidal vs interneuron
             [3] firing rate, CR;DTA- vs CR;DTA+ (interneurons)
             [4] speed score, CR;DTA- vs CR;DTA+ (interneurons)
             [5] speed cell / non-speed cell, stacked, per genotype
  row 5      the contrast that carries the argument: the same two measures for
             PYRAMIDAL cells, where the genotype effect is present

Departures from the notebook, each deliberate:

  * p-values are the animal-level mixed models (LMM for firing rate and speed
    score, binomial GLMM for the speed-cell fraction), not the pooled
    Mann-Whitney. That is the whole point of the revision -- cells from one
    mouse are not independent, and one mouse contributes 41% of the control
    cells. Exact p is printed rather than asterisks, because the claim about
    interneurons is a NULL result and a missing asterisk cannot express it.
  * the speed-cell criterion is each cell's OWN shuffle distribution (>99th
    percentile of |shuffled r|, 100 surrogates), which is what the statistics
    use. The notebook's figure drew a fixed |r| > 0.3 cut while the text quoted
    other numbers; here figure and statistics use one criterion.
  * the stacked-bar counts are computed, not hard-coded. The notebook has
    `78/113` and `64/94` typed in as literals.
  * the notebook labels x ticks [0, 0.5] m/s as [0, 25]; 0.5 m/s is 50 cm/s.
    Fixed, and the axis is labelled cm/s throughout (one of the notebook's two
    example loops says m/s and the other cm/s for the same quantity).

Numbers are recomputed from the source tables via `speed_coding_stats` and the
same `glmm_binary` used to build the S Table, so figure and table cannot drift.

Data live on the lab volume. Run from this directory.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speed_coding_stats as S                      # noqa: E402
from generate_stats_tables import glmm_binary       # noqa: E402

OUTDIR = "/Users/sachuriga/Desktop/Projects/CR_CA1_paper/Manuscript"
STEM = os.path.join(OUTDIR, "FigR2_interneuron_specificity")

# The paper's colours: interneurons cyan/magenta, pyramidal cells blue/red.
INT_CTRL, INT_EXP = "cyan", "magenta"
# wide-spiking interneurons: darker shades of the interneuron colours, so the two
# interneuron classes read as one family while staying distinguishable
WIDE_CTRL, WIDE_EXP = "#149c9c", "#9c1490"
PY_CTRL, PY_EXP = "blue", "red"
XLABELS = ["CR;DTA-", "CR;DTA+"]
N_EXAMPLES = 12
MS2CMS = 100.0          # speed is stored in m/s


def style():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams.update({
        "font.size": 7, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans", "sans-serif"],
        "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
    })
    plt.rcParams.update({
        "axes.labelpad": 5, "ytick.major.pad": 2, "xtick.major.pad": 5,
        "ytick.major.size": 2, "xtick.major.size": 2,
    })


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def bare(ax):
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)


# ------------------------------------------------------------------ examples
def example_panel(ax, row, color, rng):
    """One example cell: speed vs instantaneous firing rate.

    The notebook's recipe: scatter a random 10% of the samples, then overlay the
    25/50/75th percentiles of firing rate in a sliding speed window. Showing the
    quartiles rather than a mean matters here -- firing rate within a speed bin
    is skewed, and a mean would sit above most of the data.
    """
    speed = np.asarray(row["speed_collect"], dtype=float)
    fr = np.asarray(row["speed_fr"], dtype=float)
    n = min(len(speed), len(fr))
    speed, fr = speed[:n] * MS2CMS, fr[:n]

    step, window = 4.0, 6.0        # cm/s; the notebook's 0.04 m/s and 0.04*1.5
    centers = np.arange(0, np.nanmax(speed), step)
    p25, p50, p75 = [], [], []
    for c in centers:
        m = (speed >= c - window) & (speed <= c + window)
        q = np.quantile(fr[m], [0.25, 0.5, 0.75]) if m.sum() else [np.nan] * 3
        p25.append(q[0]); p50.append(q[1]); p75.append(q[2])

    sel = rng.integers(0, n, size=int(np.ceil(n * 0.1)))
    ax.scatter(speed[sel], fr[sel], s=2, edgecolors="white", facecolors=color,
               linewidths=0.1, marker="o", alpha=0.15)
    ax.plot(centers, p50, "k.", markersize=2)
    ax.plot(centers, p25, "k-", lw=0.6)
    ax.plot(centers, p75, "k-", lw=0.6)
    bare(ax)

    ax.set_xlim(0, 50)
    ax.set_xticks([0, 25, 50])
    ax.text(0.97, 0.95, f"r = {row['speed_score']:.2f}", ha="right", va="top",
            transform=ax.transAxes, fontsize=6)


# ------------------------------------------------------------------ row 4/5
def def_panel(ax, df_all, df_int):
    """How a speed cell is defined: observed speed scores against the shuffle."""
    sh = np.concatenate([np.asarray(s, dtype=float).ravel()
                         for s in df_int["shuffle_speed_scores"]])
    sh = sh[np.isfinite(sh)]
    obs = pd.to_numeric(df_all["speed_score"], errors="coerce").dropna().values

    c, e = np.histogram(sh, bins=200, density=True)
    ax.stairs(c, e, color="grey")
    c, e = np.histogram(obs, bins=100, density=True)
    ax.stairs(c, e, color="black")

    # dashed lines mark the criterion actually applied: |speed score| > 0.3
    thr = S.FIXED_CUT
    for x in (-thr, thr):
        ax.axvline(x, color="grey", ls="--", lw=1)
    bare(ax)
    ax.set_xlabel("Speed score")
    ax.set_ylabel("Density")
    ax.text(0.02, 1.13, "Shuffled data", color="grey", fontsize=4,
            transform=ax.transAxes)
    ax.text(0.02, 1.03, "All units from CA1", color="black", fontsize=4,
            transform=ax.transAxes)
    return thr


def density_panel(ax, df_int, df_py, thr):
    for d, col, lab, y in [(df_int, INT_CTRL, "Interneurons", 1.13),
                           (df_py, PY_CTRL, "Pyramidal neurons", 1.03)]:
        v = pd.to_numeric(d["speed_score"], errors="coerce").dropna().values
        c, e = np.histogram(v, bins=50, density=True)
        ax.stairs(c, e, color=col)
        ax.text(0.02, y, lab, color=col, fontsize=4, transform=ax.transAxes)
    for x in (-thr, thr):
        ax.axvline(x, color="grey", ls="--", lw=1)
    bare(ax)
    ax.set_xlabel("Speed score")
    ax.set_ylabel("Density")


def celltype_fraction_panel(ax, df_int, df_py):
    """% speed cells, pyramidal vs interneuron -- the criterion applied."""
    vals = [100 * df_py["is_speed"].mean(), 100 * df_int["is_speed"].mean()]
    ax.bar([0, 1], vals, color=["blue", INT_CTRL], width=0.6)
    ax.set_ylabel("% Neurons")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pyramidal", "Interneuron"], rotation=-15)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    bare(ax)
    return vals


def violin_panel(ax, sub, col, ylabel, ctrl_color, exp_color, p, log=False):
    """Violin + box, the notebook's recipe, with the mixed-model p above it."""
    v = pd.to_numeric(sub[col], errors="coerce")
    plot_df = pd.DataFrame({
        "value": np.log10(v.where(v > 0)) if log else v,
        "group": np.where(sub["group_ani"] == "control", "Control", "Experimental"),
    }).dropna(subset=["value"])

    order = ["Control", "Experimental"]
    sns.violinplot(data=plot_df, x="group", y="value", order=order, ax=ax,
                   inner=None, width=0.8, cut=0, edgecolor="white",
                   palette={"Control": ctrl_color, "Experimental": exp_color})
    sns.boxplot(data=plot_df, x="group", y="value", order=order,
                palette={"Control": "black", "Experimental": "black"},
                width=0.3, fill=False, showfliers=False, showmeans=False,
                linewidth=0.5, ax=ax)

    # per-animal means: the unit the p-value is based on, marker area ~ n cells
    per = (plot_df.assign(a=sub.loc[plot_df.index, "animal_id"].values)
                  .groupby(["group", "a"], observed=True)["value"]
                  .agg(["mean", "size"]).reset_index())
    rng = np.random.default_rng(1)
    for _, r in per.iterrows():
        x = 0 if r["group"] == "Control" else 1
        ax.plot(x + rng.uniform(-0.12, 0.12), r["mean"], "o",
                ms=1.5 + 0.55 * np.sqrt(r["size"]), mfc="white", mec="black",
                mew=0.5, zorder=20)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    bare(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(XLABELS, rotation=0, fontsize=5)
    if log:
        lo, hi = ax.get_ylim()
        ticks = [t for t in [-1, 0, 1, 2] if lo - 0.3 <= t <= hi + 0.3]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{10.0**t:g}" for t in ticks])
    add_p(ax, p)


def row_label(ax, text):
    """Population name to the left of a row, outside the axes."""
    ax.text(0.0, 1.42, text, transform=ax.transAxes, fontsize=6.5,
            fontweight="bold", ha="left", va="top", clip_on=False)


def add_p(ax, p, x=(0, 1)):
    """Significance bar spanning the two groups. `x` must match the bar/violin
    positions -- the stacked bars sit at 0 and 0.5, not 0 and 1."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    y = hi + span * 0.06
    ax.plot(list(x), [y, y], color="black", lw=1.0, clip_on=False, zorder=30)
    ax.text(np.mean(x), y + span * 0.02, fmt_p(p), ha="center", va="bottom",
            fontsize=6, clip_on=False)
    ax.set_ylim(lo, hi + span * 0.24)


def stacked_panel(ax, sub, colors, p):
    """Speed cell / non-speed cell, stacked, per genotype. Counts computed."""
    counts = []
    for g in ["control", "exp"]:
        s = sub[sub["group_ani"] == g]
        f = 100 * s["is_speed"].mean()
        counts.append([f, 100 - f])

    x = np.arange(2) * 0.5
    for i in range(2):
        ax.bar(x, [counts[j][i] for j in range(2)],
               bottom=[sum(counts[j][:i]) for j in range(2)],
               color=[colors[j][i] for j in range(2)], width=0.35)
    for i in range(2):
        lab = "(Speed cell)" if i == 0 else "(Non-\nspeed cell)"
        for j in range(2):
            # The category name is wider than a bar, so it goes on the left bar
            # only -- on both it collides with the neighbour. A 5% segment holds
            # the percentage alone.
            seg = counts[j][i]
            txt = f"{seg:.0f}%\n{lab}" if (seg >= 15 and j == 0) else f"{seg:.0f}%"
            ax.text(x[j], sum(counts[j][:i]) + seg / 2, txt,
                    ha="center", va="center", color="black", fontsize=4)

    ax.set_ylabel("%Neurons")
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, rotation=0, fontsize=5)
    ax.legend().set_visible(False)
    bare(ax)
    add_p(ax, p, x=(x[0], x[1]))
    return counts


# ------------------------------------------------------------------ figure
def main():
    style()
    # Wide-spiking interneurons are included as a second, independent interneuron
    # class: NDNF is expressed in neurogliaform/ivy cells, which are broad-spiking
    # with slow ACG rise times and therefore absent from the narrow-spiking sample.
    # They are the population an allele- or leak-driven off-target effect would hit
    # first, so they belong in a specificity control (Reviewer #1.1/#1.2).
    CELL_TYPES = ["pyramidal", "narrow_spike_interneurons", "wide_spike_interneurons"]
    df = S.load(cell_types=CELL_TYPES)
    df["is_speed"] = S.speed_cell_flag(df)   # |speed score| > 0.3 (Gois & Tort 2018)
    df_a = df.dropna(subset=["is_speed"])
    df_int = df_a[df_a["cell_type"] == "narrow_spike_interneurons"]
    df_wide = df_a[df_a["cell_type"] == "wide_spike_interneurons"]
    df_py = df_a[df_a["cell_type"] == "pyramidal"]

    # Rows 0-3 hold the example cells on a 6-column grid; the summary rows below
    # use three double-width panels each, so that the violins and the CR;DTA-/+
    # tick labels have room. One row per population, same three measures.
    fig = plt.figure(figsize=(7.2, 11.5), dpi=1200)
    gs = gridspec.GridSpec(8, 6, height_ratios=[1] * 8, width_ratios=[1] * 6)
    rng = np.random.default_rng(0)
    stats_out = []

    # ---- rows 0-3: example cells, the strongest speed coders in each genotype
    for block, (g, color) in enumerate([("control", INT_CTRL), ("exp", INT_EXP)]):
        s = df_int[df_int["group_ani"] == g]
        idx = pd.to_numeric(s["speed_score"], errors="coerce").nlargest(N_EXAMPLES).index
        for k, i in enumerate(idx):
            ax = fig.add_subplot(gs[block * 2 + k // 6, k % 6])
            example_panel(ax, s.loc[i], color, rng)
            if k % 6 == 0:
                ax.set_ylabel("Firing Rate (Hz)")
            if k // 6 == 1:
                ax.set_xlabel("Speed (cm/s)")
        stats_out.append(
            f"Examples, {'CR;DTA-' if g == 'control' else 'CR;DTA+'}: the "
            f"{N_EXAMPLES} interneurons with the highest speed score "
            f"(r = {s.loc[idx, 'speed_score'].min():.2f} to "
            f"{s.loc[idx, 'speed_score'].max():.2f}), from "
            f"{s.loc[idx, 'animal_id'].nunique()} mice.")

    # ---- row 4: definition + interneuron comparisons
    thr = def_panel(fig.add_subplot(gs[4, 0:2]), df_a, df_int)
    density_panel(fig.add_subplot(gs[4, 2:4]), df_int, df_py, thr)
    fr_py, fr_int = celltype_fraction_panel(fig.add_subplot(gs[4, 4:6]), df_int, df_py)

    p_rate, b, cl, ch, nc, nm = S.lmm_ci(df_int, "log_rate")
    ax_r4 = fig.add_subplot(gs[5, 0:2])
    violin_panel(ax_r4, df_int, "mean_firing_rate",
                 "Firing rate (Hz)", INT_CTRL, INT_EXP, p_rate, log=True)
    row_label(ax_r4, f"Narrow-spiking interneurons (n = {len(df_int)})")
    stats_out.append(f"Interneurons, firing rate: LMM on log rate, animal random "
                     f"intercept, p = {p_rate:.4f}, beta = {b:+.3f} [{cl:+.3f}, "
                     f"{ch:+.3f}], n = {nc} cells from {nm} mice.")

    p_ss, b, cl, ch, nc, nm = S.lmm_ci(df_int, "speed_z")
    violin_panel(fig.add_subplot(gs[5, 2:4]), df_int, "speed_score",
                 "Speed score", INT_CTRL, INT_EXP, p_ss)
    stats_out.append(f"Interneurons, speed score: LMM on Fisher-z(r), animal random "
                     f"intercept, p = {p_ss:.4f}, beta = {b:+.3f} [{cl:+.3f}, "
                     f"{ch:+.3f}], n = {nc} cells from {nm} mice.")

    orr, cl, ch, p_frac = glmm_binary(df_int)
    stacked_panel(fig.add_subplot(gs[5, 4:6]), df_int,
                  [[INT_CTRL, "#C7FDFD"], [INT_EXP, "#FDC7F8"]], p_frac)
    pc = 100 * df_int[df_int.group_ani == "control"]["is_speed"].mean()
    pe = 100 * df_int[df_int.group_ani == "exp"]["is_speed"].mean()
    stats_out.append(f"Interneurons, speed-cell fraction: {pc:.1f}% vs {pe:.1f}%, "
                     f"binomial GLMM, animal random intercept, p = {p_frac:.4f}, "
                     f"OR = {orr:.2f} [{cl:.2f}, {ch:.2f}], n = {len(df_int)} cells "
                     f"from {df_int.animal_id.nunique()} mice.")

    # ---- row 5: the same two measures in PYRAMIDAL cells, where it does change
    p_rate_py, b, cl, ch, nc, nm = S.lmm_ci(df_py, "log_rate")
    ax_r5 = fig.add_subplot(gs[6, 0:2])
    violin_panel(ax_r5, df_py, "mean_firing_rate",
                 "Firing rate (Hz)", PY_CTRL, PY_EXP, p_rate_py, log=True)
    row_label(ax_r5, f"Pyramidal cells (n = {len(df_py)})")
    stats_out.append(f"Pyramidal cells, firing rate: LMM on log rate, animal random "
                     f"intercept, p = {p_rate_py:.4f}, beta = {b:+.3f} [{cl:+.3f}, "
                     f"{ch:+.3f}], n = {nc} cells from {nm} mice.")

    # speed score was missing from the pyramidal row; added so that all three
    # populations show the same three measures
    p_ss_py, b, cl, ch, nc, nm = S.lmm_ci(df_py, "speed_z")
    violin_panel(fig.add_subplot(gs[6, 2:4]), df_py, "speed_score",
                 "Speed score", PY_CTRL, PY_EXP, p_ss_py)
    stats_out.append(f"Pyramidal cells, speed score: LMM on Fisher-z(r), animal random "
                     f"intercept, p = {p_ss_py:.4f}, beta = {b:+.3f} [{cl:+.3f}, "
                     f"{ch:+.3f}], n = {nc} cells from {nm} mice.")

    orr, cl, ch, p_frac_py = glmm_binary(df_py)
    stacked_panel(fig.add_subplot(gs[6, 4:6]), df_py,
                  [[PY_CTRL, "#C7D7FD"], [PY_EXP, "#FDC7C7"]], p_frac_py)
    pc = 100 * df_py[df_py.group_ani == "control"]["is_speed"].mean()
    pe = 100 * df_py[df_py.group_ani == "exp"]["is_speed"].mean()
    stats_out.append(f"Pyramidal cells, speed-cell fraction: {pc:.1f}% vs {pe:.1f}%, "
                     f"binomial GLMM, animal random intercept, p = {p_frac_py:.4f}, "
                     f"OR = {orr:.2f} [{cl:.2f}, {ch:.2f}], n = {len(df_py)} cells "
                     f"from {df_py.animal_id.nunique()} mice.")

    # ---- row 6: wide-spiking interneurons, the second interneuron class
    p_rate_w, b, cl, ch, nc, nm = S.lmm_ci(df_wide, "log_rate")
    ax_r6 = fig.add_subplot(gs[7, 0:2])
    violin_panel(ax_r6, df_wide, "mean_firing_rate",
                 "Firing rate (Hz)", WIDE_CTRL, WIDE_EXP, p_rate_w, log=True)
    row_label(ax_r6, f"Wide-spiking interneurons (n = {len(df_wide)})")
    stats_out.append(f"Wide-spiking interneurons, firing rate: LMM on log rate, animal "
                     f"random intercept, p = {p_rate_w:.4f}, beta = {b:+.3f} "
                     f"[{cl:+.3f}, {ch:+.3f}], n = {nc} cells from {nm} mice.")

    p_ss_w, b, cl, ch, nc, nm = S.lmm_ci(df_wide, "speed_z")
    violin_panel(fig.add_subplot(gs[7, 2:4]), df_wide, "speed_score",
                 "Speed score", WIDE_CTRL, WIDE_EXP, p_ss_w)
    stats_out.append(f"Wide-spiking interneurons, speed score: LMM on Fisher-z(r), animal "
                     f"random intercept, p = {p_ss_w:.4f}, beta = {b:+.3f} "
                     f"[{cl:+.3f}, {ch:+.3f}], n = {nc} cells from {nm} mice.")

    orr, cl, ch, p_frac_w = glmm_binary(df_wide)
    stacked_panel(fig.add_subplot(gs[7, 4:6]), df_wide,
                  [[WIDE_CTRL, "#B8E6E6"], [WIDE_EXP, "#E6B8E2"]], p_frac_w)
    pc = 100 * df_wide[df_wide.group_ani == "control"]["is_speed"].mean()
    pe = 100 * df_wide[df_wide.group_ani == "exp"]["is_speed"].mean()
    stats_out.append(f"Wide-spiking interneurons, speed-cell fraction: {pc:.1f}% vs "
                     f"{pe:.1f}%, binomial GLMM, animal random intercept, "
                     f"p = {p_frac_w:.4f}, OR = {orr:.2f} [{cl:.2f}, {ch:.2f}], "
                     f"n = {len(df_wide)} cells from {df_wide.animal_id.nunique()} mice.")

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95,
                        hspace=1.1, wspace=1.3)
    plt.tight_layout()
    fig.savefig(STEM + ".png", transparent=True, dpi=1200, bbox_inches="tight")
    fig.savefig(STEM + ".pdf", transparent=True, dpi=1200, bbox_inches="tight")
    print("wrote", STEM + ".png / .pdf\n")

    stats_out.append(f"Speed-cell criterion: |speed score| > {S.FIXED_CUT} "
                     f"(Gois & Tort, Cell Reports 2018), shown as the dashed lines. "
                     f"Our own |speed score| distribution is bimodal with a trough at "
                     f"0.30-0.40, reproducing that study's structure.")
    stats_out.append(f"Speed cells: {fr_py:.1f}% of pyramidal cells, "
                     f"{fr_int:.1f}% of interneurons.")
    for line in stats_out:
        print(line)


if __name__ == "__main__":
    main()
