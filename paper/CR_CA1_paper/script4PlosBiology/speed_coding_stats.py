"""
Mixed-effects statistics for speed coding and functional properties (Supp Fig 4).

Why this exists
---------------
The Supp-Fig-4 notebooks pool every neuron and run a Mann-Whitney U between
genotypes. Cells from one mouse are not independent observations, and here the
cell counts are badly unbalanced -- in session A one control mouse contributes
154 of 378 control cells (41%). A pooled test is therefore weighted by how many
cells each mouse happened to yield, not by how many mice were studied. These
models make the **animal** the unit of inference.

Models
------
speed score (a correlation, bounded -1..1)
    atanh(r) ~ genotype [* cell_type] + (1 | animal)          LMM
firing rate (right-skewed, approximately lognormal)
    log(rate) ~ genotype [* cell_type] + (1 | animal)         LMM
speed-cell proportion (binary per cell)
    is_speed ~ genotype, binomial GEE, clustered on animal

Fisher's z (``atanh``) is used for the speed score because it is a correlation:
it is bounded and its variance depends on the value, both of which break the
LMM's assumptions. Firing rate is logged because brain firing rates are
lognormal -- the mean of a raw rate is dominated by a fast-firing minority and
misrepresents the slow majority (Buzsaki & Mizuseki 2014, Nat Rev Neurosci).

Speed-cell criterion
--------------------
By default each cell is tested against its OWN shuffle distribution
(``shuffle_speed_scores``, n = 100 surrogates) at ``SHUFFLE_PCT``. The fixed
|r| > 0.3 cut used in the figures is also available via ``method="fixed"``.
Thresholds are conventions, not constants -- report whichever you actually use,
and prefer the shuffle-based one because it adapts to each cell's own noise.

Caveat worth stating in the Methods
-----------------------------------
With 5 mice per group the random-effect variance can sit near the boundary;
report the number of mice and the cells contributed by each (see design_summary).

The data tables are NOT in the repo; edit SPEED_TABLE to point to your copy.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Reuse the shared helpers rather than redefining them.
from mixed_effects_stats import CONTROL_IDS, genotype

warnings.filterwarnings("ignore")  # silence LMM boundary/convergence chatter

# NB: a different table from the one Fig 4 uses. The `_pycirc_infield` table
# carries in/out-of-field columns but drops the speed columns; only this one has
# `speed_score` + `shuffle_speed_scores`.
SPEED_TABLE = ("/Users/sachuriga/Desktop/Projects/CR_CA1_paper/tables/"
               "functional_properties_with_python_measurements.pkl")

SHUFFLE_PCT = 99      # percentile of a cell's own shuffle distribution
FIXED_CUT = 0.3       # the |r| cut used in the original figures
SESSION = "A"
CELL_TYPES = ["pyramidal", "narrow_spike_interneurons"]


# ------------------------------------------------------------------ loading
def load(session=SESSION, cell_types=CELL_TYPES, table=SPEED_TABLE):
    """Load good units for one session, annotated with genotype and transforms."""
    df = pd.read_pickle(table)
    df = df[df["unit_quality"] == "good"].copy()
    if session is not None:
        df = df[df["session"] == session]
    if cell_types is not None:
        df = df[df["buzaki_py_cell_type"].isin(cell_types)]

    df["animal_id"] = df["animal_id"].astype(str)
    df["group_ani"] = pd.Categorical(genotype(df["animal_id"]), categories=["control", "exp"])
    df["cell_type"] = pd.Categorical(df["buzaki_py_cell_type"], categories=cell_types)

    r = pd.to_numeric(df["speed_score"], errors="coerce")
    # clip strictly inside (-1, 1) so atanh stays finite
    df["speed_z"] = np.arctanh(r.clip(-0.999999, 0.999999))
    df["log_rate"] = np.log(pd.to_numeric(df["mean_firing_rate"], errors="coerce")
                            .where(lambda s: s > 0))
    return df


def speed_cell_flag(df, method="shuffle", pct=SHUFFLE_PCT, fixed=FIXED_CUT):
    """Flag speed cells, either against each cell's own shuffle null or a fixed cut.

    Returns an int Series (1 = speed cell). Cells lacking a usable shuffle
    distribution get NaN under ``method="shuffle"`` so they drop out of the model
    rather than being silently counted as non-speed cells.
    """
    if method == "fixed":
        return (pd.to_numeric(df["speed_score"], errors="coerce").abs() > fixed).astype(float)

    out = []
    for r, sh in zip(df["speed_score"], df["shuffle_speed_scores"]):
        try:
            s = np.asarray(sh, dtype=float)
            s = s[np.isfinite(s)]
        except (TypeError, ValueError):
            s = np.array([])
        if not np.isfinite(r) or s.size < 20:
            out.append(np.nan)
        else:
            # two-sided: exceed the upper percentile of |shuffled r|
            out.append(float(abs(r) > np.percentile(np.abs(s), pct)))
    return pd.Series(out, index=df.index)


# ------------------------------------------------------------------ models
def lmm_ci(d, dv, formula="y ~ group_ani", term="group_ani[T.exp]"):
    """LMM with animal random intercept. Returns (p, beta, lo, hi, n_cells, n_mice)."""
    d = d.dropna(subset=[dv, "animal_id"]).copy()
    d["y"] = d[dv]
    m = smf.mixedlm(formula, d, groups=d["animal_id"]).fit(reml=True)
    ci = m.conf_int().loc[term]
    return (m.pvalues[term], m.params[term], ci[0], ci[1],
            int(m.model.endog.shape[0]), d["animal_id"].nunique())


def prop_gee(d, flag_col="is_speed"):
    """Binomial GEE for a per-cell binary outcome, clustered on animal.

    MixedLM cannot fit a binomial family; GEE with an exchangeable working
    correlation gives population-averaged genotype effects with animal-robust
    standard errors, which is what we want for "what fraction of cells".
    Returns (p, odds_ratio, lo, hi, n_cells, n_mice).
    """
    d = d.dropna(subset=[flag_col, "animal_id"]).copy()
    d["y"] = d[flag_col].astype(float)
    m = smf.gee("y ~ group_ani", groups="animal_id", data=d,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    t = "group_ani[T.exp]"
    ci = m.conf_int().loc[t]
    return (m.pvalues[t], np.exp(m.params[t]), np.exp(ci[0]), np.exp(ci[1]),
            len(d), d["animal_id"].nunique())


# ------------------------------------------------------------------ reports
def _fdr(rows, pcol=2):
    """Benjamini-Hochberg across a family of tests; appends q to each row."""
    ps = [r[pcol] for r in rows]
    q = multipletests(ps, method="fdr_bh")[1] if ps else []
    return [r + (qi,) for r, qi in zip(rows, q)]


def speed_and_rate(df):
    """Genotype effect on speed score and firing rate, within each cell type."""
    print("\n(1) Genotype effect within cell type   [metric ~ genotype + (1|animal)]")
    print("    speed score: Fisher-z transformed;  firing rate: log transformed")
    hdr = f"{'metric':14s}{'cell type':26s}{'p':>8s}{'q(FDR)':>8s}{'beta [95% CI]':>26s}{'cells':>7s}{'mice':>6s}"
    print(hdr)
    rows = []
    for dv, lab in [("speed_z", "speed score"), ("log_rate", "firing rate")]:
        for ct in CELL_TYPES:
            s = df[df["cell_type"] == ct]
            p, b, lo, hi, nc, nm = lmm_ci(s, dv)
            rows.append((lab, ct, p, b, lo, hi, nc, nm))
    for lab, ct, p, b, lo, hi, nc, nm, q in _fdr(rows, pcol=2):
        print(f"{lab:14s}{ct:26s}{p:8.4f}{q:8.4f}"
              f"{f'{b:+.3f} [{lo:+.3f},{hi:+.3f}]':>26s}{nc:7d}{nm:6d}")

    print("\n(2) Genotype x cell-type INTERACTION   [metric ~ genotype*cell_type + (1|animal)]")
    print("    significant => the genotype effect differs between cell types")
    print(f"{'metric':14s}{'interaction p':>15s}")
    for dv, lab in [("speed_z", "speed score"), ("log_rate", "firing rate")]:
        d = df.dropna(subset=[dv]).copy()
        d["y"] = d[dv]
        m = smf.mixedlm("y ~ group_ani*cell_type", d, groups=d["animal_id"]).fit(reml=True)
        k = [t for t in m.pvalues.index if ":" in t][0]
        print(f"{lab:14s}{m.pvalues[k]:15.4f}")


def speed_cell_proportion(df):
    """Fraction of speed cells per genotype, by both criteria."""
    print("\n(3) Speed-cell proportion   [is_speed ~ genotype, binomial GEE clustered on animal]")
    print(f"{'criterion':22s}{'cell type':26s}{'p(GEE)':>8s}{'OR [95% CI]':>24s}{'%ctrl':>7s}{'%exp':>7s}{'cells':>7s}{'mice':>6s}")
    for method, name in [("shuffle", f"own shuffle >{SHUFFLE_PCT}th"), ("fixed", f"|r| > {FIXED_CUT}")]:
        d = df.copy()
        d["is_speed"] = speed_cell_flag(d, method=method)
        for ct in CELL_TYPES:
            s = d[d["cell_type"] == ct].dropna(subset=["is_speed"])
            if s["is_speed"].nunique() < 2:
                print(f"{name:22s}{ct:26s}{'n/a (no variation in outcome)':>50s}")
                continue
            p, orr, lo, hi, nc, nm = prop_gee(s)
            pc = 100 * s[s.group_ani == "control"]["is_speed"].mean()
            pe = 100 * s[s.group_ani == "exp"]["is_speed"].mean()
            print(f"{name:22s}{ct:26s}{p:8.4f}"
                  f"{f'{orr:.2f} [{lo:.2f},{hi:.2f}]':>24s}{pc:7.1f}{pe:7.1f}{nc:7d}{nm:6d}")


def pooled_mwu(df):
    """Reproduce the ORIGINAL notebook's pooled Mann-Whitney U (unit = cell).

    Every cell is treated as an independent observation, which it is not -- kept
    here only so the pooled p can be compared directly against the animal-level
    models on exactly the same subsets. MWU is rank-based, so the log / Fisher-z
    transforms used by the LMMs do not change these p-values.
    """
    from scipy.stats import mannwhitneyu

    print("\n(4) ORIGINAL pooled Mann-Whitney U  [unit = cell, animals ignored]")
    print("    side by side with the animal-level tests on the same subsets")
    print(f"{'metric':14s}{'cell type':26s}{'p(pooled MWU)':>15s}{'p(LMM)':>9s}"
          f"{'med ctrl':>10s}{'med exp':>9s}{'cells':>7s}{'mice':>6s}")
    for raw, dv, lab in [("speed_score", "speed_z", "speed score"),
                         ("mean_firing_rate", "log_rate", "firing rate")]:
        for ct in CELL_TYPES:
            s = df[df["cell_type"] == ct].copy()
            s[raw] = pd.to_numeric(s[raw], errors="coerce")
            s = s.dropna(subset=[raw])
            c = s[s.group_ani == "control"][raw].values
            e = s[s.group_ani == "exp"][raw].values
            p_pool = mannwhitneyu(c, e, alternative="two-sided")[1]
            p_lmm = lmm_ci(s, dv)[0]
            print(f"{lab:14s}{ct:26s}{p_pool:15.4g}{p_lmm:9.4f}"
                  f"{np.median(c):10.3f}{np.median(e):9.3f}{len(s):7d}{s['animal_id'].nunique():6d}")

    print("\n    Original speed-cell criterion, signed speed_score > "
          f"{FIXED_CUT} (as plotted in the notebook):")
    print(f"{'cell type':26s}{'%ctrl':>8s}{'%exp':>8s}{'p(pooled Fisher)':>18s}")
    from scipy.stats import fisher_exact
    for ct in CELL_TYPES:
        s = df[df["cell_type"] == ct].copy()
        s["is_speed_signed"] = (pd.to_numeric(s["speed_score"], errors="coerce") > FIXED_CUT).astype(float)
        c = s[s.group_ani == "control"]["is_speed_signed"]
        e = s[s.group_ani == "exp"]["is_speed_signed"]
        tbl = [[int(c.sum()), int((1 - c).sum())], [int(e.sum()), int((1 - e).sum())]]
        p_f = fisher_exact(tbl)[1]
        print(f"{ct:26s}{100 * c.mean():8.1f}{100 * e.mean():8.1f}{p_f:18.4g}")


def design_summary(df):
    """Sampling table -- makes the pseudoreplication problem explicit."""
    print("\n(0) Design: cells contributed per animal")
    for g in ["control", "exp"]:
        s = df[df["group_ani"] == g]
        n = s.groupby("animal_id").size().sort_index()
        print(f"  {g:8s}: {s['animal_id'].nunique()} mice, {len(s)} cells -> {dict(n)}")
        if len(n):
            print(f"{'':12s}largest single mouse = {100 * n.max() / n.sum():.0f}% of the group's cells")


if __name__ == "__main__":
    df = load()
    print("=" * 104)
    print(f"SPEED CODING / FUNCTIONAL PROPERTIES (Supp Fig 4) -- session {SESSION}")
    print(f"control mice: {CONTROL_IDS}")
    print("=" * 104)
    design_summary(df)
    speed_and_rate(df)
    speed_cell_proportion(df)
    pooled_mwu(df)
    print("\nReminder: report n mice (not n cells) as the sample size in the text.")
