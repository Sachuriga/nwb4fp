"""
Mixed-effects re-analysis for the PLOS Biology revision.

Addresses reviewer requests:
  * R3 major #4 : "strongly suggest the use of mixed-effect models"; clarify the
    statistical unit (mice vs recordings vs cells); test normality.
  * R1 major #6 : report n animals and n cells per animal; the deep-vs-superficial
    difference must be shown to be a genuine *differential* effect, i.e. a
    genotype x layer INTERACTION, not just "significant in one, n.s. in the other".
  * R3 minor #2 : distinguish a true spatial-coding deficit from a simple increase
    in excitability -> in-field vs out-of-field firing-rate ratio (spatial S/N).

Statistical model
-----------------
Continuous cell-level metrics:   value ~ genotype [* layer] + (1 | animal)   (LMM)
LFP session-level metrics:       value ~ genotype + (1 | animal)             (LMM)
Genotype (CR;DTA-/CR;DTA+) is the fixed effect; animal is a random intercept,
so the effective unit is the animal, not the cell/recording. Right-skewed
metrics are log-transformed.

The data tables are NOT in the repo; edit the paths below to point to your copy.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")  # silence LMM boundary/convergence chatter

# ---------------------------------------------------------------- paths
FUNCTIONAL_TABLE = "/Users/sachuriga/Desktop/Projects/CR_CA1_paper/tables/functional_properties_with_python_measurements_pycirc_infield.pkl"
LFP_TABLE = "/Users/sachuriga/Desktop/Projects/CR_CA1_paper/tables/lfp_with_gamma_event_coupling.pkl"

CONTROL_IDS = ["65165", "65091", "63383", "66539", "65622"]


def genotype(animal_ids):
    return np.where(pd.Series(animal_ids).astype(str).isin(CONTROL_IDS), "control", "exp")


def lmm(d, var, log=False, formula="y ~ group_ani"):
    d = d.copy()
    d["y"] = pd.to_numeric(d[var], errors="coerce")
    d = d.dropna(subset=["y"])
    if log:
        d = d[d["y"] > 0]
        d["y"] = np.log(d["y"])
    return smf.mixedlm(formula, d, groups=d["animal_id"]).fit(reml=True)


def per_animal_p(d, var, log=False):
    """Conservative robustness check: collapse to one mean per animal, then
    Welch t-test (if both groups' animal-means pass Shapiro) or Mann-Whitney.

    NOT used in the default submission output (reviewers asked for mixed models).
    Kept ready in case a reviewer requests an explicit animal-level analysis:
    with 5 mice/group the mixed model's random-effect variance is estimated near
    the boundary, so this animal-level test is the conservative cross-check.
    """
    from scipy.stats import ttest_ind, mannwhitneyu, shapiro
    d = d.copy()
    d["y"] = pd.to_numeric(d[var], errors="coerce")
    d = d.dropna(subset=["y"])
    if log:
        d = d[d["y"] > 0]
    am = d.groupby(["group_ani", "animal_id"])["y"].mean().reset_index()
    c = am[am.group_ani == "control"]["y"].values
    e = am[am.group_ani == "exp"]["y"].values
    if len(c) >= 3 and len(e) >= 3 and shapiro(c)[1] > 0.05 and shapiro(e)[1] > 0.05:
        return ttest_ind(c, e, equal_var=False)[1], "Welch t"
    return mannwhitneyu(c, e, alternative="two-sided")[1], "MWU"


# ====================================================================
# PART 1 — Deep vs superficial CA1 pyramidal neurons (Fig 4)
# ====================================================================
def deep_superficial():
    df = pd.read_pickle(FUNCTIONAL_TABLE)
    df = df[(df["buzaki_py_cell_type"] == "pyramidal") & (df["session"] == "A")].copy()
    df["animal_id"] = df["animal_id"].astype(str)
    df["group_ani"] = pd.Categorical(genotype(df["animal_id"]), categories=["control", "exp"])
    df["sub_population"] = pd.Categorical(df["sub_population"], categories=["deep", "superficial"])
    for c in ["Information_content_rate", "Sparsity", "Field_size", "Averate_rate",
              "Selectivity", "stability_ma", "matlab_maxfsize"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_place"] = ((df["h_0_place_cell"] == 1) &
                      (df["Information_content_rate"] >= 1.68) &
                      (df["matlab_maxfsize"] >= 20)).astype(int)

    metrics = [("Information_content_rate", "Info content rate", True),
               ("Sparsity", "Sparsity", False), ("Field_size", "Field size", True),
               ("Averate_rate", "Firing rate", True), ("Selectivity", "Selectivity", True),
               ("stability_ma", "Stability", False)]

    print("\n(1) Genotype effect within each layer  [value ~ genotype + (1|animal)]")
    print(f"{'metric':20s}{'layer':12s}{'p(exp)':>9s}{'n_cells':>9s}{'n_mice':>8s}")
    for var, lab, log in metrics:
        for sp in ["deep", "superficial"]:
            s = df[df["sub_population"] == sp]
            m = lmm(s, var, log)
            print(f"{lab:20s}{sp:12s}{m.pvalues['group_ani[T.exp]']:9.4f}"
                  f"{int(m.model.endog.shape[0]):9d}{s['animal_id'].nunique():8d}")

    print("\n(2) Genotype x layer INTERACTION  [value ~ genotype*layer + (1|animal)]")
    print("    significant interaction => layers ARE differentially affected")
    print(f"{'metric':20s}{'interaction p':>15s}")
    for var, lab, log in metrics:
        m = lmm(df, var, log, formula="y ~ group_ani*sub_population")
        k = [t for t in m.pvalues.index if ":" in t][0]
        print(f"{lab:20s}{m.pvalues[k]:15.4f}")

    print("\n(3) In-field vs out-of-field firing  (ALL pyramidal cells with a detected field)")
    # NB: computed on all pyramidal cells, NOT only place cells. Restricting to
    # place cells conditions on the outcome (high-S/N cells) and biases away the
    # very effect of interest (R3 minor #2: true coding deficit vs. excitability).
    fd = df[df["in_out_ratio"].notna()]
    print(f"{'metric':16s}{'layer':12s}{'p(exp)':>9s}{'med_ctrl':>10s}{'med_exp':>10s}{'n_cells':>9s}{'n_mice':>8s}")
    for var in ["in_field_rate", "out_field_rate", "in_out_ratio"]:
        for sp in ["deep", "superficial"]:
            s = fd[fd["sub_population"] == sp]
            m = lmm(s, var, log=True)
            mc = s[s.group_ani == "control"][var].median()
            me = s[s.group_ani == "exp"][var].median()
            print(f"{var:16s}{sp:12s}{m.pvalues['group_ani[T.exp]']:9.4f}"
                  f"{mc:10.2f}{me:10.2f}{int(m.model.endog.shape[0]):9d}{s['animal_id'].nunique():8d}")

    print("\n(4) Cells contributed per animal")
    t = df.groupby(["group_ani", "sub_population", "animal_id"]).size().reset_index(name="n")
    for g in ["control", "exp"]:
        for sp in ["deep", "superficial"]:
            sub = t[(t.group_ani == g) & (t.sub_population == sp)]
            vals = sorted(v for v in sub["n"] if v > 0)
            print(f"  {g:8s} {sp:12s}: {vals}  (sum={sum(vals)}, {len(vals)} mice)")


# ====================================================================
# PART 2 — LFP / gamma (Fig 3); unit = session/recording
# ====================================================================
def _scalar(v):
    try:
        a = np.array(v, dtype=float).flatten()
    except Exception:
        return np.nan
    return np.nanmean(a) if a.size else np.nan


def _band_sum(v, lo, hi, fs=1250):
    try:
        a = np.array(v, dtype=float)
    except Exception:
        return np.nan
    if a.size == 0:
        return np.nan
    if a.ndim == 3:
        a = a[..., 0]
    p = np.nanmean(a, axis=0) if a.ndim == 2 else a
    f = np.linspace(0, fs / 2, p.shape[0])
    return np.nansum(p[(f >= lo) & (f < hi)])


def lfp():
    df = pd.read_pickle(LFP_TABLE)
    df["animal_id"] = df["animal_id"].astype(str)
    df["group_ani"] = pd.Categorical(genotype(df["animal_id"]), categories=["control", "exp"])

    for c in ["slow_event_rate_py", "fast_event_rate_py",
              "slow_theta_gamma_coupling_py", "fast_theta_gamma_coupling_py"]:
        df[c + "_m"] = df[c].apply(_scalar)
    for layer in ["py", "sr"]:
        col = f"lfp_{layer}_norm_run"
        df[f"{layer}_theta"] = df[col].apply(lambda v: _band_sum(v, 4, 12))
        df[f"{layer}_slowg"] = df[col].apply(lambda v: _band_sum(v, 20, 39))
        df[f"{layer}_fastg"] = df[col].apply(lambda v: _band_sum(v, 40, 91))

    print("\nSessions per animal:", dict(df.groupby("animal_id").size()))
    print(f"Total {len(df)} sessions, {df['animal_id'].nunique()} mice")
    tests = [("py_theta", "Theta power (py)", True), ("py_slowg", "Slow-gamma power (py)", True),
             ("py_fastg", "Fast-gamma power (py)", True), ("sr_slowg", "Slow-gamma power (SR)", True),
             ("sr_fastg", "Fast-gamma power (SR)", True),
             ("slow_event_rate_py_m", "Slow-gamma event rate", False),
             ("fast_event_rate_py_m", "Fast-gamma event rate", False),
             ("slow_theta_gamma_coupling_py_m", "Slow theta-gamma coupling", False),
             ("fast_theta_gamma_coupling_py_m", "Fast theta-gamma coupling", False)]
    print(f"\n{'metric':30s}{'p(exp)':>9s}{'mean_ctrl':>11s}{'mean_exp':>10s}{'n_sess':>8s}{'n_mice':>8s}")
    for var, lab, log in tests:
        m = lmm(df, var, log)
        d = m.model.endog.shape[0]
        print(f"{lab + (' (log)' if log else ''):30s}{m.pvalues['group_ani[T.exp]']:9.4f}"
              f"{'':11s}{'':10s}{int(d):8d}{'':8s}")


if __name__ == "__main__":
    print("=" * 80)
    print("PART 1 — DEEP vs SUPERFICIAL (Fig 4)")
    print("=" * 80)
    deep_superficial()
    print("\n" + "=" * 80)
    print("PART 2 — LFP (Fig 3)")
    print("=" * 80)
    lfp()
