"""
Mixed-effects statistics for the deep/superficial (Fig 4) and LFP (Fig 3) analyses.

What this does:
  * uses mixed-effects models so the statistical unit is the animal, not the cell
    or the recording (cells from one mouse aren't independent);
  * reports n animals and n cells per animal;
  * tests whether deep and superficial are *differentially* affected via a
    genotype x layer interaction (not just "significant in one layer, n.s. in the other");
  * separates a real spatial-coding deficit from a plain rise in excitability using
    the in-field vs out-of-field firing-rate ratio (spatial signal-to-noise).

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

    Not in the default output (the figures use the mixed model). Handy as a
    cross-check: with only 5 mice/group the mixed model's random-effect variance
    sits near the boundary, so collapsing to per-animal means is the conservative
    sanity check.
    """
    from scipy.stats import ttest_ind, mannwhitneyu, shapiro
    d = d.copy()
    d["y"] = pd.to_numeric(d[var], errors="coerce")
    d = d.dropna(subset=["y"])
    if log:
        d = d[d["y"] > 0]
    # observed=True: group_ani is Categorical, so the default would emit every
    # group x animal combination (including impossible ones) as NaN means.
    am = d.groupby(["group_ani", "animal_id"], observed=True)["y"].mean().reset_index()
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
    # NB: all pyramidal cells, NOT only place cells. Restricting to place cells
    # conditions on the outcome (high-S/N cells) and biases away the very effect
    # we're after (true coding deficit vs. just higher excitability).
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
# PART 1b — Composite spatial-coding index (power for the interaction)
# ====================================================================
# Motivation: the six spatial-coding metrics are NOT six independent pieces of
# evidence. On these data Information_content_rate and Sparsity correlate at
# r = -0.99, and info/selectivity/in_out_ratio all sit at r ~ 0.9 -- they largely
# measure ONE latent axis. Testing them separately therefore spends power on
# redundancy and then pays an unnecessary multiple-comparison penalty. Collapsing
# them into a single pre-specified endpoint is the standard remedy.
#
# HONESTY REQUIREMENTS for using this:
#   * This is ONE endpoint, decided a priori -- not a fishing expedition run
#     after seeing which individual metrics came out lowest. Say so in Methods.
#   * The composite is built WITHOUT using genotype or layer labels (z-scoring
#     and PCA are unsupervised), so it cannot manufacture a group difference.
#   * It does not replace the per-metric table; report both.
#   * A non-significant interaction here is still a non-significant interaction.

# (metric, label, log-transform?, sign so that +1 => BETTER spatial coding)
SPATIAL_METRICS = [
    ("Information_content_rate", "Info content", True, +1),
    ("Sparsity", "Sparsity", False, -1),          # higher sparsity = less selective
    ("Field_size", "Field size", True, -1),       # bigger field = less precise
    ("Selectivity", "Selectivity", True, +1),
    ("stability_ma", "Stability", False, +1),
    ("in_out_ratio", "In/out ratio", True, +1),   # spatial signal-to-noise
]


def build_composite(df, metrics=SPATIAL_METRICS):
    """Add `coding_z` (mean of aligned z-scores) and `coding_pc1` to df.

    Signs are aligned so that HIGHER = BETTER spatial coding. Transforms match
    those used for the per-metric LMMs. Complete cases only.
    Returns (df_with_index, info_dict).
    """
    from sklearn.decomposition import PCA

    d = df.copy()
    cols = []
    for var, _lab, log, sign in metrics:
        x = pd.to_numeric(d[var], errors="coerce")
        if log:
            x = np.log(x.where(x > 0))
        col = f"_z_{var}"
        d[col] = sign * (x - x.mean()) / x.std()   # unsupervised standardisation
        cols.append(col)

    d = d.dropna(subset=cols).copy()
    d["coding_z"] = d[cols].mean(axis=1)

    pca = PCA(n_components=1).fit(d[cols].values)
    pc1 = pca.transform(d[cols].values)[:, 0]
    # orient PC1 to agree with the z-mean (PCA sign is arbitrary)
    if np.corrcoef(pc1, d["coding_z"])[0, 1] < 0:
        pc1, load = -pc1, -pca.components_[0]
    else:
        load = pca.components_[0]
    d["coding_pc1"] = pc1

    info = {"n": len(d), "var_explained": pca.explained_variance_ratio_[0],
            "loadings": dict(zip([m[1] for m in metrics], load.round(3))),
            "r_z_pc1": float(np.corrcoef(d["coding_pc1"], d["coding_z"])[0, 1])}
    return d, info


def composite_interaction(df=None):
    """Genotype x layer interaction on the single composite endpoint."""
    if df is None:
        df = pd.read_pickle(FUNCTIONAL_TABLE)
        df = df[(df["buzaki_py_cell_type"] == "pyramidal") & (df["session"] == "A")].copy()
        df["animal_id"] = df["animal_id"].astype(str)
        df["group_ani"] = pd.Categorical(genotype(df["animal_id"]), categories=["control", "exp"])
        df["sub_population"] = pd.Categorical(df["sub_population"], categories=["deep", "superficial"])

    d, info = build_composite(df)
    print(f"\nComposite built on {info['n']} complete-case cells; "
          f"PC1 explains {100 * info['var_explained']:.1f}% of variance; "
          f"r(PC1, z-mean) = {info['r_z_pc1']:.3f}")
    print("  PC1 loadings (aligned, + = better coding):", info["loadings"])

    for idx in ["coding_z", "coding_pc1"]:
        print(f"\n--- {idx} ---")
        for sp in ["deep", "superficial"]:
            s = d[d.sub_population == sp]
            m = lmm(s, idx)
            t = "group_ani[T.exp]"
            ci = m.conf_int().loc[t]
            print(f"  {sp:12s} genotype effect p={m.pvalues[t]:.4f}  "
                  f"beta={m.params[t]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  "
                  f"n={int(m.model.endog.shape[0])} cells, {s.animal_id.nunique()} mice")
        m = lmm(d, idx, formula="y ~ group_ani*sub_population")
        k = [t for t in m.pvalues.index if ":" in t][0]
        ci = m.conf_int().loc[k]
        print(f"  {'INTERACTION':12s} p={m.pvalues[k]:.4f}  "
              f"beta={m.params[k]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")


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
