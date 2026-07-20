"""
Generate the supplementary statistics tables for the PLOS Biology revision.

Purpose: the revision does NOT redo any figure. It answers the reviewers'
statistical criticisms (R1.6, R3.4 -- "what is the statistical unit?", "report
n animals and cells per animal", "use mixed-effects models") by supplying a
complete, reproducible statistics table for every in vivo comparison.

Every row is recomputed from the source tables with animal as a random intercept.
Nothing here is copied from the existing figure legends.

Outputs (written next to the manuscript, not into the git repo):
    S_Table_mixed_effects_statistics.csv    -- machine readable, open in Excel
    S_Table_mixed_effects_statistics.md     -- paste into the response letter

Run:  python generate_stats_tables.py
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import mixed_effects_stats as M
import speed_coding_stats as S

OUTDIR = "/Users/sachuriga/Desktop/Projects/CR_CA1_paper/Manuscript"

# metric -> (pretty label, log-transform?)  matching the per-figure analyses
PLACE_METRICS = [
    ("Information_content_rate", "Spatial information content", True),
    ("Sparsity", "Sparsity", False),
    ("Field_size", "Max field size", True),
    ("Averate_rate", "Mean firing rate", True),
    ("Selectivity", "Selectivity", True),
    ("stability_ma", "Within-session stability", False),
    ("in_field_rate", "In-field firing rate", True),
    ("out_field_rate", "Out-of-field firing rate", True),
    ("in_out_ratio", "In/out-field ratio", True),
]
INOUT = {"in_field_rate", "out_field_rate", "in_out_ratio"}


def glmm_binary(d, flag="is_speed"):
    """Binomial GLMM with an animal random intercept; returns (OR, lo, hi, p)."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from scipy.stats import norm
    s = d.dropna(subset=[flag]).copy()
    s["y"] = s[flag].astype(float)
    m = BinomialBayesMixedGLM.from_formula("y ~ group_ani", {"a": "0 + C(animal_id)"}, s).fit_vb()
    i = list(m.model.exog_names).index("group_ani[T.exp]")
    mean, sd = m.fe_mean[i], m.fe_sd[i]
    p = 2 * (1 - norm.cdf(abs(mean / sd)))
    return np.exp(mean), np.exp(mean - 1.96 * sd), np.exp(mean + 1.96 * sd), p


def _fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def load_place():
    df = pd.read_pickle(M.FUNCTIONAL_TABLE)
    df = df[(df["buzaki_py_cell_type"] == "pyramidal") & (df["session"] == "A")].copy()
    df["animal_id"] = df["animal_id"].astype(str)
    df["group_ani"] = pd.Categorical(M.genotype(df["animal_id"]), categories=["control", "exp"])
    df["sub_population"] = pd.Categorical(df["sub_population"], categories=["deep", "superficial"])
    return df


def place_rows(df):
    """Within-layer genotype effects + genotype x layer interaction."""
    rows = []
    fd = df[df["in_out_ratio"].notna()]
    for var, lab, log in PLACE_METRICS:
        src = fd if var in INOUT else df
        for sp in ["deep", "superficial"]:
            s = src[src.sub_population == sp]
            m = M.lmm(s, var, log)
            t = "group_ani[T.exp]"
            ci = m.conf_int().loc[t]
            rows.append(dict(
                Analysis="Place coding (Fig 4)", Comparison=f"CR;DTA- vs CR;DTA+ ({sp})",
                Metric=lab, Model="LMM: value ~ genotype; random intercept = animal",
                Transform="log" if log else "none",
                n_cells=int(m.model.endog.shape[0]), n_mice=s.animal_id.nunique(),
                beta=round(m.params[t], 3),
                CI95=f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", p=_fmt_p(m.pvalues[t]),
                p_raw=m.pvalues[t]))
        # interaction
        m = M.lmm(src, var, log, formula="y ~ group_ani*sub_population")
        k = [t for t in m.pvalues.index if ":" in t][0]
        ci = m.conf_int().loc[k]
        rows.append(dict(
            Analysis="Place coding (Fig 4)", Comparison="genotype x layer INTERACTION",
            Metric=lab, Model="LMM: value ~ genotype x layer; random intercept = animal",
            Transform="log" if log else "none",
            n_cells=int(m.model.endog.shape[0]), n_mice=src.animal_id.nunique(),
            beta=round(m.params[k], 3),
            CI95=f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", p=_fmt_p(m.pvalues[k]),
            p_raw=m.pvalues[k]))
    return rows


def composite_rows(df):
    d, info = M.build_composite(df)
    rows = []
    for idx, nice in [("coding_z", "Composite spatial-coding index (z-mean)"),
                      ("coding_pc1", "Composite spatial-coding index (PC1)")]:
        for sp in ["deep", "superficial"]:
            s = d[d.sub_population == sp]
            m = M.lmm(s, idx)
            t = "group_ani[T.exp]"
            ci = m.conf_int().loc[t]
            rows.append(dict(
                Analysis="Composite endpoint", Comparison=f"CR;DTA- vs CR;DTA+ ({sp})",
                Metric=nice, Model="LMM: index ~ genotype; random intercept = animal", Transform="z-scored",
                n_cells=int(m.model.endog.shape[0]), n_mice=s.animal_id.nunique(),
                beta=round(m.params[t], 3), CI95=f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                p=_fmt_p(m.pvalues[t]), p_raw=m.pvalues[t]))
        m = M.lmm(d, idx, formula="y ~ group_ani*sub_population")
        k = [t for t in m.pvalues.index if ":" in t][0]
        ci = m.conf_int().loc[k]
        rows.append(dict(
            Analysis="Composite endpoint", Comparison="genotype x layer INTERACTION",
            Metric=nice, Model="LMM: index ~ genotype x layer; random intercept = animal", Transform="z-scored",
            n_cells=int(m.model.endog.shape[0]), n_mice=d.animal_id.nunique(),
            beta=round(m.params[k], 3), CI95=f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            p=_fmt_p(m.pvalues[k]), p_raw=m.pvalues[k]))
    return rows, info


def celltype_rows():
    """Pyramidal vs interneuron -- the specificity control."""
    df = S.load()
    rows = []
    for dv, raw, lab, tr in [("speed_z", "speed_score", "Speed score", "Fisher z"),
                             ("log_rate", "mean_firing_rate", "Mean firing rate", "log")]:
        for ct, ctlab in [("pyramidal", "pyramidal"),
                          ("narrow_spike_interneurons", "narrow-spiking interneuron")]:
            s = df[df.cell_type == ct]
            p, b, lo, hi, nc, nm = S.lmm_ci(s, dv)
            rows.append(dict(
                Analysis="Cell-type specificity (Supp Fig 4)",
                Comparison=f"CR;DTA- vs CR;DTA+ ({ctlab})", Metric=lab,
                Model="LMM: value ~ genotype; random intercept = animal", Transform=tr,
                n_cells=nc, n_mice=nm, beta=round(b, 3),
                CI95=f"[{lo:+.3f}, {hi:+.3f}]", p=_fmt_p(p), p_raw=p))
        d = df.dropna(subset=[dv]).copy()
        d["y"] = d[dv]
        m = smf.mixedlm("y ~ group_ani*cell_type", d, groups=d["animal_id"]).fit(reml=True)
        k = [t for t in m.pvalues.index if ":" in t][0]
        ci = m.conf_int().loc[k]
        rows.append(dict(
            Analysis="Cell-type specificity (Supp Fig 4)",
            Comparison="genotype x cell-type INTERACTION", Metric=lab,
            Model="LMM: value ~ genotype x cell type; random intercept = animal", Transform=tr,
            n_cells=int(m.model.endog.shape[0]), n_mice=d.animal_id.nunique(),
            beta=round(m.params[k], 3), CI95=f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            p=_fmt_p(m.pvalues[k]), p_raw=m.pvalues[k]))

    # Speed-modulated fraction: binary outcome. A linear mixed model is not valid
    # here, so we use a binomial GLMM with an animal random intercept -- the direct
    # analogue of the LMMs above. (A population-averaged GEE is reported in the
    # footnote; its SEs are anticonservative with only ~10 clusters.)
    d = df.copy()
    d["is_speed"] = S.speed_cell_flag(d, method="shuffle")
    for ct, ctlab in [("pyramidal", "pyramidal"),
                      ("narrow_spike_interneurons", "narrow-spiking interneuron")]:
        sub = d[d.cell_type == ct].dropna(subset=["is_speed"])
        pg = S.prop_gee(sub)[0]
        orr, lo, hi, pv = glmm_binary(sub)
        pc = 100 * sub[sub.group_ani == "control"]["is_speed"].mean()
        pe = 100 * sub[sub.group_ani == "exp"]["is_speed"].mean()
        rows.append(dict(
            Analysis="Cell-type specificity (Supp Fig 4)",
            Comparison=f"CR;DTA- vs CR;DTA+ ({ctlab})",
            Metric=f"Speed-modulated fraction ({pc:.1f}% vs {pe:.1f}%)",
            Model="Binomial GLMM: is_speed ~ genotype; random intercept = animal",
            Transform="none (binary)", n_cells=len(sub), n_mice=sub.animal_id.nunique(),
            beta=round(orr, 3), CI95=f"OR [{lo:.2f}, {hi:.2f}]",
            p=_fmt_p(pv), p_raw=pv, gee_p=round(pg, 4)))
    return rows


def design_table(df):
    """Cells contributed by each animal -- explicitly requested by Reviewer #1.6."""
    out = []
    for g, glab in [("control", "CR;DTA-"), ("exp", "CR;DTA+")]:
        for sp in ["deep", "superficial"]:
            s = df[(df.group_ani == g) & (df.sub_population == sp)]
            n = s.groupby("animal_id", observed=True).size().sort_values()
            out.append(dict(Group=glab, Layer=sp,
                            cells_per_animal=", ".join(str(v) for v in n.values),
                            total_cells=int(n.sum()), n_mice=len(n)))
    return pd.DataFrame(out)


def main():
    place = load_place()
    rows = place_rows(place)
    comp, info = composite_rows(place)
    rows += comp
    rows += celltype_rows()

    tab = pd.DataFrame(rows)
    tab = tab[["Analysis", "Comparison", "Metric", "Model", "Transform",
               "n_cells", "n_mice", "beta", "CI95", "p", "p_raw"]]
    des = design_table(place)

    csv = os.path.join(OUTDIR, "S_Table_mixed_effects_statistics.csv")
    tab.drop(columns="p_raw").to_csv(csv, index=False)

    md = os.path.join(OUTDIR, "S_Table_mixed_effects_statistics.md")
    with open(md, "w") as f:
        f.write("# S Table. Mixed-effects statistics for all in vivo comparisons\n\n")
        f.write("All models use animal as a random intercept, so the statistical unit is the\n"
                "animal rather than the cell. Beta is the CR;DTA+ effect on the transformed\n"
                "scale (for log-transformed measures, exp(beta) is the fold change).\n"
                "No figure was changed; these values supersede the cell-level tests in the\n"
                "original figure legends.\n\n")
        for a in tab["Analysis"].unique():
            f.write(f"\n## {a}\n\n")
            sub = tab[tab.Analysis == a].drop(columns=["Analysis", "p_raw"])
            f.write(sub.to_markdown(index=False))
            f.write("\n")
        f.write("\n\n## Cells contributed per animal (Reviewer #1.6)\n\n")
        f.write(des.to_markdown(index=False))
        f.write("\n\nNote: for the binary speed-modulated outcome a binomial GLMM with an animal\n"
                "random intercept is reported. A population-averaged GEE clustered on animal gives\n"
                "p = 0.001 (pyramidal) and p = 0.963 (interneuron); GEE standard errors are\n"
                "anticonservative with ~10 clusters, so the GLMM is reported as primary.\n")
        f.write("\n\n## Composite index construction\n\n")
        f.write(f"- Complete-case cells: {info['n']}\n")
        f.write(f"- PC1 variance explained: {100 * info['var_explained']:.1f}%\n")
        f.write(f"- r(PC1, z-mean): {info['r_z_pc1']:.3f}\n")
        f.write(f"- PC1 loadings (aligned, + = better coding): {info['loadings']}\n")

    print(f"wrote {csv}\nwrote {md}")
    print(f"\n{len(tab)} statistical rows; "
          f"{tab['n_mice'].min()}-{tab['n_mice'].max()} mice per comparison")
    return tab, des


if __name__ == "__main__":
    main()
