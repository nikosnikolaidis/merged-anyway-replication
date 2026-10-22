"""
Quality-gate override analysis -- reproduces every number, table and figure in the paper
"Merged Anyway: How Open-Source Projects Respond When Quality Gates Fail".

Input : data/quality_gates.csv.gz
Output: results/results.json, results/paper_numbers.md, results/tables/*.csv, results/figures/*.pdf|png

Run from any directory:  python scripts/analysis.py
(prep.py is executed automatically on the first run; intermediate files go to build/.)
"""
import json, sys, warnings
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = ROOT / "data" / "quality_gates.csv.gz"
OUT = ROOT / "results"
BUILD = ROOT / "build"
for _p in (OUT, BUILD): _p.mkdir(exist_ok=True)
from fe import lpm_fe

warnings.filterwarnings("ignore")
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
TAB = OUT / "tables"; TAB.mkdir(exist_ok=True)
R = {}                                   # all reported numbers

import time
T0 = time.time()


def step(title):
    print(f"\n=== {title} ===", flush=True)


def say(msg):
    print(f"  [{time.time() - T0:5.1f}s] {msg}", flush=True)


def ci(d):
    return f"{d['pp']:+.2f} pp [95% CI {d['lo']:+.2f}, {d['hi']:+.2f}], p = {d['p']:.2g}"


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig); say(f"saved figures/{name}.pdf (+ .png)")


BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W = 5.4  # acmsmall text width (inches)


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return lo, hi


# ---------------------------------------------------------------------------
# 0. Data & cleaning
# ---------------------------------------------------------------------------
step("Step 1/6 - Data loading and cleaning")
if not ((BUILD / "prs.parquet").exists() and (BUILD / "conds.parquet").exists()):
    say("prs.parquet / conds.parquet not found -> running prep.py first")
    import prep
    _d, _c = prep.load(DATA)
    _d.drop(columns=["failed_conditions"]).to_parquet(BUILD / "prs.parquet"); _c.to_parquet(BUILD / "conds.parquet")
say("reading data/quality_gates.csv.gz and build/*.parquet")
raw = pd.read_csv(DATA, usecols=["repo_name", "pr_number"], low_memory=False)
df = pd.read_parquet(BUILD / "prs.parquet")
c = pd.read_parquet(BUILD / "conds.parquet")
R["raw_rows"] = int(len(raw)); R["raw_repos"] = int(raw.repo_name.nunique())
R["duplicates_removed"] = int(len(raw) - len(df))

repo_merge = df.groupby("repo_name").merged.mean()
excl = repo_merge[repo_merge < 0.10].index
R["excluded_repos"] = int(len(excl))
R["excluded_prs"] = int(df.repo_name.isin(excl).sum())
R["excluded_fail_prs"] = int((df.repo_name.isin(excl) & df.fail).sum())
df = df[~df.repo_name.isin(excl)].copy()
c = c[~c.repo_name.isin(excl)].copy()
df["merged_i"] = df.merged.astype(int); df["fail_i"] = df.fail.astype(int)
df["fail_nocond"] = (df.fail & (df.n_cond == 0)).astype(int)
c["merged_i"] = c.merged.astype(int)

R["n_prs"] = int(len(df)); R["n_repos"] = int(df.repo_name.nunique())
R["n_fail"] = int(df.fail.sum()); R["n_pass"] = int((~df.fail).sum())
R["n_fail_nocond"] = int(df.fail_nocond.sum()); R["n_fail_cond"] = R["n_fail"] - R["n_fail_nocond"]
R["n_conditions"] = int(len(c))
R["repos_with_fail"] = int(df[df.fail].repo_name.nunique())
pr_per_repo = df.groupby("repo_name").size()
R["prs_per_repo_median"] = float(pr_per_repo.median()); R["prs_per_repo_max"] = int(pr_per_repo.max())
R["prs_per_repo_q1"] = float(pr_per_repo.quantile(.25)); R["prs_per_repo_q3"] = float(pr_per_repo.quantile(.75))
fpr = df[df.fail].groupby("repo_name").size()
R["fail_per_repo_median"] = float(fpr.median())
R["n_cond_dist"] = {int(k): int(v) for k, v in df[df.fail & (df.n_cond > 0)].n_cond.value_counts().sort_index().items()}
say(f"raw: {R['raw_rows']:,} rows from {R['raw_repos']:,} repos; duplicates removed: {R['duplicates_removed']:,}")
say(f"excluded {R['excluded_repos']} repos with <10% merged PRs ({R['excluded_prs']:,} PRs, {R['excluded_fail_prs']:,} failing)")
say(f"analysis set: {R['n_prs']:,} PRs in {R['n_repos']:,} repos; PASS {R['n_pass']:,}, FAIL {R['n_fail']:,} ({R['n_fail']/R['n_prs']:.1%})")
say(f"FAIL with parsed conditions: {R['n_fail_cond']:,}; without details: {R['n_fail_nocond']:,}; parsed conditions: {R['n_conditions']:,}")
say(f"PRs per repo: median {R['prs_per_repo_median']:.0f} (IQR {R['prs_per_repo_q1']:.0f}-{R['prs_per_repo_q3']:.0f}), max {R['prs_per_repo_max']:,}")

# ---------------------------------------------------------------------------
# RQ1  Prevalence and within-project effect of a failed gate
# ---------------------------------------------------------------------------
step("Step 2/6 - RQ1: prevalence and within-project effect of a failing gate")
R["pass_rate"] = 1 - df.fail.mean()
R["merge_pass"] = df[~df.fail].merged.mean()
R["merge_fail"] = df[df.fail].merged.mean()
R["n_bypassed"] = int(df[df.fail].merged.sum()); R["n_blocked"] = R["n_fail"] - R["n_bypassed"]
R["merge_fail_cond"] = df[df.fail & (df.n_cond > 0)].merged.mean()
R["merge_fail_nocond"] = df[df.fail_nocond == 1].merged.mean()
R["reject_pass"] = 1 - R["merge_pass"]; R["reject_fail"] = 1 - R["merge_fail"]
R["rr_reject"] = R["reject_fail"] / R["reject_pass"]
tab = pd.crosstab(df.fail, df.merged)
chi2, p, *_ = stats.chi2_contingency(tab)
R["chi2_pass_fail"] = float(chi2)
R["cramers_v"] = float(np.sqrt(chi2 / len(df)))

g = df.groupby(["repo_name", "fail"]).merged.agg(["sum", "count"]).unstack(fill_value=0)
g.columns = ["kp", "kf", "np", "nf"]
both = g[(g.np > 0) & (g.nf > 0)]
st = StratifiedTable([np.array([[r.kf, r.nf - r.kf], [r.kp, r.np - r.kp]], float) for r in both.itertuples()])
R["mh_or"] = float(st.oddsratio_pooled); R["mh_or_ci"] = [float(x) for x in st.oddsratio_pooled_confint()]
R["mh_repos"] = int(len(both))

fe1 = lpm_fe(df, "merged_i", ["fail_i"], "repo_name")
R["fe_fail_pp"] = fe1.loc["fail_i"].to_dict()
R["gate_yield"] = -fe1.loc["fail_i", "pp"] / 100 / R["merge_pass"]   # share of failing PRs held back
fe1b = lpm_fe(df[df.fail_nocond == 0], "merged_i", ["fail_i"], "repo_name")
R["fe_fail_pp_condonly"] = fe1b.loc["fail_i"].to_dict()
# sensitivity: including excluded repos
dfall = pd.read_parquet(BUILD / "prs.parquet"); dfall["merged_i"] = dfall.merged.astype(int); dfall["fail_i"] = dfall.fail.astype(int)
R["fe_fail_pp_allrepos"] = lpm_fe(dfall, "merged_i", ["fail_i"], "repo_name").loc["fail_i"].to_dict()
R["merge_fail_allrepos"] = dfall[dfall.fail].merged.mean()

# project-weighted view
t5 = g[(g.np >= 5) & (g.nf >= 5)].copy()
t5["mp"] = t5.kp / t5.np; t5["mf"] = t5.kf / t5.nf; t5["delta"] = t5.mp - t5.mf
R["repos_5_5"] = int(len(t5))
R["delta_median_pp"] = float(t5.delta.median() * 100); R["delta_mean_pp"] = float(t5.delta.mean() * 100)
R["share_repos_fail_ge_pass"] = float((t5.delta <= 0).mean())
say(f"merge rate PASS {R['merge_pass']:.1%} vs FAIL {R['merge_fail']:.1%} -> {R['n_bypassed']:,} failing PRs merged, {R['n_blocked']:,} held back")
say(f"rejection risk ratio {R['rr_reject']:.2f}; chi2 = {R['chi2_pass_fail']:,.0f}; Cramer's V = {R['cramers_v']:.3f}")
say(f"Mantel-Haenszel OR (stratified by repo, {R['mh_repos']:,} repos) = {R['mh_or']:.3f} [{R['mh_or_ci'][0]:.3f}, {R['mh_or_ci'][1]:.3f}]")
say(f"LPM with repo FE, FAIL effect: {ci(R['fe_fail_pp'])}")
say(f"gate yield = {R['gate_yield']:.2%} (about 1 in {1/R['gate_yield']:.0f} failing PRs held back)")
say(f"sensitivity - only FAIL with conditions: {ci(R['fe_fail_pp_condonly'])}")
say(f"sensitivity - incl. excluded repos:    {ci(R['fe_fail_pp_allrepos'])}")
say(f"{R['repos_5_5']:,} repos with >=5 PASS & >=5 FAIL: median delta {R['delta_median_pp']:.1f} pp, {R['share_repos_fail_ge_pass']:.1%} merge FAIL >= PASS")

# ---------------------------------------------------------------------------
# RQ2  Heterogeneity across projects
# ---------------------------------------------------------------------------
step("Step 3/6 - RQ2: heterogeneity across projects")
f = g[g.nf > 0].copy(); f["raw"] = f.kf / f.nf
R["repos_bypass100"] = int((f.raw == 1).sum()); R["repos_bypass0"] = int((f.raw == 0).sum())
R["repos_nf1"] = int((f.nf == 1).sum())
R["repos_bypass100_nf_median"] = float(f[f.raw == 1].nf.median())
k, n = f.kf.values, f.nf.values
res = optimize.minimize(lambda q: -np.sum(stats.betabinom.logpmf(k, n, *np.exp(q))), [1, 0], method="Nelder-Mead")
a, b = np.exp(res.x); R["eb_alpha"], R["eb_beta"] = float(a), float(b); R["eb_prior_mean"] = float(a / (a + b))
f["eb"] = (f.kf + a) / (f.nf + a + b)
s10 = f[f.nf >= 10]
R["repos_nf10"] = int(len(s10))
R["nf10_median"] = float(s10.raw.median()); R["nf10_q1"] = float(s10.raw.quantile(.25)); R["nf10_q3"] = float(s10.raw.quantile(.75))
R["nf10_le20"] = float((s10.raw <= .2).mean()); R["nf10_le50"] = float((s10.raw < .5).mean()); R["nf10_ge90"] = float((s10.raw >= .9).mean())
R["eb_le50"] = float((f.eb < .5).mean()); R["eb_ge90"] = float((f.eb >= .9).mean())
R["dip_note"] = "unimodal"

# typology (>=5 PASS, >=5 FAIL)
t5["p"] = [stats.fisher_exact([[r.kp, r.np - r.kp], [r.kf, r.nf - r.kf]])[1] for r in t5.itertuples()]
t5["q"] = multipletests(t5.p, method="fdr_bh")[1]
sig = t5.q < 0.05


def tost(r, margin=0.10):
    p1, p2 = r.mp, r.mf
    se = np.sqrt(max(p1 * (1 - p1), 1e-4) / r.np + max(p2 * (1 - p2), 1e-4) / r.nf)
    d = p1 - p2
    return max(stats.norm.sf((d + margin) / se), stats.norm.cdf((d - margin) / se))


t5["p_tost"] = [tost(r) for r in t5.itertuples()]
t5["type"] = np.select(
    [sig & (t5.delta > 0) & (t5.mf < 0.5), sig & (t5.delta > 0), sig & (t5.delta < 0), t5.p_tost < 0.05],
    ["Enforcing", "Discounting", "Inverted", "Indifferent"], "Inconclusive")
typ = t5.groupby("type").agg(repos=("delta", "size"), mf=("mf", "median"), mp=("mp", "median"),
                             fail_prs=("nf", "sum"), blocked=("kf", lambda s: 0))
typ["blocked"] = t5.assign(bl=t5.nf - t5.kf).groupby("type").bl.sum()
typ["share_repos"] = typ.repos / typ.repos.sum(); typ["share_fail"] = typ.fail_prs / typ.fail_prs.sum()
typ["share_blocked"] = typ.blocked / typ.blocked.sum()
typ.to_csv(TAB / "typology.csv")
R["typology"] = typ.to_dict(orient="index")
t5.to_csv(TAB / "repo_level.csv")

# configuration: customised thresholds
DEF = {"Coverage": 80.0, "Duplication": 3.0}
cust = np.zeros(len(c), bool)
for cat_, v in DEF.items():
    m = c["cat"].eq(cat_) & c.req.notna(); cust |= (m & c.req.ne(v)).to_numpy()
rat = c["cat"].isin(["Reliability rating", "Security rating", "Maintainability rating"]) & c.req_grade.notna()
cust |= (rat & c.req_grade.ne(1)).to_numpy()
cust |= c["cat"].isin(["Issues (severity)", "PR size"]).to_numpy()
c["custom"] = cust
rc = c.groupby("repo_name").custom.max()
R["repos_obs_config"] = int(len(rc)); R["repos_custom"] = int(rc.sum())
d = df[df.repo_name.isin(rc.index)].copy(); d["custom"] = d.repo_name.map(rc).astype(int)
d["fail_x_custom"] = d.fail_i * d.custom
fe_c = lpm_fe(d, "merged_i", ["fail_i", "fail_x_custom"], "repo_name")
R["fe_custom"] = fe_c.round(4).to_dict(orient="index")
# reported in the paper: gate effect estimated separately for customised and default gates,
# and the share of Enforcing projects in each group (among the classified >=5/>=5 projects)
R["fe_by_gate"] = {lab: lpm_fe(d[d.custom == k], "merged_i", ["fail_i"], "repo_name").loc["fail_i"].round(4).to_dict()
                   for k, lab in [(1, "customised"), (0, "default")]}
_t = t5.assign(custom=t5.index.map(rc))
R["enforcing_share_by_gate"] = {lab: float((_t[_t.custom == k].type == "Enforcing").mean())
                                for k, lab in [(True, "customised"), (False, "default")]}
R["cov_thresholds"] = {str(k): int(v) for k, v in c[c["cat"] == "Coverage"].req.value_counts().head(6).items()}
R["dup_thresholds"] = {str(k): int(v) for k, v in c[c["cat"] == "Duplication"].req.value_counts().head(6).items()}
R["share_cov_default"] = float((c[c["cat"] == "Coverage"].req == 80).mean())
R["share_dup_default"] = float((c[c["cat"] == "Duplication"].req == 3).mean())
say(f"{R['repos_bypass100']} repos override 100%, {R['repos_bypass0']} override 0% (median n_fail of the 100% repos: {R['repos_bypass100_nf_median']:.0f})")
say(f"{R['repos_nf10']:,} repos with >=10 failing PRs: median override {R['nf10_median']:.1%} (IQR {R['nf10_q1']:.1%}-{R['nf10_q3']:.1%}); <50%: {R['nf10_le50']:.1%}; >=90%: {R['nf10_ge90']:.1%}")
say(f"empirical Bayes beta-binomial prior: alpha={R['eb_alpha']:.2f}, beta={R['eb_beta']:.2f}, mean {R['eb_prior_mean']:.1%}; shrunken <50%: {R['eb_le50']:.1%}")
say("typology (Fisher + BH-FDR 5%, TOST +/-10 pp):")
for _k, _v in R["typology"].items():
    say(f"   {_k:<13} {int(_v['repos']):>4} repos ({_v['share_repos']:.1%})  median merge PASS {_v['mp']:.1%} / FAIL {_v['mf']:.1%}  share of held-back {_v['share_blocked']:.1%}")
say(f"customised gates: {R['repos_custom']} of {R['repos_obs_config']} repos; FAIL x custom interaction {R['fe_custom']['fail_x_custom']['pp']:+.2f} pp [{R['fe_custom']['fail_x_custom']['lo']:+.2f}, {R['fe_custom']['fail_x_custom']['hi']:+.2f}], p = {R['fe_custom']['fail_x_custom']['p']:.2f}")

# ---------------------------------------------------------------------------
# RQ3  Which conditions are enforced?
# ---------------------------------------------------------------------------
step("Step 4/6 - RQ3: which failed conditions are enforced")
CATS = ["Coverage", "Duplication", "Maintainability rating", "Reliability rating", "Security rating",
        "Hotspots review", "Issues (severity)", "New issues", "PR size"]
ind = pd.crosstab(c.pr_id, c["cat"]).clip(upper=1).reindex(columns=CATS, fill_value=0)
a_ = df.set_index("pr_id").join(ind); a_[CATS] = a_[CATS].fillna(0)
fc = a_[a_.fail & (a_.n_cond > 0)]
rows = []
for k_ in CATS:
    s = fc[fc[k_] == 1]
    only = fc[(fc[k_] == 1) & (fc[CATS].sum(axis=1) == 1)]
    lo, hi = wilson(s.merged.sum(), len(s))
    rows.append(dict(cat=k_, prs=len(s), share_failed=len(s) / len(fc), repos=s.repo_name.nunique(),
                     bypass=s.merged.mean(), lo=lo, hi=hi, only_n=len(only), only_bypass=only.merged.mean()))
cat_tab = pd.DataFrame(rows).set_index("cat")
fe3 = lpm_fe(a_.reset_index(), "merged_i", ["fail_nocond"] + CATS, "repo_name")
cat_tab = cat_tab.join(fe3[["pp", "lo", "hi", "p"]].rename(columns=lambda x: "fe_" + x))
cat_tab.to_csv(TAB / "conditions.csv")
R["conditions"] = cat_tab.to_dict(orient="index")
R["fe_nocond"] = fe3.loc["fail_nocond"].to_dict()
a2 = a_.reset_index(); a2["k2"] = (a2.n_cond >= 2).astype(int); a2["k3"] = (a2.n_cond >= 3).astype(int)
R["fe_multi"] = lpm_fe(a2, "merged_i", ["fail_i", "k2", "k3"], "repo_name").round(4).to_dict(orient="index")
mc = fc.groupby(np.minimum(fc.n_cond, 4)).merged.agg(["mean", "size"])
R["bypass_by_ncond"] = {int(k): [float(v["mean"]), int(v["size"])] for k, v in mc.iterrows()}

# pooled vs within: the sandbox artefact
cs_all = pd.read_parquet(BUILD / "conds.parquet")
se = cs_all[(cs_all["cat"] == "Security rating") & (cs_all.grade == 5)]
R["secE_with_sandbox"] = [float(se.merged.mean()), int(len(se))]
sx = se[~se.repo_name.isin(excl)]
R["secE_without"] = [float(sx.merged.mean()), int(len(sx))]
R["sandbox_secE_prs"] = int(se.repo_name.str.startswith("PixeeSandbox").sum())
for _k, _v in cat_tab.iterrows():  # full precision
    say(f"   {_k:<23} n={int(_v['prs']):>6,}  override {_v['bypass']:.1%}  FE {_v['fe_pp']:+.2f} pp [{_v['fe_lo']:+.2f}, {_v['fe_hi']:+.2f}]")
say("override by # failed conditions: " + ", ".join(f"{k}{'+' if k == 4 else ''}: {v[0]:.1%}" for k, v in R["bypass_by_ncond"].items()))
say(f"FE: 2nd condition {R['fe_multi']['k2']['pp']:+.2f} pp, 3rd+ {R['fe_multi']['k3']['pp']:+.2f} pp (on top of first failure {R['fe_multi']['fail_i']['pp']:+.2f} pp)")
say(f"sandbox check: E security rating override {R['secE_with_sandbox'][0]:.1%} (n={R['secE_with_sandbox'][1]:,}) with sandbox vs {R['secE_without'][0]:.1%} (n={R['secE_without'][1]:,}) without")

# ---------------------------------------------------------------------------
# RQ4  Does the magnitude of the violation matter?
# ---------------------------------------------------------------------------
step("Step 5/6 - RQ4: does the magnitude of the violation matter")
cov = c[(c["cat"] == "Coverage") & c.margin.notna() & c.value.notna()].copy()
R["cov_zero_share"] = float((cov.value == 0).mean()); R["cov_zero_n"] = int((cov.value == 0).sum())
R["cov_zero_bypass"] = float(cov[cov.value == 0].merged.mean())
R["cov_nonzero_bypass"] = float(cov[cov.value > 0].merged.mean())
zz = cov.groupby("repo_name").agg(n=("value", "size"), z=("value", lambda s: (s == 0).mean()))
zz5 = zz[zz.n >= 5]; R["cov_repos5"] = int(len(zz5)); R["cov_repos_allzero"] = int((zz5.z >= .9).sum())
R["cov_gap_median"] = float(cov.margin.median())
dup = c[(c["cat"] == "Duplication") & c.margin.notna()].copy()
R["dup_excess_median"] = float(dup.margin.median())


def pr_level(dd):
    return dd.groupby("pr_id").agg(margin=("margin", "max"), merged_i=("merged_i", "first"),
                                   repo_name=("repo_name", "first")).reset_index()


bins_cov = [-0.01, 5, 10, 20, 40, 60, 79.99, 100]
lab_cov = ["≤5", "5–10", "10–20", "20–40", "40–60", "60–80", "0% cov."]
pc = pr_level(cov)
pc_val = cov.groupby("pr_id").value.min()
pc["zero"] = pc.pr_id.map(pc_val).eq(0)
pc["bin"] = pd.cut(pc.margin, bins_cov[:-1] + [100], labels=lab_cov[:-1] + ["≥80"]).astype(str)
pc.loc[pc.zero, "bin"] = "0% cov."
# reported in the paper (PR level, worst coverage value per PR)
R["cov_conditions_total"] = int(len(cov)); R["cov_prs_total"] = int(len(pc))
R["cov_zero_prs"] = int(pc.zero.sum()); R["cov_zero_prs_share"] = float(pc.zero.mean())
R["cov_zero_prs_override"] = float(pc[pc.zero].merged_i.mean())
R["cov_gap_median_pr"] = float(pc.margin.median())
_zz = pc.groupby("repo_name").agg(n=("zero", "size"), z=("zero", "mean"))
R["cov_repos5_pr"] = int((_zz.n >= 5).sum()); R["cov_repos_allzero_pr"] = int(((_zz.n >= 5) & (_zz.z >= .9)).sum())
pc["m10"] = pc.margin / 10; pc["near"] = (pc.margin <= 5).astype(int); pc["zero_i"] = pc.zero.astype(int)
R["fe_cov_slope10"] = lpm_fe(pc, "merged_i", ["m10"], "repo_name").loc["m10"].to_dict()
R["fe_cov_slope10_nonzero"] = lpm_fe(pc[~pc.zero], "merged_i", ["m10"], "repo_name").loc["m10"].to_dict()
R["fe_cov_near"] = lpm_fe(pc, "merged_i", ["near"], "repo_name").loc["near"].to_dict()
pdp = pr_level(dup)
pdp["bin"] = pd.cut(pdp.margin, [-0.01, 2, 5, 10, 20, 40, 1e4], labels=["≤2", "2–5", "5–10", "10–20", "20–40", ">40"]).astype(str)
pdp["m10"] = pdp.margin.clip(upper=100) / 10; pdp["near"] = (pdp.margin <= 2).astype(int)
R["fe_dup_slope10"] = lpm_fe(pdp, "merged_i", ["m10"], "repo_name").loc["m10"].to_dict()
R["fe_dup_near"] = lpm_fe(pdp, "merged_i", ["near"], "repo_name").loc["near"].to_dict()
R["dup_bins"] = pdp.groupby("bin").merged_i.agg(["mean", "size"]).round(4).to_dict(orient="index")
R["cov_bins"] = pc.groupby("bin").merged_i.agg(["mean", "size"]).round(4).to_dict(orient="index")
# --- actual observed values per PR (used for Fig. 4 a/b, so all four panels show observed values)
#     coverage: lowest observed coverage among the PR's coverage conditions; duplication: highest.
#     Regressions above stay threshold-relative; the figure is descriptive.
pcv = cov.groupby("pr_id").agg(value=("value", "min"), req=("req", "first"), merged_i=("merged_i", "first"),
                               repo_name=("repo_name", "first")).reset_index()
pcv["bin"] = pd.cut(pcv.value, [0, 20, 40, 60, 70, 80, 1e4], labels=["0–20", "20–40", "40–60", "60–70", "70–80", "≥80"],
                    include_lowest=False).astype(str)
pcv.loc[pcv.value == 0, "bin"] = "0"
dupv = c[(c["cat"] == "Duplication") & c.value.notna()]
pdv = dupv.groupby("pr_id").agg(value=("value", "max"), req=("req", "first"), merged_i=("merged_i", "first"),
                                repo_name=("repo_name", "first")).reset_index()
pdv["bin"] = pd.cut(pdv.value, [-0.01, 5, 10, 20, 40, 1e7], labels=["≤5", "5–10", "10–20", "20–40", ">40"]).astype(str)
R["cov_value_bins"] = pcv.groupby("bin").merged_i.agg(["mean", "size"]).round(4).to_dict(orient="index")
R["dup_value_bins"] = pdv.groupby("bin").merged_i.agg(["mean", "size"]).round(4).to_dict(orient="index")
# robustness: same bins restricted to the Sonar way default thresholds (coverage 80%, duplication 3%)
R["cov_value_bins_default"] = pcv[pcv.req == 80].groupby("bin").merged_i.agg(["mean", "size"]).round(4).to_dict(orient="index")
R["dup_value_bins_default"] = pdv[pdv.req == 3].groupby("bin").merged_i.agg(["mean", "size"]).round(4).to_dict(orient="index")
R["rating_steps"] = {}
grade_tab = {}
for k_ in ["Reliability rating", "Security rating", "Maintainability rating"]:
    dd = c[(c["cat"] == k_) & c.margin.notna()]
    R["rating_steps"][k_] = lpm_fe(dd, "merged_i", ["margin"], "repo_name").loc["margin"].to_dict()
    grade_tab[k_] = dd.groupby("grade").merged_i.agg(["mean", "size"])
    R["grades_" + k_] = {int(i): [float(v["mean"]), int(v["size"])] for i, v in grade_tab[k_].iterrows()}
hs = c[c["cat"] == "Hotspots review"].copy()
hs["bin"] = pd.cut(hs.value, [0, 1, 2, 5, 10, 1e7], labels=["1", "2", "3–5", "6–10", ">10"]).astype(str)
hs["log_n"] = np.log2(hs.value)
R["fe_hotspot_log2"] = lpm_fe(hs, "merged_i", ["log_n"], "repo_name").loc["log_n"].to_dict()
R["hotspot_bins"] = hs.groupby("bin").merged_i.agg(["mean", "size"]).round(4).to_dict(orient="index")
say(f"coverage: median gap {R['cov_gap_median']:.1f} pp; 0% coverage in {R['cov_zero_share']:.1%} of conditions ({R['cov_zero_n']:,}), override {R['cov_zero_bypass']:.1%}")
say(f"coverage slope per 10 pp gap: {ci(R['fe_cov_slope10'])}")
say(f"coverage near-miss (<=5 pp): {ci(R['fe_cov_near'])}")
say(f"duplication: median excess {R['dup_excess_median']:.1f} pp; slope per 10 pp: {ci(R['fe_dup_slope10'])}")
for _k, _v in R["rating_steps"].items():
    say(f"{_k} per grade step: {ci(_v)}")
say(f"hotspots per doubling: {ci(R['fe_hotspot_log2'])}")
for _nm in ["cov_value_bins", "cov_value_bins_default", "dup_value_bins", "dup_value_bins_default"]:
    say(_nm + ": " + ", ".join(f"{k} {v['mean']:.1%} (n={int(v['size']):,})" for k, v in R[_nm].items()))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
step("Step 6/6 - Figures and outputs")
# Fig 1: per-project merge rate of PASS vs FAIL PRs (>=5/>=5) + delta distribution
fig, ax = plt.subplots(1, 2, figsize=(W, 2.1), gridspec_kw={"width_ratios": [1, 1.25]})
ax[0].scatter(t5.mp * 100, t5.mf * 100, s=np.clip(t5.nf, 5, 400) / 8, color=BLUE, alpha=.35, lw=0)
ax[0].plot([0, 100], [0, 100], color=MUTED, lw=.8, ls="--")
ax[0].set_xlim(0, 101); ax[0].set_ylim(-1, 101)
ax[0].set_xlabel("Merge rate of passing PRs (%)"); ax[0].set_ylabel("Merge rate of failing PRs (%)")
ax[0].set_title("(a) One dot per project", loc="left")
ax[0].text(8, 92, "failing PRs merged\nmore often", fontsize=6.5, color=MUTED, va="top")
ax[0].text(97, 5, "gate\nenforced", fontsize=6.5, color=MUTED, ha="right")
bins = np.arange(-50, 101, 5)
ax[1].hist(t5.delta * 100, bins=bins, color=BLUE, edgecolor="white", lw=.6)
ax[1].axvline(0, color=MUTED, lw=.8, ls="--")
ax[1].axvline(t5.delta.median() * 100, color=ORANGE, lw=1.2)
ax[1].text(t5.delta.median() * 100 + 2, ax[1].get_ylim()[1] * .92,
           f"median {t5.delta.median()*100:.1f} pp", color=INK, fontsize=6.5)
ax[1].set_xlabel("Gate effect: passing − failing merge rate (pp)"); ax[1].set_ylabel("Projects")
ax[1].set_title("(b) Within-project gate effect", loc="left")
fig.tight_layout(); savefig(fig, "fig_gate_effect")

# Fig 2: per-project override-rate distribution: all projects vs projects with >=10 failing PRs
fig, ax = plt.subplots(figsize=(W * 0.62, 1.9))
bb = np.linspace(0, 1, 21)
ax.hist(f.raw * 100, bins=bb * 100, color=GREY, alpha=.55, label=f"all {len(f):,} projects", edgecolor="white", lw=.5)
ax.hist(s10.raw * 100, bins=bb * 100, color=BLUE, label=f"≥10 failing PRs ({len(s10):,})", edgecolor="white", lw=.5)
ax.set_xlabel("Project-level override rate (%)"); ax.set_ylabel("Projects")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); savefig(fig, "fig_projects")

# Fig 3: forest plot of within-project effects per condition type, annotated with raw override rates
lab = {"Coverage": "Coverage", "Duplication": "Duplication", "Maintainability rating": "Maintainability rating",
       "Reliability rating": "Reliability rating", "Security rating": "Security rating",
       "Hotspots review": "Security hotspots review", "Issues (severity)": "Issues by severity",
       "New issues": "New issues (any)", "PR size": "New lines (PR size)"}
ct = cat_tab.sort_values("fe_pp", ascending=False)
fig, ax = plt.subplots(figsize=(W, 2.15))
yy = np.arange(len(ct))
ax.errorbar(ct.fe_pp, yy, xerr=[ct.fe_pp - ct.fe_lo, ct.fe_hi - ct.fe_pp], fmt="o", color=BLUE,
            ms=4, elinewidth=1, capsize=0)
ax.axvline(0, color=MUTED, lw=.8, ls="--")
ax.axvline(R["fe_fail_pp"]["pp"], color=ORANGE, lw=.8, ls=":")
ax.set_yticks(yy); ax.set_yticklabels([f"{lab[i]}  (n={int(ct.loc[i,'prs']):,})" for i in ct.index])
ax.set_xlabel("Within-project change in merge probability vs. passing PRs (pp, 95% CI)")
ax.grid(axis="y", visible=False)
xmax = ax.get_xlim()[1]
for j, i in enumerate(ct.index):
    ax.text(xmax + 0.5, j, f"{ct.loc[i,'bypass']*100:.0f}%", va="center", fontsize=6.5, color=INK)
ax.text(xmax + 0.5, len(ct) - 0.2, "override\nrate", va="bottom", fontsize=6.5, color=MUTED)
ax.set_ylim(-0.6, len(ct) + 0.4)
fig.tight_layout(); savefig(fig, "fig_conditions")

# Fig 4: magnitude panels
fig, axs = plt.subplots(2, 2, figsize=(W, 2.9), sharey=True); axs = axs.ravel()


def panel(ax, d, order_, title, xlabel, ycol="merged_i", disp=None):
    gg = d.groupby("bin")[ycol].agg(["sum", "size"]).reindex(order_)
    m = gg["sum"] / gg["size"]; lo, hi = wilson(gg["sum"], gg["size"])
    xx = np.arange(len(order_))
    ax.errorbar(xx, m * 100, yerr=[(m - lo) * 100, (hi - m) * 100], fmt="o-", color=BLUE, ms=3, lw=1, elinewidth=.8)
    ax.set_xticks(xx); ax.set_xticklabels(disp or order_)
    ax.set_title(title, loc="left"); ax.set_xlabel(xlabel)


panel(axs[0], pcv, ["0", "0–20", "20–40", "40–60", "60–70", "70–80", "≥80"], "(a) Coverage",
      "Coverage on new code (%)", disp=["0", "≤20", "≤40", "≤60", "≤70", "≤80", ">80"])
panel(axs[1], pdv, ["≤5", "5–10", "10–20", "20–40", ">40"], "(b) Duplication",
      "Duplication on new code (%)", disp=["≤5", "≤10", "≤20", "≤40", ">40"])
rr = c[c["cat"].isin(["Reliability rating", "Maintainability rating", "Security rating"]) & c.grade.notna()].copy()
for k_, col, mk in [("Reliability rating", BLUE, "o"), ("Maintainability rating", ORANGE, "s"), ("Security rating", "#1baf7a", "^")]:
    gg = rr[rr["cat"] == k_].groupby("grade").merged_i.agg(["sum", "size"]).reindex([2, 3, 4, 5])
    m = gg["sum"] / gg["size"]
    axs[2].plot(np.arange(4), m * 100, marker=mk, color=col, ms=3, lw=1, label=k_.split()[0])
axs[2].set_xticks(range(4)); axs[2].set_xticklabels(list("BCDE")); axs[2].legend(frameon=False, fontsize=6.5, loc="lower center", ncol=3,
               handlelength=1.4, handletextpad=0.4, columnspacing=0.9, borderaxespad=0.2)
axs[2].set_title("(c) Ratings", loc="left"); axs[2].set_xlabel("Observed grade (required A)")
panel(axs[3], hs, ["1", "2", "3–5", "6–10", ">10"], "(d) Hotspots", "Unreviewed hotspots")
axs[0].set_ylabel("Override rate (%)"); axs[2].set_ylabel("Override rate (%)")
for ax in axs: ax.set_ylim(50, 100)
fig.tight_layout(h_pad=.8); savefig(fig, "fig_magnitude")


# ---------------------------------------------------------------------------
def tojson(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return str(o)


(OUT / "results.json").write_text(json.dumps(R, indent=1, default=tojson, ensure_ascii=False), encoding="utf-8")
say("saved results/results.json and results/tables/typology.csv, repo_level.csv, conditions.csv")
import paper_numbers
paper_numbers.write(R, OUT / "paper_numbers.md")
say("saved results/paper_numbers.md (each number in the paper, with its key in results.json)")
print(f"\nDone in {time.time() - T0:.1f}s.")
