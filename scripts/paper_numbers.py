"""Write results/paper_numbers.md: every number reported in the paper, next to the
results.json key it comes from, so that reviewers can check each one."""


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _pp(e, d=1):
    """Format a within-project effect (stored as signed pp) as 'X.X pp [lo, hi]'."""
    return f"{e['pp']:+.{d}f} pp (95% CI [{e['lo']:+.{d}f}, {e['hi']:+.{d}f}])"


def write(R, path):
    R = {k: ({str(i): x for i, x in v.items()} if isinstance(v, dict) else v) for k, v in R.items()}
    T = R["typology"]
    C = R["conditions"]
    rows = []

    def add(section, statement, value, key):
        rows.append((section, statement, value, key))

    # ---- Section III-D / Table I: dataset
    s = "III-D, Table I"
    add(s, "Raw PR records / projects", f"{R['raw_rows']:,} / {R['raw_repos']:,}", "raw_rows, raw_repos")
    add(s, "Duplicates removed", f"{R['duplicates_removed']:,}", "duplicates_removed")
    add(s, "Excluded projects (<10% merged) / their PRs", f"{R['excluded_repos']} / {R['excluded_prs']:,}", "excluded_repos, excluded_prs")
    add(s, "Final PRs / projects", f"{R['n_prs']:,} / {R['n_repos']:,}", "n_prs, n_repos")
    add(s, "Projects with >=1 failing PR", f"{R['repos_with_fail']:,}", "repos_with_fail")
    add(s, "Passing / failing PRs", f"{R['n_pass']:,} / {R['n_fail']:,} ({_pct(R['n_fail'] / R['n_prs'])} failing)", "n_pass, n_fail")
    add(s, "Failing PRs with / without condition details", f"{R['n_fail_cond']:,} / {R['n_fail_nocond']:,}", "n_fail_cond, n_fail_nocond")
    add(s, "Parsed failed conditions", f"{R['n_conditions']:,}", "n_conditions")
    nd = {int(k): v for k, v in R["n_cond_dist"].items()}
    add(s, "Failing PRs with 1 / 2 / >=3 conditions", f"{nd[1]:,} / {nd[2]:,} / {sum(v for k, v in nd.items() if k >= 3):,}", "n_cond_dist")
    add(s, "PRs per project: median / max", f"{R['prs_per_repo_median']:.0f} / {R['prs_per_repo_max']:,}", "prs_per_repo_median, prs_per_repo_max")

    # ---- RQ1
    s = "IV-A (RQ1)"
    add(s, "Failing PRs merged (override rate)", f"{R['n_bypassed']:,} ({_pct(R['merge_fail'])})", "n_bypassed, merge_fail")
    add(s, "Failing PRs not merged", f"{R['n_blocked']:,}", "n_blocked")
    add(s, "Passing PRs merged", _pct(R["merge_pass"]), "merge_pass")
    add(s, "Simple difference (passing - failing)", f"{100 * (R['merge_pass'] - R['merge_fail']):.1f} pp", "merge_pass - merge_fail")
    add(s, "Gate effect within projects (Eq. 1, FAIL only)", _pp(R["fe_fail_pp"]), "fe_fail_pp")
    add(s, "Share held back (gate effect / passing merge rate), 'one in seventeen'", _pct(R["gate_yield"]), "gate_yield")
    add(s, "Robustness: without failing PRs lacking condition details", _pp(R["fe_fail_pp_condonly"]), "fe_fail_pp_condonly")
    add(s, "Robustness: including the 24 excluded projects", _pp(R["fe_fail_pp_allrepos"]), "fe_fail_pp_allrepos")
    add(s, "Override rate with / without condition details", f"{_pct(R['merge_fail_cond'])} / {_pct(R['merge_fail_nocond'])}", "merge_fail_cond, merge_fail_nocond")
    add(s, "Projects with >=5 passing and >=5 failing PRs", f"{R['repos_5_5']:,}", "repos_5_5")
    add(s, "Median per-project gate effect (Fig. 1b)", f"{R['delta_median_pp']:.1f} pp", "delta_median_pp")
    add(s, "Projects merging failing PRs at least as often as passing", _pct(R["share_repos_fail_ge_pass"]), "share_repos_fail_ge_pass")

    # ---- RQ2
    s = "IV-B (RQ2)"
    add(s, "Projects merging every / no failing PR", f"{R['repos_bypass100']} / {R['repos_bypass0']}", "repos_bypass100, repos_bypass0")
    add(s, "Median failing PRs of the 100%-override projects", f"{R['repos_bypass100_nf_median']:.0f}", "repos_bypass100_nf_median")
    add(s, "Projects with >=10 failing PRs", f"{R['repos_nf10']:,}", "repos_nf10")
    add(s, "... merging fewer than half / at least 90% of failing PRs", f"{_pct(R['nf10_le50'])} / {_pct(R['nf10_ge90'])}", "nf10_le50, nf10_ge90")
    for prof in ["Enforcing", "Discounting", "Inverted", "Indifferent", "Inconclusive"]:
        t = T[prof]
        add(s + ", Table II", f"{prof}: projects (share), median merge passing / failing",
            f"{t['repos']} ({_pct(t['share_repos'])}), {_pct(t['mp'])} / {_pct(t['mf'])}", f"typology.{prof}")
        add(s + ", Table II", f"{prof}: share of failing / held-back PRs",
            f"{_pct(t['share_fail'])} / {_pct(t['share_blocked'])}", f"typology.{prof}")
    add(s, "Projects with observable thresholds / customised gate", f"{R['repos_obs_config']:,} / {R['repos_custom']} ({_pct(R['repos_custom'] / R['repos_obs_config'])})", "repos_obs_config, repos_custom")
    add(s, "Failed coverage / duplication conditions using the default threshold", f"{_pct(R['share_cov_default'])} / {_pct(R['share_dup_default'])}", "share_cov_default, share_dup_default")
    add(s, "Gate effect, customised gates", _pp(R["fe_by_gate"]["customised"]), "fe_by_gate.customised")
    add(s, "Gate effect, default gates", _pp(R["fe_by_gate"]["default"]), "fe_by_gate.default")
    add(s, "Share of Enforcing projects, customised / default", f"{_pct(R['enforcing_share_by_gate']['customised'])} / {_pct(R['enforcing_share_by_gate']['default'])}", "enforcing_share_by_gate")

    # ---- RQ3
    s = "IV-C (RQ3), Table III"
    for cat, c in C.items():
        add(s, f"{cat}: failing PRs, share, projects, override rate",
            f"{c['prs']:,}, {_pct(c['share_failed'])}, {c['repos']:,}, {_pct(c['bypass'])}", f"conditions.{cat}")
        add(s, f"{cat}: gate effect (Eq. 1, one indicator per condition type)",
            f"{c['fe_pp']:+.1f} pp (95% CI [{c['fe_lo']:+.1f}, {c['fe_hi']:+.1f}])", f"conditions.{cat}.fe_pp")
    bn = R["bypass_by_ncond"]
    add("IV-C (RQ3)", "Override rate with 1 / 2 / 3 / >=4 failed conditions",
        " / ".join(_pct(bn[k][0]) for k in ["1", "2", "3", "4"] if k in bn), "bypass_by_ncond")
    fm = R["fe_multi"]
    add("IV-C (RQ3)", "Within projects: first / second / third-or-more failed condition",
        f"{fm['fail_i']['pp']:+.1f} / {fm['k2']['pp']:+.1f} / {fm['k3']['pp']:+.1f} pp", "fe_multi")
    add("IV-C (RQ3)", "E security rating, uncleaned data: override (n)", f"{_pct(R['secE_with_sandbox'][0])} ({R['secE_with_sandbox'][1]:,})", "secE_with_sandbox")
    add("IV-C (RQ3)", "PRs of the sandbox repository", f"{R['sandbox_secE_prs']:,}", "sandbox_secE_prs")
    add("IV-C (RQ3)", "E security rating without the sandbox: override (n)", f"{_pct(R['secE_without'][0])} ({R['secE_without'][1]:,})", "secE_without")

    # ---- RQ4
    s = "IV-D (RQ4)"
    add(s, "Median coverage gap of failing PRs (PR level)", f"{R['cov_gap_median_pr']:.1f} pp", "cov_gap_median_pr")
    add(s, "Coverage: within-project slope per 10 pp of gap", _pp(R["fe_cov_slope10"]), "fe_cov_slope10")
    add(s, "Coverage: near misses (<=5 pp)", _pp(R["fe_cov_near"]), "fe_cov_near")
    add(s, "PRs failing a coverage condition with 0.0% coverage",
        f"{R['cov_zero_prs']:,} of {R['cov_prs_total']:,} ({_pct(R['cov_zero_prs_share'])})", "cov_zero_prs, cov_prs_total")
    add(s, "Override rate of these PRs", _pct(R["cov_zero_prs_override"]), "cov_zero_prs_override")
    add(s, "Projects with >=5 coverage-failing PRs / with >=90% of them at 0%",
        f"{R['cov_repos5_pr']} / {R['cov_repos_allzero_pr']} ({_pct(R['cov_repos_allzero_pr'] / R['cov_repos5_pr'])})", "cov_repos5_pr, cov_repos_allzero_pr")
    add(s, "Median duplication excess", f"{R['dup_excess_median']:.1f} pp", "dup_excess_median")
    add(s, "Duplication: slope per 10 pp / near misses (<=2 pp)", f"{_pp(R['fe_dup_slope10'], 2)} / {_pp(R['fe_dup_near'])}", "fe_dup_slope10, fe_dup_near")
    for k in ["Reliability rating", "Security rating", "Maintainability rating"]:
        g = R["grades_" + k]
        add(s, f"{k}: override at grade B / C / D / E", " / ".join(_pct(g[x][0]) for x in ["2", "3", "4", "5"] if x in g), f"grades_{k}")
        add(s, f"{k}: within-project change per grade", _pp(R["rating_steps"][k]), f"rating_steps.{k}")
    hb = R["hotspot_bins"]
    add(s, "Hotspots: override with 1 / 2 / 3-5 / 6-10 / >10 hotspots", " / ".join(_pct(hb[x]["mean"]) for x in ["1", "2", "3–5", "6–10", ">10"] if x in hb), "hotspot_bins")
    add(s, "Hotspots: within-project change per doubling", _pp(R["fe_hotspot_log2"]), "fe_hotspot_log2")

    lines = ["# Numbers reported in the paper",
             "",
             "Generated by `scripts/analysis.py`. Each row gives a number as reported in the paper and the key in "
             "`results/results.json` it comes from. Effects are within-project differences in merge probability, "
             "in percentage points (pp), with 95% confidence intervals (cluster-robust by project).",
             "",
             "| Section | Statement | Value | results.json key |",
             "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | `{d}` |" for a, b, c, d in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
