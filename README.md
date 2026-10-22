# Replication package: *Merged Anyway: How Open-Source Projects Respond When Quality Gates Fail*

This package contains the dataset, the condition parser, and the analysis scripts that reproduce every
number, table, and figure in the paper, together with the results they produce.

## Contents

```
.
├── data/
│   └── quality_gates.csv.gz      # 1,636,609 closed PRs from 3,497 projects (raw, before cleaning)
├── scripts/
│   ├── prep.py                   # parses the SonarQube Cloud bot comments into PR- and condition-level tables
│   ├── fe.py                     # linear probability model with project fixed effects, cluster-robust (CR1) SEs
│   ├── analysis.py               # all analyses (RQ1-RQ4), tables, and figures
│   └── paper_numbers.py          # writes results/paper_numbers.md
├── results/                      # output of scripts/analysis.py (included, so nothing needs to be run)
│   ├── paper_numbers.md          # every number reported in the paper, with its source key in results.json
│   ├── results.json              # all computed quantities
│   ├── analysis_log.txt          # console output of the run that produced these results
│   ├── tables/                   # typology.csv (Table II), conditions.csv (Table III), repo_level.csv
│   └── figures/                  # Fig. 1-4 (PDF as used in the paper, PNG for quick viewing)
├── requirements.txt
└── README.md
```

## How to reproduce

Requirements: Python 3.10 or newer (tested with Python 3.13).

```bash
pip install -r requirements.txt
python scripts/analysis.py
```

The script can be started from any directory. On the first run it calls `prep.py`, which parses the raw data
and stores two intermediate files in `build/`; later runs reuse them. A full run takes about 30 seconds on a
laptop and regenerates everything in `results/`. To start from scratch, delete `build/` and `results/`.

## Data

`data/quality_gates.csv.gz` (gzip-compressed CSV, read directly by pandas) has one row per closed pull request:

| Column | Description |
|---|---|
| `repo_name` | GitHub repository (`owner/name`) |
| `pr_number` | Pull-request number within the repository |
| `status` | Quality-gate verdict of the most recent SonarQube Cloud bot comment on the PR: `PASS` or `FAIL` |
| `is_merged` | Whether the PR was merged (GitHub `merged` flag) |
| `failed_conditions` | Failed conditions as listed in the bot comment, separated by `;` (empty for passing PRs and for failing PRs whose comment lists no conditions) |
| `pr_url` | Link to the pull request |

**Collection.** Candidate projects were selected from GH Archive using Google BigQuery: all repositories in
which the SonarQube Cloud bot (`sonarqubecloud[bot]`) authored more than 100 issue or review comments during
2025. For each project, all closed PRs were enumerated with the GitHub REST API; PRs authored by
`dependabot[bot]` and `github-actions[bot]` were skipped. For every remaining PR, the most recent comment by
`sonarcloud[bot]` or `sonarqubecloud[bot]` was kept, and PRs without such a comment were discarded. Data were
collected between February and March 2026.

**Cleaning (done by the scripts, not in the file).** The file is the raw collection. `analysis.py` removes
duplicate records (536) and excludes the 24 projects in which fewer than 10% of all PRs were merged, which
yields the analysis set of 1,630,607 PRs from 3,473 projects (Table I).

All data come from public GitHub repositories and public bot comments. The file contains no personal data
beyond repository names and links to public pull requests.

## Mapping to the paper

| Paper | Produced by | Output |
|---|---|---|
| Table I (dataset) | `analysis.py`, step 1 | `results.json` (`raw_rows` ... `n_cond_dist`) |
| RQ1, Fig. 1 | step 2 | `results.json` (`fe_fail_pp`, ...), `figures/fig_gate_effect.pdf` |
| RQ2, Table II, Fig. 2 | step 3 | `tables/typology.csv`, `tables/repo_level.csv`, `figures/fig_projects.pdf` |
| RQ3, Table III, Fig. 3 | step 4 | `tables/conditions.csv`, `figures/fig_conditions.pdf` |
| RQ4, Fig. 4 | step 5 | `results.json` (`cov_*`, `dup_*`, `rating_steps`, `fe_hotspot_log2`), `figures/fig_magnitude.pdf` |

`results/paper_numbers.md` lists each number of the paper next to the key it comes from.
`results/tables/repo_level.csv` contains, for each of the 1,890 classified projects, the merge counts of passing
and failing PRs, the Fisher test p-value, the Benjamini-Hochberg q-value, the equivalence-test (TOST) p-value,
and the assigned enforcement profile.

`results.json` also contains a few quantities that were computed during the analysis but are not reported in
the paper (e.g., a chi-square test, a Mantel-Haenszel odds ratio, and empirical-Bayes estimates of project
override rates). They are kept for completeness and do not affect any reported result.


