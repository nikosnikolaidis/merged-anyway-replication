"""Parse SonarCloud quality-gate data into PR-level and condition-level tables."""
import re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "quality_gates.csv.gz"
BUILD = ROOT / "build"

NOISE = re.compile(r"!\[.*?\]\(.*?\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
REQ = re.compile(r"\(required\s*(\S+)\s*([A-E]|\d+(?:\.\d+)?)\s*(%?)\)", re.I)
GR = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

# fine-grained condition taxonomy (order matters)
TAX = [
    ("Hotspots review",  r"security hotspots?\b(?!.*rating)"),
    ("Security rating",  r"security (review )?rating|vulnerabilit"),
    ("Reliability rating", r"reliability rating|new bugs?"),
    ("Maintainability rating", r"maintainability rating|technical debt|code smells?"),
    ("Coverage",        r"coverage|lines to cover|conditions to cover|uncovered"),
    ("Duplication",     r"duplicat"),
    ("Issues (severity)", r"new (blocker|critical|major|minor|info) issues?"),
    ("New issues",      r"new issues?"),
    ("PR size",         r"new lines"),
]
# coarse family (ISO 25010-ish)
FAMILY = {
    "Hotspots review": "Security", "Security rating": "Security",
    "Reliability rating": "Reliability", "Maintainability rating": "Maintainability",
    "Issues (severity)": "Issues", "New issues": "Issues",
    "Coverage": "Testing", "Duplication": "Maintainability", "PR size": "Size",
}


def parse_segment(seg):
    s = LINK.sub(r"\1", NOISE.sub("", seg)).replace("���", "≥?").strip()
    cat = next((c for c, p in TAX if re.search(p, s, re.I)), "Other")
    out = dict(text=s, cat=cat, value=np.nan, grade=np.nan, req=np.nan, req_grade=np.nan,
               has_req=False)
    m = REQ.search(s)
    if m:
        out["has_req"] = True
        if m.group(2) in GR:
            out["req_grade"] = GR[m.group(2)]
        else:
            out["req"] = float(m.group(2))
    g = re.match(r"^([A-E])\s", s)
    if g:
        out["grade"] = GR[g.group(1)]
    v = re.match(r"^(\d+(?:\.\d+)?)", s)
    if v and np.isnan(out["grade"]):
        out["value"] = float(v.group(1))
    return out


def load(path=DATA):
    df = pd.read_csv(path, low_memory=False)
    df = df.drop_duplicates(["repo_name", "pr_number"]).reset_index(drop=True)
    df["merged"] = df.is_merged.astype(str).str.upper().eq("TRUE")
    df["fail"] = df.status.eq("FAIL")
    df["pr_id"] = np.arange(len(df))
    rows = []
    for pid, txt in df.loc[df.fail & df.failed_conditions.notna(), ["pr_id", "failed_conditions"]].itertuples(index=False):
        for seg in str(txt).split(";"):
            if seg.strip():
                r = parse_segment(seg); r["pr_id"] = pid; rows.append(r)
    cond = pd.DataFrame(rows)
    cond = cond.merge(df[["pr_id", "repo_name", "merged"]], on="pr_id")
    cond["family"] = cond["cat"].map(FAMILY).fillna("Other")
    # threshold-relative margin (how far beyond the threshold), in natural units
    m = pd.Series(np.nan, index=cond.index)
    cv = cond["cat"].eq("Coverage"); dp = cond["cat"].eq("Duplication")
    m[cv] = cond.req[cv] - cond.value[cv]          # pp below required coverage
    m[dp] = cond.value[dp] - cond.req[dp]          # pp above allowed duplication
    gr = cond.grade.notna() & cond.req_grade.notna()
    m[gr] = cond.grade[gr] - cond.req_grade[gr]    # rating steps below required
    cond["margin"] = m
    df["n_cond"] = df.pr_id.map(cond.groupby("pr_id").size()).fillna(0).astype(int)
    return df, cond


if __name__ == "__main__":
    BUILD.mkdir(exist_ok=True)
    df, cond = load()
    df.drop(columns=["failed_conditions"]).to_parquet(BUILD / "prs.parquet")
    cond.to_parquet(BUILD / "conds.parquet")
    print(df.shape, cond.shape)
    print(cond["cat"].value_counts())
