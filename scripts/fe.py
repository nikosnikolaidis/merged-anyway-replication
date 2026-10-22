import numpy as np, pandas as pd


def lpm_fe(df, y, xs, group):
    """Linear probability model with group fixed effects (within estimator)
    and cluster-robust (CR1) standard errors by group. Returns pp effects."""
    d = df[[y, group] + xs].dropna().copy()
    d[y] = d[y].astype(float)
    for c in [y] + xs:
        d[c] = d[c].astype(float) - d.groupby(group)[c].transform("mean")
    X = d[xs].to_numpy(); Y = d[y].to_numpy()
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b
    G = d[group].to_numpy()
    meat = np.zeros((len(xs), len(xs)))
    for _, idx in pd.Series(np.arange(len(d))).groupby(G):
        s = X[idx.values].T @ e[idx.values]
        meat += np.outer(s, s)
    n, k, g = len(d), len(xs), d[group].nunique()
    adj = g / (g - 1) * (n - 1) / (n - k - g)
    V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    from scipy.stats import t
    p = 2 * t.sf(np.abs(b / se), g - 1)
    return pd.DataFrame({"pp": b * 100, "lo": (b - 1.96 * se) * 100, "hi": (b + 1.96 * se) * 100, "p": p}, index=xs)
