import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model_eval.metrics import with_reverse


def node_bootstrap(frame, n, rng):
    """Resample neurons; a pair appears count(pre)·count(post) times (Snijders & Borgatti 1999)."""
    nodes, codes = np.unique(np.r_[frame["pre_id"], frame["post_id"]], return_inverse=True)
    pre, post = np.split(codes, 2)
    rows = np.arange(len(frame))
    for _ in range(n):
        count = np.bincount(rng.integers(0, len(nodes), len(nodes)), minlength=len(nodes))
        yield frame.iloc[np.repeat(rows, count[pre] * count[post])]


def permuted(frame, rng):
    label = frame.groupby("distance_bin")["label"].transform(lambda y: rng.permutation(y.to_numpy()))
    return with_reverse(frame.assign(label=label), ["label"])


def lr_test(frame, score, covariates):
    def loglik(cols):
        X = StandardScaler().fit_transform(frame[cols])
        p = LogisticRegression(penalty=None, max_iter=1000).fit(X, frame["label"]).predict_proba(X)[:, 1]
        return np.sum(np.where(frame["label"] == 1, np.log(p), np.log1p(-p)))

    stat = max(0.0, 2 * (loglik([*covariates, score]) - loglik(covariates)))
    return {"lr_stat": stat, "p": chi2.sf(stat, df=1)}


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = np.argsort(p[ok])
    q = p[ok][order] * len(ok) / np.arange(1, len(ok) + 1)
    out[ok[order]] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1)
    return out
