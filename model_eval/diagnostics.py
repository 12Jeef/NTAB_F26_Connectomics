import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def motifs(frame):
    return pd.DataFrame({
        "cascade": frame["cascade"] > 0,
        "fan_out": frame["fan_out"] > 0,
        "fan_in": frame["fan_in"] > 0,
        "reverse": frame["label_rev"] == 1,
    })


def motif_errors(frame, scores):
    """How over-represented each motif is among the top-k false positives (k = #positives), relative to all negatives."""
    k = int(frame["label"].sum())
    base = motifs(frame[frame["label"] == 0]).mean()
    rows = {}
    for s in scores:
        top = frame.nlargest(k, s)
        rows[s] = motifs(top[top["label"] == 0]).mean() / base
    return pd.DataFrame(rows).T


def calibration(frame, s, n_bins=10):
    """Scores must be probabilities at the sampled prevalence; they are shifted to the natural prevalence (Saerens et al. 2002)."""
    y, w = frame["label"], frame["weight"]
    p = frame[s].clip(1e-6, 1 - 1e-6)
    logit = lambda q: np.log(q / (1 - q))
    z = logit(p) + logit(np.average(y, weights=w)) - logit(y.mean())
    fit = LogisticRegression(penalty=None).fit(z.to_frame(), y, sample_weight=w)

    p = 1 / (1 + np.exp(-z))
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    sums = pd.DataFrame({"p": p * w, "y": y * w, "w": w}).groupby(bins).sum()
    curve = pd.DataFrame({"predicted": sums["p"] / sums["w"], "observed": sums["y"] / sums["w"], "weight": sums["w"] / sums["w"].sum()})
    ece = (curve["weight"] * (curve["predicted"] - curve["observed"]).abs()).sum()
    return {"intercept": fit.intercept_[0], "slope": fit.coef_[0, 0], "ece": ece, "curve": curve}
