"""Every metric is f(frame, score_column) -> float. Grouped metrics are computed per group and pooled weighted by positives."""

from functools import partial

import numpy as np
from scipy.special import digamma


def with_reverse(frame, cols):
    rev = frame["rev"].to_numpy()
    return frame.assign(**{f"{c}_rev": np.where(rev >= 0, frame[c].to_numpy(float)[rev], np.nan) for c in cols})


def pool(values, n_pos, ok):
    w = n_pos.where(ok, 0)
    return (values.where(ok, 0) * w).sum() / w.sum()


def ranked(frame, s, by):
    d = frame.sort_values(s, ascending=False, kind="stable")
    g = d.groupby(by)
    return d, g, g["label"].sum(), g.size()


def random_ap(n_pos, n):
    """Exact E[AP] of a random ranking; exceeds π when there are few positives."""
    h = digamma(n + 1) + np.euler_gamma
    return (h + (n_pos - 1) / (n - 1) * (n - h)) / n


def ap_ratio(frame, s, by="all"):
    d, g, n_pos, n = ranked(frame, s, by)
    precision = g["label"].cumsum() / (g.cumcount() + 1)
    ap = (precision * d["label"]).groupby(d[by]).sum() / n_pos
    return pool(ap / random_ap(n_pos, n), n_pos, (n_pos > 0) & (n_pos < n))


def epr(frame, s, by="all"):
    d, g, n_pos, n = ranked(frame, s, by)
    top = g.cumcount() < g["label"].transform("sum")
    precision = (d["label"] * top).groupby(d[by]).sum() / n_pos
    return pool(precision / (n_pos / n), n_pos, (n_pos > 0) & (n_pos < n))


def auroc(frame, s, by="all"):
    """Mann–Whitney: (rank sum of positives − n₊(n₊+1)/2) / (n₊n₋)."""
    g = frame.groupby(by)
    n_pos, n = g["label"].sum(), g.size()
    rank_sum = (g[s].rank() * frame["label"]).groupby(frame[by]).sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * (n - n_pos))
    return pool(auc, n_pos, (n_pos > 0) & (n_pos < n))


def direction(frame, s):
    one_way = (frame["label"] == 1) & (frame["label_rev"] == 0)
    a, b = frame.loc[one_way, s], frame.loc[one_way, f"{s}_rev"]
    return ((a > b) + 0.5 * (a == b)).mean()


def strength(frame, s):
    pos = frame[frame["label"] == 1]
    return pos[s].corr(pos["n_synapses"], method="spearman")


def residual_distance(frame, s, width):
    bins = frame["distance_um"] // width
    r = frame[[s, "distance_um"]].groupby(bins).rank(pct=True)
    r = r - r.groupby(bins).transform("mean")
    return r[s].corr(r["distance_um"])


METRICS = {  # name -> (fn, chance)
    "ap_ratio_binned": (partial(ap_ratio, by="distance_bin"), 1.0),
    "ap_ratio": (ap_ratio, 1.0),
    "epr": (epr, 1.0),
    "auroc": (auroc, 0.5),
    "auroc_binned": (partial(auroc, by="distance_bin"), 0.5),
    "auroc_sender": (partial(auroc, by="pre_id"), 0.5),
    "auroc_receiver": (partial(auroc, by="post_id"), 0.5),
    "direction": (direction, 0.5),
    "strength": (strength, 0.0),
    "residual_distance_10": (partial(residual_distance, width=10.0), 0.0),
    "residual_distance_25": (partial(residual_distance, width=25.0), 0.0),
    "residual_distance_50": (partial(residual_distance, width=50.0), 0.0),
}
PRIMARY = "ap_ratio_binned"
