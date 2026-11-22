from dataclasses import dataclass

import numpy as np
import pandas as pd

from model_eval.diagnostics import calibration, motif_errors
from model_eval.metrics import METRICS, PRIMARY, with_reverse
from model_eval.stats import bh, lr_test, node_bootstrap, permuted

MIN_POS = 20
CHANCE = np.array([chance for _, chance in METRICS.values()])


@dataclass(frozen=True)
class Result:
    value: float
    chance: float
    ci: tuple
    p: float
    n_pos: int
    vs: dict


def baselines(pairs, functional_cols):
    return pd.DataFrame({
        "distance": -pairs["distance_um"],
        "degree": pairs["out_degree_pre"] * pairs["in_degree_post"],
        **{f"abs_{c}": pairs[c].abs().fillna(0) for c in functional_cols},
    })


def values(frame, scores):
    return np.array([[fn(frame, s) for fn, _ in METRICS.values()] for s in scores])


def results(frame, scores, n_boot, rng):
    n_pos = int(frame["label"].sum())
    point = np.where(n_pos >= MIN_POS, values(frame, scores), np.nan)
    boot = np.stack([values(f, scores) for f in node_bootstrap(frame, n_boot, rng)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    p = np.minimum(1, 2 * np.minimum((boot <= CHANCE).mean(0), (boot >= CHANCE).mean(0)))
    return pd.DataFrame([
        {"score": s, "metric": m, "value": point[i, j], "chance": CHANCE[j],
         "ci_low": lo[i, j], "ci_high": hi[i, j], "p": p[i, j], "n_pos": n_pos}
        for i, s in enumerate(scores) for j, m in enumerate(METRICS)
    ])


class Report:
    def __init__(self, frame, scores, against, n_boot, seed):
        self.frame = frame
        self.scores = scores
        self.against = against
        self.n_boot = n_boot
        self.rng = np.random.default_rng(seed)
        self._results = results(frame, scores, n_boot, self.rng)

    def to_frame(self):
        return self._results

    def result(self, metric=PRIMARY, score="model"):
        r = self._results.set_index(["metric", "score"]).loc[metric]
        return Result(
            value=r.at[score, "value"], chance=r.at[score, "chance"],
            ci=(r.at[score, "ci_low"], r.at[score, "ci_high"]), p=r.at[score, "p"], n_pos=r.at[score, "n_pos"],
            vs={b: r.at[score, "value"] - r.at[b, "value"] for b in self.scores if b != score},
        )

    def summary(self):
        table = self._results.pivot(index="metric", columns="score", values="value").loc[list(METRICS), self.scores]
        return table.assign(chance=CHANCE)

    def by(self, col):
        out = pd.concat([
            results(f, self.scores, self.n_boot, self.rng).assign(stratum=k) for k, f in self.frame.groupby(col)
        ])
        out["p_adj"] = out.groupby(["score", "metric"])["p"].transform(bh)
        return out

    def incremental(self, score="model"):
        return lr_test(self.frame, score, ["distance", self.against])

    def null(self, n=100, score="model"):
        observed = values(self.frame, [score])[0]
        null = np.stack([values(permuted(self.frame, self.rng), [score])[0] for _ in range(n)])
        return pd.DataFrame({
            "observed": observed,
            "null_mean": null.mean(0),
            "p": (1 + (null >= observed).sum(0)) / (n + 1),
        }, index=list(METRICS))

    def motif_errors(self):
        return motif_errors(self.frame, self.scores)

    def calibration(self, score="model"):
        return calibration(self.frame, score)


def evaluate(bench, scores, split="test", against="abs_strength", n_boot=200, seed=0):
    """`scores`: pd.Series over bench.pairs(split).index, higher = more likely pre→post."""
    pairs = bench.pairs(split)
    s = baselines(pairs, bench.functional_cols).assign(model=scores.reindex(pairs.index))
    assert s["model"].notna().all(), "scores must cover every pair in bench.pairs(split)"
    rev = pairs.index.get_indexer(pairs.index.swaplevel())
    frame = pairs.join(s).reset_index().assign(rev=rev, all=0)
    frame = with_reverse(frame, ["label", *s.columns])
    return Report(frame, ["model", *s.columns.drop("model")], against, n_boot, seed)
