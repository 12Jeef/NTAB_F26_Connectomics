"""Measure whether functional similarity lines up with proofread synapses.

This is the check NCN needs before claiming the calcium graph is informative.
"""

from __future__ import annotations

import json

import numpy as np

from microns import config
from microns.models.data import load_link_task, neighbor_sets, pearson_scores, synapse_set
from microns.models.metrics import summarize
from microns.models.seal import common_neighbor_scores


def main() -> None:
    task = load_link_task()
    syn = synapse_set(task)
    n = len(task.ids)
    rng = np.random.default_rng(0)

    pos = np.array(list(syn), dtype=np.int64).T
    neg_pairs, _ = _negatives(task, syn, len(syn), rng)
    pairs = np.concatenate([pos, neg_pairs], axis=1)
    labels = np.concatenate(
        [np.ones(pos.shape[1], dtype=int), np.zeros(neg_pairs.shape[1], dtype=int)]
    )

    knn = set(zip(task.func_src.tolist(), task.func_dst.tolist()))
    knn_score = np.array([1.0 if (int(a), int(b)) in knn else 0.0 for a, b in pairs.T])
    pearson = pearson_scores(task.traces, pairs)
    cn = common_neighbor_scores(neighbor_sets(task), pairs)

    rows = {
        "n_neurons": n,
        "n_functional_edges": int(len(task.func_src)),
        "n_anatomical_pairs": int(len(syn)),
        "knn_membership": summarize(labels, knn_score),
        "pearson_trace": summarize(labels, pearson, threshold=float(np.median(pearson))),
        "common_neighbors": summarize(labels, cn, threshold=1.0),
        "mean_pearson_synapse": float(pearson[: pos.shape[1]].mean()),
        "mean_pearson_nonsynapse": float(pearson[pos.shape[1] :].mean()),
    }
    out = config.DATA_PROCESSED / "tables" / "functional_vs_anatomy_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"Wrote {out}")


def _negatives(task, syn, n_neg, rng):
    n = len(task.ids)
    neg = []
    seen = set()
    while len(neg) < n_neg:
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if a == b or (a, b) in syn or (a, b) in seen:
            continue
        seen.add((a, b))
        neg.append((a, b))
    pairs = np.array(neg, dtype=np.int64).T
    return pairs, np.zeros(n_neg)


if __name__ == "__main__":
    main()
