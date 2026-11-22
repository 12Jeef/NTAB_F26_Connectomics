"""Load the residual kNN graph, oracle traces, and anatomical synapse labels.

Functional edges are G_f (model input). Synapse pairs are labels and are never
added to the input graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microns import config


@dataclass
class LinkTask:
    ids: np.ndarray
    traces: np.ndarray
    signal: np.ndarray
    scan_key: np.ndarray
    func_src: np.ndarray
    func_dst: np.ndarray
    similarity: np.ndarray
    syn_src: np.ndarray
    syn_dst: np.ndarray
    n_synapses: np.ndarray
    id_to_idx: dict[int, int]


def load_link_task() -> LinkTask:
    neurons = pd.read_parquet(config.DATA_PROCESSED / "functional_batch_neurons.parquet")
    fun = pd.read_parquet(config.DATA_PROCESSED / "functional_edges.parquet")
    anat = pd.read_parquet(config.DATA_PROCESSED / "edges.parquet")
    bundle = np.load(config.DATA_PROCESSED / "functional_responses.npz", allow_pickle=True)

    ids = np.asarray(bundle["pt_root_id"])
    traces = np.asarray(bundle["response"], dtype=np.float32)
    signal = (
        np.asarray(bundle["signal"], dtype=np.float32)
        if "signal" in bundle.files
        else traces
    )
    id_to_idx = {int(n): i for i, n in enumerate(ids)}
    keep = neurons["pt_root_id"].isin(id_to_idx)
    neurons = neurons.loc[keep]
    meta = neurons.set_index("pt_root_id").reindex(ids)
    scan_key = (
        meta["session"].astype(str).to_numpy() + "_" + meta["scan_idx"].astype(str).to_numpy()
    )

    fun = fun[fun["pt_root_id"].isin(id_to_idx) & fun["neighbor_pt_root_id"].isin(id_to_idx)]
    fun_ids = set(map(int, ids))
    anat = anat[anat["pre_pt_root_id"].isin(fun_ids) & anat["post_pt_root_id"].isin(fun_ids)]

    return LinkTask(
        ids=ids,
        traces=traces,
        signal=signal,
        scan_key=scan_key,
        func_src=fun["pt_root_id"].map(id_to_idx).to_numpy(np.int64),
        func_dst=fun["neighbor_pt_root_id"].map(id_to_idx).to_numpy(np.int64),
        similarity=fun["similarity"].to_numpy(np.float32),
        syn_src=anat["pre_pt_root_id"].map(id_to_idx).to_numpy(np.int64),
        syn_dst=anat["post_pt_root_id"].map(id_to_idx).to_numpy(np.int64),
        n_synapses=anat["n_synapses"].to_numpy(np.int32),
        id_to_idx=id_to_idx,
    )


def synapse_set(task: LinkTask) -> set[tuple[int, int]]:
    return set(zip(task.syn_src.tolist(), task.syn_dst.tolist()))


def same_scan(task: LinkTask, a: int, b: int) -> bool:
    return str(task.scan_key[a]) == str(task.scan_key[b])


def sample_pairs(
    task: LinkTask,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
    exclude: set[tuple[int, int]] | None = None,
    same_scan_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (2, P) pair indices and (P,) binary labels. Negatives are random non-synapses."""
    exclude = exclude or set()
    pos = []
    for a, b in zip(task.syn_src.tolist(), task.syn_dst.tolist()):
        a, b = int(a), int(b)
        if (a, b) in exclude:
            continue
        if same_scan_only and not same_scan(task, a, b):
            continue
        pos.append((a, b))
    rng.shuffle(pos)
    pos = pos[:n_pos]

    n = len(task.ids)
    syn = synapse_set(task) | exclude
    neg = []
    seen = set(pos)
    keys = task.scan_key
    by_scan: dict[str, np.ndarray] = {}
    if same_scan_only:
        for key in np.unique(keys):
            by_scan[str(key)] = np.flatnonzero(keys == key)

    while len(neg) < n_neg:
        if same_scan_only:
            key = str(rng.choice(list(by_scan)))
            idx = by_scan[key]
            if len(idx) < 2:
                continue
            a, b = (int(x) for x in rng.choice(idx, size=2, replace=False))
        else:
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n))
        if a == b or (a, b) in syn or (a, b) in seen:
            continue
        seen.add((a, b))
        neg.append((a, b))

    pairs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.float32)
    order = rng.permutation(len(pairs))
    pairs = np.array(pairs, dtype=np.int64)[order]
    labels = labels[order]
    return pairs.T, labels


def pearson_scores(traces: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    src, dst = pairs
    x = traces.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return ((x[src] * x[dst]).sum(axis=1) / (norms[src] * norms[dst])).astype(np.float32)


def pair_corr_features(task: LinkTask, pairs: np.ndarray) -> np.ndarray:
    """c_ij = (noise correlation, signal correlation) for each query pair."""
    noise = pearson_scores(task.traces, pairs)
    signal = pearson_scores(task.signal, pairs)
    return np.stack([noise, signal], axis=1)


def neighbor_sets(task: LinkTask, undirected: bool = True) -> list[set[int]]:
    adj = [set() for _ in range(len(task.ids))]
    for a, b in zip(task.func_src.tolist(), task.func_dst.tolist()):
        adj[a].add(b)
        if undirected:
            adj[b].add(a)
    return adj
