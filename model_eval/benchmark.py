import numpy as np
import pandas as pd
import scipy.sparse as sp

SPLITS = ("train", "val", "test")


def lookup(M, rows, cols):
    return np.asarray(M[rows, cols]).ravel()


def shared(X, Y, rows, cols):
    """Number of k with X[row, k] and Y[col, k]."""
    return np.asarray(X[rows].multiply(Y[cols]).sum(1)).ravel().astype(int)


class Benchmark:
    def __init__(self, ds, split_seed=0, split=(0.7, 0.15, 0.15), bin_um=25.0, neg_per_pos=10, trusted_only=True):
        self.bin_um = bin_um
        self.neg_per_pos = neg_per_pos
        self.trusted_only = trusted_only
        self.neurons = ds.neurons
        self.functional = ds.functional
        self.functional_cols = list(ds.functional.columns)

        ids = ds.neurons.index
        pre = ids.get_indexer(ds.synapses.index.get_level_values("pre_id"))
        post = ids.get_indexer(ds.synapses.index.get_level_values("post_id"))
        known = (pre >= 0) & (post >= 0)
        self.n = sp.csr_matrix((ds.synapses["n"].to_numpy()[known], (pre[known], post[known])), shape=(len(ids),) * 2)
        self.A = (self.n > 0).astype(float).tocsr()
        self.At = self.A.T.tocsr()

        rng = np.random.default_rng(split_seed)
        coreg = rng.permutation(np.flatnonzero(ids.isin(ds.coreg.index)))
        cuts = (np.cumsum(split)[:-1] * len(coreg)).round().astype(int)
        nodes = dict(zip(SPLITS, np.split(coreg, cuts)))
        self.splits = {name: ids[pos] for name, pos in nodes.items()}
        self._pairs = {name: self._build(pos, rng) for name, pos in nodes.items()}

    def pairs(self, split):
        return self._pairs[split]

    def _build(self, nodes, rng):
        out_deg = np.asarray(self.A.sum(1)).ravel()
        in_deg = np.asarray(self.A.sum(0)).ravel()
        trusted = (self.neurons["axon_proof"] != "none").to_numpy()
        xyz = self.neurons[["x", "y", "z"]].to_numpy()

        senders = nodes[trusted[nodes] | (not self.trusted_only)]
        P, Q = (a.ravel() for a in np.meshgrid(senders, nodes, indexing="ij"))
        P, Q = P[P != Q], Q[P != Q]

        n_syn = lookup(self.n, P, Q)
        pos = n_syn > 0
        distance = np.linalg.norm(xyz[P] - xyz[Q], axis=1)
        bins = (distance // self.bin_um).astype(int)
        n_bins = bins.max() + 1

        # Efraimidis–Spirakis: the top k of log(u)/w is a weighted sample without replacement.
        # w = out(i)·in(j) gives degree-corrected negatives (Aiyappa et al. 2025).
        key = np.log(rng.random(len(P))) / np.maximum(out_deg[P] * in_deg[Q], 1e-12)
        rank = pd.Series(np.where(pos, -np.inf, key)).groupby(bins).rank(ascending=False).to_numpy()
        k = self.neg_per_pos * np.bincount(bins[pos], minlength=n_bins)[bins]
        keep = pos | (lookup(self.n, Q, P) > 0) | (rank <= k)

        # Inverse sampling rate per distance bin; ignores the degree weighting.
        n_neg = np.bincount(bins[~pos], minlength=n_bins)
        kept_neg = np.bincount(bins[~pos & keep], minlength=n_bins)
        weight = np.where(pos, 1.0, n_neg[bins] / np.maximum(kept_neg[bins], 1))

        P, Q = P[keep], Q[keep]
        ids = self.neurons.index
        attrs = self.neurons.drop(columns=["x", "y", "z"])
        df = pd.DataFrame({
            "label": pos[keep].astype(int),
            "n_synapses": n_syn[keep],
            "distance_um": distance[keep],
            "distance_bin": bins[keep],
            "out_degree_pre": out_deg[P],
            "in_degree_post": in_deg[Q],
            "trusted": trusted[P],
            "weight": weight[keep],
            "cascade": shared(self.A, self.At, P, Q),
            "fan_out": shared(self.At, self.At, P, Q),
            "fan_in": shared(self.A, self.A, P, Q),
        }, index=pd.MultiIndex.from_arrays([ids[P], ids[Q]], names=["pre_id", "post_id"]))
        df = df.join(attrs.add_suffix("_pre"), on="pre_id").join(attrs.add_suffix("_post"), on="post_id")
        return df.join(self.functional)
