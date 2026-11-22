"""SEAL-style enclosing-subgraph link predictor on the functional graph.

Baseline: undirected common-neighbor count on G_f (no training).
Trained model: 1-hop enclosing subgraph, DRNL labels, two GCN layers, MLP.
The query anatomical edge is never inserted into the subgraph.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def common_neighbor_scores(adj: list[set[int]], pairs: np.ndarray) -> np.ndarray:
    src, dst = pairs
    scores = np.empty(src.shape[0], dtype=np.float32)
    for i, (a, b) in enumerate(zip(src.tolist(), dst.tolist())):
        scores[i] = len(adj[a] & adj[b])
    return scores


def enclosing_nodes(adj: list[set[int]], a: int, b: int, hops: int = 1) -> list[int]:
    nodes = {a, b}
    frontier = {a, b}
    for _ in range(hops):
        nxt: set[int] = set()
        for u in frontier:
            nxt |= adj[u]
        nodes |= nxt
        frontier = nxt
    order = [a, b] + sorted(nodes - {a, b})
    return order


def drnl(adj: list[set[int]], nodes: list[int], a: int, b: int) -> np.ndarray:
    """Double-radius node labeling relative to the query pair (a, b)."""

    def dist(src: int) -> dict[int, int]:
        out = {src: 0}
        q = [src]
        for u in q:
            for v in adj[u]:
                if v in out or v not in node_set:
                    continue
                out[v] = out[u] + 1
                q.append(v)
        return out

    node_set = set(nodes)
    da, db = dist(a), dist(b)
    labels = np.zeros(len(nodes), dtype=np.int64)
    for i, u in enumerate(nodes):
        di = da.get(u, 0 if u == a else 99)
        dj = db.get(u, 0 if u == b else 99)
        if u in (a, b):
            labels[i] = 1
        else:
            labels[i] = 1 + min(di, dj) + (di + dj) * (di + dj + 1) // 2
    return labels


def subgraph_tensors(
    adj: list[set[int]],
    node_feat: np.ndarray,
    a: int,
    b: int,
    hops: int,
    max_label: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = enclosing_nodes(adj, a, b, hops)
    idx = {n: i for i, n in enumerate(nodes)}
    labels = drnl(adj, nodes, a, b)
    labels = np.clip(labels, 1, max_label)
    x = np.zeros((len(nodes), max_label + node_feat.shape[1]), dtype=np.float32)
    x[np.arange(len(nodes)), labels - 1] = 1.0
    x[:, max_label:] = node_feat[nodes]
    edges = []
    for u in nodes:
        for v in adj[u]:
            if v not in idx:
                continue
            if {u, v} == {a, b}:
                continue
            edges.append((idx[u], idx[v]))
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.from_numpy(x), edge_index, torch.tensor([0, 1], dtype=torch.long)


class SubgraphGCN(nn.Module):
    def __init__(self, n_in: int, n_hidden: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(n_in, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Sequential(
            nn.Linear(3 * n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, 1),
        )

    def _conv(self, x: torch.Tensor, edge_index: torch.Tensor, lin: nn.Linear) -> torch.Tensor:
        h = lin(x)
        if edge_index.numel() == 0:
            return F.relu(h)
        src, dst = edge_index
        msg = h[src]
        agg = h.new_zeros(h.shape)
        agg.index_add_(0, dst, msg)
        deg = h.new_zeros((h.size(0), 1))
        deg.index_add_(0, dst, h.new_ones((dst.numel(), 1)))
        return F.relu(h + agg / deg.clamp(min=1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        h = self._conv(x, edge_index, self.lin1)
        h = self._conv(h, edge_index, self.lin2)
        pooled = h.mean(dim=0)
        src, dst = h[pair[0]], h[pair[1]]
        return self.out(torch.cat([src, dst, pooled])).squeeze(-1)


def pca_feat(traces: np.ndarray, k: int = 16) -> np.ndarray:
    x = traces.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    feat = (x @ vt[:k].T).astype(np.float32)
    feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-6)
    return feat


class SEAL(nn.Module):
    def __init__(self, n_feat: int, max_label: int = 16, n_hidden: int = 64, hops: int = 1):
        super().__init__()
        self.max_label = max_label
        self.hops = hops
        self.gcn = SubgraphGCN(max_label + n_feat, n_hidden)

    def encode_pair(
        self,
        adj: list[set[int]],
        node_feat: np.ndarray,
        a: int,
        b: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return subgraph_tensors(adj, node_feat, a, b, self.hops, self.max_label)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        return self.gcn(x, edge_index, pair)
