"""Directed Neural Common Neighbor on the residual functional graph.

Slide 11: two 1D-CNN streams (tuning / residual), GNN on G_f, directed source
and target maps, common-neighbor pooling, pair features = signal and noise
correlation. Anatomical edges are labels only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MERGE_SUM = 1
MERGE_MEAN = 2
N_PAIR_FEAT = 2


class TraceEncoder(nn.Module):
    def __init__(self, n_trace_emb: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, n_trace_emb),
            nn.ReLU(),
        )

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        return self.encoder(traces.unsqueeze(1))


class FunctionalGNNLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, method: int = MERGE_MEAN):
        super().__init__()
        self.method = method
        self.self_linear = nn.Linear(n_in, n_out)
        self.nh_linear = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        messages = x[src]
        nh_sum = x.new_zeros(x.shape)
        nh_sum.index_add_(0, dst, messages)
        degree = x.new_zeros((x.size(0), 1))
        degree.index_add_(0, dst, x.new_ones((dst.numel(), 1)))
        nh = nh_sum if self.method == MERGE_SUM else nh_sum / degree.clamp(min=1)
        return F.relu(self.self_linear(x) + self.nh_linear(nh))


class DirectedNCNPredictor(nn.Module):
    def __init__(self, n_neuron_emb: int, n_pair_feat: int = N_PAIR_FEAT, method: int = MERGE_MEAN):
        super().__init__()
        self.method = method
        self.W_s = nn.Linear(n_neuron_emb, n_neuron_emb)
        self.W_t = nn.Linear(n_neuron_emb, n_neuron_emb)
        self.corr_skip = nn.Linear(n_pair_feat, 1)
        self.mlp = nn.Sequential(
            nn.Linear(3 * n_neuron_emb + n_pair_feat, n_neuron_emb),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_neuron_emb, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        q_pairs: torch.Tensor,
        neighbors: torch.Tensor,
        pair_feat: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = q_pairs
        src_h = self.W_s(h[src])
        dst_h = self.W_t(h[dst])
        common = neighbors[src] & neighbors[dst]
        nh_h = common.to(h.dtype) @ h
        if self.method == MERGE_MEAN:
            nh_h = nh_h / common.sum(dim=1, keepdim=True).clamp(min=1)
        # c_ij (noise, signal corr) is a slide pair feature; skip so the head
        # cannot ignore residual Pearson, which is the current bar to beat.
        gnn_logit = self.mlp(torch.cat([src_h, dst_h, nh_h, pair_feat], dim=-1))
        return (gnn_logit + self.corr_skip(pair_feat)).squeeze(-1)


class DirectedNCN(nn.Module):
    """Two-stream NCN: residual + signal encoders, GNN on G_f, directed decoder."""

    def __init__(
        self,
        n_trace_emb: int = 64,
        n_neuron_emb: int = 64,
        method: int = MERGE_MEAN,
    ):
        super().__init__()
        hidden = 2 * n_trace_emb
        self.residual_encoder = TraceEncoder(n_trace_emb)
        self.signal_encoder = TraceEncoder(n_trace_emb)
        self.feature_linear = nn.Linear(hidden, hidden)
        self.gnn = FunctionalGNNLayer(hidden, n_neuron_emb, method)
        self.predictor = DirectedNCNPredictor(n_neuron_emb, N_PAIR_FEAT, method)

    def encode(self, residual: torch.Tensor, signal: torch.Tensor, f_edges: torch.Tensor) -> torch.Tensor:
        h = torch.cat(
            [self.residual_encoder(residual), self.signal_encoder(signal)],
            dim=-1,
        )
        h = F.relu(self.feature_linear(h))
        return self.gnn(h, f_edges)

    def forward(
        self,
        residual: torch.Tensor,
        signal: torch.Tensor,
        q_pairs: torch.Tensor,
        f_edges: torch.Tensor,
        neighbors: torch.Tensor,
        pair_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(residual, signal, f_edges)
        return self.predictor(h, q_pairs, neighbors, pair_feat)


def neighbor_matrix(n: int, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    mat = torch.zeros((n, n), dtype=torch.bool, device=src.device)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    mat[src, dst] = True
    mat[dst, src] = True
    return mat
