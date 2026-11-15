"""Pairwise residual-similarity map with anatomical overlay.

Similarity is Pearson of oracle-repeat residuals (noise correlation), the
same definition as G_f. Cross-scan entries are blank: those cells were not
recorded together. Anatomical synapses are proofread directed pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from microns import config

BG = "#f7f4ef"


def cosine_map(traces: np.ndarray) -> np.ndarray:
    x = np.asarray(traces, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    x = x / norms
    return x @ x.T


def block_cosine(traces: np.ndarray, scan_key: np.ndarray) -> np.ndarray:
    sim = np.full((len(traces), len(traces)), np.nan, dtype=np.float64)
    for key in np.unique(scan_key):
        idx = np.flatnonzero(scan_key == key)
        if idx.size < 2:
            continue
        block = cosine_map(traces[idx])
        sim[np.ix_(idx, idx)] = block
    return sim


def ordered_ids(neurons: pd.DataFrame, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta = neurons.set_index("pt_root_id").reindex(ids)
    session = meta["session"].to_numpy()
    scan = meta["scan_idx"].to_numpy()
    unit = meta["unit_id"].to_numpy()
    order = np.lexsort((unit, scan, session))
    scan_key = session.astype(str) + "_" + scan.astype(str)
    return order, scan_key[order]


def scan_breaks(scan_key: np.ndarray) -> np.ndarray:
    change = np.flatnonzero(scan_key[1:] != scan_key[:-1]) + 1
    return change


def synapse_matrix(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float32)
    a[src, dst] = 1.0
    return a


def draw_breaks(ax, breaks: np.ndarray, n: int) -> None:
    for b in breaks:
        ax.axhline(b - 0.5, color="white", linewidth=0.4, alpha=0.85)
        ax.axvline(b - 0.5, color="white", linewidth=0.4, alpha=0.85)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)


def colorbar(fig, ax, image, label: str, ticks=None) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    fig.colorbar(image, cax=cax, label=label, ticks=ticks)


def main() -> None:
    neurons = pd.read_parquet(config.DATA_PROCESSED / "functional_batch_neurons.parquet")
    anat = pd.read_parquet(config.DATA_PROCESSED / "edges.parquet")
    bundle = np.load(config.DATA_PROCESSED / "functional_responses.npz", allow_pickle=True)
    ids = np.asarray(bundle["pt_root_id"])
    traces = bundle["response"]
    order, scan_key = ordered_ids(neurons, ids)
    ids = ids[order]
    traces = traces[order]
    id_to_idx = {int(n): i for i, n in enumerate(ids)}
    breaks = scan_breaks(scan_key)

    fun_ids = set(map(int, ids))
    anat = anat[anat["pre_pt_root_id"].isin(fun_ids) & anat["post_pt_root_id"].isin(fun_ids)]
    src = anat["pre_pt_root_id"].map(id_to_idx).to_numpy(np.int64)
    dst = anat["post_pt_root_id"].map(id_to_idx).to_numpy(np.int64)

    source = str(bundle["source"]) if "source" in bundle.files else "activity"
    sim = block_cosine(traces, scan_key)
    np.fill_diagonal(sim, np.nan)
    adj = synapse_matrix(len(ids), src, dst)
    np.fill_diagonal(adj, np.nan)
    n = len(ids)

    valid = np.isfinite(sim)
    syn_mask = (adj == 1) & valid
    nonsyn_mask = (adj != 1) & valid
    mean_syn = float(np.nanmean(sim[syn_mask])) if syn_mask.any() else float("nan")
    mean_nonsyn = float(np.nanmean(sim[nonsyn_mask])) if nonsyn_mask.any() else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    im0 = axes[0].imshow(sim, cmap="RdBu_r", vmin=-0.4, vmax=0.8, interpolation="nearest")
    draw_breaks(axes[0], breaks, n)
    axes[0].set_title("Residual noise correlation", fontsize=11)
    axes[0].set_xlabel("neurons, ordered by session / scan")
    colorbar(fig, axes[0], im0, "cosine")

    im1 = axes[1].imshow(adj, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    draw_breaks(axes[1], breaks, n)
    axes[1].set_title("Anatomical synapses", fontsize=11)
    axes[1].set_xlabel("directed proofread pairs")
    colorbar(fig, axes[1], im1, "synapse", ticks=[0, 1])

    tri_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    tri_lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    upper = np.ma.masked_where(~tri_upper, sim)
    lower = np.ma.masked_where(~tri_lower, adj)
    im2 = axes[2].imshow(upper, cmap="RdBu_r", vmin=-0.4, vmax=0.8, interpolation="nearest")
    axes[2].imshow(lower, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    draw_breaks(axes[2], breaks, n)
    axes[2].set_title("Overlay  ·  upper residual / lower synapse", fontsize=11)
    axes[2].set_xlabel("same neuron order")
    colorbar(fig, axes[2], im2, "cosine (upper)")

    fig.suptitle(
        f"Pairwise maps  ·  {n:,} neurons  ·  {int(np.nansum(adj)):,} anatomical pairs",
        fontsize=13,
        y=1.03,
    )
    fig.text(
        0.5,
        -0.06,
        f"Similarity = within-scan Pearson of oracle-repeat residuals ({source}). "
        "Anatomy = proofread directed synapses (pre axon, post dendrite, both coregistered). "
        f"Mean residual corr on synapses {mean_syn:.3f} vs non-synapses {mean_nonsyn:.3f}.",
        ha="center",
        fontsize=8,
        color="#5c5a56",
    )
    out = config.DATA_PROCESSED / "correlation_anatomy_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"cosine off-diagonal mean {np.nanmean(sim):.4f}  median {np.nanmedian(sim):.4f}")
    print(f"mean cosine synapses {mean_syn:.4f}  non-synapses {mean_nonsyn:.4f}")
    print(f"anatomical pairs plotted {int(np.nansum(adj)):,}")
    print(f"scan blocks {len(breaks) + 1}")


if __name__ == "__main__":
    main()
