"""Draw the current functional kNN graph and write a results table.

Layout of the full graph is the first two principal components of the
activity vectors, so nearby nodes have similar responses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from microns import config

K = 15
BG = "#f7f4ef"
TEAL = "#1d7a6d"
NAVY = "#3d5a80"
CORAL = "#e07a5f"
SLATE = "#6d597a"
AREA_COLORS = {
    "V1": "#1d7a6d",
    "RL": "#3d5a80",
    "AL": "#e07a5f",
    "unknown": "#b8b2a7",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    neurons = pd.read_parquet(config.DATA_PROCESSED / "functional_batch_neurons.parquet")
    edges = pd.read_parquet(config.DATA_PROCESSED / "functional_edges.parquet")
    anat = pd.read_parquet(config.DATA_PROCESSED / "edges.parquet")
    bundle = np.load(config.DATA_PROCESSED / "functional_responses.npz", allow_pickle=True)
    ids = bundle["pt_root_id"]
    responses = bundle["response"].astype(np.float64)
    source = (
        str(bundle["source"])
        if "source" in bundle.files
        else "activity"
    )
    ding_path = config.DATA_RAW / "node_data_v1.pkl"
    if ding_path.exists():
        ding = pd.read_pickle(ding_path).rename(columns={"scan_session": "session"})
        ding["session"] = ding["session"].astype("Int64")
        ding["scan_idx"] = ding["scan_idx"].astype("Int64")
        ding["unit_id"] = ding["unit_id"].astype("Int64")
        meta = ding[["session", "scan_idx", "unit_id", "brain_area", "layer"]].drop_duplicates(
            ["session", "scan_idx", "unit_id"]
        )
        neurons = neurons.merge(meta, on=["session", "scan_idx", "unit_id"], how="left")
    neurons["brain_area"] = neurons.get("brain_area", pd.Series(index=neurons.index)).astype(
        "string"
    ).fillna("unknown")
    return neurons, edges, anat, ids, responses, source


def pca_positions(responses: np.ndarray) -> np.ndarray:
    x = responses - responses.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    xy = x @ vt[:2].T
    xy = xy - np.median(xy, axis=0)
    scale = np.quantile(np.abs(xy), 0.99)
    return xy / scale


def id_map(ids: np.ndarray) -> dict[int, int]:
    return {int(n): i for i, n in enumerate(ids)}


def pick_center(edges: pd.DataFrame, anat: pd.DataFrame, ids: np.ndarray) -> int:
    fun_ids = set(map(int, ids))
    anat_sub = anat[
        anat["pre_pt_root_id"].isin(fun_ids) & anat["post_pt_root_id"].isin(fun_ids)
    ]
    anat_pairs = set(
        zip(anat_sub["pre_pt_root_id"].astype("int64"), anat_sub["post_pt_root_id"].astype("int64"))
    )
    hits = []
    for nid, grp in edges.groupby("pt_root_id"):
        n = sum(
            (int(nid), int(nb)) in anat_pairs for nb in grp["neighbor_pt_root_id"]
        )
        hits.append((n, int(nid)))
    hits.sort(reverse=True)
    return hits[0][1]


def draw_full_graph(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    ids: np.ndarray,
    xy: np.ndarray,
    path: Path,
    source: str,
) -> None:
    index = id_map(ids)
    area = (
        neurons.set_index("pt_root_id")
        .reindex(ids)["brain_area"]
        .fillna("unknown")
        .astype(str)
        .to_numpy()
    )
    colors = [AREA_COLORS.get(a, AREA_COLORS["unknown"]) for a in area]
    src = edges["pt_root_id"].map(index)
    dst = edges["neighbor_pt_root_id"].map(index)
    keep = src.notna() & dst.notna()
    src = src[keep].astype(int).to_numpy()
    dst = dst[keep].astype(int).to_numpy()
    segs = np.stack([xy[src], xy[dst]], axis=1)

    fig, ax = plt.subplots(figsize=(11.2, 10.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.add_collection(
        LineCollection(segs, colors="#8aa0a8", linewidths=0.18, alpha=0.07, zorder=1)
    )
    ax.scatter(xy[:, 0], xy[:, 1], s=22, c=colors, linewidths=0, zorder=2, alpha=0.92)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    n_nodes = len(ids)
    n_edges = len(edges)
    ax.set_title(f"Functional graph  ·  {n_nodes:,} neurons  ·  {n_edges:,} kNN edges", pad=18, fontsize=15)
    caption = (
        "Layout is PCA of in-vivo calcium traces. Nearby nodes have similar activity."
        if "fluorescence" in source or "calcium" in source
        else "Layout is PCA of activity vectors. Nearby nodes have similar activity."
    )
    ax.text(
        0.5,
        1.015,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#5c5a56",
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=8, label=name)
        for name, c in [("V1", AREA_COLORS["V1"]), ("RL", AREA_COLORS["RL"]), ("AL", AREA_COLORS["AL"]), ("unlabeled", AREA_COLORS["unknown"])]
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def draw_one_neuron(
    center: int,
    edges: pd.DataFrame,
    anat: pd.DataFrame,
    path: Path,
    source: str,
) -> None:
    nbrs = edges.loc[edges["pt_root_id"] == center].sort_values("rank")
    fun_pairs = set(zip(nbrs["pt_root_id"].astype("int64"), nbrs["neighbor_pt_root_id"].astype("int64")))
    anat_pairs = set(zip(anat["pre_pt_root_id"].astype("int64"), anat["post_pt_root_id"].astype("int64")))
    n = len(nbrs)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    fig, ax = plt.subplots(figsize=(10.4, 11.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    ring = plt.Circle((0, 0), 1.0, fill=False, color="#ddd6cc", linewidth=1.0)
    ax.add_patch(ring)
    ax.scatter([0], [0], s=620, c=TEAL, zorder=4, linewidths=3.5, edgecolors="white")

    synapse_n = 0
    for i, row in enumerate(nbrs.itertuples(index=False)):
        x, y = np.cos(angles[i]), np.sin(angles[i])
        is_syn = (int(center), int(row.neighbor_pt_root_id)) in anat_pairs
        synapse_n += int(is_syn)
        color = CORAL if is_syn else NAVY
        ax.annotate(
            "",
            xy=(x * 0.86, y * 0.86),
            xytext=(0.12 * x, 0.12 * y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, mutation_scale=10),
        )
        ax.scatter([x], [y], s=210, c=color, zorder=4, linewidths=2.2, edgecolors="white")
        label = f"{int(row.neighbor_pt_root_id) % 10000:04d}"
        ax.text(x * 1.16, y * 1.16, label, ha="center", va="center", fontsize=8, color="#2f2e2c")
        ax.text(
            x * 1.32,
            y * 1.32,
            f"{row.similarity:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color=color,
        )

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.62, 1.55)
    ax.set_title("Functional neighborhood of one neuron", pad=28, fontsize=16)
    fig.text(
        0.5,
        0.935,
        (
            f"{n} nearest neighbors by in-vivo calcium trace  ·  {synapse_n} also share a proofread synapse"
            if "fluorescence" in source or "calcium" in source
            else f"{n} nearest neighbors by activity  ·  {synapse_n} also share a proofread synapse"
        ),
        ha="center",
        fontsize=10,
        color="#5c5a56",
    )
    fig.text(
        0.5,
        0.91,
        "Edge labels are Pearson correlation. Labels are the last four digits of each neuron ID.",
        ha="center",
        fontsize=9,
        color="#7a766f",
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markersize=10, label="This neuron"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NAVY, markersize=9, label="Functional neighbor only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL, markersize=9, label="Also an anatomical synapse"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    _ = fun_pairs


def write_results(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    anat: pd.DataFrame,
    responses: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    nodes = pd.read_parquet(config.DATA_PROCESSED / "nodes.parquet", columns=["pt_root_id"])
    proof = pd.read_parquet(config.DATA_RAW / "proofreading.parquet", columns=["pt_root_id", "status_axon"])
    axon = set(proof.loc[proof["status_axon"].astype(bool), "pt_root_id"])
    imaged = set(nodes["pt_root_id"])
    fun_ids = set(neurons["pt_root_id"])
    axon_imaged = axon & imaged
    sub_anat = anat[anat["pre_pt_root_id"].isin(fun_ids) & anat["post_pt_root_id"].isin(fun_ids)]
    fun_pairs = set(zip(edges["pt_root_id"].astype("int64"), edges["neighbor_pt_root_id"].astype("int64")))
    anat_pairs = set(zip(sub_anat["pre_pt_root_id"].astype("int64"), sub_anat["post_pt_root_id"].astype("int64")))
    overlap = fun_pairs & anat_pairs
    sim = edges["similarity"]
    rows = [
        ("imaged_neurons", len(imaged)),
        ("proofread_axon", len(axon)),
        ("target_imaged_and_proofread_axon", len(axon_imaged)),
        ("functional_neurons", len(fun_ids)),
        ("missing_neurons", len(axon_imaged - fun_ids)),
        ("K", K),
        ("functional_edges", len(edges)),
        ("response_length", int(responses.shape[1])),
        ("similarity_min", float(sim.min())),
        ("similarity_median", float(sim.median())),
        ("similarity_max", float(sim.max())),
        ("anatomical_pairs", len(anat)),
        ("anatomical_synapses", int(anat["n_synapses"].sum())),
        ("anatomical_pairs_in_functional_neurons", len(anat_pairs)),
        ("edges_in_both_graphs", len(overlap)),
        ("fraction_knn_that_are_synapses", len(overlap) / len(fun_pairs)),
        ("fraction_synapses_recovered_as_knn", len(overlap) / len(anat_pairs) if anat_pairs else 0),
    ]
    table = pd.DataFrame(rows, columns=["metric", "value"])
    table.to_csv(path, index=False)
    return table


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    tables = config.DATA_PROCESSED / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    neurons, edges, anat, ids, responses, source = load()
    xy = pca_positions(responses)
    center = pick_center(edges, anat, ids)

    full_path = config.DATA_PROCESSED / "functional_graph_full.png"
    ego_path = config.DATA_PROCESSED / "one_neuron_functional.png"
    results_path = tables / "full_graph_results.csv"

    draw_full_graph(neurons, edges, ids, xy, full_path, source)
    draw_one_neuron(center, edges, anat, ego_path, source)
    table = write_results(neurons, edges, anat, responses, results_path)

    print(f"Center neuron {center}")
    print(table.to_string(index=False))
    print(f"Wrote {full_path}")
    print(f"Wrote {ego_path}")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
