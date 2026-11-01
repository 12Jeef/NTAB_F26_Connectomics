"""Join calcium units onto EM neurons, and build graphs on that node list.

``build_nodes`` makes one node per EM neuron. ``build_edges`` attaches
proofread synapses, which are the anatomical target graph. ``knn_edges``
builds the functional input graph: each neuron points to the ``k`` other
neurons with the most similar residual activity.
"""

import numpy as np
import pandas as pd

from microns import config


def build_nodes(coreg: pd.DataFrame) -> pd.DataFrame:
    """Return one row per EM neuron, with its chosen calcium unit."""
    df = coreg.loc[coreg["pt_root_id"] != 0].copy()
    manual_table = config.COREG_TABLES[0]
    df["manual"] = df["coreg_source"].eq(manual_table)
    df["unit_key"] = (
        df["session"].astype(str)
        + "_"
        + df["scan_idx"].astype(str)
        + "_"
        + df["unit_id"].astype(str)
    )
    df = df.sort_values(
        ["pt_root_id", "manual", "score"],
        ascending=[True, False, False],
        kind="mergesort",
    )

    units = (
        df.groupby("pt_root_id", sort=False)["unit_key"]
        .agg(n_units="size", unit_keys=list)
        .reset_index()
    )
    primary = df.drop_duplicates("pt_root_id", keep="first")
    nodes = primary.merge(units, on="pt_root_id", how="left")
    columns = [
        "pt_root_id",
        "session",
        "scan_idx",
        "unit_id",
        "field",
        "coreg_source",
        "score",
        "residual",
        "n_units",
        "unit_keys",
    ]
    return nodes[columns].reset_index(drop=True)


def build_edges(pairs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Join directed synapse pairs onto the chosen calcium unit of each node."""
    pre = nodes.rename(
        columns={
            "pt_root_id": "pre_pt_root_id",
            "session": "pre_session",
            "scan_idx": "pre_scan_idx",
            "unit_id": "pre_unit_id",
        }
    )[["pre_pt_root_id", "pre_session", "pre_scan_idx", "pre_unit_id"]]
    post = nodes.rename(
        columns={
            "pt_root_id": "post_pt_root_id",
            "session": "post_session",
            "scan_idx": "post_scan_idx",
            "unit_id": "post_unit_id",
        }
    )[["post_pt_root_id", "post_session", "post_scan_idx", "post_unit_id"]]

    edges = pairs.merge(pre, on="pre_pt_root_id", how="inner")
    edges = edges.merge(post, on="post_pt_root_id", how="inner")
    columns = [
        "pre_pt_root_id",
        "post_pt_root_id",
        "pre_session",
        "pre_scan_idx",
        "pre_unit_id",
        "post_session",
        "post_scan_idx",
        "post_unit_id",
        "n_synapses",
        "total_size",
    ]
    return edges[columns].reset_index(drop=True)


def knn_edges(residual: np.ndarray, node_ids: np.ndarray, k: int = 15) -> pd.DataFrame:
    """Directed k-nearest neighbors from residual-activity correlation.

    ``residual`` is one row per neuron. An edge ``i -> j`` means ``j`` is among
    the ``k`` neurons most correlated with ``i``. The diagonal is excluded.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    x = np.asarray(residual, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1)
    keep = norms > 0
    x = x[keep]
    node_ids = np.asarray(node_ids)[keep]
    norms = norms[keep]
    corr = (x @ x.T) / np.outer(norms, norms)
    np.fill_diagonal(corr, -np.inf)

    k = min(k, corr.shape[0] - 1)
    neighbors = np.argpartition(corr, -k, axis=1)[:, -k:]
    rows = []
    for i, cols in enumerate(neighbors):
        cols = cols[np.argsort(corr[i, cols])[::-1]]
        for rank, j in enumerate(cols, start=1):
            rows.append(
                {
                    "pt_root_id": node_ids[i],
                    "neighbor_pt_root_id": node_ids[j],
                    "similarity": float(corr[i, j]),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def mark_connected_nodes(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Flag nodes that appear in at least one proofread synapse edge."""
    connected = set(edges["pre_pt_root_id"]) | set(edges["post_pt_root_id"])
    nodes = nodes.copy()
    nodes["has_proofread_edge"] = nodes["pt_root_id"].isin(connected)
    return nodes
