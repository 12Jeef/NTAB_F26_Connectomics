"""Build the functional input graph in batches.

The first batch uses trial-averaged in vivo responses from Ding et al. 2025
(node_data_v1.pkl, 280 MB). Each neuron has a 120-sample mean response to
naturalistic movies. The graph connects each neuron to its K nearest neighbors
by correlation of those vectors. Synapses are not edges.

This is the signal-correlation graph. The residual (noise-correlation) graph
still needs per-repeat traces from microns_phase3_nda, which can be added as
later batches without changing the kNN step.

Outputs (data/processed/):
  functional_responses.npz   activity vectors for this batch
  functional_edges.parquet   directed k-nearest-neighbor edges
"""

import argparse

import numpy as np
import pandas as pd

from microns import config
from microns.graph import knn_edges

K = 15
DING_NODES = config.DATA_RAW / "node_data_v1.pkl"


def proofread_axon_nodes() -> pd.DataFrame:
    nodes = pd.read_parquet(
        config.DATA_PROCESSED / "nodes.parquet",
        columns=["pt_root_id", "session", "scan_idx", "unit_id"],
    )
    proof = pd.read_parquet(
        config.DATA_RAW / "proofreading.parquet",
        columns=["pt_root_id", "status_axon"],
    )
    axon = proof.loc[proof["status_axon"].astype(bool), "pt_root_id"]
    return nodes[nodes["pt_root_id"].isin(set(axon))].reset_index(drop=True)


def stack_vectors(series: pd.Series) -> np.ndarray:
    rows = [np.asarray(v, dtype=np.float64).ravel() for v in series]
    length = min(len(r) for r in rows)
    return np.stack([r[:length] for r in rows])


def load_ding_batch(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ding = pd.read_pickle(DING_NODES).rename(columns={"scan_session": "session"})
    ding["session"] = ding["session"].astype("Int64")
    ding["scan_idx"] = ding["scan_idx"].astype("Int64")
    ding["unit_id"] = ding["unit_id"].astype("Int64")
    matched = nodes.merge(
        ding[["session", "scan_idx", "unit_id", "in_vivo_mean_resp", "brain_area", "layer"]],
        on=["session", "scan_idx", "unit_id"],
        how="inner",
    )
    if matched.empty:
        raise RuntimeError("No proofread-axon neurons matched the Ding response table")
    vectors = stack_vectors(matched["in_vivo_mean_resp"])
    ids = matched["pt_root_id"].to_numpy()
    return vectors, ids, matched.drop(columns=["in_vivo_mean_resp"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    nodes = proofread_axon_nodes()
    print(f"{len(nodes):,} neurons with a proofread axon and a calcium unit")

    if not DING_NODES.exists():
        raise RuntimeError(
            f"Missing {DING_NODES}. Download node_data_v1.pkl from the "
            "MICrONS functional connectomics release."
        )

    vectors, ids, matched = load_ding_batch(nodes)
    print(
        f"Batch 1 (trial-averaged in vivo responses): {len(ids):,} / {len(nodes):,} neurons"
    )
    print(f"  vector length {vectors.shape[1]}, scans {matched.groupby(['session','scan_idx']).ngroups}")

    edges = knn_edges(vectors, ids, k=args.k)
    edges_path = config.DATA_PROCESSED / "functional_edges.parquet"
    traces_path = config.DATA_PROCESSED / "functional_responses.npz"
    matched_path = config.DATA_PROCESSED / "functional_batch_neurons.parquet"
    edges.to_parquet(edges_path, index=False)
    matched.to_parquet(matched_path, index=False)
    np.savez_compressed(traces_path, pt_root_id=ids, response=vectors.astype(np.float32))
    print(f"Functional graph: {len(ids):,} neurons, {len(edges):,} edges, K={args.k}")
    print("  Similarity is signal correlation of 120-sample mean movie responses.")
    print(f"Wrote {traces_path}")
    print(f"Wrote {edges_path}")


if __name__ == "__main__":
    main()
