"""Build the functional kNN graph from calcium traces, or Ding means as fallback.

Prefer DANDI fluorescence batches in data/raw/calcium/. If those are missing,
use Ding et al. trial-averaged movie responses. Synapses are not edges.

    python scripts/download_calcium_traces.py   # fetch traces, then this graph
    python scripts/build_functional_knn.py      # rebuild from whatever is on disk
    python scripts/build_functional_knn.py --source ding
"""

import argparse

import numpy as np
import pandas as pd

from microns import config
from microns.graph import knn_edges
from microns.traces import load_batches

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


def write_graph(vectors, ids, matched, source: str, k: int) -> None:
    edges = knn_edges(vectors, ids, k=k)
    edges_path = config.DATA_PROCESSED / "functional_edges.parquet"
    traces_path = config.DATA_PROCESSED / "functional_responses.npz"
    matched_path = config.DATA_PROCESSED / "functional_batch_neurons.parquet"
    edges.to_parquet(edges_path, index=False)
    matched.to_parquet(matched_path, index=False)
    np.savez_compressed(
        traces_path,
        pt_root_id=ids,
        response=np.asarray(vectors, dtype=np.float32),
        source=np.array(source),
    )
    print(f"Functional graph: {len(ids):,} neurons, {len(edges):,} edges, K={k}")
    print(f"  source={source}")
    print(f"Wrote {traces_path}")
    print(f"Wrote {edges_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--source", choices=["auto", "calcium", "ding"], default="auto")
    args = parser.parse_args()

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    nodes = proofread_axon_nodes()
    print(f"{len(nodes):,} neurons with a proofread axon and a calcium unit")

    have_calcium = bool(list(config.DATA_CALCIUM.glob("session*_scan*.npz")))
    use_calcium = args.source == "calcium" or (args.source == "auto" and have_calcium)
    if use_calcium:
        vectors, ids, matched = load_batches()
        print(f"Calcium traces: {len(ids):,} / {len(nodes):,} neurons")
        write_graph(vectors, ids, matched, "dandi_000402_fluorescence", args.k)
        return

    if not DING_NODES.exists():
        raise RuntimeError(
            "No calcium batches and no Ding table. "
            "Run scripts/download_calcium_traces.py"
        )

    vectors, ids, matched = load_ding_batch(nodes)
    print(
        f"Ding in-vivo means: {len(ids):,} / {len(nodes):,} neurons, "
        f"vector length {vectors.shape[1]}"
    )
    write_graph(vectors, ids, matched, "ding_in_vivo_mean_resp", args.k)


if __name__ == "__main__":
    main()
