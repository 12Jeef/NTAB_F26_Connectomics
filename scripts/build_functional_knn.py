"""Build the functional kNN graph G_f from oracle-repeat residuals.

Default (the NT@B / Nature definition):
  six 10 s clips × ten repeats → trial average (signal) and trial-minus-mean
  (residual). G_f is a directed kNN of residual noise correlation, computed
  only among neurons recorded in the same scan.

Ablations:
    python scripts/build_functional_knn.py                  # oracle residual
    python scripts/build_functional_knn.py --source raw     # full-trace Pearson
    python scripts/build_functional_knn.py --source ding
    python scripts/download_oracle_traces.py                # fetch oracle windows
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from microns import config
from microns.graph import knn_edges, knn_edges_grouped
from microns.oracle import download_oracle_scans, load_oracle_batches
from microns.traces import load_batches

K = 15
DING_NODES = config.DATA_RAW / "node_data_v1.pkl"
SOURCE_ORACLE = "dandi_000402_oracle_residual"
SOURCE_RAW = "dandi_000402_fluorescence"
SOURCE_DING = "ding_in_vivo_mean_resp"


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


def write_graph(
    vectors: np.ndarray,
    ids: np.ndarray,
    matched: pd.DataFrame,
    source: str,
    k: int,
    signal: np.ndarray | None = None,
    grouped: bool = False,
) -> None:
    if grouped:
        groups = (
            matched.set_index("pt_root_id")
            .reindex(ids)[["session", "scan_idx"]]
            .astype(str)
            .agg("_".join, axis=1)
            .to_numpy()
        )
        edges = knn_edges_grouped(vectors, ids, groups, k=k)
    else:
        edges = knn_edges(vectors, ids, k=k)
    edges_path = config.DATA_PROCESSED / "functional_edges.parquet"
    traces_path = config.DATA_PROCESSED / "functional_responses.npz"
    matched_path = config.DATA_PROCESSED / "functional_batch_neurons.parquet"
    edges.to_parquet(edges_path, index=False)
    matched.to_parquet(matched_path, index=False)
    payload = {
        "pt_root_id": ids,
        "response": np.asarray(vectors, dtype=np.float32),
        "source": np.array(source),
    }
    if signal is not None:
        payload["signal"] = np.asarray(signal, dtype=np.float32)
    np.savez_compressed(traces_path, **payload)
    same = 0
    if grouped and len(edges):
        meta = matched.set_index("pt_root_id")
        left = meta.reindex(edges["pt_root_id"])
        right = meta.reindex(edges["neighbor_pt_root_id"])
        same = int(
            (
                (left["session"].to_numpy() == right["session"].to_numpy())
                & (left["scan_idx"].to_numpy() == right["scan_idx"].to_numpy())
            ).sum()
        )
    print(f"Functional graph: {len(ids):,} neurons, {len(edges):,} edges, K={k}")
    print(f"  source={source}")
    if grouped:
        print(f"  within-scan edges {same:,} / {len(edges):,}")
    print(f"Wrote {traces_path}")
    print(f"Wrote {edges_path}")


def write_oracle_graph(k: int = K) -> None:
    residual, ids, matched, signal = load_oracle_batches()
    print(f"Oracle residuals: {len(ids):,} neurons, vector length {residual.shape[1]}")
    write_graph(residual, ids, matched, SOURCE_ORACLE, k, signal=signal, grouped=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument(
        "--source",
        choices=["oracle", "raw", "calcium", "ding", "auto"],
        default="oracle",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download missing oracle windows from DANDI before building",
    )
    args = parser.parse_args()

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    nodes = proofread_axon_nodes()
    print(f"{len(nodes):,} neurons with a proofread axon and a calcium unit")

    source = args.source
    if source == "auto":
        source = "oracle"

    if source == "oracle":
        if args.fetch:
            download_oracle_scans()
        try:
            write_oracle_graph(args.k)
        except FileNotFoundError:
            print("Oracle batches missing; fetching from DANDI")
            download_oracle_scans()
            write_oracle_graph(args.k)
        return

    if source in {"raw", "calcium"}:
        vectors, ids, matched = load_batches()
        print(f"Raw fluorescence: {len(ids):,} / {len(nodes):,} neurons")
        write_graph(vectors, ids, matched, SOURCE_RAW, args.k, grouped=False)
        return

    if not DING_NODES.exists():
        raise RuntimeError("No Ding table. Run scripts/download_oracle_traces.py")
    vectors, ids, matched = load_ding_batch(nodes)
    print(
        f"Ding in-vivo means: {len(ids):,} / {len(nodes):,} neurons, "
        f"vector length {vectors.shape[1]}"
    )
    write_graph(vectors, ids, matched, SOURCE_DING, args.k, grouped=False)


if __name__ == "__main__":
    main()
