"""Download in-vivo calcium traces one scan at a time and rebuild the kNN graph.

Each batch streams fluorescence for our proofread-axon neurons from DANDI
000402 (same traces as microns_phase3_nda). Synapses are not edges.

    python scripts/download_calcium_traces.py              # all published scans
    python scripts/download_calcium_traces.py --scan 8-7   # one batch
    python scripts/download_calcium_traces.py --rebuild    # graph from batches already on disk
"""

from __future__ import annotations

import argparse

import numpy as np

from microns import config
from microns.graph import knn_edges
from microns.traces import (
    DANDI_SCANS,
    batch_path,
    ensure_scanunit,
    fetch_scan_traces,
    load_batches,
    save_batch,
    units_for_scan,
)

K = 15


def parse_scan(text: str) -> tuple[int, int]:
    session, scan_idx = text.replace(",", "-").split("-")
    return int(session), int(scan_idx)


def rebuild(k: int) -> None:
    vectors, ids, neurons = load_batches()
    edges = knn_edges(vectors, ids, k=k)
    resp_path = config.DATA_PROCESSED / "functional_responses.npz"
    edge_path = config.DATA_PROCESSED / "functional_edges.parquet"
    neuron_path = config.DATA_PROCESSED / "functional_batch_neurons.parquet"
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        resp_path,
        pt_root_id=ids,
        response=vectors.astype(np.float32),
        source=np.array("dandi_000402_fluorescence"),
    )
    edges.to_parquet(edge_path, index=False)
    neurons.to_parquet(neuron_path, index=False)
    print(f"Functional graph: {len(ids):,} neurons, {len(edges):,} edges, K={k}")
    print(f"Wrote {resp_path}")
    print(f"Wrote {edge_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", help="One batch, like 8-7")
    parser.add_argument("--rebuild", action="store_true", help="Skip download, rebuild graph")
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    if args.rebuild:
        rebuild(args.k)
        return

    scanunit = ensure_scanunit()
    scans = [parse_scan(args.scan)] if args.scan else list(DANDI_SCANS)

    for session, scan_idx in scans:
        path = batch_path(session, scan_idx)
        if path.exists():
            print(f"skip {session}-{scan_idx} (already have {path.name})")
            continue
        units = units_for_scan(scanunit, session, scan_idx)
        print(f"scan {session}-{scan_idx}: {len(units)} neurons")
        if units.empty:
            continue
        fetched = fetch_scan_traces(session, scan_idx, units)
        written = save_batch(session, scan_idx, fetched)
        print(f"  wrote {written}")

    rebuild(args.k)


if __name__ == "__main__":
    main()
