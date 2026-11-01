"""Fetch digital-twin movie responses in row batches and rebuild the kNN graph.

The in-vivo NDA traces are a 97 GB database. The digital-twin export stores a
predicted response for every unit in 13 scans as rows of a 31 GB array. This
script downloads only the rows that match our proofread-axon neurons (~360 MB),
downsamples 30 Hz to 1 Hz, and rebuilds the functional graph in that shared
response space.

Source:
  gs/s3 digital_twin_properties/v2/responses/responses.npy
  shape (104171, 75000) float32, C-order, 30 Hz in-silico movie responses
"""

from __future__ import annotations

import ast
import struct
import time
import urllib.request

import numpy as np
import pandas as pd

from microns import config
from microns.graph import knn_edges

RESPONSES_URL = (
    "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
    "functional_data/digital_twin_properties/v2/responses/responses.npy"
)
UNITS_CSV = config.DATA_RAW / "digital_twin" / "responses_units.csv"
K = 15
DOWNSAMPLE = 30  # 30 Hz -> 1 Hz
BATCH_ROWS = 32
RETRIES = 5


def npy_layout(url: str) -> tuple[int, tuple[int, int]]:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-255"})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
    if data[:6] != b"\x93NUMPY":
        raise RuntimeError("responses.npy is not a numpy file")
    header_len = struct.unpack_from("<H", data, 8)[0]
    header = data[10 : 10 + header_len].decode("latin1")
    meta = ast.literal_eval(header)
    if meta["descr"] != "<f4" or meta["fortran_order"]:
        raise RuntimeError(f"unexpected npy layout: {meta}")
    offset = 10 + header_len
    offset += (64 - offset % 64) % 64
    return offset, meta["shape"]


def fetch_range(url: str, start: int, end: int) -> bytes:
    last_error = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end - 1}"}
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                payload = response.read()
            if len(payload) != end - start:
                raise RuntimeError(
                    f"short read: got {len(payload)} expected {end - start}"
                )
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed range {start}-{end}: {last_error}")


def proofread_axon_units() -> pd.DataFrame:
    nodes = pd.read_parquet(
        config.DATA_PROCESSED / "nodes.parquet",
        columns=["pt_root_id", "session", "scan_idx", "unit_id", "unit_keys"],
    )
    proof = pd.read_parquet(
        config.DATA_RAW / "proofreading.parquet",
        columns=["pt_root_id", "status_axon"],
    )
    axon = set(proof.loc[proof["status_axon"].astype(bool), "pt_root_id"])
    ours = nodes[nodes["pt_root_id"].isin(axon)].copy()
    rows = []
    for row in ours.itertuples(index=False):
        keys = row.unit_keys if isinstance(row.unit_keys, list) else list(row.unit_keys)
        primary = (int(row.session), int(row.scan_idx), int(row.unit_id))
        ranked = [primary]
        for key in keys:
            session, scan_idx, unit_id = map(int, key.split("_"))
            candidate = (session, scan_idx, unit_id)
            if candidate not in ranked:
                ranked.append(candidate)
        for rank, (session, scan_idx, unit_id) in enumerate(ranked):
            rows.append(
                {
                    "pt_root_id": row.pt_root_id,
                    "session": session,
                    "scan_idx": scan_idx,
                    "unit_id": unit_id,
                    "preference": rank,
                }
            )
    return pd.DataFrame(rows)


def neuron_rows(units: pd.DataFrame, axon_units: pd.DataFrame) -> pd.DataFrame:
    units = units.copy()
    units["row"] = np.arange(len(units))
    matched = axon_units.merge(units, on=["session", "scan_idx", "unit_id"], how="inner")
    matched = matched.sort_values(["pt_root_id", "preference"])
    return matched.drop_duplicates("pt_root_id", keep="first").reset_index(drop=True)


def download_rows(row_index: np.ndarray, offset: int, n_times: int) -> np.ndarray:
    row_bytes = n_times * 4
    order = np.argsort(row_index)
    sorted_rows = row_index[order]
    traces = np.empty((len(row_index), n_times), dtype=np.float32)
    i = 0
    fetched = 0
    while i < len(sorted_rows):
        j = i + 1
        while (
            j < len(sorted_rows)
            and sorted_rows[j] == sorted_rows[j - 1] + 1
            and j - i < BATCH_ROWS
        ):
            j += 1
        first, last = int(sorted_rows[i]), int(sorted_rows[j - 1])
        start = offset + first * row_bytes
        end = offset + (last + 1) * row_bytes
        payload = fetch_range(RESPONSES_URL, start, end)
        block = np.frombuffer(payload, dtype="<f4").reshape(last - first + 1, n_times)
        for local, source_row in enumerate(sorted_rows[i:j]):
            traces[order[i + local]] = block[int(source_row) - first]
        fetched += j - i
        print(f"  fetched {fetched:,} / {len(sorted_rows):,} rows (npy {first}-{last})")
        i = j
    return traces


def downsample(traces: np.ndarray, factor: int) -> np.ndarray:
    usable = traces.shape[1] - (traces.shape[1] % factor)
    stacked = traces[:, :usable].reshape(traces.shape[0], usable // factor, factor)
    return stacked.mean(axis=2)


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if not UNITS_CSV.exists():
        raise RuntimeError(f"Missing {UNITS_CSV}")

    offset, shape = npy_layout(RESPONSES_URL)
    n_units, n_times = shape
    print(f"responses.npy  {n_units:,} units x {n_times:,} frames, data at byte {offset}")

    units = pd.read_csv(UNITS_CSV)
    if len(units) != n_units:
        raise RuntimeError(f"units.csv has {len(units)} rows, npy has {n_units}")

    chosen = neuron_rows(units, proofread_axon_units())
    axon_n = pd.read_parquet(
        config.DATA_PROCESSED / "nodes.parquet", columns=["pt_root_id"]
    )
    proof = pd.read_parquet(
        config.DATA_RAW / "proofreading.parquet", columns=["pt_root_id", "status_axon"]
    )
    n_axon = int(proof.loc[proof["status_axon"].astype(bool), "pt_root_id"].nunique())
    print(f"Proofread-axon neurons with a digital-twin row: {len(chosen):,} / {n_axon:,}")

    traces = download_rows(chosen["row"].to_numpy(), offset, n_times)
    vectors = downsample(traces, DOWNSAMPLE)
    print(f"Downsampled {n_times} -> {vectors.shape[1]} samples")

    ids = chosen["pt_root_id"].to_numpy()
    edges = knn_edges(vectors, ids, k=K)
    out_resp = config.DATA_PROCESSED / "functional_responses.npz"
    out_edges = config.DATA_PROCESSED / "functional_edges.parquet"
    out_neurons = config.DATA_PROCESSED / "functional_batch_neurons.parquet"
    np.savez_compressed(
        out_resp,
        pt_root_id=ids,
        response=vectors.astype(np.float32),
        source=np.array("digital_twin_v2_in_silico"),
    )
    edges.to_parquet(out_edges, index=False)
    chosen.drop(columns=["row"]).to_parquet(out_neurons, index=False)
    print(f"Functional graph: {len(ids):,} neurons, {len(edges):,} edges, K={K}")
    print(f"Wrote {out_resp}")
    print(f"Wrote {out_edges}")


if __name__ == "__main__":
    main()
