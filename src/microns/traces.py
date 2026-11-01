"""Calcium traces for the functional kNN graph.

Traces come from DANDI 000402 NWB files (the published form of the NDA
fluorescence). ``ScanUnit.pkl`` maps ``(session, scan_idx, unit_id)`` to
``(field, mask_id)`` inside each file. Residual / oracle-repeat correlation
can replace the full-trace vector later without changing the kNN step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from microns import config

DANDISET = "000402"
DANDI_VERSION = "0.230307.2132"
NWB_PATH = "sub-17797/sub-17797_ses-{session}-scan-{scan}_behavior+image+ophys.nwb"
SCANUNIT_URL = (
    "https://raw.githubusercontent.com/aplbrain/neurodatarehack2023/main/ScanUnit.pkl"
)
TARGET_LEN = 2500

# Scans published in DANDI 000402. (5, 4), (5, 5), and (8, 4) are not.
DANDI_SCANS = [
    (4, 7),
    (4, 9),
    (4, 10),
    (5, 3),
    (5, 6),
    (5, 7),
    (6, 2),
    (6, 4),
    (6, 6),
    (6, 7),
    (7, 3),
    (7, 4),
    (7, 5),
    (8, 5),
    (8, 7),
    (8, 9),
    (9, 3),
    (9, 4),
    (9, 6),
]


def proofread_axon_units() -> pd.DataFrame:
    """One row per (neuron, calcium unit), ranked so the primary unit is first."""
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
            candidate = tuple(int(part) for part in str(key).split("_"))
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


def ensure_scanunit() -> pd.DataFrame:
    path = config.SCANUNIT_PATH
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SCANUNIT_URL, path)
    scanunit = pd.read_pickle(path)
    return scanunit[["session", "scan_idx", "unit_id", "field", "mask_id"]]


def units_for_scan(scanunit: pd.DataFrame, session: int, scan_idx: int) -> pd.DataFrame:
    axon = proofread_axon_units()
    mapped = axon.merge(scanunit, on=["session", "scan_idx", "unit_id"], how="inner")
    mapped = mapped[(mapped["session"] == session) & (mapped["scan_idx"] == scan_idx)]
    mapped = mapped.sort_values(["pt_root_id", "preference"])
    return mapped.drop_duplicates("pt_root_id", keep="first").reset_index(drop=True)


def nwb_url(session: int, scan_idx: int) -> str:
    from dandi.dandiapi import DandiAPIClient

    path = NWB_PATH.format(session=session, scan=scan_idx)
    with DandiAPIClient() as client:
        asset = client.get_dandiset(DANDISET, DANDI_VERSION).get_asset_by_path(path)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def fetch_scan_traces(session: int, scan_idx: int, units: pd.DataFrame) -> pd.DataFrame:
    """Stream one scan from DANDI and return fluorescence for the given units."""
    import h5py
    import remfile

    if units.empty:
        return units.assign(trace=pd.Series(dtype=object))

    h5 = h5py.File(remfile.File(nwb_url(session, scan_idx)), "r")
    traces = []
    by_field = units.groupby("field", sort=True)
    for field, grp in by_field:
        series = h5[f"processing/ophys/Fluorescence/RoiResponseSeries{int(field)}/data"]
        for row in grp.itertuples(index=False):
            traces.append(np.asarray(series[:, int(row.mask_id) - 1], dtype=np.float32))
    h5.close()

    out = units.copy()
    out["trace"] = traces
    return out


def downsample(trace: np.ndarray, target: int = TARGET_LEN) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32).ravel()
    if len(trace) < target:
        pad = np.full(target - len(trace), trace[-1] if len(trace) else 0, dtype=np.float32)
        return np.concatenate([trace, pad])
    factor = len(trace) // target
    usable = factor * target
    return trace[:usable].reshape(target, factor).mean(axis=1)


def batch_path(session: int, scan_idx: int) -> Path:
    return config.DATA_CALCIUM / f"session{session}_scan{scan_idx}.npz"


def save_batch(session: int, scan_idx: int, units: pd.DataFrame) -> Path:
    config.DATA_CALCIUM.mkdir(parents=True, exist_ok=True)
    path = batch_path(session, scan_idx)
    stacked = np.stack([downsample(t) for t in units["trace"]])
    np.savez_compressed(
        path,
        pt_root_id=units["pt_root_id"].to_numpy(),
        session=units["session"].to_numpy(),
        scan_idx=units["scan_idx"].to_numpy(),
        unit_id=units["unit_id"].to_numpy(),
        field=units["field"].to_numpy(),
        mask_id=units["mask_id"].to_numpy(),
        response=stacked.astype(np.float32),
        source=np.array("dandi_000402_fluorescence"),
    )
    return path


def load_batches() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    files = sorted(config.DATA_CALCIUM.glob("session*_scan*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No calcium batches in {config.DATA_CALCIUM}. "
            "Run scripts/download_calcium_traces.py"
        )
    frames = []
    vectors = []
    for path in files:
        bundle = np.load(path, allow_pickle=True)
        frame = pd.DataFrame(
            {
                "pt_root_id": bundle["pt_root_id"],
                "session": bundle["session"],
                "scan_idx": bundle["scan_idx"],
                "unit_id": bundle["unit_id"],
            }
        )
        frames.append(frame)
        vectors.append(bundle["response"])
    neurons = pd.concat(frames, ignore_index=True)
    stacked = np.vstack(vectors)
    keep = ~neurons.duplicated("pt_root_id", keep="first")
    neurons = neurons.loc[keep].reset_index(drop=True)
    stacked = stacked[keep.to_numpy()]
    return stacked, neurons["pt_root_id"].to_numpy(), neurons
