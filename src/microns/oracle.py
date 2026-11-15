"""Oracle-repeat signal and residual traces for the functional graph.

The Nature resource paper repeats six 10 s natural-movie clips, in the same
order, ten times per scan. The NT@B spec uses those repeats to split calcium
into:

  signal   trial average over the ten oracle blocks
  residual each block minus that average (noise)

G_f is a kNN of residual (noise) correlation among neurons recorded in the
same scan. Cross-scan residual correlation is undefined: those cells were
not imaged together. DANDI 000402 has fluorescence, not CaImAn deconvolution,
so residuals are computed on an AR1 spike inference of the oracle windows
(GCaMP6s, same algorithm family as CaImAn OASIS).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from microns import config
from microns.traces import (
    DANDI_SCANS,
    ensure_scanunit,
    nwb_url,
    units_for_scan,
)

N_CLIPS = 6
N_REPEATS = 10
N_BINS = 63  # ~10 s at 6.3 Hz
GCAMP6S_TAU_S = 1.0
ORACLE_DIR = config.DATA_CALCIUM / "oracle"


def oracle_batch_path(session: int, scan_idx: int) -> Path:
    return ORACLE_DIR / f"session{session}_scan{scan_idx}.npz"


def _decode(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return str(value)


def read_clip_table(h5) -> pd.DataFrame:
    clip = h5["intervals/Clip"]
    return pd.DataFrame(
        {
            "start_time": clip["start_time"][:],
            "stop_time": clip["stop_time"][:],
            "condition_hash": [_decode(x) for x in clip["condition_hash"][:]],
            "movie_name": [_decode(x) for x in clip["movie_name"][:]],
            "short_movie_name": [_decode(x) for x in clip["short_movie_name"][:]],
        }
    )


def oracle_blocks(clips: pd.DataFrame) -> pd.DataFrame:
    """Return the ten contiguous 6-clip oracle blocks (paper definition)."""
    hashes = clips["condition_hash"].tolist()
    grams = [tuple(hashes[i : i + N_CLIPS]) for i in range(len(hashes) - N_CLIPS + 1)]
    gram, count = Counter(grams).most_common(1)[0]
    if count != N_REPEATS or len(set(gram)) != N_CLIPS:
        raise RuntimeError(
            f"expected {N_REPEATS} repeats of {N_CLIPS} unique clips, "
            f"got count={count} unique={len(set(gram))}"
        )
    rows = []
    for repeat, i in enumerate(i for i, g in enumerate(grams) if g == gram):
        block = clips.iloc[i : i + N_CLIPS].copy()
        block["repeat"] = repeat
        block["clip"] = np.arange(N_CLIPS)
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    return out


def deconvolve_ar1(y: np.ndarray, g: float) -> np.ndarray:
    """Non-negative AR1 inverse, CaImAn/OASIS family, fixed decay."""
    y = np.asarray(y, dtype=np.float64)
    baseline = np.nanpercentile(y, 8)
    if not np.isfinite(baseline):
        baseline = 0.0
    y = np.nan_to_num(y - baseline)
    s = np.empty_like(y)
    s[0] = max(y[0], 0.0)
    s[1:] = np.maximum(y[1:] - g * y[:-1], 0.0)
    return s.astype(np.float32)


def _resample(values: np.ndarray, times: np.ndarray, t0: float, t1: float, n_bins: int) -> np.ndarray:
    if t1 <= t0 or len(times) < 2:
        return np.zeros(n_bins, dtype=np.float32)
    grid = np.linspace(t0, t1, n_bins, endpoint=False)
    return np.interp(grid, times, values).astype(np.float32)


def _timestamps(h5) -> np.ndarray:
    fluor = h5["processing/ophys/Fluorescence"]
    for key in fluor.keys():
        if key.startswith("RoiResponseSeries") and "timestamps" in fluor[key]:
            return np.asarray(fluor[key]["timestamps"][:], dtype=np.float64)
    raise RuntimeError("no fluorescence timestamps in NWB")


def fetch_oracle_scan(session: int, scan_idx: int, units: pd.DataFrame) -> dict:
    """Stream one scan and return oracle signal / residual arrays."""
    import h5py
    import remfile

    if units.empty:
        raise ValueError("no units for scan")

    h5 = h5py.File(remfile.File(nwb_url(session, scan_idx)), "r")
    clips = oracle_blocks(read_clip_table(h5))
    ts = _timestamps(h5)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0 / 6.3
    g = float(np.exp(-dt / GCAMP6S_TAU_S))

    n = len(units)
    repeats = np.zeros((n, N_REPEATS, N_CLIPS * N_BINS), dtype=np.float32)
    index_lookup = {idx: i for i, idx in enumerate(units.index)}

    for field, grp in units.groupby("field", sort=True):
        series = h5[f"processing/ophys/Fluorescence/RoiResponseSeries{int(field)}/data"]
        cols = (grp["mask_id"].astype(int) - 1).to_numpy()
        uniq, inverse = np.unique(cols, return_inverse=True)
        for repeat, block in clips.groupby("repeat"):
            t0 = float(block["start_time"].iloc[0])
            t1 = float(block["stop_time"].iloc[-1])
            i0 = int(np.searchsorted(ts, t0, side="left"))
            i1 = int(np.searchsorted(ts, t1, side="left"))
            i0 = max(i0, 0)
            i1 = min(max(i1, i0 + 2), len(ts))
            chunk = np.asarray(series[i0:i1, uniq.tolist()], dtype=np.float32)
            times = ts[i0:i1]
            for row_idx, inv in zip(grp.index, inverse):
                spikes = deconvolve_ar1(chunk[:, inv], g)
                pieces = []
                for _, clip in block.iterrows():
                    pieces.append(
                        _resample(
                            spikes,
                            times,
                            float(clip.start_time),
                            float(clip.stop_time),
                            N_BINS,
                        )
                    )
                repeats[index_lookup[row_idx], int(repeat)] = np.concatenate(pieces)

    h5.close()
    signal = repeats.mean(axis=1)
    residual = repeats - signal[:, None, :]
    residual_flat = residual.reshape(n, -1)
    hashes = clips.loc[clips["repeat"] == 0, "condition_hash"].tolist()
    movies = clips.loc[clips["repeat"] == 0, "movie_name"].tolist()
    oracle = jackknife_oracle_scores(repeats)
    return {
        "pt_root_id": units["pt_root_id"].to_numpy(),
        "session": units["session"].to_numpy(),
        "scan_idx": units["scan_idx"].to_numpy(),
        "unit_id": units["unit_id"].to_numpy(),
        "field": units["field"].to_numpy(),
        "mask_id": units["mask_id"].to_numpy(),
        "preference": units["preference"].to_numpy(),
        "repeats": repeats,
        "signal": signal.astype(np.float32),
        "residual": residual.astype(np.float32),
        "residual_flat": residual_flat.astype(np.float32),
        "oracle_score": oracle,
        "clip_hash": np.array(hashes),
        "movie_name": np.array(movies),
        "dt": np.array(dt),
        "source": np.array("dandi_000402_oracle_residual"),
    }


def jackknife_oracle_scores(repeats: np.ndarray) -> np.ndarray:
    """Paper oracle score: mean leave-one-repeat correlation with the others."""
    n, r, t = repeats.shape
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        trials = repeats[i]
        vals = []
        for k in range(r):
            loo = np.delete(trials, k, axis=0).mean(axis=0)
            a = trials[k] - trials[k].mean()
            b = loo - loo.mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 0:
                vals.append(float(np.dot(a, b) / denom))
        scores[i] = float(np.mean(vals)) if vals else 0.0
    return scores


def save_oracle_batch(session: int, scan_idx: int, bundle: dict) -> Path:
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    path = oracle_batch_path(session, scan_idx)
    np.savez_compressed(path, **bundle)
    return path


def load_oracle_batches() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    files = sorted(ORACLE_DIR.glob("session*_scan*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No oracle batches in {ORACLE_DIR}. "
            "Run scripts/download_oracle_traces.py"
        )
    frames = []
    residual = []
    signal = []
    for path in files:
        bundle = np.load(path, allow_pickle=True)
        frames.append(
            pd.DataFrame(
                {
                    "pt_root_id": bundle["pt_root_id"],
                    "session": bundle["session"],
                    "scan_idx": bundle["scan_idx"],
                    "unit_id": bundle["unit_id"],
                    "preference": bundle["preference"] if "preference" in bundle.files else 0,
                    "oracle_score": bundle["oracle_score"] if "oracle_score" in bundle.files else 0,
                }
            )
        )
        residual.append(np.asarray(bundle["residual_flat"], dtype=np.float32))
        signal.append(np.asarray(bundle["signal"], dtype=np.float32))
    neurons = pd.concat(frames, ignore_index=True)
    width = max(a.shape[1] for a in residual)
    stacked = []
    for a in residual:
        if a.shape[1] < width:
            a = np.pad(a, ((0, 0), (0, width - a.shape[1])))
        stacked.append(a)
    residual_arr = np.vstack(stacked)
    sig_width = max(a.shape[1] for a in signal)
    sig_stacked = []
    for a in signal:
        if a.shape[1] < sig_width:
            a = np.pad(a, ((0, 0), (0, sig_width - a.shape[1])))
        sig_stacked.append(a)
    signal_arr = np.vstack(sig_stacked)
    order = np.argsort(neurons["preference"].to_numpy(), kind="mergesort")
    neurons = neurons.iloc[order].reset_index(drop=True)
    residual_arr = residual_arr[order]
    signal_arr = signal_arr[order]
    keep = ~neurons.duplicated("pt_root_id", keep="first")
    neurons = neurons.loc[keep].reset_index(drop=True)
    residual_arr = residual_arr[keep.to_numpy()]
    signal_arr = signal_arr[keep.to_numpy()]
    return residual_arr, neurons["pt_root_id"].to_numpy(), neurons, signal_arr


def download_oracle_scans(scans: list[tuple[int, int]] | None = None) -> list[Path]:
    scanunit = ensure_scanunit()
    written = []
    for session, scan_idx in scans or list(DANDI_SCANS):
        path = oracle_batch_path(session, scan_idx)
        if path.exists():
            print(f"skip {session}-{scan_idx} (already have {path.name})", flush=True)
            written.append(path)
            continue
        units = units_for_scan(scanunit, session, scan_idx)
        print(f"oracle {session}-{scan_idx}: {len(units)} neurons", flush=True)
        if units.empty:
            continue
        try:
            bundle = fetch_oracle_scan(session, scan_idx, units)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {session}-{scan_idx}: {exc}", flush=True)
            continue
        path = save_oracle_batch(session, scan_idx, bundle)
        print(
            f"  wrote {path}  repeats {bundle['repeats'].shape}  "
            f"mean oracle {bundle['oracle_score'].mean():.3f}",
            flush=True,
        )
        written.append(path)
    return written
