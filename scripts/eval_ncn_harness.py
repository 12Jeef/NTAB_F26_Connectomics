"""Rescore residual / signal / NCN with the NTAB model_eval harness.

    PYTHONPATH=. python scripts/eval_ncn_harness.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from microns import config
from microns.models.data import load_link_task, pair_corr_features
from microns.models.ncn import DirectedNCN, neighbor_matrix
from model_eval import Benchmark, evaluate
from model_eval.report import Report, baselines
from model_eval.metrics import with_reverse

OUT = Path("results/ncn_baseline/harness")


@dataclass
class MicronsDataset:
    neurons: pd.DataFrame
    synapses: pd.DataFrame
    functional: pd.DataFrame
    coreg: pd.DataFrame


def cosine_block(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms) @ (x / norms).T


def load_xyz(ids: np.ndarray) -> np.ndarray:
    coreg = pd.read_parquet(
        config.DATA_RAW / "coregistered_neurons.parquet",
        columns=["pt_root_id", "pt_position"],
    )
    coreg = coreg.drop_duplicates("pt_root_id", keep="first").set_index("pt_root_id")
    pos = coreg.reindex(ids)["pt_position"]
    xyz_nm = np.stack(pos.map(lambda p: np.asarray(p, dtype=np.float64)))
    return xyz_nm / 1000.0


def build_dataset() -> tuple[MicronsDataset, object]:
    task = load_link_task()
    ids = task.ids.astype(np.int64)
    xyz = load_xyz(ids)
    proof = pd.read_parquet(
        config.DATA_RAW / "proofreading.parquet",
        columns=["pt_root_id", "status_axon", "strategy_axon"],
    ).drop_duplicates("pt_root_id").set_index("pt_root_id")
    proof = proof.reindex(ids)
    axon_proof = proof["strategy_axon"].fillna("none").astype(str)
    axon_proof = axon_proof.where(proof["status_axon"].fillna(False).astype(bool), "none")

    neurons = pd.DataFrame(
        {
            "axon_proof": axon_proof.to_numpy(),
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "scan": task.scan_key,
        },
        index=pd.Index(ids, name="id"),
    )
    synapses = pd.DataFrame(
        {"n": task.n_synapses.astype(np.int32)},
        index=pd.MultiIndex.from_arrays(
            [ids[task.syn_src], ids[task.syn_dst]], names=["pre_id", "post_id"]
        ),
    )

    rows = []
    for key in np.unique(task.scan_key):
        idx = np.flatnonzero(task.scan_key == key)
        if idx.size < 2:
            continue
        noise = cosine_block(task.traces[idx])
        sig = cosine_block(task.signal[idx])
        n = idx.size
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                rows.append((ids[idx[a]], ids[idx[b]], float(noise[a, b]), float(sig[a, b])))
    functional = pd.DataFrame(
        rows, columns=["pre_id", "post_id", "residual_corr", "signal_corr"]
    ).set_index(["pre_id", "post_id"])
    coreg = pd.DataFrame({"unit_id": np.arange(len(ids))}, index=neurons.index)
    return MicronsDataset(neurons, synapses, functional, coreg), task


def ncn_scores(task, pairs: pd.MultiIndex) -> pd.Series:
    path = Path("results/ncn_baseline/ncn.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DirectedNCN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    id_to_idx = task.id_to_idx
    src = np.array([id_to_idx[int(a)] for a in pairs.get_level_values("pre_id")])
    dst = np.array([id_to_idx[int(b)] for b in pairs.get_level_values("post_id")])
    q = np.stack([src, dst])
    residual = torch.tensor(task.traces)
    signal = torch.tensor(task.signal)
    f_edges = torch.tensor(np.stack([task.func_src, task.func_dst]), dtype=torch.long)
    neighbors = neighbor_matrix(len(task.ids), f_edges[0], f_edges[1])
    feat = torch.tensor(pair_corr_features(task, q))
    same = task.scan_key[src] == task.scan_key[dst]
    feat[~torch.tensor(same)] = 0
    with torch.no_grad():
        logits = model(
            residual, signal, torch.tensor(q, dtype=torch.long), f_edges, neighbors, feat
        )
        scores = torch.sigmoid(logits).numpy()
    scores = np.where(same, scores, 0.5)
    return pd.Series(scores, index=pairs, name="model")


def run_report(bench, scores: pd.Series, split: str, against: str, n_boot: int, same_scan: bool) -> Report:
    pairs = bench.pairs(split)
    if same_scan:
        pre = bench.neurons.loc[pairs.index.get_level_values("pre_id"), "scan"].to_numpy()
        post = bench.neurons.loc[pairs.index.get_level_values("post_id"), "scan"].to_numpy()
        pairs = pairs.loc[pre == post]
        scores = scores.reindex(pairs.index)
    s = baselines(pairs, bench.functional_cols).assign(model=scores.reindex(pairs.index))
    if s["model"].isna().any():
        missing = int(s["model"].isna().sum())
        raise RuntimeError(f"{missing} pairs missing NCN scores")
    rev = pairs.index.get_indexer(pairs.index.swaplevel())
    frame = pairs.join(s).reset_index().assign(rev=rev, all=0)
    frame = with_reverse(frame, ["label", *s.columns])
    return Report(frame, ["model", *s.columns.drop("model")], against, n_boot, 0), pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds, task = build_dataset()
    print(
        f"dataset  neurons={len(ds.neurons):,}  synapses={len(ds.synapses):,}  "
        f"functional pairs={len(ds.functional):,}"
    )
    bench = Benchmark(ds, split_seed=0, bin_um=25.0, neg_per_pos=10, trusted_only=True)
    for name, ids in bench.splits.items():
        print(f"  {name} nodes {len(ids):,}  pairs {len(bench.pairs(name)):,}  pos {int(bench.pairs(name).label.sum()):,}")

    test_idx = bench.pairs("test").index
    print("scoring NCN on test pairs...", flush=True)
    scores = ncn_scores(task, test_idx)

    summaries = {}
    for tag, same_scan in [("all_pairs", False), ("same_scan", True)]:
        print(f"\n=== {tag}  against=abs_signal_corr ===", flush=True)
        report, pairs = run_report(
            bench, scores, "test", against="abs_signal_corr", n_boot=40, same_scan=same_scan
        )
        table = report.summary()
        print(table.round(3).to_string())
        primary = report.result()
        print(
            f"primary ap_ratio_binned  model={primary.value:.3f}  "
            f"chance={primary.chance:.1f}  vs={ {k: round(v, 3) for k, v in primary.vs.items()} }"
        )
        try:
            inc = report.incremental()
            print(f"incremental LR beyond distance+abs_signal_corr  p={inc['p']:.4g}  stat={inc['lr_stat']:.2f}")
        except Exception as exc:  # noqa: BLE001
            inc = {"error": str(exc)}
            print(f"incremental failed: {exc}")
        try:
            motifs = report.motif_errors()
            print("motif FP ratios (1 = not over-represented)")
            print(motifs.round(2).to_string())
        except Exception as exc:  # noqa: BLE001
            motifs = None
            print(f"motifs failed: {exc}")
        summaries[tag] = {
            "n_pairs": int(len(pairs)),
            "n_pos": int(pairs.label.sum()),
            "summary": table.replace({np.nan: None}).to_dict(),
            "primary": {
                "value": None if primary.value != primary.value else float(primary.value),
                "chance": float(primary.chance),
                "p": float(primary.p),
                "n_pos": int(primary.n_pos),
                "vs": {k: (None if v != v else float(v)) for k, v in primary.vs.items()},
            },
            "incremental": inc,
            "motifs": None if motifs is None else motifs.replace({np.nan: None}).to_dict(),
        }

    out = OUT / "ncn_harness.json"
    out.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
