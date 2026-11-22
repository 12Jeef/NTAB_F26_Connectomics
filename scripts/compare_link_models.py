"""Baselines vs two-stream NCN on residual G_f.

    python scripts/compare_link_models.py
    python scripts/compare_link_models.py --same-scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from microns.models.data import (
    load_link_task,
    neighbor_sets,
    pair_corr_features,
    pearson_scores,
    sample_pairs,
)
from microns.models.metrics import summarize
from microns.models.ncn import DirectedNCN, neighbor_matrix
from microns.models.seal import common_neighbor_scores

OUT_DIR = Path("results/ncn_baseline")
BG = "#f7f4ef"
COLORS = {
    "residual (noise)": "#1d7a6d",
    "signal (trial average)": "#e07a5f",
    "common neighbors": "#3d5a80",
    "ncn": "#6d597a",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=2048)
    parser.add_argument("--same-scan", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    task = load_link_task()
    n = len(task.ids)
    pairs, labels = sample_pairs(
        task, args.pairs, args.pairs, rng, same_scan_only=args.same_scan
    )
    adj = neighbor_sets(task)

    scores = {
        "residual (noise)": pearson_scores(task.traces, pairs),
        "signal (trial average)": pearson_scores(task.signal, pairs),
        "common neighbors": common_neighbor_scores(adj, pairs),
    }
    results = {
        name: summarize(labels, s, threshold=float(np.median(s)))
        for name, s in scores.items()
    }
    results["common neighbors"] = summarize(labels, scores["common neighbors"], threshold=1.0)

    ncn_path = OUT_DIR / "ncn.pt"
    if ncn_path.exists():
        ckpt = torch.load(ncn_path, map_location="cpu", weights_only=False)
        model = DirectedNCN()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        residual = torch.tensor(task.traces)
        signal = torch.tensor(task.signal)
        f_edges = torch.tensor(np.stack([task.func_src, task.func_dst]), dtype=torch.long)
        neighbors = neighbor_matrix(n, f_edges[0], f_edges[1])
        q = torch.tensor(pairs, dtype=torch.long)
        pair_feat = torch.tensor(pair_corr_features(task, pairs))
        with torch.no_grad():
            ncn_scores = torch.sigmoid(
                model(residual, signal, q, f_edges, neighbors, pair_feat)
            ).numpy()
        scores["ncn"] = ncn_scores
        results["ncn"] = summarize(labels, ncn_scores)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot([0, 1], [0, 1], color="#b8b2a7", linestyle="--", linewidth=1, label="chance  0.50")
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(labels, s)
        auroc = float(roc_auc_score(labels, s))
        ax.plot(
            fpr,
            tpr,
            color=COLORS.get(name, "#1f1c18"),
            linewidth=2,
            label=f"{name}  {auroc:.3f}",
        )
        results[name]["auroc"] = auroc
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    title = "Same-scan pairs" if args.same_scan else "Trusted pairs (mixed scans)"
    ax.set_title(f"NCN vs correlation baselines  ·  {title}")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "same_scan" if args.same_scan else "mixed"
    fig.savefig(OUT_DIR / f"ncn_roc_{tag}.png", dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    payload = {
        "same_scan_only": args.same_scan,
        "n_neurons": n,
        "n_pairs": int(len(labels)),
        "n_pos": int(labels.sum()),
        "metrics": results,
    }
    out = OUT_DIR / f"comparison_{tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
