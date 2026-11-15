"""ROC of functional similarity against proofread synapses.

Scores are Pearson/cosine of oracle residuals (noise correlation) and of
trial-averaged oracle responses (signal correlation). Only same-scan pairs
are scored: residual similarity is undefined across scans.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from microns import config
from microns.models.metrics import summarize

BG = "#f7f4ef"
TEAL = "#1d7a6d"
CORAL = "#e07a5f"
NAVY = "#3d5a80"


def pearson_block(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms) @ (x / norms).T


def score_scan_pairs(
    residual: np.ndarray,
    signal: np.ndarray,
    scan_key: np.ndarray,
    syn: set[tuple[int, int]],
    knn: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = []
    noise = []
    sig = []
    knn_s = []
    for key in np.unique(scan_key):
        idx = np.flatnonzero(scan_key == key)
        if idx.size < 2:
            continue
        r = pearson_block(residual[idx])
        s = pearson_block(signal[idx])
        n = idx.size
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                ia, ib = int(idx[a]), int(idx[b])
                y.append(1 if (ia, ib) in syn else 0)
                noise.append(float(r[a, b]))
                sig.append(float(s[a, b]))
                knn_s.append(1.0 if (ia, ib) in knn else 0.0)
    return (
        np.asarray(y, dtype=int),
        np.asarray(noise, dtype=np.float64),
        np.asarray(sig, dtype=np.float64),
        np.asarray(knn_s, dtype=np.float64),
    )


def main() -> None:
    neurons = pd.read_parquet(config.DATA_PROCESSED / "functional_batch_neurons.parquet")
    fun = pd.read_parquet(config.DATA_PROCESSED / "functional_edges.parquet")
    anat = pd.read_parquet(config.DATA_PROCESSED / "edges.parquet")
    bundle = np.load(config.DATA_PROCESSED / "functional_responses.npz", allow_pickle=True)
    ids = np.asarray(bundle["pt_root_id"])
    residual = np.asarray(bundle["response"], dtype=np.float32)
    signal = np.asarray(bundle["signal"], dtype=np.float32) if "signal" in bundle.files else residual
    source = str(bundle["source"]) if "source" in bundle.files else "activity"

    meta = neurons.set_index("pt_root_id").reindex(ids)
    scan_key = meta["session"].astype(str).to_numpy() + "_" + meta["scan_idx"].astype(str).to_numpy()
    id_to_idx = {int(n): i for i, n in enumerate(ids)}

    fun_ids = set(map(int, ids))
    anat = anat[anat["pre_pt_root_id"].isin(fun_ids) & anat["post_pt_root_id"].isin(fun_ids)]
    syn = set(
        zip(
            anat["pre_pt_root_id"].map(id_to_idx).to_numpy(),
            anat["post_pt_root_id"].map(id_to_idx).to_numpy(),
        )
    )
    knn = set(
        zip(
            fun["pt_root_id"].map(id_to_idx).to_numpy(),
            fun["neighbor_pt_root_id"].map(id_to_idx).to_numpy(),
        )
    )

    y, noise, sig, knn_s = score_scan_pairs(residual, signal, scan_key, syn, knn)
    curves = {
        "noise (residual)": (noise, TEAL),
        "signal (trial average)": (sig, CORAL),
        "kNN membership": (knn_s, NAVY),
    }

    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot([0, 1], [0, 1], color="#b8b2a7", linestyle="--", linewidth=1, label="chance  0.50")
    rows = {}
    for name, (scores, color) in curves.items():
        fpr, tpr, _ = roc_curve(y, scores)
        auroc = float(roc_auc_score(y, scores))
        ap = float(average_precision_score(y, scores))
        ax.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{name}  AUROC {auroc:.3f}")
        rows[name] = {
            **summarize(y, scores, threshold=float(np.median(scores))),
            "auroc": auroc,
            "ap": ap,
        }

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Does functional similarity recover synapses?")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.text(
        0.5,
        -0.04,
        f"{source}  ·  {y.sum():,} same-scan synapses / {len(y):,} same-scan pairs  ·  "
        f"prevalence {y.mean():.3f}",
        ha="center",
        fontsize=8,
        color="#5c5a56",
    )
    out = config.DATA_PROCESSED / "similarity_roc.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    table = config.DATA_PROCESSED / "tables" / "similarity_roc.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(
        json.dumps(
            {
                "source": source,
                "n_neurons": int(len(ids)),
                "n_same_scan_pairs": int(len(y)),
                "n_same_scan_synapses": int(y.sum()),
                "metrics": rows,
            },
            indent=2,
        )
    )
    print(f"Wrote {out}")
    print(f"Wrote {table}")
    for name, m in rows.items():
        print(f"  {name}: AUROC {m['auroc']:.4f}  AP {m['ap']:.4f}")


if __name__ == "__main__":
    main()
