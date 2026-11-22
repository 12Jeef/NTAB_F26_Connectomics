"""Distance-matched AUROC of residual cosine vs proofread synapses.

Random same-scan non-synapses are farther apart than wired pairs, so a
plain AUROC can pick up proximity. This script matches each same-scan
synapse to an unused non-synapse in the same scan with the closest
soma-soma distance, then ranks those pairs. Also reports a 25 μm window.
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
NEAR_UM = 25.0


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
    if pos.isna().any():
        missing = int(pos.isna().sum())
        raise RuntimeError(f"{missing} G_f neurons have no pt_position")
    xyz_nm = np.stack(pos.map(lambda p: np.asarray(p, dtype=np.float64)))
    return xyz_nm / 1000.0  # nm → μm


def scan_pairs(
    residual: np.ndarray,
    signal: np.ndarray,
    xyz: np.ndarray,
    scan_key: np.ndarray,
    syn: set[tuple[int, int]],
    knn: set[tuple[int, int]],
) -> pd.DataFrame:
    rows = []
    for key in np.unique(scan_key):
        idx = np.flatnonzero(scan_key == key)
        if idx.size < 2:
            continue
        noise = cosine_block(residual[idx])
        sig = cosine_block(signal[idx])
        delta = xyz[idx][:, None, :] - xyz[idx][None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        n = idx.size
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                ia, ib = int(idx[a]), int(idx[b])
                rows.append(
                    (
                        key,
                        ia,
                        ib,
                        1 if (ia, ib) in syn else 0,
                        float(noise[a, b]),
                        float(sig[a, b]),
                        1.0 if (ia, ib) in knn else 0.0,
                        float(dist[a, b]),
                    )
                )
    return pd.DataFrame(
        rows,
        columns=["scan", "src", "dst", "y", "residual", "signal", "knn", "dist_um"],
    )


def match_distances(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    """Greedy same-scan match: each synapse gets one unused non-synapse."""
    picked = []
    for scan, pos_scan in pos.groupby("scan", sort=False):
        neg_scan = neg.loc[neg["scan"] == scan].copy()
        if neg_scan.empty:
            continue
        used = np.zeros(len(neg_scan), dtype=bool)
        d_neg = neg_scan["dist_um"].to_numpy()
        neg_idx = neg_scan.index.to_numpy()
        for _, row in pos_scan.sort_values("dist_um").iterrows():
            remain = np.flatnonzero(~used)
            if remain.size == 0:
                break
            j = remain[int(np.argmin(np.abs(d_neg[remain] - row.dist_um)))]
            used[j] = True
            picked.append(
                {
                    "scan": scan,
                    "y": 1,
                    "residual": row.residual,
                    "signal": row.signal,
                    "knn": row.knn,
                    "dist_um": row.dist_um,
                }
            )
            n = neg_scan.loc[neg_idx[j]]
            picked.append(
                {
                    "scan": scan,
                    "y": 0,
                    "residual": n.residual,
                    "signal": n.signal,
                    "knn": n.knn,
                    "dist_um": n.dist_um,
                }
            )
    return pd.DataFrame(picked)


def metrics(df: pd.DataFrame) -> dict:
    y = df["y"].to_numpy(dtype=int)
    out = {}
    for name, col in [
        ("noise (residual)", "residual"),
        ("signal (trial average)", "signal"),
        ("kNN membership", "knn"),
    ]:
        scores = df[col].to_numpy(dtype=np.float64)
        out[name] = {
            **summarize(y, scores, threshold=float(np.median(scores))),
            "auroc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
        }
    out["distance_um"] = {
        "synapse_median": float(df.loc[df.y == 1, "dist_um"].median()),
        "nonsynapse_median": float(df.loc[df.y == 0, "dist_um"].median()),
        "synapse_mean": float(df.loc[df.y == 1, "dist_um"].mean()),
        "nonsynapse_mean": float(df.loc[df.y == 0, "dist_um"].mean()),
    }
    return out


def draw(matched: pd.DataFrame, near: pd.DataFrame, source: str, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), facecolor=BG)
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.plot([0, 1], [0, 1], color="#b8b2a7", linestyle="--", linewidth=1, label="chance  0.50")
    colors = {
        "noise (residual)": TEAL,
        "signal (trial average)": CORAL,
        "kNN membership": NAVY,
    }
    cols = {
        "noise (residual)": "residual",
        "signal (trial average)": "signal",
        "kNN membership": "knn",
    }
    y = matched["y"].to_numpy(dtype=int)
    for name, col in cols.items():
        scores = matched[col].to_numpy(dtype=np.float64)
        fpr, tpr, _ = roc_curve(y, scores)
        auroc = float(roc_auc_score(y, scores))
        ax.plot(fpr, tpr, color=colors[name], linewidth=2.0, label=f"{name}  {auroc:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Distance-matched  ·  same-scan")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.set_facecolor(BG)
    bins = np.linspace(0, 400, 41)
    ax.hist(
        matched.loc[matched.y == 1, "dist_um"],
        bins=bins,
        color=TEAL,
        alpha=0.55,
        label="synapses",
    )
    ax.hist(
        matched.loc[matched.y == 0, "dist_um"],
        bins=bins,
        color=CORAL,
        alpha=0.45,
        label="matched non-synapses",
    )
    ax.axvline(NEAR_UM, color=NAVY, linestyle="--", linewidth=1, label=f"{NEAR_UM:.0f} μm")
    ax.set_xlabel("soma–soma distance (μm)")
    ax.set_ylabel("pairs")
    ax.set_title("Matched distance distributions")
    ax.legend(frameon=False, fontsize=8)
    n_pos = int((matched.y == 1).sum())
    n_near = int((near.y == 1).sum()) if len(near) else 0
    fig.suptitle("Does residual cosine recover synapses after distance matching?", y=1.02)
    fig.text(
        0.5,
        -0.06,
        f"{source}  ·  {n_pos:,} matched pairs  ·  {n_near:,} synapses with both cells within {NEAR_UM:.0f} μm",
        ha="center",
        fontsize=8,
        color="#5c5a56",
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


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
    xyz = load_xyz(ids)
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

    pairs = scan_pairs(residual, signal, xyz, scan_key, syn, knn)
    pos = pairs.loc[pairs.y == 1]
    neg = pairs.loc[pairs.y == 0]
    matched = match_distances(pos, neg)
    near = pairs.loc[pairs.dist_um <= NEAR_UM]

    out_png = config.DATA_PROCESSED / "distance_matched_roc.png"
    out_json = config.DATA_PROCESSED / "tables" / "distance_matched_roc.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    draw(matched, near, source, out_png)

    payload = {
        "source": source,
        "n_neurons": int(len(ids)),
        "near_um": NEAR_UM,
        "random_same_scan": metrics(pairs),
        "distance_matched": metrics(matched),
        "within_25um": metrics(near) if near["y"].nunique() == 2 else {"error": "one class"},
        "n_same_scan_pairs": int(len(pairs)),
        "n_same_scan_synapses": int(pos.shape[0]),
        "n_matched_synapses": int((matched.y == 1).sum()),
        "n_within_25um_pairs": int(len(near)),
        "n_within_25um_synapses": int((near.y == 1).sum()),
        "unmatched_median_dist_um": {
            "synapse": float(pos.dist_um.median()),
            "nonsynapse": float(neg.dist_um.median()),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")
    for split in ("random_same_scan", "distance_matched", "within_25um"):
        print(f"\n{split}")
        block = payload[split]
        if "error" in block:
            print(" ", block)
            continue
        d = block["distance_um"]
        print(
            f"  median dist synapse {d['synapse_median']:.1f} μm  "
            f"non-synapse {d['nonsynapse_median']:.1f} μm"
        )
        for name in ("noise (residual)", "signal (trial average)", "kNN membership"):
            m = block[name]
            print(f"  {name}: AUROC {m['auroc']:.4f}  AP {m['ap']:.4f}")


if __name__ == "__main__":
    main()
