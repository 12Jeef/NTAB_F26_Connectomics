"""Train directed two-stream NCN on residual G_f.

    python scripts/train_ncn.py
    python scripts/train_ncn.py --epochs 25 --pairs 4096 --same-scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from microns.models.data import load_link_task, pair_corr_features, sample_pairs
from microns.models.metrics import summarize
from microns.models.ncn import DirectedNCN, neighbor_matrix

OUT_DIR = Path("results/ncn_baseline")


def tensors(task, pairs):
    residual = torch.tensor(task.traces)
    signal = torch.tensor(task.signal)
    f_edges = torch.tensor(np.stack([task.func_src, task.func_dst]), dtype=torch.long)
    neighbors = neighbor_matrix(len(task.ids), f_edges[0], f_edges[1])
    q = torch.tensor(pairs, dtype=torch.long)
    pair_feat = torch.tensor(pair_corr_features(task, pairs))
    return residual, signal, f_edges, neighbors, q, pair_feat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--same-scan",
        action="store_true",
        help="Sample only co-recorded pairs (where G_f edges exist)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    task = load_link_task()

    n_avail = 0
    for a, b in zip(task.syn_src.tolist(), task.syn_dst.tolist()):
        if (not args.same_scan) or str(task.scan_key[a]) == str(task.scan_key[b]):
            n_avail += 1
    n_pos = min(args.pairs, max(n_avail // 2, 1))
    print(f"positives available {n_avail:,}  using {n_pos:,} train / {min(n_pos, n_avail - n_pos):,} test")

    train_pairs, train_y = sample_pairs(
        task, n_pos, n_pos, rng, same_scan_only=args.same_scan
    )
    held = set(map(tuple, train_pairs.T.tolist()))
    n_test = min(n_pos, max(n_avail - n_pos, 1))
    test_pairs, test_y = sample_pairs(
        task,
        n_test,
        n_test,
        rng,
        exclude=held,
        same_scan_only=args.same_scan,
    )
    scan_pairs, scan_y = sample_pairs(
        task, min(1024, n_test), min(1024, n_test), rng, exclude=held, same_scan_only=True
    )

    residual, signal, f_edges, neighbors, q_train, feat_train = tensors(task, train_pairs)
    _, _, _, _, q_test, feat_test = tensors(task, test_pairs)
    _, _, _, _, q_scan, feat_scan = tensors(task, scan_pairs)
    y_train = torch.tensor(train_y)
    y_test = torch.tensor(test_y)

    model = DirectedNCN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best = None
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(residual, signal, q_train, f_edges, neighbors, feat_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(residual, signal, q_test, f_edges, neighbors, feat_test)
            test_loss = loss_fn(test_logits, y_test).item()
            probs = torch.sigmoid(test_logits).numpy()
            metrics = summarize(test_y, probs)
            scan_probs = torch.sigmoid(
                model(residual, signal, q_scan, f_edges, neighbors, feat_scan)
            ).numpy()
            scan_metrics = summarize(scan_y, scan_probs)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(loss.item()),
                "test_loss": test_loss,
                **metrics,
                "same_scan_ap": scan_metrics["ap"],
                "same_scan_auroc": scan_metrics["auroc"],
            }
        )
        print(
            f"epoch {epoch+1:03d}  train {loss.item():.4f}  test {test_loss:.4f}  "
            f"AP {metrics['ap']:.4f}  AUROC {metrics['auroc']:.4f}  "
            f"same-scan AUROC {scan_metrics['auroc']:.4f}"
        )
        if best is None or metrics["ap"] > best:
            best = metrics["ap"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": "two_stream_directed"}, OUT_DIR / "ncn.pt")
    (OUT_DIR / "ncn_history.json").write_text(
        json.dumps(
            {
                "best_ap": best,
                "same_scan_only": args.same_scan,
                "n_neurons": int(len(task.ids)),
                "n_functional_edges": int(len(task.func_src)),
                "history": history,
            },
            indent=2,
        )
    )
    print(f"best AP {best:.4f}")
    print(f"Wrote {OUT_DIR / 'ncn.pt'}")


if __name__ == "__main__":
    main()
