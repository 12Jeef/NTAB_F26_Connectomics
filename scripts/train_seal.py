"""Train a SEAL enclosing-subgraph GNN on the calcium functional graph.

    python scripts/train_seal.py
    python scripts/train_seal.py --epochs 5 --pairs 1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from microns.models.data import load_link_task, neighbor_sets, sample_pairs
from microns.models.metrics import summarize
from microns.models.seal import SEAL, pca_feat


def run_epoch(model, adj, feat, pairs, labels, opt, loss_fn, train: bool):
    scores = []
    total = 0.0
    if train:
        model.train()
    else:
        model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (a, b), y in zip(pairs.T.tolist(), labels.tolist()):
            x, edges, pair = model.encode_pair(adj, feat, a, b)
            logit = model(x, edges, pair)
            target = torch.tensor(y)
            loss = loss_fn(logit, target)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += float(loss.item())
            scores.append(float(torch.sigmoid(logit).detach()))
    return total / max(len(labels), 1), np.array(scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    task = load_link_task()
    adj = neighbor_sets(task)
    feat = pca_feat(task.traces, 16)

    train_pairs, train_y = sample_pairs(task, args.pairs, args.pairs, rng)
    held = set(map(tuple, train_pairs.T.tolist()))
    test_pairs, test_y = sample_pairs(task, min(512, args.pairs), min(512, args.pairs), rng, exclude=held)

    model = SEAL(n_feat=feat.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best = None
    best_state = None
    for epoch in range(args.epochs):
        train_loss, _ = run_epoch(model, adj, feat, train_pairs, train_y, opt, loss_fn, True)
        test_loss, probs = run_epoch(model, adj, feat, test_pairs, test_y, opt, loss_fn, False)
        metrics = summarize(test_y, probs)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss, **metrics})
        print(
            f"epoch {epoch+1:03d}  train {train_loss:.4f}  test {test_loss:.4f}  "
            f"AP {metrics['ap']:.4f}  AUROC {metrics['auroc']:.4f}"
        )
        if best is None or metrics["ap"] > best:
            best = metrics["ap"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    out_dir = Path("results/model_functional")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_feat": feat.shape[1]}, out_dir / "seal.pt")
    (out_dir / "seal_history.json").write_text(json.dumps({"best_ap": best, "history": history}, indent=2))
    print(f"best AP {best:.4f}")
    print(f"Wrote {out_dir / 'seal.pt'}")


if __name__ == "__main__":
    main()
