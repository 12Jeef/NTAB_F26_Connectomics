"""Classification metrics for directed synapse prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


def summarize(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pred = scores >= threshold
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else 0.0,
        "ap": float(average_precision_score(y_true, scores)) if y_true.min() != y_true.max() else float("nan"),
        "auroc": float(roc_auc_score(y_true, scores)) if y_true.min() != y_true.max() else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)) if y_true.min() != y_true.max() else float("nan"),
        "recall": float(recall_score(y_true, pred, zero_division=0)) if y_true.min() != y_true.max() else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)) if y_true.min() != y_true.max() else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if y_true.min() != y_true.max() else float("nan"),
    }
    return out
