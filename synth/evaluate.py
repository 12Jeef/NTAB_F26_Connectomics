"""Usage: python -m synth.evaluate   (|corr| baseline: coupling=0 control vs coupling on)"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from synth.config import Config
from synth.dataset import build


def matched_auc(labels, scores, distances, bin_um=10.0):
    """ROC-AUC with negatives reweighted to the positives' distance distribution."""
    bins = (distances // bin_um).astype(int)
    pos = np.bincount(bins, labels)
    neg = np.bincount(bins, ~labels)
    weights = np.where(labels, (neg > 0)[bins], (pos / np.maximum(neg, 1))[bins])
    return roc_auc_score(labels, scores, sample_weight=weights)


def pairs(ds):
    """Every ordered pair: (connected either direction, functional strength, distance µm)."""
    index = ds.functional.index
    pre, post = (index.get_level_values(level) for level in ("pre_id", "post_id"))
    xyz = ds.neurons[["x", "y", "z"]]
    distances = np.linalg.norm(xyz.loc[pre].to_numpy() - xyz.loc[post].to_numpy(), axis=1)
    reverse = pd.MultiIndex.from_arrays([post, pre])
    connected = index.isin(ds.synapses.index) | reverse.isin(ds.synapses.index)
    return connected, ds.functional["strength"].to_numpy(), distances


if __name__ == "__main__":
    for coupling in (0.0, Config.coupling):
        connected, strength, distances = pairs(build(Config(coupling=coupling)))
        print(f"coupling={coupling}: distance-matched AUC (|corr|) = "
              f"{matched_auc(connected, np.abs(strength), distances):.3f}")
