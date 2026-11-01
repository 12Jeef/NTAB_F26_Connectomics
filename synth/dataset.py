"""Usage: python -m synth.dataset <out_dir> [key=value ...]   e.g. coupling=0 seed=1"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from synth.activity import simulate
from synth.config import Config
from synth.network import CLASSES, place, wire

ID_OFFSET = 9 * 10**17  # far from real MICrONS root ids
TABLES = ("neurons", "synapses", "functional", "coreg")


@dataclass
class Dataset:
    neurons: pd.DataFrame
    synapses: pd.DataFrame
    functional: pd.DataFrame
    coreg: pd.DataFrame
    traces: np.ndarray  # (frames, unit_id), single session/scan
    config: Config

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            getattr(self, name).to_parquet(path / f"{name}.parquet")
        np.save(path / "traces.npy", self.traces)
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls(
            **{name: pd.read_parquet(path / f"{name}.parquet") for name in TABLES},
            traces=np.load(path / "traces.npy"),
            config=Config(**json.loads((path / "config.json").read_text())),
        )


def pair_index(ids, pre, post):
    return pd.MultiIndex.from_arrays([ids[pre], ids[post]], names=["pre_id", "post_id"])


def build(cfg=Config()):
    rng = np.random.default_rng(cfg.seed)
    xyz, cls = place(cfg, rng)
    n, W = wire(cfg, xyz, cls, rng)
    traces = simulate(cfg, W, rng)

    ids = ID_OFFSET + np.arange(cfg.n_neurons, dtype=np.int64)
    index = pd.Index(ids, name="id")

    neurons = pd.DataFrame({
        "cell_class": CLASSES[cls],
        "cell_type": "none",
        "axon_proof": "axon_fully_extended",
        "dend_proof": "dendrite_extended",
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
    }, index=index)

    pre, post = np.nonzero(n)
    synapses = pd.DataFrame({"n": n[pre, post]}, index=pair_index(ids, pre, post))

    pre, post = np.nonzero(~np.eye(cfg.n_neurons, dtype=bool))
    corr = np.corrcoef(traces.T)
    functional = pd.DataFrame({"strength": corr[pre, post]}, index=pair_index(ids, pre, post))

    coreg = pd.DataFrame({"session": 1, "scan_idx": 1, "unit_id": np.arange(cfg.n_neurons)}, index=index)

    return Dataset(neurons, synapses, functional, coreg, traces, cfg)


if __name__ == "__main__":
    overrides = {k: json.loads(v) for k, v in (arg.split("=", 1) for arg in sys.argv[2:])}
    build(Config(**overrides)).save(sys.argv[1])
