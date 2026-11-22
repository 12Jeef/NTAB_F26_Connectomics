"""Plot one neuron's oracle calcium on a time axis.

Ten repeats of the six 10 s clips, the trial average (signal), and the
residual used for G_f.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from microns.oracle import ORACLE_DIR, oracle_batch_path

BG = "#f7f4ef"
INK = "#1f1c18"
MUTED = "#5c5a56"
TEAL = "#1d7a6d"
CORAL = "#c45c44"
NAVY = "#3d5a80"
SLATE = "#6d597a"
CLIP_COLORS = ["#1d7a6d22", "#3d5a8022", "#c45c4422", "#6d597a22", "#81b29a33", "#e07a5f22"]
SHORT_NAMES = {
    "run82-ancientTomb-1-0": "ancient tomb",
    "Mad Max: Fury Road (2015)": "Mad Max",
    "americandiner_005-1-5-70-0_54.165.28.255": "diner",
    "2009 Krugers Cross Singlespeed Cyclocross Race - Helmet Cam": "cyclocross",
    "Star Wars: Episode VII - The Force Awakens (2015)": "Star Wars",
    "2012 USA Basketball vs The Dream Team 1992 USA Basketball - NBA 2K13": "basketball",
}


def short_movie(name: str) -> str:
    name = str(name)
    return SHORT_NAMES.get(name, name.split("-")[0][:18])


def load_scan(session: int = 4, scan_idx: int = 7):
    path = oracle_batch_path(session, scan_idx)
    if not path.exists():
        files = sorted(ORACLE_DIR.glob("session*_scan*.npz"))
        if not files:
            raise FileNotFoundError(f"No oracle batches in {ORACLE_DIR}")
        path = files[0]
    return np.load(path, allow_pickle=True), path


def pick_neuron(bundle) -> int:
    scores = np.asarray(bundle["oracle_score"], dtype=np.float64)
    return int(np.argmax(scores))


def style(ax) -> None:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c2b6")
    ax.spines["bottom"].set_color("#c9c2b6")
    ax.tick_params(colors=MUTED, labelsize=9)


def shade_clips(ax, t: np.ndarray, n_clips: int, movies: np.ndarray) -> None:
    n = len(t)
    width = n // n_clips
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    for i in range(n_clips):
        t0 = t[i * width]
        t1 = t[min((i + 1) * width, n - 1)]
        ax.axvspan(t0, t1, color=CLIP_COLORS[i % len(CLIP_COLORS)], lw=0)
        ax.text(
            (t0 + t1) / 2,
            ymax,
            short_movie(movies[i]),
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
            clip_on=False,
        )
        if i > 0:
            ax.axvline(t0, color="#d9d2c6", lw=0.8, ls="--")
    ax.set_ylim(ymin, ymax)


def main() -> None:
    bundle, src = load_scan()
    i = pick_neuron(bundle)
    repeats = np.asarray(bundle["repeats"][i], dtype=np.float64)
    signal = np.asarray(bundle["signal"][i], dtype=np.float64)
    residual = np.asarray(bundle["residual"][i], dtype=np.float64)
    dt = float(np.asarray(bundle["dt"]))
    movies = bundle["movie_name"]
    n_bins = repeats.shape[1]
    n_clips = len(movies)
    t = np.arange(n_bins) * dt
    root = int(bundle["pt_root_id"][i])
    unit = int(bundle["unit_id"][i])
    session = int(bundle["session"][i])
    scan = int(bundle["scan_idx"][i])
    score = float(bundle["oracle_score"][i])

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.2, 8.4),
        facecolor=BG,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.15], "hspace": 0.38},
    )
    fig.suptitle(
        f"Oracle calcium  ·  session {session} scan {scan}  ·  unit {unit}",
        fontsize=14,
        color=INK,
        fontweight="medium",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        f"EM {root}   ·   6 clips × 10 repeats   ·   oracle score {score:.2f}   ·   {src.name}",
        ha="center",
        fontsize=9,
        color=MUTED,
    )

    ax = axes[0]
    style(ax)
    for r in range(repeats.shape[0]):
        ax.plot(t, repeats[r], color=TEAL, lw=0.7, alpha=0.22)
    ax.plot(t, signal, color=TEAL, lw=2.0, label="trial average (signal)")
    ax.set_ylabel("inferred activity", color=INK, fontsize=10)
    ax.set_xlim(t[0], t[-1])
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    shade_clips(ax, t, n_clips, movies)
    ax.set_title("Ten oracle repeats (faint) and the mean", loc="left", fontsize=11, color=INK, pad=18)

    ax = axes[1]
    style(ax)
    for r in range(min(4, residual.shape[0])):
        ax.plot(t, residual[r], color=CORAL, lw=0.85, alpha=0.55)
    ax.axhline(0, color="#c9c2b6", lw=0.8)
    ax.set_ylabel("residual", color=INK, fontsize=10)
    ax.set_xlim(t[0], t[-1])
    ax.set_title(
        "Residual = each repeat minus the mean  (this is what G_f correlates)",
        loc="left",
        fontsize=11,
        color=INK,
        pad=8,
    )

    ax = axes[2]
    style(ax)
    scores = np.asarray(bundle["oracle_score"], dtype=np.float64)
    others = np.argsort(scores)[::-1][:6]
    raster = np.asarray(bundle["signal"][others], dtype=np.float64)
    raster = raster - raster.mean(axis=1, keepdims=True)
    std = raster.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    raster = raster / std
    offset = 0
    for row, idx in enumerate(others):
        y = raster[row] + offset
        color = TEAL if idx == i else NAVY
        ax.plot(t, y, color=color, lw=1.15 if idx == i else 0.9, alpha=0.95)
        ax.text(
            t[-1] + 0.6,
            offset,
            f"unit {int(bundle['unit_id'][idx])}",
            va="center",
            fontsize=8,
            color=TEAL if idx == i else MUTED,
        )
        offset -= 3.4
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time within one 60 s oracle block (s)", color=INK, fontsize=10)
    ax.set_ylabel("z-scored signal\n(offset)", color=INK, fontsize=10)
    ax.set_yticks([])
    ax.set_title("Same scan, six neurons  (highlighted cell on top)", loc="left", fontsize=11, color=INK, pad=8)

    out = Path("data/processed/calcium_trace_timeline.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"neuron {root}  session {session}-{scan}  unit {unit}  oracle {score:.3f}")


if __name__ == "__main__":
    main()
