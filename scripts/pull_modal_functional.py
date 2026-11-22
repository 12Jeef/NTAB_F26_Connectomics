"""Copy the shared functional graph from Modal volume ntab-functional.

Requires `modal` CLI login to workspace ntab-connectomics:

    modal token new
    python scripts/pull_modal_functional.py
"""

from __future__ import annotations

import subprocess
import sys

from microns import config


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.DATA_CALCIUM.mkdir(parents=True, exist_ok=True)
    cmds = [
        [
            "modal",
            "volume",
            "get",
            "--env",
            "main",
            "ntab-functional",
            "/processed",
            str(config.DATA_PROCESSED),
        ],
        [
            "modal",
            "volume",
            "get",
            "--env",
            "main",
            "ntab-functional",
            "/calcium",
            str(config.DATA_CALCIUM),
        ],
    ]
    for cmd in cmds:
        print(" ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
