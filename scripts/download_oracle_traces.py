"""Download oracle-repeat calcium windows from DANDI 000402, then rebuild G_f.

    python scripts/download_oracle_traces.py
    python scripts/download_oracle_traces.py --scan 4-7
    python scripts/download_oracle_traces.py --rebuild
"""

from __future__ import annotations

import argparse

from microns.oracle import DANDI_SCANS, download_oracle_scans


def parse_scan(text: str) -> tuple[int, int]:
    session, scan_idx = text.replace(",", "-").split("-")
    return int(session), int(scan_idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", help="One batch, like 4-7")
    parser.add_argument("--rebuild", action="store_true", help="Skip download, rebuild graph")
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    if not args.rebuild:
        scans = [parse_scan(args.scan)] if args.scan else list(DANDI_SCANS)
        download_oracle_scans(scans)

    if args.scan and not args.rebuild:
        print("Single-scan download done. Rebuild G_f after all scans with:")
        print("  python scripts/download_oracle_traces.py --rebuild")
        return

    from build_functional_knn import write_oracle_graph

    write_oracle_graph(args.k)


if __name__ == "__main__":
    main()
