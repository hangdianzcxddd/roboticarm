"""Collect TCP calibration pose pairs.

This module is a placeholder for the Linux-side data collection workflow. The
real collector should record robot TCP poses together with the corresponding
camera-frame points received from the Windows vision host.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).with_name("tcp_samples.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect TCP calibration samples.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(
        "TCP calibration collection is not implemented yet. "
        f"Planned output path: {args.output}"
    )


if __name__ == "__main__":
    main()
