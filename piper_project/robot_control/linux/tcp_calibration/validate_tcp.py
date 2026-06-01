"""Validate a solved TCP calibration on the Linux robot host."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("tcp.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate TCP calibration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(
        "TCP calibration validation is not implemented yet. "
        f"Config path: {args.config}"
    )


if __name__ == "__main__":
    main()
