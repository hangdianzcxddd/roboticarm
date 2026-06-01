"""Solve TCP calibration from collected pose pairs."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INPUT = Path(__file__).with_name("tcp_samples.yaml")
DEFAULT_OUTPUT = Path(__file__).with_name("tcp.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve TCP calibration.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(
        "TCP calibration solving is not implemented yet. "
        f"Input: {args.input}; output: {args.output}"
    )


if __name__ == "__main__":
    main()
