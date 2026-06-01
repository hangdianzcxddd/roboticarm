"""Download the YOLO26n pretrained model into the project models directory."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from robot_control.shared.paths import MODELS_DIR


DEFAULT_MODEL_PATH = MODELS_DIR / "yolo26n.pt"
DEFAULT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the YOLO26n pretrained model into piper_project/models."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"Model already exists: {output}")
        return

    with urllib.request.urlopen(args.url, timeout=60) as response:
        with output.open("wb") as model_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                model_file.write(chunk)

    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
