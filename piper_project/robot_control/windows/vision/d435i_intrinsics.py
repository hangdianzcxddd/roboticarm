"""Load D435i rectified intrinsics exported from calibration parameters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_D435I_PARA = Path(__file__).with_name("d435i_para.json")


@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    source: str = "unknown"


def load_d435i_rectified_intrinsics(path: Path = DEFAULT_D435I_PARA) -> list[PinholeIntrinsics]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"D435i parameter file does not exist: {path}") from None
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")

    result: list[PinholeIntrinsics] = []
    indices = sorted(
        {
            key.split(".")[1]
            for key in document
            if key.startswith("rectified.") and key.endswith(".width")
        },
        key=lambda value: int(value),
    )
    for index in indices:
        prefix = f"rectified.{index}"
        intrinsics = _intrinsics_from_section(document, prefix)
        if intrinsics is not None:
            result.append(intrinsics)
    if not result:
        raise ValueError(f"{path} does not contain any usable rectified intrinsics")
    return result


def find_d435i_intrinsics(
    width: int,
    height: int,
    path: Path = DEFAULT_D435I_PARA,
) -> PinholeIntrinsics:
    matches = [
        intrinsics
        for intrinsics in load_d435i_rectified_intrinsics(path)
        if intrinsics.width == int(width) and intrinsics.height == int(height)
    ]
    if not matches:
        available = ", ".join(
            f"{item.width}x{item.height}" for item in load_d435i_rectified_intrinsics(path)
        )
        raise ValueError(
            f"no D435i rectified intrinsics for {width}x{height}; available: {available}"
        )
    return matches[0]


def format_intrinsics(intrinsics: PinholeIntrinsics) -> str:
    return (
        f"{intrinsics.source}: {intrinsics.width}x{intrinsics.height} "
        f"fx={intrinsics.fx:.6f} fy={intrinsics.fy:.6f} "
        f"ppx={intrinsics.ppx:.6f} ppy={intrinsics.ppy:.6f}"
    )


def _intrinsics_from_section(
    document: dict[str, Any],
    prefix: str,
) -> PinholeIntrinsics | None:
    width = _number(document.get(f"{prefix}.width"))
    height = _number(document.get(f"{prefix}.height"))
    fx = _number(document.get(f"{prefix}.fx"))
    fy = _number(document.get(f"{prefix}.fy"))
    ppx = _number(document.get(f"{prefix}.ppx"))
    ppy = _number(document.get(f"{prefix}.ppy"))
    values = (width, height, fx, fy, ppx, ppy)
    if any(value is None for value in values):
        return None
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        return None
    return PinholeIntrinsics(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        ppx=float(ppx),
        ppy=float(ppy),
        source=prefix,
    )


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
