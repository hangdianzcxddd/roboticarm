"""Image-space conveyor belt depth plane calibration and compensation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BeltDepthPlaneResiduals:
    mean_mm: float
    rms_mm: float
    max_abs_mm: float


@dataclass(frozen=True)
class BeltDepthPlane:
    width: int
    height: int
    a: float
    b: float
    c: float
    roi_xywh: tuple[int, int, int, int] | None = None
    patch_size: int = 15
    sample_count: int = 0
    residuals: BeltDepthPlaneResiduals | None = None
    model: str = "image_depth_plane_v1"

    def depth_at(self, u: float, v: float) -> float:
        return float(self.a) * float(u) + float(self.b) * float(v) + float(self.c)

    def validate_resolution(self, width: int, height: int) -> None:
        if int(width) != self.width or int(height) != self.height:
            raise ValueError(
                "belt depth plane resolution "
                f"{self.width}x{self.height} does not match frame {width}x{height}"
            )


def depth_array_meters(depth_frame: Any) -> np.ndarray:
    if hasattr(depth_frame, "get_data") and hasattr(depth_frame, "get_units"):
        raw = np.asanyarray(depth_frame.get_data())
        return raw.astype(np.float32) * float(depth_frame.get_units())

    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    depth_m = np.zeros((height, width), dtype=np.float32)
    for v in range(height):
        for u in range(width):
            depth_m[v, u] = float(depth_frame.get_distance(u, v))
    return depth_m


def valid_depth_values(
    depth_m: np.ndarray,
    mask: np.ndarray,
    *,
    max_depth_m: float = 0.0,
) -> np.ndarray:
    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError(f"mask shape {mask.shape[:2]} does not match depth shape {depth_m.shape[:2]}")
    mask_bool = mask.astype(bool)
    valid = mask_bool & np.isfinite(depth_m) & (depth_m > 0.0)
    if max_depth_m > 0.0:
        valid &= depth_m <= float(max_depth_m)
    return depth_m[valid].astype(float)


def mask_depth_percentile(
    depth_frame: Any,
    mask: np.ndarray,
    *,
    percentile: float,
    max_depth_m: float = 0.0,
) -> tuple[float, int]:
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be in [0, 100]")
    values = valid_depth_values(
        depth_array_meters(depth_frame),
        mask,
        max_depth_m=max_depth_m,
    )
    if values.size == 0:
        raise ValueError("mask has no valid depth pixels")
    return float(np.percentile(values, float(percentile))), int(values.size)


def patch_depth_median(
    depth_frame: Any,
    u: int,
    v: int,
    *,
    patch_size: int,
    max_depth_m: float = 0.0,
) -> float:
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    depth_m = depth_array_meters(depth_frame)
    height, width = depth_m.shape[:2]
    u = int(u)
    v = int(v)
    if u < 0 or u >= width or v < 0 or v >= height:
        raise ValueError(f"pixel ({u}, {v}) is outside depth frame {width}x{height}")
    radius = int(patch_size) // 2
    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0.0)
    if max_depth_m > 0.0:
        valid &= patch <= float(max_depth_m)
    values = patch[valid]
    if values.size == 0:
        raise ValueError(f"patch around ({u}, {v}) has no valid depth pixels")
    return float(np.median(values))


def fit_belt_depth_plane(
    samples: list[tuple[float, float, float]],
    *,
    width: int,
    height: int,
    roi_xywh: tuple[int, int, int, int] | None = None,
    patch_size: int = 15,
) -> BeltDepthPlane:
    if len(samples) < 3:
        raise ValueError("at least 3 samples are required to fit a belt depth plane")
    matrix = np.asarray([[float(u), float(v), 1.0] for u, v, _ in samples], dtype=float)
    values = np.asarray([float(depth_m) for _, _, depth_m in samples], dtype=float)
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    predicted = matrix @ coefficients
    residual_m = predicted - values
    residuals = BeltDepthPlaneResiduals(
        mean_mm=float(np.mean(residual_m) * 1000.0),
        rms_mm=float(np.sqrt(np.mean(np.square(residual_m))) * 1000.0),
        max_abs_mm=float(np.max(np.abs(residual_m)) * 1000.0),
    )
    return BeltDepthPlane(
        width=int(width),
        height=int(height),
        a=float(coefficients[0]),
        b=float(coefficients[1]),
        c=float(coefficients[2]),
        roi_xywh=roi_xywh,
        patch_size=int(patch_size),
        sample_count=len(samples),
        residuals=residuals,
    )


def save_belt_depth_plane(path: Path, plane: BeltDepthPlane) -> None:
    payload: dict[str, Any] = asdict(plane)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_belt_depth_plane(path: Path) -> BeltDepthPlane:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != "image_depth_plane_v1":
        raise ValueError(f"unsupported belt depth plane model: {payload.get('model')!r}")
    residual_payload = payload.get("residuals")
    residuals = None
    if residual_payload is not None:
        residuals = BeltDepthPlaneResiduals(
            mean_mm=float(residual_payload["mean_mm"]),
            rms_mm=float(residual_payload["rms_mm"]),
            max_abs_mm=float(residual_payload["max_abs_mm"]),
        )
    roi_payload = payload.get("roi_xywh")
    roi_xywh = None if roi_payload is None else tuple(int(item) for item in roi_payload)
    if roi_xywh is not None and len(roi_xywh) != 4:
        raise ValueError("belt depth plane roi_xywh must contain 4 values")
    return BeltDepthPlane(
        width=int(payload["width"]),
        height=int(payload["height"]),
        a=float(payload["a"]),
        b=float(payload["b"]),
        c=float(payload["c"]),
        roi_xywh=roi_xywh,
        patch_size=int(payload.get("patch_size", 15)),
        sample_count=int(payload.get("sample_count", 0)),
        residuals=residuals,
        model=str(payload.get("model", "image_depth_plane_v1")),
    )
