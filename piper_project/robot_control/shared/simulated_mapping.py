"""Temporary camera-to-robot mapping used before real spatial calibration exists."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from robot_control.shared.geometry import EndPose


@dataclass(frozen=True)
class SimulatedMappingConfig:
    """Small-amplitude mapping from camera-frame meters to robot-frame millimeters."""

    base_x_mm: float = 150.0
    base_y_mm: float = 0.0
    base_z_mm: float = 220.0
    scale_x: float = 120.0
    scale_y: float = 120.0
    scale_z: float = 80.0
    max_delta_x_mm: float = 40.0
    max_delta_y_mm: float = 40.0
    max_delta_z_mm: float = 30.0
    rx_deg: float = 0.0
    ry_deg: float = 85.0
    rz_deg: float = 0.0


def _clamp(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def camera_xyz_to_simulated_robot_pose(
    x_m: float,
    y_m: float,
    z_m: float,
    config: SimulatedMappingConfig | None = None,
) -> EndPose:
    """Map camera XYZ to a conservative robot pose for communication testing.

    This is not a calibrated transform. It intentionally creates small, bounded
    robot-frame changes around a known nominal pose so the TCP flow can be tested
    before hand-eye or point-pair calibration is available.
    """

    cfg = config or SimulatedMappingConfig()
    dx = _clamp(x_m * cfg.scale_x, cfg.max_delta_x_mm)
    dy = _clamp(y_m * cfg.scale_y, cfg.max_delta_y_mm)
    dz = _clamp((z_m - 0.30) * cfg.scale_z, cfg.max_delta_z_mm)
    return EndPose(
        x_mm=cfg.base_x_mm + dx,
        y_mm=cfg.base_y_mm + dy,
        z_mm=cfg.base_z_mm + dz,
        rx_deg=cfg.rx_deg,
        ry_deg=cfg.ry_deg,
        rz_deg=cfg.rz_deg,
    )


def pose_to_dict(pose: EndPose) -> dict[str, float]:
    return asdict(pose)
