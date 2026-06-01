"""Shared geometry dataclasses used on both Windows and Linux hosts."""

from __future__ import annotations

from dataclasses import dataclass


SDK_DEG_FACTOR = 1000


@dataclass(frozen=True)
class EndPose:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    def as_sdk_units(self) -> tuple[int, int, int, int, int, int]:
        return (
            round(self.x_mm * SDK_DEG_FACTOR),
            round(self.y_mm * SDK_DEG_FACTOR),
            round(self.z_mm * SDK_DEG_FACTOR),
            round(self.rx_deg * SDK_DEG_FACTOR),
            round(self.ry_deg * SDK_DEG_FACTOR),
            round(self.rz_deg * SDK_DEG_FACTOR),
        )
