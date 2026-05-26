
"""OpenCV-based target detectors that return image-space target centers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A target candidate detected in RGB image coordinates."""

    u: int
    v: int
    x: int
    y: int
    width: int
    height: int
    area: float
    label: str = "target"
    confidence: float | None = None
    class_id: int | None = None


@dataclass(frozen=True)
class ContourDetectionConfig:
    min_area: float = 500.0
    max_area: Optional[float] = None
    max_results: int = 1
    morphology_kernel_size: int = 5


def clean_binary_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Remove small noise and fill small holes in a binary foreground mask."""

    if mask.ndim != 2:
        raise ValueError("binary mask must be a single-channel image")

    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return binary

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def detections_from_mask(
    mask: np.ndarray,
    config: ContourDetectionConfig | None = None,
    label: str = "target",
) -> list[Detection]:
    """Find contour bounding boxes and center points from a foreground mask."""

    config = config or ContourDetectionConfig()
    clean_mask = clean_binary_mask(mask, config.morphology_kernel_size)
    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections: list[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.min_area:
            continue
        if config.max_area is not None and area > config.max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            u = int(round(moments["m10"] / moments["m00"]))
            v = int(round(moments["m01"] / moments["m00"]))
        else:
            u = int(round(x + width / 2.0))
            v = int(round(y + height / 2.0))

        detections.append(
            Detection(
                u=u,
                v=v,
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                area=area,
                label=label,
            )
        )

    detections.sort(key=lambda item: item.area, reverse=True)
    if config.max_results > 0:
        detections = detections[: config.max_results]
    return detections


class BackgroundSubtractorDetector:
    """Detect foreground targets using OpenCV MOG2 background subtraction."""

    def __init__(
        self,
        config: ContourDetectionConfig | None = None,
        history: int = 120,
        var_threshold: float = 25.0,
        detect_shadows: bool = True,
    ) -> None:
        self.config = config or ContourDetectionConfig()
        self.history = int(history)
        self.var_threshold = float(var_threshold)
        self.detect_shadows = bool(detect_shadows)
        self._subtractor = self._create_subtractor()
        self.last_mask: Optional[np.ndarray] = None

    def _create_subtractor(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    def reset(self) -> None:
        self._subtractor = self._create_subtractor()
        self.last_mask = None

    def apply(self, frame_bgr: np.ndarray, learning_rate: float = -1.0) -> np.ndarray:
        """Update the background model and return a cleaned foreground mask."""

        raw_mask = self._subtractor.apply(frame_bgr, learningRate=learning_rate)
        if self.detect_shadows:
            _, raw_mask = cv2.threshold(raw_mask, 244, 255, cv2.THRESH_BINARY)

        self.last_mask = clean_binary_mask(
            raw_mask,
            kernel_size=self.config.morphology_kernel_size,
        )
        return self.last_mask

    def detect(
        self,
        frame_bgr: np.ndarray,
        learning_rate: float = -1.0,
    ) -> list[Detection]:
        mask = self.apply(frame_bgr, learning_rate=learning_rate)
        return detections_from_mask(mask, self.config, label="foreground")


class ColorThresholdDetector:
    """Detect targets by HSV color range, useful for colored blocks or markers."""

    def __init__(
        self,
        lower_hsv: tuple[int, int, int],
        upper_hsv: tuple[int, int, int],
        config: ContourDetectionConfig | None = None,
    ) -> None:
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
        self.config = config or ContourDetectionConfig()
        self.last_mask: Optional[np.ndarray] = None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        self.last_mask = clean_binary_mask(
            raw_mask,
            kernel_size=self.config.morphology_kernel_size,
        )
        return detections_from_mask(self.last_mask, self.config, label="color")
