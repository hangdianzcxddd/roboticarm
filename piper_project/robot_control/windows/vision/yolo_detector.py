"""YOLO detection utilities for RGB target center extraction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from robot_control.shared.paths import MODELS_DIR
from robot_control.windows.vision.detector import Detection


DEFAULT_MODEL_PATH = MODELS_DIR / "yolo26n.pt"


def configure_ultralytics_config_dir(models_dir: Path = MODELS_DIR) -> Path:
    """Keep Ultralytics runtime settings inside the project models directory."""

    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(models_dir)
    return models_dir


def require_yolo_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {path}. Download it to models first."
        )
    return path


def _load_ultralytics_yolo() -> Any:
    configure_ultralytics_config_dir()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO detection. "
            "Install project requirements first."
        ) from exc
    return YOLO


class YOLOObjectDetector:
    """Run Ultralytics YOLO and return image-space bbox center detections."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_results: int = 1,
        imgsz: int = 640,
        device: str | None = None,
        class_ids: list[int] | None = None,
    ) -> None:
        YOLO = _load_ultralytics_yolo()
        self.model_path = require_yolo_model(model_path)
        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_results = int(max_results)
        self.imgsz = int(imgsz)
        self.device = device
        self.class_ids = class_ids
        self.names = getattr(self.model, "names", {}) or {}

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.class_ids,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for bbox, confidence, class_id in zip(xyxy, confidences, classes):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            u = int(round((x1 + x2) / 2.0))
            v = int(round((y1 + y2) / 2.0))
            label = str(self.names.get(int(class_id), f"class_{int(class_id)}"))
            detections.append(
                Detection(
                    u=u,
                    v=v,
                    x=int(round(x1)),
                    y=int(round(y1)),
                    width=int(round(width)),
                    height=int(round(height)),
                    area=float(width * height),
                    label=label,
                    confidence=float(confidence),
                    class_id=int(class_id),
                )
            )

        detections.sort(
            key=lambda item: (
                item.confidence if item.confidence is not None else 0.0,
                item.area,
            ),
            reverse=True,
        )
        if self.max_results > 0:
            detections = detections[: self.max_results]
        return detections
