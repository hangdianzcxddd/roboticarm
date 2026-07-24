"""Adapter from project point clouds to the official GraspNet baseline.

This module is imported by ``graspnet_grasp_send_tcp.py`` via the default
``--graspnet-module graspnet_backend`` argument.

It intentionally does not vendor the official GraspNet model code. Put the
official ``graspnet-baseline`` repository at ``models/graspnet-baseline`` or set
``GRASPNET_BASELINE_DIR`` to its location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "models" / "checkpoint-rs.tar"


def candidate_dicts_from_grasp_array(
    grasp_array: np.ndarray,
    *,
    max_candidates: int = 50,
) -> list[dict[str, Any]]:
    """Convert official GraspNet rows to the local protocol-friendly shape."""
    rows = np.asarray(grasp_array, dtype=np.float32)
    if rows.size == 0:
        return []
    if rows.ndim != 2 or rows.shape[1] < 16:
        raise ValueError(f"expected GraspNet rows shaped (N, >=16), got {rows.shape}")
    order = np.argsort(rows[:, 0])[::-1]
    rows = rows[order]
    if max_candidates > 0:
        rows = rows[:max_candidates]
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "position_camera_m": row[13:16].astype(float).tolist(),
                "rotation_camera": row[4:13].reshape(3, 3).astype(float).tolist(),
                "width_m": float(row[1]),
                "score": float(row[0]),
                "approach_depth_m": float(row[3]),
                "source": "graspnet-baseline",
            }
        )
    return result


def sample_points_for_graspnet(
    points_xyz_m: np.ndarray,
    num_point: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    points = np.asarray(points_xyz_m, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected point cloud shaped (N, 3), got {points.shape}")
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if points.shape[0] >= num_point:
        indexes = rng.choice(points.shape[0], int(num_point), replace=False)
    else:
        existing = np.arange(points.shape[0])
        padded = rng.choice(
            points.shape[0],
            int(num_point) - points.shape[0],
            replace=True,
        )
        indexes = np.concatenate([existing, padded], axis=0)
    return points[indexes].astype(np.float32, copy=False)


class GraspNetPredictor:
    """Predict 6D grasp candidates with the official GraspNet baseline."""

    def __init__(
        self,
        checkpoint: str | os.PathLike[str] | None = None,
        device: str | None = None,
        root: str | os.PathLike[str] | None = None,
        num_point: int = 20000,
        num_view: int = 300,
        max_candidates: int = 50,
    ) -> None:
        self.root = _resolve_graspnet_root(root)
        _add_graspnet_paths(self.root)
        try:
            from graspnet import GraspNet, pred_decode
        except ImportError as exc:
            raise RuntimeError(_install_message(self.root)) from exc

        self.pred_decode = pred_decode
        self.num_point = int(num_point)
        self.max_candidates = int(max_candidates)
        if self.num_point <= 0:
            raise ValueError("num_point must be > 0")

        self.device = torch.device(
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        checkpoint_path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_DIR / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"GraspNet checkpoint does not exist: {checkpoint_path}")

        self.net = GraspNet(
            input_feature_dim=0,
            num_view=int(num_view),
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        self.net.to(self.device)
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.net.load_state_dict(checkpoint_payload["model_state_dict"])
        self.net.eval()
        epoch = checkpoint_payload.get("epoch", "unknown")
        print(
            f"Loaded GraspNet baseline checkpoint {checkpoint_path} "
            f"(epoch={epoch}, device={self.device})",
            flush=True,
        )

    def predict(self, points_xyz_m: np.ndarray) -> list[dict[str, Any]]:
        sampled = sample_points_for_graspnet(points_xyz_m, self.num_point)
        if sampled.size == 0:
            return []
        cloud = torch.from_numpy(sampled[np.newaxis, ...]).to(self.device)
        end_points = {"point_clouds": cloud}
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = self.pred_decode(end_points)
        grasp_array = grasp_preds[0].detach().cpu().numpy()
        return candidate_dicts_from_grasp_array(
            grasp_array,
            max_candidates=self.max_candidates,
        )


def _resolve_graspnet_root(root: str | os.PathLike[str] | None) -> Path:
    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root))
    env_root = os.environ.get("GRASPNET_BASELINE_DIR")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            PROJECT_DIR / "models" / "graspnet-baseline",
            PROJECT_DIR / "models" / "graspnet_baseline",
            PROJECT_DIR.parent / "sdk" / "graspnet-baseline",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "models" / "graspnet.py").exists():
            return resolved
    searched = "\n  ".join(str(item) for item in candidates)
    raise RuntimeError(
        "Cannot find official graspnet-baseline source code. Searched:\n"
        f"  {searched}\n"
        "Clone it to models\\graspnet-baseline or set GRASPNET_BASELINE_DIR."
    )


def _add_graspnet_paths(root: Path) -> None:
    for path in (
        root,
        root / "models",
        root / "dataset",
        root / "utils",
        root / "pointnet2",
        root / "knn",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _install_message(root: Path) -> str:
    return (
        "Found graspnet-baseline source code but could not import its model. "
        "Compile/install its custom operators first:\n"
        f"  cd {root}\\pointnet2\n"
        "  python setup.py install\n"
        f"  cd {root}\\knn\n"
        "  python setup.py install\n"
        "Then rerun the grasp client."
    )
