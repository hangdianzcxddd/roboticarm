#!/usr/bin/env python3
"""Offline multi-start Piper IK diagnostic (never connects to CAN)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Permit direct execution by path from any working directory, matching the
# other repository command-line scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_control.linux.arm.kinematics import forward_kinematics_pose
from robot_control.linux.catch.tcp_offset import rotation_matrix_xyz_deg
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.shared.geometry import EndPose

TOOL_Z_OFFSET_MM = 82.0


@dataclass(frozen=True)
class Candidate:
    q: tuple[float, ...]
    seed: int
    pos_mm: float
    ori_deg: float
    condition: float
    nfev: int


def pose_rotation(pose: EndPose) -> np.ndarray:
    return rotation_matrix_xyz_deg(pose.rx_deg, pose.ry_deg, pose.rz_deg)


def corrected_fk(q: Sequence[float], offset_mm: float) -> EndPose:
    wrist = forward_kinematics_pose(q)
    offset = pose_rotation(wrist) @ np.array([0.0, 0.0, offset_mm])
    return EndPose(
        wrist.x_mm + float(offset[0]), wrist.y_mm + float(offset[1]),
        wrist.z_mm + float(offset[2]), wrist.rx_deg, wrist.ry_deg, wrist.rz_deg,
    )


def residual(q: np.ndarray, target: EndPose, offset_mm: float) -> np.ndarray:
    solved = corrected_fk(q, offset_mm)
    relative = Rotation.from_matrix(
        pose_rotation(solved).T @ pose_rotation(target)
    ).as_rotvec()
    return np.r_[
        (solved.x_mm - target.x_mm) / 100.0,
        (solved.y_mm - target.y_mm) / 100.0,
        (solved.z_mm - target.z_mm) / 100.0,
        relative,
    ]


def joint_bounds() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([item.min_value for item in ROBOT_CONFIG.joint_limits_rad]),
        np.array([item.max_value for item in ROBOT_CONFIG.joint_limits_rad]),
    )


def condition(q: Sequence[float], target: EndPose, offset_mm: float) -> float:
    q = np.asarray(q, dtype=float)
    jac = np.empty((6, 6), dtype=float)
    eps = 1e-5
    for i in range(6):
        plus, minus = q.copy(), q.copy()
        plus[i] += eps
        minus[i] -= eps
        jac[:, i] = (residual(plus, target, offset_mm) - residual(minus, target, offset_mm)) / (2 * eps)
    return float(np.linalg.cond(jac))


def parse_degrees(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected six comma-separated degrees") from exc
    if len(values) != 6 or not all(math.isfinite(x) for x in values):
        raise argparse.ArgumentTypeError("expected six finite joint angles")
    return values


def seeds(count: int, random_seed: int, start_deg: tuple[float, ...] | None) -> tuple[tuple[float, ...], ...]:
    lower, upper = joint_bounds()
    center = (lower + upper) / 2.0
    reference = center if start_deg is None else np.radians(start_deg)
    reference = np.clip(reference, lower + 1e-7, upper - 1e-7)
    masks: tuple[object, ...] = (
        None, "center", (1, 2), (3, 4, 5), (1, 2, 3, 4, 5),
        "j1_plus", "j1_minus_elbow", "j1_plus_wrist",
    )
    result: list[tuple[float, ...]] = []
    for mask in masks:
        if len(result) >= count:
            break
        q = center.copy() if mask == "center" else reference.copy()
        if mask == "j1_plus":
            q[0] += math.radians(90)
        elif mask == "j1_minus_elbow":
            q[0] -= math.radians(90)
            q[[1, 2]] = lower[[1, 2]] + upper[[1, 2]] - q[[1, 2]]
        elif mask == "j1_plus_wrist":
            q[0] += math.radians(90)
            q[[3, 4, 5]] = lower[[3, 4, 5]] + upper[[3, 4, 5]] - q[[3, 4, 5]]
        elif mask is not None and mask != "center":
            index = np.asarray(mask, dtype=int)
            q[index] = lower[index] + upper[index] - q[index]
        result.append(tuple(float(x) for x in np.clip(q, lower + 1e-7, upper - 1e-7)))
    rng = np.random.default_rng(random_seed)
    while len(result) < count:
        q = rng.uniform(lower, upper) if len(result) % 2 else np.clip(
            reference + rng.normal(0, math.radians(35), 6), lower + 1e-7, upper - 1e-7
        )
        result.append(tuple(float(x) for x in q))
    return tuple(result)


def solve(target: EndPose, initial_seeds: Sequence[tuple[float, ...]], args: argparse.Namespace) -> tuple[list[Candidate], list[tuple]]:
    lower, upper = joint_bounds()
    candidates: list[Candidate] = []
    attempts: list[tuple] = []
    for seed_number, initial in enumerate(initial_seeds, 1):
        result = least_squares(
            residual, np.asarray(initial), args=(target, args.tool_z_offset_mm),
            bounds=(lower, upper), max_nfev=args.max_nfev,
            xtol=1e-9, ftol=1e-9, gtol=1e-9,
        )
        q = tuple(float(x) for x in result.x)
        error = residual(np.asarray(q), target, args.tool_z_offset_mm)
        pos = float(np.linalg.norm(error[:3]) * 100)
        ori = math.degrees(float(np.linalg.norm(error[3:])))
        cond = condition(q, target, args.tool_z_offset_mm)
        valid = pos <= args.position_tolerance_mm and ori <= args.orientation_tolerance_deg and math.isfinite(cond) and cond <= args.max_condition
        duplicate = any(max(abs(a - b) for a, b in zip(q, old.q)) <= math.radians(args.dedupe_deg) for old in candidates)
        reason = "valid" if valid and not duplicate else ("duplicate" if valid else "rejected")
        attempts.append((seed_number, q, pos, ori, cond, result.nfev, reason))
        if valid and not duplicate:
            candidates.append(Candidate(q, seed_number, pos, ori, cond, result.nfev))
    candidates.sort(key=lambda item: (item.pos_mm, item.ori_deg, item.condition))
    return candidates, attempts


def current_waypoints() -> tuple[tuple[str, EndPose], ...]:
    return (
        ("pick_above", EndPose(360.600, 0, 253.252, -175.958, 1.353, -102.579)),
        ("pick_grasp", EndPose(360.600, 0, 155.391, -175.958, 1.353, -91.996)),
        ("pick_lift", EndPose(360.600, 0, 253.252, -175.958, 1.353, -91.996)),
    )


def path_safe(start: Sequence[float], goal: Sequence[float], step_deg: float) -> bool:
    lower, upper = joint_bounds()
    start, goal = np.asarray(start), np.asarray(goal)
    count = max(1, int(math.ceil(np.max(np.abs(np.degrees(goal - start))) / step_deg)))
    for fraction in np.linspace(0, 1, count + 1):
        q = start + fraction * (goal - start)
        if np.any(q < lower - 1e-9) or np.any(q > upper + 1e-9):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-current-waypoints", action="store_true")
    parser.add_argument("--x", type=float, default=360.6)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=253.252)
    parser.add_argument("--rx", type=float, default=-175.958)
    parser.add_argument("--ry", type=float, default=1.353)
    parser.add_argument("--rz", type=float, default=-102.579)
    parser.add_argument("--seed-count", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=20260805)
    parser.add_argument("--start-joints-deg", type=parse_degrees)
    parser.add_argument("--tool-z-offset-mm", type=float, default=TOOL_Z_OFFSET_MM)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.2)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.2)
    parser.add_argument("--max-condition", type=float, default=500.0)
    parser.add_argument("--dedupe-deg", type=float, default=0.1)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--path-step-deg", type=float, default=2.0)
    args = parser.parse_args()
    if args.seed_count < 8:
        parser.error("--seed-count must be at least 8")
    waypoints = current_waypoints() if args.all_current_waypoints else ((
        "target", EndPose(args.x, args.y, args.z, args.rx, args.ry, args.rz)
    ),)
    initial_seeds = seeds(args.seed_count, args.random_seed, args.start_joints_deg)
    print(f"Offline IK only; no CAN/Piper commands. seeds={args.seed_count} tool_z_offset={args.tool_z_offset_mm:.1f}mm")
    all_candidates: list[list[Candidate]] = []
    for name, pose in waypoints:
        started = time.perf_counter()
        candidates, attempts = solve(pose, initial_seeds, args)
        all_candidates.append(candidates)
        print(f"\n[{name}] elapsed={time.perf_counter()-started:.3f}s candidates={len(candidates)}")
        for i, item in enumerate(candidates, 1):
            print(f"  candidate={i} seed={item.seed} q_deg={tuple(round(math.degrees(x),3) for x in item.q)} pos_err={item.pos_mm:.4f}mm ori_err={item.ori_deg:.4f}deg cond={item.condition:.2f}")
        if not candidates:
            print("  no valid full-pose solution; best attempts:")
            for item in sorted(attempts, key=lambda x: (x[2], x[3]))[:8]:
                print(f"  seed={item[0]} q_deg={tuple(round(math.degrees(x),3) for x in item[1])} pos_err={item[2]:.4f}mm ori_err={item[3]:.4f}deg cond={item[4]:.2f} {item[6]}")
    if all(all_candidates):
        start = np.radians(args.start_joints_deg) if args.start_joints_deg else all_candidates[0][0].q
        route_ok = True
        print("\nJoint-limit-only route connectivity:")
        for (name, _), candidates in zip(waypoints, all_candidates):
            choices = [candidate for candidate in candidates if path_safe(start, candidate.q, args.path_step_deg)]
            if not choices:
                print(f"  previous -> {name}: NO")
                route_ok = False
                break
            selected = min(choices, key=lambda candidate: np.linalg.norm(np.asarray(candidate.q) - start))
            print(f"  previous -> {name}: YES (candidate seed={selected.seed})")
            start = selected.q
        print("PATH RESULT: " + ("a joint-limit-safe route exists" if route_ok else "no complete route found"))
        print("Note: collision/fixture clearance and Piper firmware branch selection are not modelled.")
    else:
        print("\nPATH RESULT: no route can satisfy every requested full pose because at least one endpoint has no valid candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
