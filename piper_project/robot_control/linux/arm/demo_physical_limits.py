"""Conservative, supervised joint-range and speed demonstration for Piper.

This is deliberately *not* a homing program or a hardware-limit finder.  It
only commands the approved, interior envelope in the profile.  A profile must
be checked on the installed robot, including its tool, cables, table, camera,
and any guarding, before ``--execute`` is used.

The program is dry-run by default.  During an execution it moves to zero at
low speed first, tests one joint at a time, and returns to zero between tests.
On Ctrl-C it overwrites the active target with the live joint angles and keeps
the drives enabled; it deliberately does not send a software emergency-stop,
because that can remove holding torque on a gravity-loaded arm.  The physical
E-stop must remain reachable throughout the demonstration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.config.robot_config import ROBOT_CONFIG


JOINT_NAMES = tuple(f"j{index}" for index in range(1, 7))
ZERO_HOME_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
MULTI_SIZE_FACTORS = {"small": 0.25, "normal": 0.50}
DEFAULT_PROFILE: dict[str, Any] = {
    "safe_home_deg": list(ZERO_HOME_DEG),
    "demo_ranges_deg": {
        "j1": [-140.0, 140.0],
        "j2": [10.0, 170.0],
        "j3": [-160.0, -10.0],
        "j4": [-90.0, 90.0],
        "j5": [-60.0, 60.0],
        "j6": [-50.0, 50.0],
    },
    # This is deliberately wider than the displayed range for J2/J3 only so
    # that the approved zero posture can be reached.  It too needs a site
    # clearance check; it is not a substitute for a collision model.
    "safe_motion_ranges_deg": {
        "j1": [-140.0, 140.0],
        "j2": [0.0, 170.0],
        "j3": [-160.0, 0.0],
        "j4": [-90.0, 90.0],
        "j5": [-60.0, 60.0],
        "j6": [-50.0, 50.0],
    },
    "speed_zones_deg": {
        "j1": [-60.0, 60.0],
        "j2": [55.0, 125.0],
        "j3": [-120.0, -50.0],
        "j4": [-45.0, 45.0],
        "j5": [-30.0, 30.0],
        "j6": [-30.0, 30.0],
    },
    "multi_speed_scenarios": [],
}


@dataclass(frozen=True)
class Profile:
    safe_home_deg: tuple[float, ...]
    demo_ranges_deg: dict[str, tuple[float, float]]
    safe_motion_ranges_deg: dict[str, tuple[float, float]]
    speed_zones_deg: dict[str, tuple[float, float]]
    multi_speed_scenarios: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RunOptions:
    speed: int
    home_speed: int
    tolerance_deg: float
    timeout_s: float
    feedback_timeout_s: float
    max_range_speed: int
    max_observed_speed_factor: float
    abort_hold_speed: int
    abort_hold_duration_s: float
    target_command_interval_s: float
    transition_step_deg: float


class DemoSafetyError(RuntimeError):
    """Raised when live feedback means the demonstration must stop."""


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _six_degrees(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{label} must be a list of six degrees values")
    return tuple(_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _ranges(value: object, label: str) -> dict[str, tuple[float, float]]:
    if not isinstance(value, dict) or set(value) != set(JOINT_NAMES):
        raise ValueError(f"{label} must contain exactly {', '.join(JOINT_NAMES)}")
    result: dict[str, tuple[float, float]] = {}
    for joint in JOINT_NAMES:
        pair = value[joint]
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{label}.{joint} must be [min_deg, max_deg]")
        low = _finite_number(pair[0], f"{label}.{joint}[0]")
        high = _finite_number(pair[1], f"{label}.{joint}[1]")
        if not low < high:
            raise ValueError(f"{label}.{joint} must have min_deg < max_deg")
        result[joint] = (low, high)
    return result


def load_profile(path: Path | None) -> Profile:
    document: dict[str, Any] = DEFAULT_PROFILE
    if path is not None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"cannot read profile {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in profile {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError("profile root must be a JSON object")
        document = {**DEFAULT_PROFILE, **loaded}

    home = _six_degrees(document["safe_home_deg"], "safe_home_deg")
    if home != ZERO_HOME_DEG:
        raise ValueError(
            "safe_home_deg must remain [0, 0, 0, 0, 0, 0]: this script always "
            "returns to the requested zero posture"
        )
    demo_ranges = _ranges(document["demo_ranges_deg"], "demo_ranges_deg")
    safe_motion_ranges = _ranges(document["safe_motion_ranges_deg"], "safe_motion_ranges_deg")
    speed_zones = _ranges(document["speed_zones_deg"], "speed_zones_deg")
    for index, joint in enumerate(JOINT_NAMES):
        safe_low, safe_high = safe_motion_ranges[joint]
        demo_low, demo_high = demo_ranges[joint]
        speed_low, speed_high = speed_zones[joint]
        if not safe_low <= demo_low < demo_high <= safe_high:
            raise ValueError(f"demo_ranges_deg.{joint} must be inside safe_motion_ranges_deg.{joint}")
        if not demo_low <= speed_low < speed_high <= demo_high:
            raise ValueError(f"speed_zones_deg.{joint} must be inside demo_ranges_deg.{joint}")
        if not safe_low <= home[index] <= safe_high:
            raise ValueError(f"safe_home_deg {joint} must be inside safe_motion_ranges_deg.{joint}")
    scenarios = document.get("multi_speed_scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("multi_speed_scenarios must be a list")
    checked_scenarios: list[dict[str, Any]] = []
    for number, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict) or not isinstance(scenario.get("name"), str):
            raise ValueError(f"multi_speed_scenarios[{number}] needs a name")
        start = _six_degrees(scenario.get("start_deg"), f"scenario {scenario['name']}.start_deg")
        target = _six_degrees(scenario.get("target_deg"), f"scenario {scenario['name']}.target_deg")
        if start == target:
            raise ValueError(f"scenario {scenario['name']} start_deg and target_deg must differ")
        for index, (start_angle, target_angle) in enumerate(zip(start, target)):
            joint = JOINT_NAMES[index]
            safe_low, safe_high = safe_motion_ranges[joint]
            if not (safe_low <= start_angle <= safe_high and safe_low <= target_angle <= safe_high):
                raise ValueError(f"scenario {scenario['name']} values for {joint} must be inside its safe motion range")
            if start_angle != target_angle:
                speed_low, speed_high = speed_zones[joint]
                if not (speed_low <= start_angle <= speed_high and speed_low <= target_angle <= speed_high):
                    raise ValueError(
                        f"moving {joint} in scenario {scenario['name']} must start and end inside its speed zone"
                    )
        checked_scenarios.append({"name": scenario["name"], "start_deg": start, "target_deg": target})
    return Profile(home, demo_ranges, safe_motion_ranges, speed_zones, tuple(checked_scenarios))


def degrees_to_radians(degrees: Iterable[float]) -> list[float]:
    return [math.radians(value) for value in degrees]


def build_selected_multi_scenario(
    profile: Profile,
    joints: Iterable[str],
    size: str,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Build a conservative round-trip for exactly the selected joints.

    Selected joints move between random points in the centre portion of their
    approved speed zones.  The supplied random generator makes a run
    reproducible from its seed.
    Unselected joints remain at the configured zero home posture throughout
    the test.  This only enforces per-joint ranges: the operator must verify
    each selected combination is collision-free on the installed robot.
    """
    selected = tuple(dict.fromkeys(joints))
    if not selected:
        raise ValueError("select at least one joint with --multi-joint")
    try:
        factor = MULTI_SIZE_FACTORS[size]
    except KeyError as exc:
        raise ValueError(f"unsupported multi-axis size {size!r}") from exc
    generator = rng or random.Random()
    start = list(profile.safe_home_deg)
    target = list(profile.safe_home_deg)
    for joint in selected:
        index = JOINT_NAMES.index(joint)
        low, high = profile.speed_zones_deg[joint]
        centre = (low + high) / 2.0
        half_span = (high - low) / 2.0 * factor
        # Pick a pair with a meaningful displacement while preserving a
        # factor-sized interior buffer on either side of the speed zone.
        start[index] = generator.uniform(centre - half_span, centre - half_span * 0.25)
        target[index] = generator.uniform(centre + half_span * 0.25, centre + half_span)
    return {
        "name": f"selected_{'_'.join(selected)}_{size}",
        "start_deg": tuple(start),
        "target_deg": tuple(target),
        "selected_joints": selected,
        "size": size,
    }


def select_multi_scenarios(profile: Profile, args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    if args.multi_joint:
        return (
            build_selected_multi_scenario(
                profile,
                args.multi_joint,
                args.multi_size,
                random.Random(args.random_seed),
            ),
        )
    scenarios = profile.multi_speed_scenarios
    if args.scenario is not None:
        scenarios = tuple(item for item in scenarios if item["name"] == args.scenario)
        if not scenarios:
            raise ValueError(f"no approved multi-axis scenario named {args.scenario!r}")
    if not scenarios:
        raise ValueError(
            "no multi-axis motion was selected; use --multi-joint j1 [j2 ...] "
            "or define multi_speed_scenarios in the profile"
        )
    return scenarios


def _joint_feedback_degrees(feedback: Any) -> tuple[float, ...] | None:
    state = getattr(feedback, "joint_state", None)
    if state is None:
        return None
    values = [getattr(state, f"joint_{index}", None) for index in range(1, 7)]
    if any(value is None for value in values):
        return None
    return tuple(float(value) / 1000.0 for value in values)


def _feedback_stamp(feedback: Any) -> float | None:
    stamp = getattr(feedback, "time_stamp", None)
    if stamp is None:
        return None
    try:
        return float(stamp)
    except (TypeError, ValueError):
        return None


class PhysicalLimitsDemo:
    def __init__(
        self,
        controller: PiperController,
        profile: Profile,
        options: RunOptions,
        csv_writer: csv.DictWriter[str] | None = None,
    ) -> None:
        self.controller = controller
        self.profile = profile
        self.options = options
        self.csv_writer = csv_writer
        self._last_feedback_at: float | None = None
        self._last_feedback_stamp: float | None = None
        self._previous_sample: tuple[float, tuple[float, ...]] | None = None
        self._last_observed_speeds = (0.0,) * 6
        self._speed_samples_by_stage: dict[str, list[tuple[float, ...]]] = {}

    def _record_feedback(self, stage: str, feedback: Any) -> tuple[float, ...]:
        joints = _joint_feedback_degrees(feedback)
        now = time.monotonic()
        if joints is None:
            if self._last_feedback_at is None or now - self._last_feedback_at > self.options.feedback_timeout_s:
                raise DemoSafetyError("joint feedback is missing or stale")
            raise DemoSafetyError("joint feedback is malformed")
        stamp = _feedback_stamp(feedback)
        if stamp is not None and stamp != self._last_feedback_stamp:
            self._last_feedback_stamp = stamp
            self._last_feedback_at = now
        elif self._last_feedback_at is None:
            # Some SDK versions do not expose a timestamp; the successful read
            # is the best available freshness signal in that case.
            self._last_feedback_at = now
        elif stamp is None:
            self._last_feedback_at = now
        elif now - self._last_feedback_at > self.options.feedback_timeout_s:
            raise DemoSafetyError("joint feedback timestamp stopped updating")

        for index, angle in enumerate(joints):
            low, high = self.profile.safe_motion_ranges_deg[JOINT_NAMES[index]]
            if not (low <= angle <= high):
                raise DemoSafetyError(
                    f"{JOINT_NAMES[index]} feedback {angle:.2f}° left approved envelope "
                    f"[{low:.2f}, {high:.2f}]°"
                )
        sample_time = stamp if stamp is not None else now
        if self._previous_sample is None:
            speeds = (0.0,) * 6
        else:
            prior_time, prior_joints = self._previous_sample
            elapsed = max(sample_time - prior_time, 1e-6)
            speeds = tuple((angle - old) / elapsed for angle, old in zip(joints, prior_joints))
        if self.csv_writer is not None:
            row: dict[str, object] = {"monotonic_s": f"{now:.6f}", "stage": stage}
            for index, angle in enumerate(joints, start=1):
                row[f"j{index}_deg"] = f"{angle:.4f}"
            for index, speed in enumerate(speeds, start=1):
                row[f"j{index}_deg_s"] = f"{speed:.4f}"
            self.csv_writer.writerow(row)
        self._last_observed_speeds = speeds
        self._speed_samples_by_stage.setdefault(stage, []).append(speeds)
        self._previous_sample = (sample_time, joints)
        return joints

    def speed_statistics(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return per-stage observed joint-speed peaks and mean absolute speeds."""
        result: dict[str, dict[str, dict[str, float]]] = {}
        for stage, samples in self._speed_samples_by_stage.items():
            result[stage] = {}
            for index, joint in enumerate(JOINT_NAMES):
                values = [abs(sample[index]) for sample in samples]
                result[stage][joint] = {
                    "peak_deg_s": max(values, default=0.0),
                    "mean_abs_deg_s": sum(values) / len(values) if values else 0.0,
                }
        return result

    def _assert_controller_status(self) -> None:
        status = self.controller.get_status()
        arm_status = getattr(status, "arm_status", status)
        code = getattr(arm_status, "arm_status", None)
        if code not in (None, 0):
            raise DemoSafetyError(f"arm status reported fault code {code}")
        if not all(self.controller.get_enable_status()):
            raise DemoSafetyError("one or more joint drives are no longer enabled")

    def _check_observed_speed(self, joints_deg: tuple[float, ...], commanded_speed: int) -> None:
        del joints_deg, commanded_speed
        for index, observed in enumerate(self._last_observed_speeds):
            configured_max = math.degrees(ROBOT_CONFIG.joint_max_speeds_rad_s[index])
            permitted_max = configured_max * self.options.max_observed_speed_factor
            if abs(observed) > permitted_max:
                raise DemoSafetyError(
                    f"{JOINT_NAMES[index]} observed speed {observed:.1f}°/s exceeds "
                    f"the configured {configured_max:.1f}°/s limit with factor "
                    f"{self.options.max_observed_speed_factor:g}"
                )

    def move_and_monitor(
        self,
        target_deg: tuple[float, ...],
        *,
        speed: int,
        stage: str,
    ) -> tuple[float, ...]:
        target_rad = degrees_to_radians(target_deg)
        self.controller.validate_joints(target_rad)
        deadline = time.monotonic() + self.options.timeout_s
        self.controller.set_motion_mode("J", speed)
        next_command = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_command:
                self.controller.send_joint_target(target_rad)
                next_command = now + self.options.target_command_interval_s
            self._assert_controller_status()
            current = self._record_feedback(stage, self.controller.get_joint_state())
            self._check_observed_speed(current, speed)
            if all(abs(actual - requested) <= self.options.tolerance_deg for actual, requested in zip(current, target_deg)):
                return current
            time.sleep(0.02)
        raise DemoSafetyError(f"{stage}: target was not reached within {self.options.timeout_s:.1f}s")

    def move_home(self, stage: str = "home") -> None:
        self.move_smoothly(
            self.profile.safe_home_deg,
            speed=self.options.home_speed,
            stage=stage,
        )

    def move_smoothly(
        self,
        target_deg: tuple[float, ...],
        *,
        speed: int,
        stage: str,
    ) -> None:
        """Reach a target through bounded joint-space waypoints.

        This avoids commanding a large multi-axis step from the current pose,
        which was the main source of observed start-of-trajectory vibration.
        """
        initial = self._record_feedback(f"{stage}_read_current", self.controller.get_joint_state())
        largest_delta = max(abs(target - current) for target, current in zip(target_deg, initial))
        segments = max(1, math.ceil(largest_delta / self.options.transition_step_deg))
        for segment in range(1, segments + 1):
            fraction = segment / segments
            waypoint = tuple(
                current + (target - current) * fraction
                for current, target in zip(initial, target_deg)
            )
            self.move_and_monitor(
                waypoint,
                speed=speed,
                stage=f"{stage}_step_{segment}_of_{segments}",
            )

    def run_range(self, joints: Iterable[str]) -> None:
        for joint in joints:
            index = JOINT_NAMES.index(joint)
            low, high = self.profile.demo_ranges_deg[joint]
            for suffix, angle in (("min", low), ("max", high)):
                target = list(self.profile.safe_home_deg)
                target[index] = angle
                self.move_and_monitor(tuple(target), speed=self.options.speed, stage=f"range_{joint}_{suffix}")
                self.move_home(stage=f"range_{joint}_{suffix}_return")

    def run_single_speed(self, joints: Iterable[str]) -> None:
        for joint in joints:
            index = JOINT_NAMES.index(joint)
            low, high = self.profile.speed_zones_deg[joint]
            start = list(self.profile.safe_home_deg)
            start[index] = low
            self.move_and_monitor(tuple(start), speed=self.options.home_speed, stage=f"speed_{joint}_start")
            end = list(self.profile.safe_home_deg)
            end[index] = high
            self.move_and_monitor(tuple(end), speed=self.options.speed, stage=f"speed_{joint}_forward")
            self.move_and_monitor(tuple(start), speed=self.options.speed, stage=f"speed_{joint}_reverse")
            self.move_home(stage=f"speed_{joint}_return")

    def run_multi_speed(self, scenarios: Iterable[dict[str, Any]]) -> None:
        for scenario in scenarios:
            self.move_smoothly(
                scenario["start_deg"], speed=self.options.home_speed, stage=f"multi_{scenario['name']}_start"
            )
            self.move_smoothly(
                scenario["target_deg"], speed=self.options.speed, stage=f"multi_{scenario['name']}_forward"
            )
            self.move_smoothly(
                scenario["start_deg"], speed=self.options.speed, stage=f"multi_{scenario['name']}_reverse"
            )
            self.move_home(stage=f"multi_{scenario['name']}_return")

    def hold_after_abort(self, reason: str) -> bool:
        """Cancel active motion by holding the live pose without dropping torque."""
        print(
            f"ABORT: {reason}; cancelling motion and holding current position with drives enabled.",
            file=sys.stderr,
        )
        try:
            target = self.controller.hold_current_position(
                speed_percent=self.options.abort_hold_speed,
                command_duration_s=self.options.abort_hold_duration_s,
                feedback_timeout_s=self.options.feedback_timeout_s,
            )
            print(
                "Current position hold sent (radians): "
                + ", ".join(f"{angle:.4f}" for angle in target),
                file=sys.stderr,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - report the original reason too
            print(
                f"Position hold failed: {exc}. If the arm is unsafe, use the physical E-stop; "
                "do not rely on software emergency stop to hold it.",
                file=sys.stderr,
            )
            return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, help="approved JSON safety profile")
    parser.add_argument("--write-default-profile", type=Path, metavar="PATH", help="write the conservative example profile, then exit")
    parser.add_argument("--phase", choices=("range", "single-speed", "multi-speed", "all"), default="range")
    parser.add_argument("--joint", choices=JOINT_NAMES, action="append", help="joint to test; repeat to select several (default: all)")
    parser.add_argument("--scenario", help="approved multi-axis scenario name from the profile")
    parser.add_argument("--multi-joint", nargs="+", choices=JOINT_NAMES, help="generate a multi-axis test for exactly these moving joints")
    parser.add_argument("--multi-size", choices=tuple(MULTI_SIZE_FACTORS), default="small", help="generated multi-axis excursion: small or normal (default: small)")
    parser.add_argument("--random-seed", type=int, help="seed for a reproducible generated multi-axis route; omit for a new route each run")
    parser.add_argument("--speed", type=int, default=10, help="test speed percentage, 0..100 (default: 10)")
    parser.add_argument("--home-speed", type=int, default=10, help="speed percentage for every move to zero (default: 10)")
    parser.add_argument("--max-range-speed", type=int, default=20, help="range phase is refused above this speed unless explicitly raised (default: 20)")
    parser.add_argument("--timeout", type=float, default=45.0, help="per-motion timeout in seconds (default: 45)")
    parser.add_argument("--feedback-timeout", type=float, default=1.0, help="maximum stale feedback age in seconds (default: 1)")
    parser.add_argument("--tolerance-deg", type=float, default=2.0, help="target/envelope tolerance in degrees (default: 2)")
    parser.add_argument("--max-observed-speed-factor", type=float, default=1.20, help="abort above configured joint max speed times this factor (default: 1.20)")
    parser.add_argument("--abort-hold-speed", type=int, default=5, help="low speed percentage used to hold the live pose after Ctrl-C (default: 5)")
    parser.add_argument("--abort-hold-duration", type=float, default=2.0, help="seconds to resend the live hold target after Ctrl-C (default: 2)")
    parser.add_argument("--target-command-interval", type=float, default=0.20, help="seconds between identical target refreshes (default: 0.20)")
    parser.add_argument("--transition-step-deg", type=float, default=5.0, help="largest joint increment for smooth transitions (default: 5)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="directory for CSV and JSON reports")
    parser.add_argument("--execute", action="store_true", help="actually enable and move the arm; omitted means dry-run")
    return parser


def _validate_options(args: argparse.Namespace) -> RunOptions:
    for label, speed in (("speed", args.speed), ("home-speed", args.home_speed), ("max-range-speed", args.max_range_speed), ("abort-hold-speed", args.abort_hold_speed)):
        if not isinstance(speed, int) or not 0 <= speed <= 100:
            raise ValueError(f"--{label} must be an integer from 0 through 100")
    if args.phase in ("range", "all") and args.speed > args.max_range_speed:
        raise ValueError("range demonstration speed exceeds --max-range-speed")
    if args.scenario is not None and args.multi_joint:
        raise ValueError("--scenario and --multi-joint cannot be used together")
    if (args.scenario is not None or args.multi_joint) and args.phase not in ("multi-speed", "all"):
        raise ValueError("--scenario and --multi-joint require --phase multi-speed or --phase all")
    if args.timeout <= 0 or args.feedback_timeout <= 0 or args.tolerance_deg <= 0 or args.abort_hold_duration <= 0 or args.target_command_interval <= 0 or args.transition_step_deg <= 0 or args.max_observed_speed_factor < 1:
        raise ValueError("timeout, feedback timeout, tolerance, command interval, transition step, and abort hold duration must be positive; speed factor must be at least 1")
    return RunOptions(args.speed, args.home_speed, args.tolerance_deg, args.timeout, args.feedback_timeout, args.max_range_speed, args.max_observed_speed_factor, args.abort_hold_speed, args.abort_hold_duration, args.target_command_interval, args.transition_step_deg)


def _print_plan(
    profile: Profile,
    args: argparse.Namespace,
    joints: tuple[str, ...],
    multi_scenarios: tuple[dict[str, Any], ...],
) -> None:
    print("DRY RUN: no CAN connection, enabling, or motion will occur.")
    print("Zero home (degrees):", list(profile.safe_home_deg))
    print("Approved demonstration envelope:")
    for joint in JOINT_NAMES:
        print(f"  {joint}: {profile.demo_ranges_deg[joint][0]:g}° .. {profile.demo_ranges_deg[joint][1]:g}°")
    print("Approved zero-return motion envelope:")
    for joint in JOINT_NAMES:
        print(f"  {joint}: {profile.safe_motion_ranges_deg[joint][0]:g}° .. {profile.safe_motion_ranges_deg[joint][1]:g}°")
    print(f"Phase: {args.phase}; joints: {', '.join(joints)}; test speed: {args.speed}%")
    if args.phase in ("multi-speed", "all"):
        print("Multi-axis plan:")
        for scenario in multi_scenarios:
            selected = scenario.get("selected_joints", "profile-defined")
            print(f"  {scenario['name']} ({selected}):")
            print(f"    start_deg : {list(scenario['start_deg'])}")
            print(f"    target_deg: {list(scenario['target_deg'])}")
        if args.multi_joint:
            print(f"  random seed: {args.random_seed if args.random_seed is not None else 'generated by Python system entropy'}")
    print("Review clearance for the installed tool, cable, table, camera, and guards before --execute.")


def _open_logs(output: Path) -> tuple[Any, csv.DictWriter[str], Path]:
    output.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output / f"physical_limits_{stamp}.csv"
    stream = csv_path.open("w", newline="", encoding="utf-8")
    fields = ["monotonic_s", "stage"] + [f"j{index}_deg" for index in range(1, 7)] + [f"j{index}_deg_s" for index in range(1, 7)]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return stream, writer, output / f"physical_limits_{stamp}.json"


def main() -> int:
    args = build_parser().parse_args()
    if args.write_default_profile:
        args.write_default_profile.write_text(json.dumps(DEFAULT_PROFILE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote conservative profile to {args.write_default_profile}")
        return 0
    try:
        profile = load_profile(args.profile)
        options = _validate_options(args)
        if args.multi_joint and args.random_seed is None:
            args.random_seed = random.SystemRandom().randrange(0, 2**63)
        multi_scenarios = (
            select_multi_scenarios(profile, args)
            if args.phase in ("multi-speed", "all")
            else ()
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    joints = tuple(args.joint) if args.joint else JOINT_NAMES
    if not args.execute:
        _print_plan(profile, args, joints, multi_scenarios)
        return 0

    log_stream, writer, summary_path = _open_logs(args.output)
    controller = PiperController()
    demo = PhysicalLimitsDemo(controller, profile, options, writer)
    summary: dict[str, Any] = {
        "phase": args.phase,
        "joints": joints,
        "speed_percent": args.speed,
        "random_seed": args.random_seed,
        "multi_scenarios": [item["name"] for item in multi_scenarios],
        "completed": False,
    }
    try:
        controller.connect(enable=True)
        # Required by the requested operating procedure: always establish the
        # zero posture at low speed before doing any range or speed test.
        demo.move_home("startup_zero_home")
        if args.phase in ("range", "all"):
            demo.run_range(joints)
        if args.phase in ("single-speed", "all"):
            demo.run_single_speed(joints)
        if args.phase in ("multi-speed", "all"):
            demo.run_multi_speed(multi_scenarios)
        demo.move_home("final_zero_home")
        summary["completed"] = True
        print("Demonstration completed; arm is at zero home. Drives remain enabled.")
        return 0
    except KeyboardInterrupt:
        summary["position_hold_sent"] = demo.hold_after_abort("Ctrl-C received")
        summary["abort_reason"] = "KeyboardInterrupt"
        return 130
    except (DemoSafetyError, SafetyError, TimeoutError, ValueError, RuntimeError) as exc:
        summary["position_hold_sent"] = demo.hold_after_abort(str(exc))
        summary["abort_reason"] = str(exc)
        return 2
    finally:
        summary["observed_speed_statistics_deg_s"] = demo.speed_statistics()
        summary["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Logs: {log_stream.name}, {summary_path}")
        finally:
            log_stream.close()
            controller.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
