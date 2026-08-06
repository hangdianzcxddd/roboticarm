from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
STATE_DIR = APP_DIR / "state"
RUNTIME_DIR = APP_DIR / "runtime"
PREVIEW_DIR = RUNTIME_DIR / "preview"
COMMANDS_PATH = STATE_DIR / "commands.json"
PROFILE_PATH = STATE_DIR / "profile.json"
PREVIEW_FILES = {
    "rgb": "opencv_rgb.jpg",
    "mask": "opencv_mask.jpg",
    "depth": "opencv_depth.jpg",
}


DEFAULT_PROFILE: dict[str, Any] = {
    "name": "动态抓取默认配置",
    "windows": {
        "python": "python",
        "conda_env": "piper",
        "mode": "dynamic",
        "host": "192.168.146.128",
        "port": 5005,
        "conveyor_speed_mm_s": 140,
        "dynamic_min_x_mm": 220,
        "dynamic_max_x_mm": 480,
        "dynamic_min_y_mm": -550,
        "dynamic_max_y_mm": -300,
        "dynamic_pick_x_from_bbox_center": True,
        "dynamic_pick_x_mm": 350,
        "dynamic_pick_y_mm": 0,
        "width": 1280,
        "height": 720,
        "conveyor_direction": "positive-y",
        "base_z_offset_mm": 0,
        "belt_lower_hsv": [35, 40, 40],
        "belt_upper_hsv": [90, 255, 255],
        "reject_highlight": True,
        "highlight_max_s": 30,
        "highlight_min_v": 225,
        "min_area_px": 1200,
        "min_width_px": 25,
        "min_height_px": 25,
        "morphology_kernel_size": 5,
        "morphology_iterations": 1,
        "roi": [250, 100, 940, 550],
        "entry_edge": "right",
        "entry_clear_margin_px": 15,
        "entry_clear_frames": 3,
        "hand_eye_result": r"robot_control\linux\hand_eye\eye_to_hand_result_auto.json",
        "dynamic_debug_candidates": True,
        "dynamic_skip_memory_s": 0,
        "dynamic_probe_interval_s": 0,
        "print_interval": 1,
        "morphology_close_kernel_size": 11,
        "morphology_open_kernel_size": 3,
        "contour_mask_mode": "convex-hull",
    },
    "linux_shell": {
        "ssh_user": "zcxddd",
        "ssh_host": "192.168.146.128",
        "batch_mode": True,
        "strict_host_key_checking": "accept-new",
        "server_alive_interval": 5,
        "remote_project": "/home/zcxddd/projects/roboticarm/piper_project",
        "conda_init": "~/miniconda3/etc/profile.d/conda.sh",
        "conda_env": "piper",
        "python": "python3",
    },
    "linux": {
        "host": "0.0.0.0",
        "port": 5005,
        "can": "can0",
        "hand_eye_result": "robot_control/linux/hand_eye/eye_to_hand_result_auto.json",
        "startup_speed": 10,
        "speed": 60,
        "rx_deg": -175.957642,
        "ry_deg": 1.353477,
        "rz_deg": -102.579260,
        "command_duration_s": 0.3,
        "gripper_settle_s": 0.1,
        "dynamic_preposition": True,
        "dynamic_preposition_clearance_mm": 40,
        "final_action_time_s": 0.5,
        "command_period_s": 0.05,
        "print_feedback": True,
        "open_command_mm": 0,
        "grasp_command_mm": -90,
        "effort_nm": 0.9,
        "min_target_x_mm": 220,
        "max_target_x_mm": 480,
        "force_pick_y_mm": 0,
        "adaptive_joint_pitch_by_x": True,
        "dynamic_pitch_mode": "planar-ik",
        "planar_ik_fk_tool_z_offset_mm": 82,
        "planar_ik_joint_tolerance_deg": 12,
        "keep_pitch_min_x_mm": 220,
        "keep_pitch_max_x_mm": 355,
        "pitch_deg_per_mm_far": 0.013,
        "pitch_deg_per_mm_near": 0.01,
        "min_dynamic_pitch_deg": -5,
        "max_dynamic_pitch_deg": 5,
        "grasp_z": 120,
        "use_detected_z": "top",
        "detected_z_offset_mm": 115,
        "min_grasp_z_mm": 120,
        "max_grasp_z_mm": 310,
        "execute": True,
        "planar_ik_grasp_z_comp_mm_per_deg": 20.0,
        "planar_ik_speed": 20,
        "dynamic_wait_placement": "nowait",
        "pick_x_offset_mm": -45,
        "t_robot_s": 2.8,
        "dynamic_t_robot_by_x": True,
        "dynamic_t_robot_reference_x_mm": 340,
        "dynamic_t_robot_s_per_mm": 0,
        "dynamic_t_robot_planar_near_s_per_mm": 0.017,
        "dynamic_t_robot_planar_far_s_per_mm": 0.010,
        "grasp_yaw_by_object_short_edge": True,
        "gripper_close_axis_offset_deg": 0,
        "grasp_yaw_max_delta_deg": 70,
        "grasp_yaw_deadband_deg": 5,
        "grasp_yaw_speed": 60,
    },
}


IMPORTANT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("error", re.compile(r"\b(error|failed|traceback|exception|timed out|abort|refused)\b", re.I)),
    ("error", re.compile(r"TARGET_POS_EXCEEDS_LIMIT|EXCEEDS_LIMIT", re.I)),
    ("warn", re.compile(r"\b(skipped|warning|not reached|no valid)\b", re.I)),
    (
        "success",
        re.compile(r"\b(ack|send completed|catch send counts|after catch|executed=True|Piper arm enabled)\b", re.I),
    ),
    ("info", re.compile(r"\b(dynamic timing|client connected|listening on|using hand-eye)\b", re.I)),
)


STEP_LABELS: dict[str, str] = {
    "initial_ready": "初始准备等待",
    "startup_ready": "启动准备位",
    "ready_open": "准备并打开夹爪",
    "pick_above": "移动到抓取上方",
    "preposition": "预定位等待",
    "dynamic_preposition": "动态预定位",
    "pick_grasp": "下探抓取",
    "pick_lift": "抓取后抬升",
    "return_ready_above": "返回准备上方",
    "place_above": "移动到放置上方",
    "place_release": "放置并释放",
    "return_ready": "返回准备位",
}


def humanize_step(step: str) -> str:
    return STEP_LABELS.get(step, step.replace("_", " "))


def parse_arm_activity(line: str) -> dict[str, str] | None:
    message = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", line).strip()
    step_match = re.search(r"\bstep=([a-zA-Z0-9_-]+)", message)
    if "Running planar IK" in message and step_match:
        step = step_match.group(1)
        return {"phase": "运动中", "step": step, "label": humanize_step(step)}
    if "Reusing planar IK" in message and step_match:
        step = step_match.group(1)
        return {"phase": "运动中", "step": step, "label": humanize_step(step)}
    if "Planar IK" in message and "feedback:" in message and step_match:
        step = step_match.group(1)
        return {"phase": "到达反馈", "step": step, "label": humanize_step(step)}

    running_match = re.search(r"\bRunning ([a-zA-Z0-9_-]+):", message)
    if running_match:
        step = running_match.group(1)
        return {"phase": "运动中", "step": step, "label": humanize_step(step)}

    waiting_match = re.search(r"Dynamic final timing: waiting [0-9.]+s at ([^(]+)", message)
    if waiting_match:
        step = waiting_match.group(1).strip().replace(" ", "_")
        return {"phase": "动态时序等待", "step": step, "label": humanize_step(step)}

    if "Client connected:" in message:
        return {"phase": "连接", "step": "client_connected", "label": "客户端已连接"}
    if "Before catch J6" in message:
        return {"phase": "抓取开始", "step": "before_catch", "label": "读取抓取前姿态"}
    if "Moving to startup ready pose" in message:
        return {"phase": "启动", "step": "startup_ready", "label": humanize_step("startup_ready")}
    if "Piper arm enabled" in message:
        return {"phase": "就绪", "step": "enabled", "label": "机械臂已使能"}
    if "listening on" in message.lower():
        return {"phase": "等待", "step": "listening", "label": "等待视觉客户端连接"}
    if "After catch J6" in message:
        return {"phase": "完成", "step": "after_catch", "label": "本轮抓取完成"}
    if "Catch send counts" in message:
        return {"phase": "完成", "step": "catch_counts", "label": "动作指令统计完成"}
    if message.startswith("task="):
        return {"phase": "完成", "step": "task_result", "label": "任务结果已输出"}
    return None


@dataclass(frozen=True)
class LogEntry:
    time: str
    level: str
    text: str


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def classify_log_line(line: str) -> str:
    for level, pattern in IMPORTANT_PATTERNS:
        if pattern.search(line):
            return level
    return "debug"


class ManagedProcess:
    def __init__(self, name: str, *, preview_enabled: bool = False) -> None:
        self.name = name
        self.preview_enabled = preview_enabled
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._logs: deque[LogEntry] = deque(maxlen=3000)
        self._important: deque[LogEntry] = deque(maxlen=300)
        self._command = ""
        self._started_at: float | None = None
        self._exit_code: int | None = None

    def start(self, command: str) -> None:
        command = command.strip()
        if not command:
            raise ValueError(f"{self.name} command is empty")
        with self._lock:
            if self.is_running_locked():
                raise RuntimeError(f"{self.name} process is already running")
            self._command = command
            self._started_at = time.time()
            self._exit_code = None
            self._logs.clear()
            self._important.clear()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if self.preview_enabled:
                PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                env["PIPER_WEB_PREVIEW_DIR"] = str(PREVIEW_DIR)
                env.setdefault("PIPER_WEB_PREVIEW_INTERVAL_S", "0.05")
            kwargs: dict[str, Any] = {
                "cwd": str(PROJECT_ROOT),
                "shell": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
                "env": env,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["preexec_fn"] = os.setsid
            self._append("info", f"$ {command}")
            self._process = subprocess.Popen(command, **kwargs)
            self._reader = threading.Thread(
                target=self._read_output,
                args=(self._process,),
                name=f"{self.name}-log-reader",
                daemon=True,
            )
            self._reader.start()

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            self._append("warn", "Stopping process...")
        terminate_process_tree(process)
        with self._lock:
            self._exit_code = process.poll()

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                if line:
                    self._append(classify_log_line(line), line)
        finally:
            exit_code = process.wait()
            level = "success" if exit_code == 0 else "error"
            self._append(level, f"Process exited with code {exit_code}")
            with self._lock:
                if self._process is process:
                    self._exit_code = exit_code

    def _append(self, level: str, text: str) -> None:
        entry = LogEntry(timestamp(), level, text)
        with self._lock:
            self._logs.append(entry)
            if level != "debug":
                self._important.append(entry)

    def append_event(self, level: str, text: str) -> None:
        self._append(level, text)

    def is_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self.is_running_locked()
            pid = self._process.pid if self._process is not None else None
            started_at = self._started_at
            return {
                "name": self.name,
                "running": running,
                "pid": pid,
                "command": self._command,
                "started_at": started_at,
                "uptime_s": time.time() - started_at if running and started_at else None,
                "exit_code": None if running else self._exit_code,
                "last_important": asdict(self._important[-1]) if self._important else None,
            }

    def activity_status(self) -> dict[str, Any]:
        with self._lock:
            running = self.is_running_locked()
            exit_code = None if running else self._exit_code
            if exit_code not in (None, 0):
                return {
                    "phase": "异常停止",
                    "step": "process_exit",
                    "label": f"Linux 进程退出，code={exit_code}",
                    "time": timestamp(),
                    "running": False,
                    "exit_code": exit_code,
                }
            for entry in reversed(self._logs):
                activity = parse_arm_activity(entry.text)
                if activity:
                    return {
                        **activity,
                        "time": entry.time,
                        "running": running,
                        "exit_code": exit_code,
                    }
            if running:
                return {
                    "phase": "运行中",
                    "step": "waiting_log",
                    "label": "等待机械臂流程日志",
                    "time": timestamp(),
                    "running": True,
                    "exit_code": None,
                }
            if exit_code is not None:
                return {
                    "phase": "已停止",
                    "step": "process_exit",
                    "label": f"Linux 进程退出，code={exit_code}",
                    "time": timestamp(),
                    "running": False,
                    "exit_code": exit_code,
                }
            return {
                "phase": "空闲",
                "step": "idle",
                "label": "等待启动 Linux 服务",
                "time": None,
                "running": False,
                "exit_code": None,
            }

    def activity_history(self, *, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            items: list[dict[str, Any]] = []
            last_key: tuple[str | None, str | None, str | None] | None = None
            for entry in reversed(self._logs):
                activity = parse_arm_activity(entry.text)
                if not activity:
                    continue
                key = (activity.get("phase"), activity.get("step"), activity.get("label"))
                if key == last_key:
                    continue
                items.append({**activity, "time": entry.time})
                last_key = key
                if len(items) >= limit:
                    break
            return list(reversed(items))

    def logs(self, *, limit: int = 500, important: bool = False) -> list[dict[str, str]]:
        with self._lock:
            source = self._important if important else self._logs
            return [asdict(item) for item in list(source)[-limit:]]


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return


def deep_merge(defaults: Any, loaded: Any) -> Any:
    if isinstance(defaults, dict):
        merged = deepcopy(defaults)
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                merged[key] = deep_merge(defaults.get(key), value)
        return merged
    if isinstance(defaults, list):
        return deepcopy(loaded) if isinstance(loaded, list) else deepcopy(defaults)
    return deepcopy(loaded) if loaded is not None else deepcopy(defaults)


def load_profile() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        return deepcopy(DEFAULT_PROFILE)
    try:
        loaded = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(DEFAULT_PROFILE)
    return deep_merge(DEFAULT_PROFILE, loaded)


def save_profile(payload: dict[str, Any]) -> dict[str, Any]:
    profile = deep_merge(DEFAULT_PROFILE, payload)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def build_generated_commands(profile: dict[str, Any] | None = None) -> dict[str, str]:
    profile = load_profile() if profile is None else deep_merge(DEFAULT_PROFILE, profile)
    return {
        "windows": build_windows_command(profile),
        "linux": build_linux_command(profile),
    }


def local_join(parts: list[Any]) -> str:
    string_parts = [str(part) for part in parts]
    if os.name == "nt":
        return subprocess.list2cmdline(string_parts)
    return shlex.join(string_parts)


def add_option(parts: list[Any], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    parts.extend([flag, value])


def add_number_option(parts: list[Any], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    parts.extend([flag, format_number(value)])


def add_tuple_option(parts: list[Any], flag: str, values: Any, *, size: int) -> None:
    if not isinstance(values, list) or len(values) != size:
        return
    parts.append(flag)
    parts.extend(format_number(item) for item in values[:size])


def add_flag(parts: list[Any], enabled: Any, flag: str) -> None:
    if bool(enabled):
        parts.append(flag)


def format_number(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def build_windows_command(profile: dict[str, Any]) -> str:
    windows = profile["windows"]
    conda_env = str(windows.get("conda_env", "")).strip()
    if conda_env:
        parts: list[Any] = ["conda", "run", "-n", conda_env, windows.get("python", "python")]
    else:
        parts = [windows.get("python", "python")]
    parts.extend(
        [
            "-m",
            "robot_control.windows.client.opencv_grasp_send_tcp",
            windows.get("mode", "dynamic"),
        ]
    )
    add_option(parts, "--host", windows.get("host"))
    port = int(windows.get("port", 5005))
    if port != 5005:
        add_number_option(parts, "--port", port)
    add_number_option(parts, "--conveyor-speed-mm-s", windows.get("conveyor_speed_mm_s"))
    add_number_option(parts, "--dynamic-min-x-mm", windows.get("dynamic_min_x_mm"))
    add_number_option(parts, "--dynamic-max-x-mm", windows.get("dynamic_max_x_mm"))
    add_number_option(parts, "--dynamic-min-y-mm", windows.get("dynamic_min_y_mm"))
    add_number_option(parts, "--dynamic-max-y-mm", windows.get("dynamic_max_y_mm"))
    if windows.get("dynamic_pick_x_from_bbox_center"):
        parts.append("--dynamic-pick-x-from-bbox-center")
    else:
        add_number_option(parts, "--dynamic-pick-x-mm", windows.get("dynamic_pick_x_mm"))
    add_number_option(parts, "--dynamic-pick-y-mm", windows.get("dynamic_pick_y_mm"))
    add_number_option(parts, "--width", windows.get("width"))
    add_number_option(parts, "--height", windows.get("height"))
    add_option(parts, "--conveyor-direction", windows.get("conveyor_direction"))
    add_number_option(parts, "--base-z-offset-mm", windows.get("base_z_offset_mm"))
    add_tuple_option(parts, "--belt-lower-hsv", windows.get("belt_lower_hsv"), size=3)
    add_tuple_option(parts, "--belt-upper-hsv", windows.get("belt_upper_hsv"), size=3)
    add_flag(parts, windows.get("reject_highlight"), "--reject-highlight")
    add_number_option(parts, "--highlight-max-s", windows.get("highlight_max_s"))
    add_number_option(parts, "--highlight-min-v", windows.get("highlight_min_v"))
    add_number_option(parts, "--min-area-px", windows.get("min_area_px"))
    add_number_option(parts, "--min-width-px", windows.get("min_width_px"))
    add_number_option(parts, "--min-height-px", windows.get("min_height_px"))
    add_number_option(parts, "--morphology-kernel-size", windows.get("morphology_kernel_size"))
    add_number_option(parts, "--morphology-iterations", windows.get("morphology_iterations"))
    add_tuple_option(parts, "--roi", windows.get("roi"), size=4)
    add_option(parts, "--entry-edge", windows.get("entry_edge"))
    add_number_option(parts, "--entry-clear-margin-px", windows.get("entry_clear_margin_px"))
    add_number_option(parts, "--entry-clear-frames", windows.get("entry_clear_frames"))
    add_option(parts, "--hand-eye-result", windows.get("hand_eye_result"))
    add_flag(parts, windows.get("dynamic_debug_candidates"), "--dynamic-debug-candidates")
    add_number_option(parts, "--dynamic-skip-memory-s", windows.get("dynamic_skip_memory_s"))
    add_number_option(parts, "--dynamic-probe-interval-s", windows.get("dynamic_probe_interval_s"))
    add_number_option(parts, "--print-interval", windows.get("print_interval"))
    add_number_option(parts, "--morphology-close-kernel-size", windows.get("morphology_close_kernel_size"))
    add_number_option(parts, "--morphology-open-kernel-size", windows.get("morphology_open_kernel_size"))
    add_option(parts, "--contour-mask-mode", windows.get("contour_mask_mode"))
    return local_join(parts)


def build_linux_python_parts(profile: dict[str, Any]) -> list[Any]:
    linux = profile["linux"]
    shell = profile["linux_shell"]
    parts: list[Any] = [
        shell.get("python", "python3"),
        "-m",
        "robot_control.linux.catch.catch_server_1",
    ]
    add_option(parts, "--host", linux.get("host"))
    add_number_option(parts, "--port", linux.get("port"))
    add_option(parts, "--can", linux.get("can"))
    add_option(parts, "--hand-eye-result", linux.get("hand_eye_result"))
    add_number_option(parts, "--startup-speed", linux.get("startup_speed"))
    add_number_option(parts, "--speed", linux.get("speed"))
    add_number_option(parts, "--rx-deg", linux.get("rx_deg"))
    add_number_option(parts, "--ry-deg", linux.get("ry_deg"))
    add_number_option(parts, "--rz-deg", linux.get("rz_deg"))
    add_number_option(parts, "--command-duration-s", linux.get("command_duration_s"))
    add_number_option(parts, "--gripper-settle-s", linux.get("gripper_settle_s"))
    add_flag(parts, linux.get("dynamic_preposition"), "--dynamic-preposition")
    add_number_option(parts, "--dynamic-preposition-clearance-mm", linux.get("dynamic_preposition_clearance_mm"))
    add_number_option(parts, "--final-action-time-s", linux.get("final_action_time_s"))
    add_number_option(parts, "--command-period-s", linux.get("command_period_s"))
    add_flag(parts, linux.get("print_feedback"), "--print-feedback")
    add_number_option(parts, "--open-command-mm", linux.get("open_command_mm"))
    add_number_option(parts, "--grasp-command-mm", linux.get("grasp_command_mm"))
    add_number_option(parts, "--effort-nm", linux.get("effort_nm"))
    add_number_option(parts, "--min-target-x-mm", linux.get("min_target_x_mm"))
    add_number_option(parts, "--max-target-x-mm", linux.get("max_target_x_mm"))
    add_number_option(parts, "--force-pick-y-mm", linux.get("force_pick_y_mm"))
    add_flag(parts, linux.get("adaptive_joint_pitch_by_x"), "--adaptive-joint-pitch-by-x")
    add_option(parts, "--dynamic-pitch-mode", linux.get("dynamic_pitch_mode"))
    add_number_option(parts, "--planar-ik-fk-tool-z-offset-mm", linux.get("planar_ik_fk_tool_z_offset_mm"))
    add_number_option(parts, "--planar-ik-joint-tolerance-deg", linux.get("planar_ik_joint_tolerance_deg"))
    add_number_option(parts, "--keep-pitch-min-x-mm", linux.get("keep_pitch_min_x_mm"))
    add_number_option(parts, "--keep-pitch-max-x-mm", linux.get("keep_pitch_max_x_mm"))
    add_number_option(parts, "--pitch-deg-per-mm-far", linux.get("pitch_deg_per_mm_far"))
    add_number_option(parts, "--pitch-deg-per-mm-near", linux.get("pitch_deg_per_mm_near"))
    add_number_option(parts, "--min-dynamic-pitch-deg", linux.get("min_dynamic_pitch_deg"))
    add_number_option(parts, "--max-dynamic-pitch-deg", linux.get("max_dynamic_pitch_deg"))
    add_number_option(parts, "--grasp-z", linux.get("grasp_z"))
    use_detected_z = linux.get("use_detected_z")
    if use_detected_z and use_detected_z != "off":
        add_option(parts, "--use-detected-z", use_detected_z)
    add_number_option(parts, "--detected-z-offset-mm", linux.get("detected_z_offset_mm"))
    add_number_option(parts, "--min-grasp-z-mm", linux.get("min_grasp_z_mm"))
    add_number_option(parts, "--max-grasp-z-mm", linux.get("max_grasp_z_mm"))
    add_flag(parts, linux.get("execute"), "--execute")
    add_number_option(parts, "--planar-ik-grasp-z-comp-mm-per-deg", linux.get("planar_ik_grasp_z_comp_mm_per_deg"))
    add_number_option(parts, "--planar-ik-speed", linux.get("planar_ik_speed"))
    add_option(parts, "--dynamic-wait-placement", linux.get("dynamic_wait_placement"))
    add_number_option(parts, "--pick-x-offset-mm", linux.get("pick_x_offset_mm"))
    add_number_option(parts, "--t-robot-s", linux.get("t_robot_s"))
    # A preposition+wait catch is already at the target-side waiting pose and
    # schedules only the final descent/close from --final-action-time-s. Raw-X
    # timing slopes are needed only when nowait schedules departure from ready.
    if linux.get("dynamic_wait_placement") == "nowait":
        add_flag(parts, linux.get("dynamic_t_robot_by_x"), "--dynamic-t-robot-by-x")
        add_number_option(parts, "--dynamic-t-robot-reference-x-mm", linux.get("dynamic_t_robot_reference_x_mm"))
        add_number_option(parts, "--dynamic-t-robot-s-per-mm", linux.get("dynamic_t_robot_s_per_mm"))
        add_number_option(parts, "--dynamic-t-robot-planar-near-s-per-mm", linux.get("dynamic_t_robot_planar_near_s_per_mm"))
        add_number_option(parts, "--dynamic-t-robot-planar-far-s-per-mm", linux.get("dynamic_t_robot_planar_far_s_per_mm"))
    add_flag(parts, linux.get("grasp_yaw_by_object_short_edge"), "--grasp-yaw-by-object-short-edge")
    add_number_option(parts, "--gripper-close-axis-offset-deg", linux.get("gripper_close_axis_offset_deg"))
    add_number_option(parts, "--grasp-yaw-max-delta-deg", linux.get("grasp_yaw_max_delta_deg"))
    add_number_option(parts, "--grasp-yaw-deadband-deg", linux.get("grasp_yaw_deadband_deg"))
    add_number_option(parts, "--grasp-yaw-speed", linux.get("grasp_yaw_speed"))
    return parts


def build_linux_command(profile: dict[str, Any]) -> str:
    shell = profile["linux_shell"]
    python_parts = build_linux_python_parts(profile)
    remote_steps = [
        f"cd {shlex.quote(str(shell.get('remote_project', '')))}",
    ]
    conda_env = str(shell.get("conda_env", "")).strip()
    conda_init = str(shell.get("conda_init", "")).strip()
    if conda_env:
        if conda_init:
            remote_steps.append(f"source {conda_init}")
        remote_steps.append(f"conda activate {shlex.quote(conda_env)}")
    remote_steps.append(shlex.join(str(part) for part in python_parts))
    remote_command = " && ".join(remote_steps)

    return local_join(build_ssh_argv(shell, remote_command))


def build_ssh_argv(shell: dict[str, Any], remote_command: str) -> list[str]:
    ssh_parts: list[Any] = ["ssh"]
    if shell.get("batch_mode", True):
        ssh_parts.extend(["-o", "BatchMode=yes"])
    strict_host_key_checking = shell.get("strict_host_key_checking")
    if strict_host_key_checking:
        ssh_parts.extend(["-o", f"StrictHostKeyChecking={strict_host_key_checking}"])
    server_alive_interval = shell.get("server_alive_interval")
    if server_alive_interval not in (None, ""):
        ssh_parts.extend(["-o", f"ServerAliveInterval={format_number(server_alive_interval)}"])
    ssh_parts.append(f"{shell.get('ssh_user')}@{shell.get('ssh_host')}")
    ssh_parts.append(remote_command)
    return [str(part) for part in ssh_parts]


def remote_setup_can(profile: dict[str, Any], sudo_password: str) -> dict[str, Any]:
    profile = deep_merge(DEFAULT_PROFILE, profile)
    password = str(sudo_password or "")
    if not password:
        return {
            "ok": False,
            "running": False,
            "message": "请输入 Linux sudo 密码。",
        }

    shell = profile["linux_shell"]
    linux = profile["linux"]
    remote_project = str(shell.get("remote_project", "")).strip()
    if not remote_project:
        return {
            "ok": False,
            "running": False,
            "message": "远端项目路径为空，无法执行 setup_can.sh。",
        }

    can_iface = str(linux.get("can") or "can0").strip() or "can0"
    bitrate = str(linux.get("can_bitrate") or "1000000").strip() or "1000000"
    remote_steps = [
        f"cd {shlex.quote(remote_project)}",
        "test -f ./setup_can.sh || { echo 'setup_can.sh not found in remote project'; exit 66; }",
        "IFS= read -r SUDO_PASS",
        (
            "printf '%s\\n' \"$SUDO_PASS\" | "
            f"sudo -S -p '' bash ./setup_can.sh {shlex.quote(can_iface)} {shlex.quote(bitrate)}"
        ),
    ]
    remote_command = " && ".join(remote_steps)

    try:
        result = subprocess.run(
            build_ssh_argv(shell, remote_command),
            input=password + "\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "running": False,
            "can": can_iface,
            "bitrate": bitrate,
            "message": "连接机械臂超时：远端 setup_can.sh 未在 20 秒内完成。",
        }
    except OSError as exc:
        return {
            "ok": False,
            "running": False,
            "can": can_iface,
            "bitrate": bitrate,
            "message": f"连接机械臂失败：{exc}",
        }

    raw = result.stdout.strip().replace(password, "******")
    ok = result.returncode == 0
    if ok:
        message = f"机械臂 CAN 已连接：{can_iface}，bitrate={bitrate}。"
    elif result.returncode == 66:
        message = "远端项目目录中没有找到 setup_can.sh，请先同步该文件到 Linux 项目目录。"
    else:
        message = raw or f"连接机械臂失败，setup_can.sh 返回 {result.returncode}。"

    return {
        "ok": ok,
        "running": False,
        "can": can_iface,
        "bitrate": bitrate,
        "exit_code": result.returncode,
        "raw": raw,
        "message": message,
    }


def linux_service_port(profile: dict[str, Any]) -> int:
    try:
        port = int(profile["linux"].get("port", 5005))
    except (TypeError, ValueError) as exc:
        raise ValueError("Linux 服务端口必须是数字") from exc
    if port < 1 or port > 65535:
        raise ValueError("Linux 服务端口必须在 1-65535 之间")
    return port


def remote_linux_port_status(profile: dict[str, Any]) -> dict[str, Any]:
    profile = deep_merge(DEFAULT_PROFILE, profile)
    shell = profile["linux_shell"]
    port = linux_service_port(profile)
    remote_command = f"ss -H -ltnp 'sport = :{port}'"
    try:
        result = subprocess.run(
            build_ssh_argv(shell, remote_command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "occupied": None,
            "port": port,
            "pids": [],
            "processes": [],
            "raw": "",
            "error": "SSH 检查远端端口超时",
            "message": f"无法确认 Linux 端口 {port} 状态：SSH 检查超时。",
        }
    except OSError as exc:
        return {
            "ok": False,
            "occupied": None,
            "port": port,
            "pids": [],
            "processes": [],
            "raw": "",
            "error": str(exc),
            "message": f"无法确认 Linux 端口 {port} 状态：{exc}",
        }

    raw = result.stdout.strip()
    if result.returncode != 0:
        error = raw or f"ssh exited with code {result.returncode}"
        return {
            "ok": False,
            "occupied": None,
            "port": port,
            "pids": [],
            "processes": [],
            "raw": raw,
            "exit_code": result.returncode,
            "error": error,
            "message": f"无法确认 Linux 端口 {port} 状态：{error}",
        }

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    pids = sorted({int(match) for match in re.findall(r"pid=(\d+)", raw)})
    processes = sorted({match for match in re.findall(r'\("([^"]+)",pid=\d+', raw)})
    occupied = bool(lines)
    if occupied:
        owner = ", ".join(
            f"{name or 'process'}:{pid}"
            for name, pid in zip(processes or ["process"] * len(pids), pids)
        )
        if not owner and pids:
            owner = ", ".join(str(pid) for pid in pids)
        message = f"Linux 端口 {port} 已被占用"
        if owner:
            message += f"（{owner}）"
        message += "。请先停止旧服务，再重新启动 Linux。"
    else:
        message = f"Linux 端口 {port} 空闲，可以启动服务。"

    return {
        "ok": True,
        "occupied": occupied,
        "port": port,
        "pids": pids,
        "processes": processes,
        "raw": raw,
        "message": message,
    }


def stop_remote_linux_port_owner(profile: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    profile = deep_merge(DEFAULT_PROFILE, profile)
    status = remote_linux_port_status(profile)
    if not status.get("ok"):
        return {**status, "stopped": False}
    if not status.get("occupied"):
        return {"ok": True, "stopped": False, "status": status, "message": status["message"]}

    pids = [int(pid) for pid in status.get("pids", [])]
    if not pids:
        return {
            "ok": False,
            "stopped": False,
            "status": status,
            "message": f"Linux 端口 {status['port']} 被占用，但没有拿到 PID，无法自动停止。",
        }

    signal_name = "-KILL" if force else "-TERM"
    remote_command = "kill " + signal_name + " " + " ".join(str(pid) for pid in pids)
    result = subprocess.run(
        build_ssh_argv(profile["linux_shell"], remote_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=8,
        check=False,
    )
    raw = result.stdout.strip()
    if result.returncode != 0:
        return {
            "ok": False,
            "stopped": False,
            "status": status,
            "raw": raw,
            "message": raw or f"停止远端旧服务失败，ssh/kill 返回 {result.returncode}",
        }

    latest = status
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        time.sleep(0.25)
        latest = remote_linux_port_status(profile)
        if latest.get("ok") and not latest.get("occupied"):
            break

    if latest.get("ok") and latest.get("occupied"):
        return {
            "ok": False,
            "stopped": True,
            "stopped_pids": pids,
            "status": latest,
            "message": f"已向 PID {', '.join(str(pid) for pid in pids)} 发送停止信号，但端口仍被占用。",
        }

    return {
        "ok": True,
        "stopped": True,
        "stopped_pids": pids,
        "status": latest,
        "message": f"已停止远端旧 Linux 服务 PID {', '.join(str(pid) for pid in pids)}。",
    }


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    profile = deep_merge(DEFAULT_PROFILE, profile)
    errors: list[str] = []
    warnings: list[str] = []
    windows = profile["windows"]
    linux = profile["linux"]
    shell = profile["linux_shell"]

    if float(windows["dynamic_min_x_mm"]) >= float(windows["dynamic_max_x_mm"]):
        errors.append("Windows 动态 X 范围需要满足 min < max")
    if float(windows["dynamic_min_y_mm"]) >= float(windows["dynamic_max_y_mm"]):
        errors.append("Windows 动态 Y 范围需要满足 min < max")
    if len(windows.get("roi", [])) != 4:
        errors.append("Windows ROI 需要包含 x/y/w/h 四个值")
    if float(linux["min_target_x_mm"]) >= float(linux["max_target_x_mm"]):
        errors.append("Linux 目标 X 范围需要满足 min < max")
    if float(linux["min_grasp_z_mm"]) >= float(linux["max_grasp_z_mm"]):
        errors.append("Linux 抓取 Z 范围需要满足 min < max")
    if not shell.get("ssh_user") or not shell.get("ssh_host"):
        errors.append("Linux SSH 用户和主机不能为空")
    if linux.get("execute"):
        warnings.append("Linux 已开启真实执行")
    if float(linux.get("command_duration_s", 0)) <= 0:
        errors.append("command-duration-s 必须大于 0")
    if float(linux.get("gripper_settle_s", 0)) < 0:
        errors.append("gripper-settle-s 不能为负数")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "commands": build_generated_commands(profile),
    }


def load_saved_commands() -> dict[str, str]:
    return build_generated_commands(load_profile())


def save_commands(payload: dict[str, Any]) -> dict[str, str]:
    commands = {
        "windows": str(payload.get("windows", "")).strip(),
        "linux": str(payload.get("linux", "")).strip(),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    COMMANDS_PATH.write_text(
        json.dumps(commands, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return commands


class PiperCommandCenter:
    def __init__(self) -> None:
        self.windows = ManagedProcess("windows", preview_enabled=True)
        self.linux = ManagedProcess("linux")

    def process(self, name: str) -> ManagedProcess:
        if name == "windows":
            return self.windows
        if name == "linux":
            return self.linux
        raise KeyError(name)

    def status(self) -> dict[str, Any]:
        return {
            "windows": self.windows.status(),
            "linux": self.linux.status(),
            "preview": preview_status(),
            "arm_activity": self.linux.activity_status(),
            "arm_activity_history": self.linux.activity_history(),
        }


def preview_status() -> dict[str, Any]:
    return {
        name: image_status(PREVIEW_DIR / filename)
        for name, filename in PREVIEW_FILES.items()
    }


def image_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "mtime": None}
    stat = path.stat()
    return {"exists": True, "mtime": stat.st_mtime, "size": stat.st_size}


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "PiperCommandCenter/2.0"

    @property
    def app(self) -> PiperCommandCenter:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {fmt % args}", file=sys.stderr)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            safe_name = path.removeprefix("/static/").replace("\\", "/")
            if ".." in Path(safe_name).parts:
                self.send_error(400, "invalid path")
                return
            self.send_file(STATIC_DIR / safe_name, cache=False)
            return
        if path == "/api/status":
            self.send_json(self.app.status())
            return
        if path == "/api/profile":
            profile = load_profile()
            self.send_json(
                {
                    "profile": profile,
                    "commands": build_generated_commands(profile),
                    "validation": validate_profile(profile),
                }
            )
            return
        if path == "/api/linux/port-status":
            self.send_json(remote_linux_port_status(load_profile()))
            return
        if path == "/api/commands":
            self.send_json(load_saved_commands())
            return
        if path.startswith("/api/logs/"):
            name = path.rsplit("/", 1)[-1]
            try:
                process = self.app.process(name)
            except KeyError:
                self.send_error(404, "unknown process")
                return
            query = parse_qs(parsed.query)
            limit = parse_int(query.get("limit", ["500"])[0], default=500)
            important = query.get("important", ["0"])[0] in ("1", "true", "yes")
            self.send_json(process.logs(limit=limit, important=important))
            return
        if path == "/api/preview/status":
            self.send_json(preview_status())
            return
        if path.startswith("/stream/"):
            filename = path.rsplit("/", 1)[-1]
            stream_name = filename.removesuffix(".mjpg")
            if stream_name not in PREVIEW_FILES or filename == stream_name:
                self.send_error(404, "unknown stream")
                return
            self.send_mjpeg_stream(PREVIEW_DIR / PREVIEW_FILES[stream_name])
            return
        if path.startswith("/preview/"):
            filename = path.rsplit("/", 1)[-1]
            allowed = {f"{name}.jpg": preview_file for name, preview_file in PREVIEW_FILES.items()}
            if filename not in allowed:
                self.send_error(404, "unknown preview")
                return
            preview_path = PREVIEW_DIR / allowed[filename]
            if not preview_path.exists():
                self.send_error(404, "preview not ready")
                return
            self.send_file(preview_path, cache=False)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/profile":
            payload = self.read_json()
            profile = save_profile(payload.get("profile", payload))
            self.send_json(
                {
                    "profile": profile,
                    "commands": build_generated_commands(profile),
                    "validation": validate_profile(profile),
                }
            )
            return
        if path == "/api/profile/validate":
            payload = self.read_json()
            profile = payload.get("profile", payload)
            self.send_json(validate_profile(profile))
            return
        if path == "/api/commands":
            payload = self.read_json()
            self.send_json(save_commands(payload))
            return
        if path == "/api/linux/port-status":
            payload = self.read_json()
            profile = payload.get("profile") or load_profile()
            self.send_json(remote_linux_port_status(profile))
            return
        if path == "/api/linux/stop-port-owner":
            payload = self.read_json()
            profile = payload.get("profile") or load_profile()
            result = stop_remote_linux_port_owner(profile, force=bool(payload.get("force")))
            self.app.linux.append_event("warn" if result.get("ok") else "error", result.get("message", ""))
            self.send_json(result, status=200 if result.get("ok") else 400)
            return
        if path == "/api/linux/setup-can":
            payload = self.read_json()
            profile = payload.get("profile") or load_profile()
            result = remote_setup_can(profile, str(payload.get("password") or ""))
            self.app.linux.append_event("success" if result.get("ok") else "error", result.get("message", ""))
            self.send_json(result, status=200 if result.get("ok") else 400)
            return

        match = re.fullmatch(r"/api/process/(windows|linux)/(start|stop)", path)
        if not match:
            self.send_error(404, "not found")
            return
        name, action = match.groups()
        process = self.app.process(name)
        try:
            if action == "start":
                payload = self.read_json()
                command = str(payload.get("command") or "").strip()
                if not command:
                    profile = payload.get("profile") or load_profile()
                    command = build_generated_commands(profile)[name]
                if name == "linux":
                    profile = payload.get("profile") or load_profile()
                    port_status = remote_linux_port_status(profile)
                    if not port_status.get("ok"):
                        self.app.linux.append_event("error", port_status.get("message", "无法检查远端端口"))
                        self.send_json({"ok": False, "error": port_status.get("message"), "port_status": port_status}, status=400)
                        return
                    if port_status.get("occupied"):
                        self.app.linux.append_event("warn", port_status.get("message", "Linux 端口已被占用"))
                        self.send_json({"ok": False, "error": port_status.get("message"), "port_status": port_status}, status=400)
                        return
                process.start(command)
            else:
                process.stop()
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
            return
        self.send_json({"ok": True, "status": process.status()})

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        try:
            payload = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON payload must be an object")
        return payload

    def send_json(self, payload: Any, *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, *, cache: bool = True) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, "file not found")
            return
        content_type, _ = mimetypes.guess_type(str(path))
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_mjpeg_stream(self, path: Path) -> None:
        boundary = "piper-preview-frame"
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        last_mtime_ns: int | None = None
        last_body: bytes | None = None
        while True:
            try:
                stat = path.stat()
                if stat.st_mtime_ns != last_mtime_ns:
                    last_body = path.read_bytes()
                    last_mtime_ns = stat.st_mtime_ns
                body = last_body
                if body:
                    self.wfile.write(
                        (
                            f"--{boundary}\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(body)}\r\n\r\n"
                        ).encode("ascii")
                    )
                    self.wfile.write(body)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(0.05)
            except FileNotFoundError:
                time.sleep(0.1)
            except PermissionError:
                time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return


def parse_int(value: str, *, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


class PiperHTTPServer(ThreadingHTTPServer):
    app: PiperCommandCenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Piper web command center.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    server = PiperHTTPServer((args.host, args.port), RequestHandler)
    server.app = PiperCommandCenter()
    url = f"http://{args.host}:{args.port}"
    print(f"Piper command center listening on {url}")
    print(f"Project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web command center...")
    finally:
        server.app.windows.stop()
        server.app.linux.stop()
        server.server_close()


if __name__ == "__main__":
    main()
