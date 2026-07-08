from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import re
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
PREVIEW_FILES = {
    "rgb": "opencv_rgb.jpg",
    "mask": "opencv_mask.jpg",
    "depth": "opencv_depth.jpg",
}


DEFAULT_WINDOWS_COMMAND = (
    "python -m robot_control.windows.client.opencv_grasp_send_tcp dynamic "
    "--host 192.168.146.128 --conveyor-speed-mm-s 140 "
    "--dynamic-min-x-mm 220 --dynamic-max-x-mm 480 "
    "--dynamic-min-y-mm -550 --dynamic-max-y-mm -300 "
    "--dynamic-pick-x-from-bbox-center --dynamic-pick-y-mm 0 "
    "--width 1280 --height 720 --conveyor-direction positive-y "
    "--base-z-offset-mm 0 --belt-lower-hsv 35 40 40 "
    "--belt-upper-hsv 90 255 255 --reject-highlight --highlight-max-s 30 "
    "--highlight-min-v 225 --min-area-px 1200 --min-width-px 25 "
    "--min-height-px 25 --morphology-kernel-size 5 --morphology-iterations 1 "
    "--roi 250 100 940 550 --entry-edge right --entry-clear-margin-px 15 "
    "--entry-clear-frames 3 "
    "--hand-eye-result robot_control\\linux\\hand_eye\\eye_to_hand_result_auto.json "
    "--dynamic-debug-candidates --dynamic-skip-memory-s 0 "
    "--dynamic-probe-interval-s 0 --print-interval 1 "
    "--morphology-close-kernel-size 11 --morphology-open-kernel-size 3 "
    "--contour-mask-mode convex-hull"
)

DEFAULT_LINUX_COMMAND = (
    "python3 -m robot_control.linux.catch.catch_server_1 "
    "--host 0.0.0.0 --port 5005 --can can0 "
    "--hand-eye-result robot_control/linux/hand_eye/eye_to_hand_result_auto.json "
    "--startup-speed 10 --speed 60 --rx-deg -175.957642 "
    "--ry-deg 1.353477 --rz-deg -102.579260 --command-duration-s 0.3 "
    "--gripper-settle-s 0.1 --dynamic-preposition "
    "--dynamic-preposition-clearance-mm 40 --final-action-time-s 0.5 "
    "--command-period-s 0.05 --print-feedback --open-command-mm 0 "
    "--grasp-command-mm -90 --effort-nm 0.9 --no-descent-protection "
    "--min-target-x-mm 220 --max-target-x-mm 480 --force-pick-y-mm 0 "
    "--adaptive-joint-pitch-by-x --dynamic-pitch-mode planar-ik "
    "--planar-ik-fk-tool-z-offset-mm 82 --planar-ik-joint-tolerance-deg 12 "
    "--keep-pitch-min-x-mm 220 --keep-pitch-max-x-mm 355 "
    "--pitch-deg-per-mm-far 0.013 --pitch-deg-per-mm-near 0.01 "
    "--min-dynamic-pitch-deg -5 --max-dynamic-pitch-deg 5 "
    "--grasp-z 120 --use-detected-z top --detected-z-offset-mm 115 "
    "--min-grasp-z-mm 120 --max-grasp-z-mm 310 --execute "
    "--planar-ik-grasp-z-comp-mm-per-deg 20.0 --planar-ik-speed 20 "
    "--dynamic-wait-placement nowait --pick-x-offset-mm -45 --t-robot-s 2.8 "
    "--dynamic-t-robot-by-x --dynamic-t-robot-reference-x-mm 340 "
    "--dynamic-t-robot-s-per-mm 0 --dynamic-t-robot-planar-near-s-per-mm 0.017 "
    "--dynamic-t-robot-planar-far-s-per-mm 0.010 "
    "--grasp-yaw-by-object-short-edge --gripper-close-axis-offset-deg 0 "
    "--grasp-yaw-max-delta-deg 70 --grasp-yaw-deadband-deg 5 "
    "--grasp-yaw-speed 60"
)


IMPORTANT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("error", re.compile(r"\b(error|failed|traceback|exception|timed out|abort|refused)\b", re.I)),
    ("error", re.compile(r"TARGET_POS_EXCEEDS_LIMIT|EXCEEDS_LIMIT", re.I)),
    ("warn", re.compile(r"\b(skipped|warning|protected abort|not reached|no valid)\b", re.I)),
    ("success", re.compile(r"\b(ack|send completed|catch send counts|after catch|executed=True|Piper arm enabled)\b", re.I)),
    ("info", re.compile(r"\b(dynamic timing|client connected|listening on|using hand-eye)\b", re.I)),
)


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


def load_saved_commands() -> dict[str, str]:
    defaults = {
        "windows": DEFAULT_WINDOWS_COMMAND,
        "linux": DEFAULT_LINUX_COMMAND,
    }
    if not COMMANDS_PATH.exists():
        return defaults
    try:
        loaded = json.loads(COMMANDS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(loaded, dict):
        return defaults
    return {
        "windows": str(loaded.get("windows") or defaults["windows"]),
        "linux": str(loaded.get("linux") or defaults["linux"]),
    }


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
    server_version = "PiperCommandCenter/1.0"

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
            self.send_file(STATIC_DIR / safe_name)
            return
        if path == "/api/status":
            self.send_json(self.app.status())
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
        if path == "/api/commands":
            payload = self.read_json()
            self.send_json(save_commands(payload))
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
                process.start(str(payload.get("command", "")))
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
