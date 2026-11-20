#!/usr/bin/env python3
"""Adaptive Studio: plan a multi-account video split from the browser.

Runs beside ComfyUI on the master machine and uses only the Python standard
library. It reads the Distributed worker fleet from ``gpu_config.json`` and
calls the same planner the nodes use (``utils/video_plan.py``), so what you see
here is exactly what each Colab account will render.

    python3 apps/adaptive_studio/server.py --port 8765

Endpoints:
    GET  /            the UI
    GET  /api/health  liveness
    GET  /api/fleet   master + workers from gpu_config.json
    POST /api/plan    plan for the given duration/fps/fleet
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
REPO_ROOT = APP_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.video_plan import (  # noqa: E402
    MAX_VIDEO_SEGMENTS,
    _round_half_up,
    plan_video_segments,
)

FPS_CHOICES = (8, 12, 16, 24, 30)
DEFAULT_CONFIG = REPO_ROOT / "gpu_config.json"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------

def read_fleet(config_path: Path | None = None) -> dict:
    """Return the configured master and workers, tolerating a missing config.

    An explicit ``config_path`` is read as-is; otherwise the live ComfyUI
    config is used so workers toggled in the panel show up immediately.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    config: dict = {}
    source = "missing"
    if config_path:
        try:
            config = json.loads(path.read_text())
            source = "file"
        except (OSError, ValueError):
            config = {}
            source = "missing"
    else:
        try:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from utils.config import load_config  # noqa: PLC0415 - optional dependency
            config = load_config() or {}
            source = "comfyui"
        except Exception:  # pragma: no cover - only when run outside the repo
            try:
                config = json.loads(path.read_text())
                source = "file"
            except (OSError, ValueError):
                config = {}
                source = "missing"

    workers = []
    for entry in config.get("workers", []) or []:
        if not isinstance(entry, dict):
            continue
        workers.append({
            "id": str(entry.get("id") or ""),
            "name": str(entry.get("name") or entry.get("id") or "worker"),
            "type": str(entry.get("type") or entry.get("worker_type") or "local"),
            "host": str(entry.get("host") or ""),
            "port": entry.get("port"),
        })
    master = config.get("master") or {}
    return {
        "source": source,
        "config_path": str(path),
        "master": {"host": str(master.get("host") or ""),
                   "delegate_only": bool((config.get("settings") or {}).get("master_delegate_only"))},
        "workers": workers,
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _round3(value: float) -> float:
    """Round to 3 decimals half-up so the browser port matches exactly."""
    return _round_half_up(value * 1000) / 1000


def _as_number(value, default, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(minimum, min(maximum, number))


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_plan(payload: dict, fleet: dict | None = None) -> dict:
    """Plan a split from a request payload; pure and side-effect free."""
    payload = payload or {}
    fleet = fleet if fleet is not None else read_fleet(payload.get("config_path"))
    workers = fleet.get("workers", []) if isinstance(fleet, dict) else []

    duration = _as_number(payload.get("duration_seconds"), 6.0, 0.04, 3600.0)
    fps = int(_as_number(payload.get("fps"), 24, 1, 240))
    min_frames = int(_as_number(payload.get("min_frames_per_segment"), 5, 1, 1000))
    master_participates = _as_bool(payload.get("master_participates"), True)
    align_to_wan = _as_bool(payload.get("align_to_wan"), True)
    share_boundary = _as_bool(payload.get("share_boundary_frames"), True)

    enabled = int(_as_number(payload.get("accounts"), 0, 0, 10_000))
    if enabled <= 0:  # auto: follow the configured fleet
        enabled = len(workers)
        auto_accounts = True
    else:
        auto_accounts = False

    participant_count = enabled + (1 if master_participates else 0)
    total_frames = max(1, int(round(duration * fps)))
    plan = plan_video_segments(
        total_frames=total_frames,
        num_segments=0,
        min_frames_per_segment=min_frames,
        align_to_wan=align_to_wan,
        share_boundary_frames=share_boundary,
        num_participants=participant_count,
    )

    accounts = []
    for position in range(plan["num_segments"]):
        is_master = master_participates and position == 0
        worker_index = position - (1 if master_participates else 0)
        if is_master:
            label, kind = "master", "master"
        elif 0 <= worker_index < len(workers):
            label = workers[worker_index]["name"] or f"worker_{worker_index + 1}"
            kind = "worker"
        else:
            label = f"worker_{worker_index + 1}"
            kind = "worker"
        segment = plan["segments"][position]
        accounts.append({
            "slot": "master" if is_master else f"worker_{worker_index + 1}",
            "label": label,
            "kind": kind,
            "index": position,
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "frames": segment["frames"],
            "start_seconds": _round3(segment["start_frame"] / fps),
            "end_seconds": _round3((segment["end_frame"] + 1) / fps),
            "seconds": _round3(segment["frames"] / fps),
        })

    warnings = list(plan["warnings"])
    idle = enabled - plan["num_segments"]
    if idle > 0:
        warnings.append(
            f"{idle} of {enabled} accounts have no segment in this plan; a longer video "
            "or fewer accounts keeps every GPU busy."
        )
    if plan["num_segments"] < participant_count:
        warnings.append(
            "The plan has fewer segments than accounts, so some accounts will idle."
        )

    return {
        "fleet": fleet,
        "accounts": accounts,
        "plan": plan,
        "warnings": warnings,
        "summary": {
            "duration_seconds": duration,
            "fps": fps,
            "total_frames": total_frames,
            "enabled_workers": enabled,
            "auto_accounts": auto_accounts,
            "master_participates": master_participates,
            "segments": plan["num_segments"],
            "keyframes": len(plan["keyframe_frames"]),
            "keyframe_times": [_round3(frame / fps) for frame in plan["keyframe_frames"]],
            "stitched_frames": plan["stitched_frames"],
            "stitched_seconds": _round3(plan["stitched_frames"] / fps),
            "frames_per_account": sorted({segment["frames"] for segment in plan["segments"]}),
            "align_to_wan": align_to_wan,
            "share_boundary_frames": share_boundary,
            "max_segments": MAX_VIDEO_SEGMENTS,
        },
    }


def ffmpeg_script(result: dict) -> str:
    """Return an FFmpeg concat recipe matching the plan."""
    summary = result["summary"]
    accounts = result["accounts"]
    fps = summary["fps"]
    share = summary["share_boundary_frames"]
    lines = ["# Save every account's clip as segment_<slot>.mp4, then run:"]
    if share and len(accounts) > 1:
        lines += ["", "# Drop the duplicated boundary frame from every segment after the first:"]
        for account in accounts[1:]:
            lines.append(
                f"ffmpeg -y -i segment_{account['slot']}.mp4 "
                f"-vf \"select='gte(n\\,1)',setpts=N/FRAME_RATE/TB\" "
                f"-r {fps} trim_{account['slot']}.mp4"
            )
        files = [f"segment_{accounts[0]['slot']}.mp4"] + [
            f"trim_{account['slot']}.mp4" for account in accounts[1:]]
    else:
        files = [f"segment_{account['slot']}.mp4" for account in accounts]
    lines += ["", "# Concatenate in plan order:"]
    lines.append("printf \"file '%s'\\n\" " + " ".join(files) + " > list.txt")
    lines.append("ffmpeg -y -f concat -safe 0 -i list.txt -c copy -movflags +faststart final.mp4")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _json_bytes(value) -> bytes:
    return json.dumps(value).encode("utf-8")


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "AdaptiveStudio/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, status: int, value):
        self._send(status, _json_bytes(value))

    def do_GET(self):  # noqa: N802 - stdlib naming
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, {"ok": True, "app": "adaptive-studio"})
            return
        if path == "/api/fleet":
            self._send_json(200, read_fleet(self.server.config_path))
            return
        if path in ("/", "/index.html"):
            self._serve_static("index.html")
            return
        self._serve_static(path.lstrip("/"))

    def do_POST(self):  # noqa: N802 - stdlib naming
        path = urlparse(self.path).path
        if path != "/api/plan":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": f"invalid JSON: {error}"})
            return
        try:
            result = build_plan(payload, read_fleet(self.server.config_path))
            result["ffmpeg"] = ffmpeg_script(result)
            self._send_json(200, result)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:  # pragma: no cover - defensive
            self._send_json(500, {"error": f"{type(error).__name__}: {error}"})

    def _serve_static(self, relative: str):
        candidate = (WEB_DIR / relative).resolve()
        if not str(candidate).startswith(str(WEB_DIR.resolve())) or not candidate.is_file():
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        content_type = CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
        self._send(200, candidate.read_bytes(), content_type)


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  config_path: Path | None = None, verbose: bool = False) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), StudioHandler)
    server.config_path = Path(config_path) if config_path else None
    server.verbose = verbose
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive Studio: plan a multi-account video split.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default 127.0.0.1; use 0.0.0.0 to reach it from another machine)")
    parser.add_argument("--port", type=int, default=8765, help="port to listen on (default 8765)")
    parser.add_argument("--config", default=None, help="path to gpu_config.json")
    parser.add_argument("--open", action="store_true", help="open the UI in a browser")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    server = create_server(args.host, args.port, args.config, args.verbose)
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '', '127.0.0.1') else args.host}:{args.port}"
    print(f"Adaptive Studio listening on {args.host}:{args.port}  ->  {url}")
    print(f"Config: {args.config or DEFAULT_CONFIG}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Adaptive Studio.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
