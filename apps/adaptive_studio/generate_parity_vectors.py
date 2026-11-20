#!/usr/bin/env python3
"""Regenerate apps/adaptive_studio/parity_vectors.json.

The vectors are produced by the reference Python planner and asserted by both
the Python tests and the browser tests, so the JS port cannot drift.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.video_plan import plan_video_segments  # noqa: E402

_spec = importlib.util.spec_from_file_location("studio_server_for_vectors", APP_DIR / "server.py")
studio = importlib.util.module_from_spec(_spec)
sys.modules["studio_server_for_vectors"] = studio
_spec.loader.exec_module(studio)

FLEET = {
    "source": "file",
    "config_path": "parity",
    "master": {"host": "", "delegate_only": False},
    "workers": [
        {"id": f"colab-{letter}", "name": f"Colab {letter.upper()}", "type": "remote",
         "host": f"https://{letter}.trycloudflare.com", "port": 443}
        for letter in ("a", "b", "c", "d", "e")
    ],
}

PLAN_TOTALS = [24, 48, 97, 144, 201, 480]


def fingerprint(plan):
    """Compact, comparison-friendly view of a plan (keeps the fixture small)."""
    return {
        "num_segments": plan["num_segments"],
        "frames": [segment["frames"] for segment in plan["segments"]],
        "starts": [segment["start_frame"] for segment in plan["segments"]],
        "ends": [segment["end_frame"] for segment in plan["segments"]],
        "keyframe_frames": plan["keyframe_frames"],
        "rendered_frames": plan["rendered_frames"],
        "stitched_frames": plan["stitched_frames"],
        "capped": plan["capped"],
        "warnings": plan["warnings"],
    }
PLAN_ACCOUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 17, 20, 33]


def plan_cases():
    cases = []
    for total in PLAN_TOTALS:
        for accounts in PLAN_ACCOUNTS:
            for align in (False, True):
                # Unshared boundaries only for small fleets; they exercise the
                # other code path without bloating the fixture.
                shares = (True, False) if accounts <= 6 else (True,)
                for share in shares:
                    plan = plan_video_segments(
                        total_frames=total,
                        num_segments=0,
                        min_frames_per_segment=5,
                        align_to_wan=align,
                        share_boundary_frames=share,
                        num_participants=accounts,
                    )
                    cases.append({
                        "input": {
                            "totalFrames": total,
                            "numSegments": 0,
                            "minFramesPerSegment": 5,
                            "alignToWan": align,
                            "shareBoundaryFrames": share,
                            "numParticipants": accounts,
                        },
                        "expected": fingerprint(plan),
                    })
    # Explicit segment counts and other minimums.
    for total, segments, minimum in [(144, 3, 5), (144, 7, 1), (144, 12, 20), (97, 4, 4), (480, 32, 5)]:
        for align in (False, True):
            plan = plan_video_segments(
                total_frames=total,
                num_segments=segments,
                min_frames_per_segment=minimum,
                align_to_wan=align,
                share_boundary_frames=True,
                num_participants=1,
            )
            cases.append({
                "input": {
                    "totalFrames": total,
                    "numSegments": segments,
                    "minFramesPerSegment": minimum,
                    "alignToWan": align,
                    "shareBoundaryFrames": True,
                    "numParticipants": 1,
                },
                "expected": fingerprint(plan),
            })
    return cases


RESPONSE_PAYLOADS = [
    {"duration_seconds": 6, "fps": 24},
    {"duration_seconds": 6, "fps": 24, "master_participates": False},
    {"duration_seconds": 6, "fps": 24, "accounts": 8},
    {"duration_seconds": 6, "fps": 24, "accounts": 8, "master_participates": False},
    {"duration_seconds": 6, "fps": 24, "accounts": 0, "align_to_wan": False},
    {"duration_seconds": 5.5, "fps": 30, "accounts": 4},
    {"duration_seconds": 12, "fps": 16, "accounts": 12, "share_boundary_frames": False},
    {"duration_seconds": 2, "fps": 24, "accounts": 40},
    {"duration_seconds": 6, "fps": 24, "accounts": 20, "min_frames_per_segment": 17},
    {"duration_seconds": 6, "fps": 24, "master_participates": False, "align_to_wan": False},
    {"duration_seconds": "x", "fps": 0, "accounts": -5, "min_frames_per_segment": 0},
    {"duration_seconds": 3.25, "fps": 12, "accounts": 2},
]


def response_cases():
    cases = []
    for payload in RESPONSE_PAYLOADS:
        result = studio.build_plan(payload, FLEET)
        cases.append({
            "input": {"payload": payload, "fleet": FLEET},
            "expected": {
                "accounts": result["accounts"],
                "summary": result["summary"],
                "warnings": result["warnings"],
            },
            "ffmpeg": studio.ffmpeg_script(result),
        })
    return cases


def main() -> int:
    vectors = {
        "generated_by": "apps/adaptive_studio/generate_parity_vectors.py",
        "plan_cases": plan_cases(),
        "response_cases": response_cases(),
    }
    out = APP_DIR / "parity_vectors.json"
    out.write_text(json.dumps(vectors, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, "
          f"{len(vectors['plan_cases'])} plan cases, "
          f"{len(vectors['response_cases'])} response cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
