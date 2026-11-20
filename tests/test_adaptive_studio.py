"""Adaptive Studio: plan endpoint, fleet reading, and HTTP surface."""
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "apps" / "adaptive_studio"
SERVER_PATH = APP_DIR / "server.py"
PARITY_PATH = APP_DIR / "parity_vectors.json"


def _load_studio():
    for name in list(sys.modules):
        if name == "studio_under_test" or name.startswith("studio_under_test."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("studio_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["studio_under_test"] = module
    spec.loader.exec_module(module)
    return module


studio = _load_studio()


FLEET = {
    "source": "file",
    "config_path": "/tmp/gpu_config.json",
    "master": {"host": "", "delegate_only": False},
    "workers": [
        {"id": f"colab-{letter}", "name": f"Colab {letter.upper()}", "type": "remote",
         "host": f"https://{letter}.trycloudflare.com", "port": 443}
        for letter in ("a", "b", "c")
    ],
}


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------

def test_reads_an_explicit_config_file(tmp_path):
    config = tmp_path / "gpu_config.json"
    config.write_text(json.dumps({
        "master": {"host": "https://master.example.com"},
        "settings": {"master_delegate_only": True},
        "workers": [
            {"id": "w1", "name": "GPU one", "type": "remote", "host": "1.2.3.4", "port": 443},
            {"id": "w2", "type": "local", "port": 8189},
            "not-a-worker",
        ],
    }))

    fleet = studio.read_fleet(config)

    assert fleet["source"] == "file"
    assert fleet["master"]["host"] == "https://master.example.com"
    assert fleet["master"]["delegate_only"] is True
    assert [worker["id"] for worker in fleet["workers"]] == ["w1", "w2"]
    assert fleet["workers"][1]["name"] == "w2"  # falls back to the id


def test_missing_config_is_not_fatal(tmp_path):
    fleet = studio.read_fleet(tmp_path / "absent.json")
    assert fleet["source"] == "missing"
    assert fleet["workers"] == []


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def test_auto_accounts_follow_the_fleet():
    result = studio.build_plan({"duration_seconds": 6, "fps": 24}, FLEET)

    assert result["summary"]["auto_accounts"] is True
    assert result["summary"]["enabled_workers"] == 3
    assert result["summary"]["segments"] == 4  # 3 workers + participating master
    assert [account["slot"] for account in result["accounts"]] == [
        "master", "worker_1", "worker_2", "worker_3"]
    assert [account["label"] for account in result["accounts"]] == [
        "master", "Colab A", "Colab B", "Colab C"]


def test_master_can_delegate_every_segment():
    result = studio.build_plan(
        {"duration_seconds": 6, "fps": 24, "master_participates": False}, FLEET)

    assert result["summary"]["segments"] == 3
    assert [account["slot"] for account in result["accounts"]] == [
        "worker_1", "worker_2", "worker_3"]
    assert all(account["kind"] == "worker" for account in result["accounts"])


def test_manual_accounts_override_the_fleet():
    result = studio.build_plan(
        {"duration_seconds": 6, "fps": 24, "accounts": 8, "master_participates": False}, FLEET)

    assert result["summary"]["auto_accounts"] is False
    assert result["summary"]["segments"] == 8
    labels = [account["label"] for account in result["accounts"]]
    assert labels[:3] == ["Colab A", "Colab B", "Colab C"]
    assert labels[3] == "worker_4"  # beyond the configured fleet


def test_segments_cover_the_timeline_without_gaps():
    result = studio.build_plan(
        {"duration_seconds": 6, "fps": 24, "accounts": 5, "master_participates": False}, FLEET)
    accounts = result["accounts"]

    assert accounts[0]["start_frame"] == 0
    for earlier, later in zip(accounts, accounts[1:]):
        assert earlier["end_frame"] == later["start_frame"]
    assert accounts[-1]["end_seconds"] == pytest.approx(
        result["summary"]["stitched_seconds"], abs=0.001)
    assert result["summary"]["stitched_frames"] == sum(
        account["frames"] for account in accounts) - (len(accounts) - 1)


def test_wan_alignment_and_free_frame_counts_differ():
    aligned = studio.build_plan(
        {"duration_seconds": 6, "fps": 24, "accounts": 3, "align_to_wan": True}, FLEET)
    free = studio.build_plan(
        {"duration_seconds": 6, "fps": 24, "accounts": 3, "align_to_wan": False}, FLEET)

    assert all(account["frames"] % 4 == 1 for account in aligned["accounts"])
    assert free["summary"]["stitched_frames"] == 144
    assert aligned["summary"]["stitched_frames"] >= 144


def test_short_segments_and_idle_accounts_are_explained():
    result = studio.build_plan(
        {"duration_seconds": 2, "fps": 24, "accounts": 40, "master_participates": False}, FLEET)

    assert result["summary"]["segments"] < 40
    assert any("only" in warning for warning in result["warnings"])
    assert any("no segment" in warning for warning in result["warnings"])


def test_bad_values_are_clamped_instead_of_raising():
    result = studio.build_plan(
        {"duration_seconds": "x", "fps": 0, "accounts": -5, "min_frames_per_segment": 0}, FLEET)

    assert result["summary"]["duration_seconds"] == 6.0
    assert result["summary"]["fps"] == 1
    assert result["summary"]["segments"] >= 1


def test_ffmpeg_recipe_trims_shared_boundaries():
    result = studio.build_plan({"duration_seconds": 6, "fps": 24, "accounts": 2}, FLEET)
    script = studio.ffmpeg_script(result)

    assert "trim_worker_1.mp4" in script
    assert "trim_worker_2.mp4" in script
    assert "-f concat -safe 0 -i list.txt" in script
    assert 'select=\'gte(n' in script


def test_ffmpeg_recipe_skips_trimming_when_boundaries_are_not_shared():
    result = studio.build_plan(
        {"duration_seconds": 6, "fps": 24, "accounts": 2, "share_boundary_frames": False}, FLEET)
    script = studio.ffmpeg_script(result)

    assert "trim_" not in script
    assert "segment_worker_1.mp4" in script and "segment_worker_2.mp4" in script


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    config = tmp_path / "gpu_config.json"
    config.write_text(json.dumps({
        "master": {"host": ""},
        "settings": {"master_delegate_only": False},
        "workers": FLEET["workers"],
    }))
    instance = studio.create_server("127.0.0.1", 0, config)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    host, port = instance.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read(), response.headers.get("Content-Type", "")


def _post(url, payload, timeout=5):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def test_health_and_ui_are_served(server):
    status, body, _ = _get(f"{server}/api/health")
    assert status == 200 and json.loads(body)["ok"] is True

    status, body, content_type = _get(f"{server}/")
    assert status == 200
    assert "text/html" in content_type
    assert b"Adaptive Studio" in body

    for asset in ("styles.css", "app.js", "planner.js"):
        status, body, content_type = _get(f"{server}/{asset}")
        assert status == 200 and len(body) > 0, asset


def test_plan_endpoint_returns_accounts_and_ffmpeg(server):
    status, result = _post(f"{server}/api/plan", {"duration_seconds": 6, "fps": 24})

    assert status == 200
    assert result["summary"]["segments"] == 4
    assert len(result["accounts"]) == 4
    assert result["fleet"]["workers"]
    assert "ffmpeg" in result


def test_fleet_endpoint_reports_the_config_file(server):
    status, body, _ = _get(f"{server}/api/fleet")
    fleet = json.loads(body)
    assert status == 200
    assert fleet["source"] == "file"
    assert len(fleet["workers"]) == 3


def test_unknown_paths_and_bad_payloads_are_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as missing:
        _get(f"{server}/does-not-exist")
    assert missing.value.code == 404

    # Path traversal must not escape the web directory.
    with pytest.raises(urllib.error.HTTPError) as traversal:
        _get(f"{server}/../server.py")
    assert traversal.value.code == 404

    request = urllib.request.Request(
        f"{server}/api/plan", data=b"not json", method="POST",
        headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as bad:
        urllib.request.urlopen(request, timeout=5)
    assert bad.value.code == 400


def test_unknown_endpoint_post_is_404(server):
    request = urllib.request.Request(f"{server}/api/nope", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 404


# ---------------------------------------------------------------------------
# Browser parity
# ---------------------------------------------------------------------------

def _fingerprint(plan):
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


def _python_plan(case_input):
    from utils.video_plan import plan_video_segments

    return plan_video_segments(
        total_frames=case_input["totalFrames"],
        num_segments=case_input["numSegments"],
        min_frames_per_segment=case_input["minFramesPerSegment"],
        align_to_wan=case_input["alignToWan"],
        share_boundary_frames=case_input["shareBoundaryFrames"],
        num_participants=case_input["numParticipants"],
    )


def test_browser_parity_vectors_match_the_python_planner():
    """The JS port is locked to this fixture; regenerate it when Python changes."""
    vectors = json.loads(PARITY_PATH.read_text())
    assert len(vectors["plan_cases"]) > 150

    for case in vectors["plan_cases"]:
        assert _fingerprint(_python_plan(case["input"])) == case["expected"], case["input"]


def test_browser_parity_vectors_match_the_python_plan_response():
    vectors = json.loads(PARITY_PATH.read_text())
    for case in vectors["response_cases"]:
        result = studio.build_plan(case["input"]["payload"], case["input"]["fleet"])
        assert result["accounts"] == case["expected"]["accounts"], case["input"]["payload"]
        assert result["summary"] == case["expected"]["summary"], case["input"]["payload"]
        assert result["warnings"] == case["expected"]["warnings"], case["input"]["payload"]
        assert studio.ffmpeg_script(result) == case["ffmpeg"], case["input"]["payload"]


def test_browser_app_ships_the_planner_and_loads_it_first():
    html = (APP_DIR / "web" / "index.html").read_text()
    planner = (APP_DIR / "web" / "planner.js").read_text()

    assert 'src="planner.js"' in html
    assert 'src="app.js"' in html
    # planner must be defined before the UI script runs
    assert html.index('src="planner.js"') < html.index('src="app.js"')
    # classic script (not a module) so file:// double-click also works
    assert 'type="module"' not in html
    assert "global.AdaptivePlanner" in planner
