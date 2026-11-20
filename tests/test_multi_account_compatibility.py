"""Multi-account compatibility: prove the adaptive plan is correct for any fleet size.

These tests simulate a fleet end to end with the production code paths:

* orchestration resolves the auto divider and injects fleet context per account
  (``api/orchestration/prompt_transform.py``);
* every account plans the same segments and picks its own window
  (``nodes/utilities.py`` planner);
* the real collector orders participant frames and maps preview slots
  (``nodes/collector.py``);
* the stitcher rebuilds the timeline (``nodes/utilities.py``).

Frame identity is tracked, so a duplicated or missing boundary frame fails the
test rather than silently producing a 1-frame glitch.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]

# Fleet sizes to prove: solo, the original 3 Colab accounts, common mid sizes,
# and both sides of the 16-slot socket limit.
FLEET_SIZES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 17, 20]

DURATION_SECONDS = 6.0
FPS = 24


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_adaptive = _load_file("adaptive_video_harness", ROOT / "tests" / "test_adaptive_video.py")
utils = _adaptive.utils
pt = _adaptive.pt
_collector_harness = _load_file("collector_harness_for_fleet", ROOT / "tests" / "test_collector_list_inputs.py")


def _load_collector(worker_count):
    """Load the real collector with ``worker_count`` configured workers."""
    module = _collector_harness._load_collector_module()
    module.load_config = lambda: {"workers": [
        {"id": worker_id, "name": f"GPU {index}"}
        for index, worker_id in enumerate(_worker_ids(worker_count))
    ]}
    return module


def _worker_ids(count):
    return [f"worker-{index}" for index in range(count)]


def _participants(worker_count, delegate_only):
    """Master plus workers, or workers only when the master delegates."""
    workers = _worker_ids(worker_count)
    return ([] if delegate_only else ["master"]) + workers


def _plan_for(participant, workers, delegate_only):
    is_master = participant == "master"
    worker_index = workers.index(participant) if not is_master else 0
    return utils.DistributedVideoPlanner().plan(
        DURATION_SECONDS, FPS, 0, 5, True, True,
        is_worker=not is_master,
        worker_id="" if is_master else f"worker_{worker_index}",
        enabled_worker_ids=json.dumps(workers),
        delegate_only=delegate_only,
    )


def _render_segment(start_frame, frames):
    """A synthetic clip whose pixel value is the global frame index."""
    ids = torch.arange(start_frame, start_frame + frames, dtype=torch.float32)
    return ids.reshape(-1, 1, 1, 1).repeat(1, 2, 2, 3)


@pytest.mark.parametrize("worker_count", FLEET_SIZES)
@pytest.mark.parametrize("delegate_only", [False, True])
def test_plan_gives_every_account_its_own_segment(worker_count, delegate_only):
    workers = _worker_ids(worker_count)
    participants = _participants(worker_count, delegate_only)
    results = {participant: _plan_for(participant, workers, delegate_only) for participant in participants}

    plans = {result[0] for result in results.values()}
    assert len(plans) == 1, "every account must compute the identical plan"
    plan = json.loads(plans.pop())

    # One segment per participant, each reached by exactly one account.
    assert plan["num_segments"] == len(participants)
    assert plan["participants"] == len(participants)
    assert sorted(result[2] for result in results.values()) == list(range(len(participants)))

    # Segment windows tile the timeline with shared boundary frames.
    segments = plan["segments"]
    assert segments[0]["start_frame"] == 0
    for earlier, later in zip(segments, segments[1:]):
        assert earlier["end_frame"] == later["start_frame"]
    assert plan["rendered_frames"] == sum(segment["frames"] for segment in segments)

    # Rendered length matches the requested duration, minus the shared frames.
    assert plan["stitched_frames"] == plan["rendered_frames"] - (plan["num_segments"] - 1)
    assert abs(plan["stitched_frames"] - DURATION_SECONDS * FPS) <= plan["num_segments"]


@pytest.mark.parametrize("worker_count", FLEET_SIZES)
@pytest.mark.parametrize("delegate_only", [False, True])
def test_rendered_segments_rebuild_a_seamless_timeline(worker_count, delegate_only):
    """The collector's order plus the plan must reproduce every frame exactly once."""
    workers = _worker_ids(worker_count)
    participants = _participants(worker_count, delegate_only)
    results = {participant: _plan_for(participant, workers, delegate_only) for participant in participants}
    plan = json.loads(next(iter(results.values()))[0])

    # Each account renders its own window; the collector concatenates master
    # first and then the enabled workers, which is exactly participant order.
    clips = []
    for participant in participants:
        _plan_json, _num, index, start_frame, _end, frames = results[participant][:6]
        clips.append(_render_segment(start_frame, frames))
    combined = torch.cat(clips, dim=0)

    stitcher = utils.DistributedVideoStitcher()
    images, frames = stitcher.stitch(
        share_boundary_frames=True, images=combined, plan=json.dumps(plan))

    assert frames == plan["stitched_frames"]
    # Frame values must be 0..stitched-1 with no duplicate and no gap.
    assert images[:, 0, 0, 0].tolist() == [float(index) for index in range(plan["stitched_frames"])]


@pytest.mark.parametrize("worker_count", FLEET_SIZES)
def test_auto_divider_resolves_to_the_fleet_size(worker_count):
    workers = _worker_ids(worker_count)
    prompt = {
        "1": {"class_type": "ImageBatchDivider", "inputs": {"divide_by": 0}},
        "2": {"class_type": "AudioBatchDivider", "inputs": {"divide_by": 0}},
        "3": {"class_type": "DistributedVideoPlanner", "inputs": {}},
        "4": {"class_type": "DistributedVideoStitcher", "inputs": {}},
        "5": {"class_type": "DistributedCollector", "inputs": {}},
    }
    index = pt.PromptIndex(prompt)
    master = pt.apply_participant_overrides(
        {key: {"class_type": node["class_type"], "inputs": dict(node["inputs"])}
         for key, node in prompt.items()},
        participant_id="master",
        enabled_worker_ids=workers,
        job_id_map=pt.generate_job_id_map(index, "run"),
        master_url="http://master",
        delegate_master=False,
        prompt_index=index,
    )
    expected = worker_count + 1
    assert int(master["1"]["inputs"]["divide_by"]) == expected
    assert int(master["2"]["inputs"]["divide_by"]) == expected
    assert int(master["4"]["inputs"]["num_participants"]) == expected
    assert master["3"]["inputs"]["worker_id"] == ""

    for position, worker in enumerate(workers):
        worker_prompt = pt.apply_participant_overrides(
            {key: {"class_type": node["class_type"], "inputs": dict(node["inputs"])}
             for key, node in prompt.items()},
            participant_id=worker,
            enabled_worker_ids=workers,
            job_id_map=pt.generate_job_id_map(index, "run"),
            master_url="http://master",
            delegate_master=False,
            prompt_index=index,
        )
        assert worker_prompt["3"]["inputs"]["worker_id"] == f"worker_{position}"
        assert worker_prompt["3"]["inputs"]["is_worker"] is True


@pytest.mark.parametrize("worker_count", [1, 3, 8, 16, 17, 20])
def test_collector_preview_slots_cover_every_account_it_can(worker_count):
    module = _load_collector(worker_count)
    assert module.PER_GPU_OUTPUT_NAMES == ("master",) + tuple(
        f"worker_{index + 1}" for index in range(16))

    slots = module.DistributedCollectorNode()._worker_output_slots(_worker_ids(worker_count))

    # Slots follow config order, so worker_N keeps its GPU as accounts toggle.
    expected = min(worker_count, 16)
    assert list(slots.values()) == [f"worker_{index + 1}" for index in range(expected)]


@pytest.mark.parametrize("worker_count", [1, 3, 8, 16])
def test_collector_combines_accounts_in_plan_order(worker_count):
    module = _load_collector(worker_count)
    participants = _worker_ids(worker_count)
    master_clip = _render_segment(0, 4)
    worker_images = {
        participant: {0: _render_segment(4 * (index + 1), 4)}
        for index, participant in enumerate(participants)
    }

    combined = module.DistributedCollectorNode()._reorder_and_combine_tensors(
        worker_images, participants, master_clip.shape[0], master_clip, False, master_clip)

    values = combined[:, 0, 0, 0].tolist()
    assert values == [float(index) for index in range(4 * (worker_count + 1))]


@pytest.mark.parametrize("worker_count", FLEET_SIZES)
def test_dropped_account_keeps_the_rest_of_the_timeline(worker_count):
    """A Colab account that dies mid-run must not shorten or corrupt the video."""
    if worker_count < 2:
        pytest.skip("needs at least two accounts")
    workers = _worker_ids(worker_count)
    participants = _participants(worker_count, False)
    results = {participant: _plan_for(participant, workers, False) for participant in participants}
    plan = json.loads(next(iter(results.values()))[0])

    clips = []
    for participant in participants[:-1]:  # the last account never reports
        _plan_json, _num, index, start_frame, _end, frames = results[participant][:6]
        clips.append(_render_segment(start_frame, frames))
    combined = torch.cat(clips, dim=0)

    stitcher = utils.DistributedVideoStitcher()
    images, frames = stitcher.stitch(
        share_boundary_frames=True, images=combined, plan=json.dumps(plan))

    # Without matching offsets the stitcher keeps every frame it received.
    assert frames == combined.shape[0]
    assert images[:, 0, 0, 0].tolist() == combined[:, 0, 0, 0].tolist()


@pytest.mark.parametrize("worker_count", [17, 20])
def test_fleets_past_the_socket_limit_still_stitch(worker_count):
    """Over 16 accounts there are no per-account sockets, but the plan path works."""
    workers = _worker_ids(worker_count)
    participants = _participants(worker_count, False)
    results = {participant: _plan_for(participant, workers, False) for participant in participants}
    plan = json.loads(next(iter(results.values()))[0])
    assert plan["num_segments"] == worker_count + 1

    combined = torch.cat([
        _render_segment(results[participant][3], results[participant][5])
        for participant in participants
    ], dim=0)
    images, frames = utils.DistributedVideoStitcher().stitch(
        share_boundary_frames=True, images=combined, plan=json.dumps(plan))
    assert frames == plan["stitched_frames"]
    assert images[:, 0, 0, 0].tolist() == [float(index) for index in range(frames)]


def test_worker_prompt_keeps_the_planner_and_drops_the_stitcher():
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        # The planner drives per-account frame windows, so it sits upstream of
        # the sampler and must survive worker pruning.
        "2": {"class_type": "DistributedVideoPlanner", "inputs": {}},
        "3": {"class_type": "ImageFromBatch",
              "inputs": {"image": ["1", 0], "batch_index": ["2", 2], "length": 1}},
        "4": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["3", 0], "latent_image": ["3", 0]}},
        "5": {"class_type": "DistributedCollector", "inputs": {"images": ["4", 0]}},
        "6": {"class_type": "DistributedVideoStitcher",
              "inputs": {"images": ["5", 0], "plan": ["2", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
    }
    pruned = pt.prune_prompt_for_worker(prompt, use_worker_sink=True)
    classes = {node["class_type"] for node in pruned.values()}
    assert "DistributedVideoPlanner" in classes, "workers must keep the planner"
    assert "DistributedVideoStitcher" not in classes, "stitching is master-only"


def test_master_delegating_still_plans_one_segment_per_worker():
    workers = _worker_ids(4)
    results = [_plan_for(worker, workers, True) for worker in workers]
    plans = {result[0] for result in results}
    assert len(plans) == 1
    plan = json.loads(plans.pop())
    assert plan["num_segments"] == 4
    assert sorted(result[2] for result in results) == [0, 1, 2, 3]


def test_segment_ceiling_caps_a_very_large_fleet():
    workers = _worker_ids(40)
    # 20s gives every account more than the 5-frame minimum, so the 32-segment
    # ceiling is what stops the plan.
    result = utils.DistributedVideoPlanner().plan(
        20.0, FPS, 0, 5, True, True, False, "", json.dumps(workers), False)
    plan = json.loads(result[0])
    assert plan["num_segments"] == utils.MAX_VIDEO_SEGMENTS == 32
    assert plan["capped"] is True
    assert any("ceiling" in warning for warning in plan["warnings"])


def test_short_videos_cap_by_the_frame_minimum():
    workers = _worker_ids(40)
    result = utils.DistributedVideoPlanner().plan(
        DURATION_SECONDS, FPS, 0, 5, True, True, False, "", json.dumps(workers), False)
    plan = json.loads(result[0])
    assert plan["num_segments"] == 144 // 5 == 28
    assert plan["capped"] is True
    assert all(segment["frames"] >= 5 for segment in plan["segments"])
