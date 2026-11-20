"""Adaptive video segmentation: planning, stitching, and orchestration wiring."""
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_package_module(relative_path, module_suffix):
    """Load a repo module with a stub ``<pkg>.utils.logging`` package."""
    module_path = Path(__file__).resolve().parents[1] / relative_path
    package_name = f"dist_adaptive_{module_suffix}"

    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]

    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = []
    sys.modules[package_name] = root_pkg

    parts = relative_path.split("/")[:-1]  # e.g. ["api", "orchestration"]
    prefix = package_name
    for part in parts:
        prefix = f"{prefix}.{part}"
        pkg = types.ModuleType(prefix)
        pkg.__path__ = []
        sys.modules[prefix] = pkg

    logging_module = types.ModuleType(f"{package_name}.utils.logging")
    logging_module.debug_log = lambda *_args, **_kwargs: None
    logging_module.log = lambda *_args, **_kwargs: None
    sys.modules[f"{package_name}.utils.logging"] = logging_module

    module_name = f"{package_name}.{'.'.join(parts + [module_path.stem])}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


utils = _load_package_module("nodes/utilities.py", "utilities")
pt = _load_package_module("api/orchestration/prompt_transform.py", "prompt_transform")


class PlanVideoSegmentsTests(unittest.TestCase):
    def test_auto_uses_participant_count(self):
        plan = utils.plan_video_segments(144, 0, num_participants=6, align_to_wan=False)
        self.assertEqual(plan["num_segments"], 6)
        self.assertTrue(plan["auto"])

    def test_three_accounts_split_a_six_second_clip(self):
        plan = utils.plan_video_segments(144, 3, align_to_wan=False)
        self.assertEqual(plan["stitched_frames"], 144)
        self.assertEqual([seg["start_frame"] for seg in plan["segments"]], [0, 48, 96])
        self.assertEqual([seg["end_frame"] for seg in plan["segments"]], [48, 96, 143])

    def test_boundary_frames_are_shared_between_neighbours(self):
        plan = utils.plan_video_segments(144, 4, align_to_wan=False)
        for earlier, later in zip(plan["segments"], plan["segments"][1:]):
            self.assertEqual(earlier["end_frame"], later["start_frame"])
        self.assertEqual(plan["rendered_frames"], plan["stitched_frames"] + 3)

    def test_no_shared_boundaries_partition_the_timeline(self):
        plan = utils.plan_video_segments(144, 3, align_to_wan=False, share_boundary_frames=False)
        self.assertEqual([seg["frames"] for seg in plan["segments"]], [48, 48, 48])
        self.assertEqual(plan["rendered_frames"], 144)

    def test_keyframes_span_the_whole_timeline(self):
        plan = utils.plan_video_segments(144, 3, align_to_wan=False)
        # One keyframe per segment start, plus the exclusive end boundary.
        self.assertEqual(plan["keyframe_frames"], [0, 48, 96, 144])

    def test_more_accounts_means_shorter_segments(self):
        three = utils.plan_video_segments(144, 3, align_to_wan=False)
        eight = utils.plan_video_segments(144, 8, align_to_wan=False)
        self.assertLess(max(seg["frames"] for seg in eight["segments"]),
                        max(seg["frames"] for seg in three["segments"]))
        self.assertEqual(eight["stitched_frames"], 144)

    def test_segment_count_is_capped_by_minimum_frames(self):
        plan = utils.plan_video_segments(20, 8, min_frames_per_segment=10)
        self.assertEqual(plan["num_segments"], 2)
        self.assertTrue(plan["capped"])
        self.assertTrue(plan["warnings"])

    def test_wan_alignment_produces_4k_plus_1_segments(self):
        plan = utils.plan_video_segments(144, 5, align_to_wan=True)
        for seg in plan["segments"]:
            self.assertEqual(seg["frames"] % 4, 1)

    def test_wan_alignment_keeps_stitched_length_close_to_target(self):
        for segments in (2, 3, 4, 6, 8, 12, 16):
            plan = utils.plan_video_segments(144, segments, align_to_wan=True)
            self.assertLessEqual(abs(plan["stitched_frames"] - 144), 1,
                                 msg=f"{segments} segments landed on {plan['stitched_frames']} frames")

    def test_wan_alignment_does_not_shorten_a_many_account_run(self):
        # Snapping each segment independently rounded 18 -> 17 and lost 15 frames.
        plan = utils.plan_video_segments(144, 8, align_to_wan=True)
        self.assertGreater(plan["stitched_frames"], 144 - 5)

    def test_rejects_non_positive_totals(self):
        with self.assertRaises(ValueError):
            utils.plan_video_segments(0, 3)


class PlannerNodeTests(unittest.TestCase):
    def _plan(self, participants, worker_id="", is_worker=False, delegate_only=False, **kwargs):
        model = utils.DistributedVideoPlanner()
        return model.plan(
            kwargs.pop("duration_seconds", 6.0),
            kwargs.pop("fps", 24),
            kwargs.pop("num_segments", 0),
            kwargs.pop("min_frames_per_segment", 5),
            kwargs.pop("align_to_wan", False),
            kwargs.pop("share_boundary_frames", True),
            is_worker=is_worker,
            worker_id=worker_id,
            enabled_worker_ids=json.dumps(participants),
            delegate_only=delegate_only,
        )

    def test_master_participating_adds_one_segment(self):
        result = self._plan(["worker-a", "worker-b"])
        self.assertEqual(result[1], 3)  # master + 2 workers
        self.assertEqual(result[2], 0)  # master renders the first segment

    def test_delegate_only_master_does_not_add_a_segment(self):
        result = self._plan(["worker-a", "worker-b"], delegate_only=True)
        self.assertEqual(result[1], 2)

    def test_worker_gets_its_own_segment_window(self):
        result = self._plan(["worker-a", "worker-b", "worker-c"], worker_id="worker_2", is_worker=True)
        plan = json.loads(result[0])
        # master + 3 workers: worker_2 renders the fourth segment.
        self.assertEqual(result[2], 3)
        self.assertEqual(result[3], plan["segments"][3]["start_frame"])
        self.assertEqual(result[4], plan["segments"][3]["end_frame"])
        self.assertEqual(result[5], plan["segments"][3]["frames"])

    def test_master_participating_shifts_worker_segments_by_one(self):
        participating = self._plan(["worker-a"], worker_id="worker_0", is_worker=True)
        delegating = self._plan(["worker-a"], worker_id="worker_0", is_worker=True,
                                delegate_only=True)
        self.assertEqual(participating[2], 1)
        self.assertEqual(delegating[2], 0)

    def test_keyframe_times_are_reported_in_seconds(self):
        # master + 2 workers = 3 segments over 6s -> keyframes at 0/2/4/6s.
        result = self._plan(["worker-a", "worker-b"])
        self.assertEqual(result[6].split(","), ["0.000", "2.000", "4.000", "6.000"])

    def test_explicit_segment_count_overrides_the_fleet_size(self):
        result = self._plan(["worker-a", "worker-b"], num_segments=6)
        self.assertEqual(result[1], 6)

    def test_keyframe_indexes_bracket_the_segment(self):
        result = self._plan(["worker-a", "worker-b"], worker_id="worker_1", is_worker=True)
        self.assertEqual(result[2], 2)
        self.assertEqual(result[8], 2)
        self.assertEqual(result[9], 3)

    def test_very_short_segments_are_warned_about(self):
        result = self._plan([f"worker-{index}" for index in range(19)])
        plan = json.loads(result[0])
        self.assertTrue(any("only" in warning for warning in plan["warnings"]))

    def test_out_of_range_worker_reuses_last_segment(self):
        result = self._plan(["worker-a"], worker_id="worker_9", is_worker=True)
        self.assertEqual(result[2], result[1] - 1)


class VideoStitcherTests(unittest.TestCase):
    def _clip(self, values, channels=3):
        frames = torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1)
        return frames.repeat(1, 2, 2, channels)

    def test_joins_segments_in_socket_order(self):
        stitcher = utils.DistributedVideoStitcher()
        images, frames = stitcher.stitch(
            num_segments=2, share_boundary_frames=False,
            segment_1=self._clip([1, 2]), segment_2=self._clip([3, 4]),
        )
        self.assertEqual(frames, 4)
        self.assertEqual(images[:, 0, 0, 0].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_shared_boundary_frame_is_dropped(self):
        stitcher = utils.DistributedVideoStitcher()
        images, frames = stitcher.stitch(
            num_segments=2, share_boundary_frames=True,
            segment_1=self._clip([1, 2]), segment_2=self._clip([2, 3]),
        )
        self.assertEqual(frames, 3)
        self.assertEqual(images[:, 0, 0, 0].tolist(), [1.0, 2.0, 3.0])

    def test_missing_accounts_are_skipped(self):
        stitcher = utils.DistributedVideoStitcher()
        images, frames = stitcher.stitch(
            num_segments=4, share_boundary_frames=False,
            segment_1=self._clip([1]), segment_3=self._clip([3]),
        )
        self.assertEqual(frames, 2)
        self.assertEqual(images[:, 0, 0, 0].tolist(), [1.0, 3.0])

    def test_empty_segment_socket_is_ignored(self):
        stitcher = utils.DistributedVideoStitcher()
        images, frames = stitcher.stitch(
            num_segments=0, share_boundary_frames=False,
            segment_1=self._clip([1]), segment_2=self._clip([1])[:0],
        )
        self.assertEqual(frames, 1)
        self.assertEqual(images.shape[0], 1)

    def test_mismatched_resolution_is_rejected(self):
        stitcher = utils.DistributedVideoStitcher()
        with self.assertRaises(ValueError):
            stitcher.stitch(
                num_segments=2,
                segment_1=torch.zeros(2, 4, 4, 3),
                segment_2=torch.zeros(2, 8, 8, 3),
            )

    def test_no_segments_returns_an_empty_batch(self):
        stitcher = utils.DistributedVideoStitcher()
        images, frames = stitcher.stitch(num_segments=0)
        self.assertEqual(frames, 0)
        self.assertEqual(images.shape[0], 0)

    def test_combined_collector_batch_is_cut_by_the_plan(self):
        stitcher = utils.DistributedVideoStitcher()
        plan = utils.plan_video_segments(144, 3, align_to_wan=True)
        # Collector order is master then workers, so the batch is chronological.
        combined = self._clip(list(range(1, plan["rendered_frames"] + 1)))
        images, frames = stitcher.stitch(
            share_boundary_frames=True, images=combined, plan=json.dumps(plan))
        self.assertEqual(frames, plan["stitched_frames"])
        # 49 + 48 + 48 frames after dropping the two shared boundaries.
        self.assertEqual(images[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(images[49, 0, 0, 0].item(), 51.0)
        self.assertEqual(images[-1, 0, 0, 0].item(), float(plan["rendered_frames"]))

    def test_combined_batch_without_plan_is_kept_whole(self):
        stitcher = utils.DistributedVideoStitcher()
        images, frames = stitcher.stitch(share_boundary_frames=True, images=self._clip([1, 2, 3]))
        self.assertEqual(frames, 3)
        self.assertEqual(images[:, 0, 0, 0].tolist(), [1.0, 2.0, 3.0])

    def test_plan_mismatch_keeps_every_frame(self):
        stitcher = utils.DistributedVideoStitcher()
        plan = utils.plan_video_segments(144, 3, align_to_wan=False)
        combined = self._clip(list(range(1, 100)))  # an account dropped out
        images, frames = stitcher.stitch(share_boundary_frames=True, images=combined,
                                         plan=json.dumps(plan))
        self.assertEqual(frames, 99)
        self.assertEqual(images.shape[0], 99)

    def test_segment_sockets_win_over_the_combined_batch(self):
        stitcher = utils.DistributedVideoStitcher()
        images, frames = stitcher.stitch(
            share_boundary_frames=False, images=self._clip([9, 9, 9]),
            segment_1=self._clip([1]), segment_2=self._clip([2]))
        self.assertEqual(images[:, 0, 0, 0].tolist(), [1.0, 2.0])

    def test_participant_count_limits_the_stitch(self):
        stitcher = utils.DistributedVideoStitcher()
        _, frames = stitcher.stitch(
            num_segments=0, num_participants=1, share_boundary_frames=False,
            segment_1=self._clip([1]), segment_2=self._clip([2]),
        )
        self.assertEqual(frames, 1)


class AutoDividerOverrideTests(unittest.TestCase):
    def _apply(self, prompt, participant_id="master", enabled=None, delegate_master=False):
        if enabled is None:
            enabled = ["worker-a", "worker-b", "worker-c"]
        index = pt.PromptIndex(prompt)
        return pt.apply_participant_overrides(
            dict(prompt),
            participant_id=participant_id,
            enabled_worker_ids=enabled,
            job_id_map=pt.generate_job_id_map(index, "run"),
            master_url="http://master.example.com",
            delegate_master=delegate_master,
            prompt_index=index,
        )

    def _divider_prompt(self, divide_by):
        return {
            "1": {"class_type": "ImageBatchDivider", "inputs": {"divide_by": divide_by}},
            "2": {"class_type": "AudioBatchDivider", "inputs": {"divide_by": divide_by}},
        }

    def test_auto_divider_resolves_to_participant_count(self):
        result = self._apply(self._divider_prompt(0))
        self.assertEqual(result["1"]["inputs"]["divide_by"], 4)
        self.assertEqual(result["2"]["inputs"]["divide_by"], 4)

    def test_auto_divider_excludes_a_delegating_master(self):
        result = self._apply(self._divider_prompt(0), delegate_master=True)
        self.assertEqual(result["1"]["inputs"]["divide_by"], 3)

    def test_explicit_divider_value_is_preserved(self):
        result = self._apply(self._divider_prompt(2))
        self.assertEqual(result["1"]["inputs"]["divide_by"], 2)

    def test_linked_divider_value_is_left_alone(self):
        prompt = {"1": {"class_type": "ImageBatchDivider", "inputs": {"divide_by": ["9", 0]}}}
        result = self._apply(prompt)
        self.assertEqual(result["1"]["inputs"]["divide_by"], ["9", 0])

    def test_planner_receives_fleet_context(self):
        prompt = {"1": {"class_type": "DistributedVideoPlanner", "inputs": {}}}
        result = self._apply(prompt, participant_id="worker-b")
        inputs = result["1"]["inputs"]
        self.assertEqual(inputs["enabled_worker_ids"], json.dumps(["worker-a", "worker-b", "worker-c"]))
        self.assertTrue(inputs["is_worker"])
        self.assertEqual(inputs["worker_id"], "worker_1")
        self.assertFalse(inputs["delegate_only"])

    def test_planner_master_has_no_worker_identity(self):
        prompt = {"1": {"class_type": "DistributedVideoPlanner", "inputs": {}}}
        result = self._apply(prompt)
        inputs = result["1"]["inputs"]
        self.assertFalse(inputs["is_worker"])
        self.assertEqual(inputs["worker_id"], "")

    def test_stitcher_receives_participant_count(self):
        prompt = {"1": {"class_type": "DistributedVideoStitcher", "inputs": {}}}
        result = self._apply(prompt, delegate_master=True)
        self.assertEqual(result["1"]["inputs"]["num_participants"], 3)


if __name__ == "__main__":
    unittest.main()
