import torch
import json

from ..utils.logging import debug_log, log
from ..utils.video_plan import (
    MAX_DIVIDER_OUTPUTS,
    MAX_VIDEO_SEGMENTS,
    _chunk_bounds,
    _count_participants,
    _parse_worker_index,
    plan_video_segments,
)


class DistributedSeed:
    """
    Distributes seed values across multiple GPUs.
    On master: passes through the original seed.
    On workers: adds offset based on worker ID.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 1125899906842, 
                    "min": 0,
                    "max": 1125899906842624,
                    "forceInput": False  # Widget by default, can be converted to input
                }),
            },
            "hidden": {
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "distribute"
    CATEGORY = "utils"
    
    def distribute(self, seed, is_worker=False, worker_id=""):
        if not is_worker:
            # Master node: pass through original values
            debug_log(f"Distributor - Master: seed={seed}")
            return (seed,)
        else:
            # Worker node: apply offset based on worker index
            # Find worker index from enabled_worker_ids
            try:
                # Worker IDs are passed as "worker_0", "worker_1", etc.
                if worker_id.startswith("worker_"):
                    worker_index = int(worker_id.split("_")[1])
                else:
                    # Fallback: try to parse as direct index
                    worker_index = int(worker_id)
                
                offset = worker_index + 1
                new_seed = seed + offset
                debug_log(f"Distributor - Worker {worker_index}: seed={seed} → {new_seed}")
                return (new_seed,)
            except (ValueError, IndexError) as e:
                debug_log(f"Distributor - Error parsing worker_id '{worker_id}': {e}")
                # Fallback: return original seed
                return (seed,)


# Define ByPassTypeTuple for flexible return types
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


class DistributedValue:
    """
    Outputs a different value per worker.
    On master: returns default_value.
    On workers: looks up the worker-specific value from a JSON map,
    falling back to default_value if not set.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "default_value": ("STRING", {"default": ""}),
                "worker_values": ("STRING", {"default": "{}"}),
            },
            "hidden": {
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "distribute"
    CATEGORY = "utils"

    @staticmethod
    def _coerce(value, value_type):
        """Convert a string value to the requested type."""
        if value_type == "INT":
            # Parse integers exactly first; float() loses precision above 2**53,
            # which corrupts large seeds such as 18446744073709551615.
            try:
                return int(str(value).strip())
            except ValueError:
                return int(float(value))
        if value_type == "FLOAT":
            return float(value)
        return value  # STRING and COMBO stay as strings

    @staticmethod
    def _coerce_safe(value, value_type):
        """Best-effort coercion with graceful fallback to original value."""
        try:
            return DistributedValue._coerce(value, value_type)
        except (TypeError, ValueError):
            return value

    def distribute(self, default_value, worker_values="{}", is_worker=False, worker_id=""):
        values = {}
        value_type = "STRING"

        try:
            values = json.loads(worker_values) if isinstance(worker_values, str) else worker_values
            if not isinstance(values, dict):
                values = {}
        except json.JSONDecodeError as e:
            debug_log(f"DistributedValue - Error parsing worker_values: {e}")
            values = {}

        value_type = values.get("_type", "STRING")
        coerced_default = self._coerce_safe(default_value, value_type)

        if not is_worker:
            debug_log(f"DistributedValue - Master: returning default '{coerced_default}'")
            return (coerced_default,)

        try:
            if worker_id.startswith("worker_"):
                idx = int(worker_id.split("_")[1])
            else:
                idx = int(worker_id)
            key = str(idx + 1)  # worker_0 → key "1" (1-indexed)
            raw = values.get(key, "")
            if raw:
                coerced = self._coerce(raw, value_type)
                debug_log(f"DistributedValue - Worker {idx}: returning '{coerced}'")
                return (coerced,)
        except (ValueError, IndexError) as e:
            debug_log(f"DistributedValue - Error: {e}")
        debug_log(f"DistributedValue - Worker fallback: returning default '{coerced_default}'")
        return (coerced_default,)

class DistributedModelName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "log_input"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    def _stringify(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, indent=4)
        except Exception:
            return str(value)

    def _update_workflow(self, extra_pnginfo, unique_id, values):
        if not extra_pnginfo:
            return
        info = extra_pnginfo[0] if isinstance(extra_pnginfo, list) else extra_pnginfo
        if not isinstance(info, dict) or "workflow" not in info:
            return
        node_id = None
        if isinstance(unique_id, list) and unique_id:
            node_id = str(unique_id[0])
        elif unique_id is not None:
            node_id = str(unique_id)
        if not node_id:
            return
        workflow = info["workflow"]
        node = next((x for x in workflow["nodes"] if str(x.get("id")) == node_id), None)
        if node:
            node["widgets_values"] = [values]

    def log_input(self, text, unique_id=None, extra_pnginfo=None):
        values = []
        if isinstance(text, list):
            for val in text:
                values.append(self._stringify(val))
        else:
            values.append(self._stringify(text))

        # Keep widget display in workflow metadata if available.
        self._update_workflow(extra_pnginfo, unique_id, values)

        if isinstance(values, list) and len(values) == 1:
            return {"ui": {"text": values}, "result": (values[0],)}
        return {"ui": {"text": values}, "result": (values,)}

class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return any_type
        return item

class ImageBatchDivider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "divide_by": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": MAX_DIVIDER_OUTPUTS,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of parts to divide the batch into. 0 = auto (uses the number of enabled participants; resolved at dispatch time)."
                }),
            }
        }

    RETURN_TYPES = ByPassTypeTuple(("IMAGE", ))  # Flexible for variable outputs
    RETURN_NAMES = ByPassTypeTuple(tuple([f"batch_{i+1}" for i in range(MAX_DIVIDER_OUTPUTS)]))
    FUNCTION = "divide_batch"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def divide_batch(self, images, divide_by):
        # divide_by == 0 (auto) is rewritten to the participant count by the
        # orchestrator before dispatch. If it still arrives as 0 (local run
        # without orchestration), fall back to a single unsplit batch.
        total_splits = max(1, min(int(divide_by), MAX_DIVIDER_OUTPUTS))
        total_frames = images.shape[0]
        empty_tensor = images[:0]
        bounds = _chunk_bounds(total_frames, total_splits)
        outputs = [images[start:end] if end > start else empty_tensor for start, end in bounds]

        while len(outputs) < MAX_DIVIDER_OUTPUTS:
            outputs.append(empty_tensor)

        return tuple(outputs[:MAX_DIVIDER_OUTPUTS])


class AudioBatchDivider:
    """Divides an audio waveform into sequential segments along the time dimension."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "divide_by": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": MAX_DIVIDER_OUTPUTS,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of sequential time segments to create. 0 = auto (uses the number of enabled participants; resolved at dispatch time)."
                }),
            }
        }

    RETURN_TYPES = ByPassTypeTuple(("AUDIO",))  # Flexible for variable outputs
    RETURN_NAMES = ByPassTypeTuple(tuple([f"audio_{i+1}" for i in range(MAX_DIVIDER_OUTPUTS)]))
    FUNCTION = "divide_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def divide_audio(self, audio, divide_by):
        import torch

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 44100)

        if waveform is None or waveform.numel() == 0:
            # Return empty audio for all outputs
            empty_audio = {"waveform": torch.zeros(1, 2, 1), "sample_rate": sample_rate}
            return tuple([empty_audio] * MAX_DIVIDER_OUTPUTS)

        total_splits = max(1, min(int(divide_by), MAX_DIVIDER_OUTPUTS))
        total_samples = int(waveform.shape[-1])
        bounds = _chunk_bounds(total_samples, total_splits)

        outputs = []
        empty_waveform = waveform[..., :0]
        for start, end in bounds:
            split_waveform = waveform[..., start:end] if end > start else empty_waveform
            outputs.append({
                "waveform": split_waveform,
                "sample_rate": sample_rate
            })

        # Pad with empty audio up to max to match RETURN_TYPES length
        empty_audio = {
            "waveform": empty_waveform,
            "sample_rate": sample_rate
        }

        while len(outputs) < MAX_DIVIDER_OUTPUTS:
            outputs.append(empty_audio)

        return tuple(outputs)


class DistributedEmptyImage:
    """Produces an empty IMAGE batch used when the master delegates all work."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "width": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "channels": ("INT", {"default": 3, "min": 1, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create"
    CATEGORY = "image"

    def create(self, height, width, channels):
        import torch

        shape = (0, height, width, channels)
        tensor = torch.zeros(shape, dtype=torch.float32)
        return (tensor,)


class DistributedVideoPlanner:
    """Plans adaptive video segments for however many participants are active.

    The plan is derived from the enabled-worker list injected by the
    orchestrator, so the same workflow scales from 1 account to the whole
    fleet without editing node values. ``num_segments = 0`` means auto.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_seconds": ("FLOAT", {
                    "default": 6.0, "min": 0.04, "max": 3600.0, "step": 0.01,
                    "tooltip": "Target length of the finished video.",
                }),
                "fps": ("INT", {
                    "default": 24, "min": 1, "max": 240, "step": 1,
                    "tooltip": "Frame rate of the finished video.",
                }),
                "num_segments": ("INT", {
                    "default": 0, "min": 0, "max": MAX_VIDEO_SEGMENTS, "step": 1,
                    "tooltip": "0 = auto (one segment per participating account). "
                               "Set a positive number to override the fleet size.",
                }),
                "min_frames_per_segment": ("INT", {
                    "default": 5, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "Segments are never planned shorter than this.",
                }),
                "align_to_wan": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Snap every segment to a Wan-valid 4k+1 frame count.",
                }),
                "share_boundary_frames": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "First/last-frame style continuity: each segment ends on the "
                               "frame the next segment starts from, and stitching drops the "
                               "duplicate.",
                }),
            },
            "hidden": {
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "delegate_only": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT", "INT", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("plan", "num_segments", "segment_index", "start_frame",
                    "end_frame", "frames", "keyframe_times", "stitched_frames",
                    "first_keyframe", "last_keyframe")
    FUNCTION = "plan"
    CATEGORY = "video"

    def plan(self, duration_seconds, fps, num_segments=0, min_frames_per_segment=5,
             align_to_wan=True, share_boundary_frames=True, is_worker=False,
             worker_id="", enabled_worker_ids="[]", delegate_only=False):
        total_frames = max(1, int(round(float(duration_seconds) * int(fps))))
        participants = _count_participants(enabled_worker_ids, delegate_only)
        plan = plan_video_segments(
            total_frames=total_frames,
            num_segments=num_segments,
            min_frames_per_segment=min_frames_per_segment,
            align_to_wan=align_to_wan,
            share_boundary_frames=share_boundary_frames,
            num_participants=participants,
        )
        plan["participants"] = participants

        # Which segment does this participant render? The master takes the
        # first one; workers use their positional index from the enabled list
        # and shift past the master's segment when it participates. That keeps
        # worker_N's collector preview slot aligned with segment N + 1.
        index = 0
        if is_worker:
            parsed = _parse_worker_index(worker_id)
            if parsed is not None:
                index = parsed + (0 if delegate_only else 1)
        if index >= plan["num_segments"]:
            plan["warnings"].append(
                f"Participant index {index} exceeds the {plan['num_segments']}-segment plan; "
                "reusing the last segment."
            )
            index = plan["num_segments"] - 1
        index = max(0, index)

        segment = plan["segments"][index]
        keyframe_times = ",".join(f"{frame / float(fps):.3f}" for frame in plan["keyframe_frames"])

        shortest_seconds = min(seg["frames"] for seg in plan["segments"]) / float(fps)
        if shortest_seconds < 0.5:
            plan["warnings"].append(
                f"Each account renders only {shortest_seconds:.2f}s; a longer video or fewer "
                "segments keeps motion quality up."
            )

        for warning in plan["warnings"]:
            log(f"[Adaptive Video] {warning}")
        debug_log(
            f"[Adaptive Video] plan participants={participants} segments={plan['num_segments']} "
            f"segment_index={index} frames={segment['frames']} stitched={plan['stitched_frames']}"
        )

        # Indexes into a keyframe strip that holds one image per boundary:
        # this segment renders from boundary k to boundary k+1 (FLF2V).
        return (
            json.dumps(plan),
            plan["num_segments"],
            index,
            segment["start_frame"],
            segment["end_frame"],
            segment["frames"],
            keyframe_times,
            plan["stitched_frames"],
            index,
            index + 1,
        )


class DistributedVideoStitcher:
    """Concatenates rendered segments into one timeline.

    Missing (unconnected or empty) segment sockets are skipped, so the same
    node stitches a 2-account run and a 16-account run. When segments share
    boundary frames, the duplicated first frame of each later segment is
    dropped so the join is seamless.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"segment_{index + 1}": ("IMAGE",)
            for index in range(MAX_DIVIDER_OUTPUTS)
        }
        return {
            "required": {
                "num_segments": ("INT", {
                    "default": 0, "min": 0, "max": MAX_VIDEO_SEGMENTS, "step": 1,
                    "tooltip": "0 = auto (stitch every connected, non-empty segment).",
                }),
                "share_boundary_frames": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Drop the first frame of each later segment, because it repeats "
                               "the previous segment's last frame.",
                }),
                "num_participants": ("INT", {
                    "default": 0, "min": 0, "max": MAX_VIDEO_SEGMENTS, "step": 1,
                    "tooltip": "Internal: filled in by the orchestrator with the number of "
                               "accounts taking part (0 = derive from connected segments).",
                }),
            },
            "optional": {
                **optional,
                "images": ("IMAGE",),
                "plan": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frames")
    FUNCTION = "stitch"
    CATEGORY = "video"

    @staticmethod
    def _parse_plan(plan):
        """Return the plan dict from a wire value, or None when unusable."""
        if isinstance(plan, dict):
            return plan
        if not isinstance(plan, str) or not plan.strip():
            return None
        try:
            parsed = json.loads(plan)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _stitch_combined(self, images, plan, share_boundary_frames):
        """Slice a collector batch into plan segments and drop shared frames."""
        segments = plan.get("segments") if isinstance(plan, dict) else None
        if not isinstance(segments, list) or not segments:
            return None
        try:
            counts = [int(segment["frames"]) for segment in segments]
        except (KeyError, TypeError, ValueError):
            return None
        if sum(counts) != int(images.shape[0]):
            # An account dropped out (or rendered something else), so segment
            # offsets no longer line up. Keep every frame rather than cutting
            # the video in the wrong place.
            log(
                f"[Adaptive Video] Stitcher expected {sum(counts)} frames from the plan but "
                f"received {int(images.shape[0])}; joining without boundary trimming."
            )
            return images

        parts = []
        offset = 0
        for position, count in enumerate(counts):
            clip = images[offset:offset + count]
            offset += count
            if position > 0 and share_boundary_frames and clip.shape[0] > 1:
                clip = clip[1:]
            parts.append(clip)
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def stitch(self, num_segments=0, share_boundary_frames=True, num_participants=0,
               images=None, plan=None, **segments):
        import torch

        clips = []
        for index in range(MAX_DIVIDER_OUTPUTS):
            clip = segments.get(f"segment_{index + 1}")
            if clip is None:
                continue
            if not hasattr(clip, "shape") or clip.shape[0] == 0:
                continue
            clips.append(clip)

        # Preferred wiring: the collector's combined batch plus the plan, which
        # the collector already orders as master then workers. Falls back to the
        # per-account sockets above when they are connected.
        if not clips and images is not None and getattr(images, "shape", (0,))[0] > 0:
            combined = self._stitch_combined(images, self._parse_plan(plan), share_boundary_frames)
            if combined is not None:
                debug_log(f"[Adaptive Video] stitched plan segments into {int(combined.shape[0])} frames")
                return (combined, int(combined.shape[0]))
            clips = [images]

        requested = int(num_segments or 0)
        if requested <= 0 and int(num_participants or 0) > 0:
            requested = int(num_participants)
        if requested > 0:
            clips = clips[:requested]

        if not clips:
            return (torch.zeros((0, 64, 64, 3), dtype=torch.float32), 0)

        reference = clips[0].shape[1:]
        stitched = []
        for position, clip in enumerate(clips):
            if clip.shape[1:] != reference:
                raise ValueError(
                    f"Segment {position + 1} has resolution {tuple(clip.shape[1:])}, "
                    f"expected {tuple(reference)}. Every account must render the same size."
                )
            if position > 0 and share_boundary_frames and clip.shape[0] > 1:
                clip = clip[1:]
            stitched.append(clip)

        images = torch.cat(stitched, dim=0) if len(stitched) > 1 else stitched[0]
        expected = sum(int(clip.shape[0]) for clip in stitched)
        assert images.shape[0] == expected
        debug_log(f"[Adaptive Video] stitched {len(stitched)} segments into {images.shape[0]} frames")
        return (images, int(images.shape[0]))
