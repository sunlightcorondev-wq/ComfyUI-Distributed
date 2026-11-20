"""Adaptive video segment planning; pure Python, no ComfyUI or torch imports.

Kept dependency-free so both the node pack (``nodes/utilities.py``) and the
standalone Adaptive Studio app (``apps/adaptive_studio``) share one planner.
"""
import json


def _chunk_bounds(total_items: int, n_splits: int) -> list[tuple[int, int]]:
    """Return contiguous [start, end) bounds for n_splits chunks."""
    split_count = max(1, int(n_splits))
    total = max(0, int(total_items))
    base, remainder = divmod(total, split_count)

    bounds: list[tuple[int, int]] = []
    start = 0
    for idx in range(split_count):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


# Batch dividers expose this many outputs. It matches the Collector's
# per-GPU preview slots so a workflow can wire one output per account.
MAX_DIVIDER_OUTPUTS = 16

# Hard ceiling for adaptive video segmentation. The planner caps auto
# segment counts here so a huge fleet cannot create absurdly short clips.
MAX_VIDEO_SEGMENTS = 32


def _snap_to_wan_frames(frames: float, minimum: int) -> int:
    """Snap a frame count to the nearest Wan-valid 4k+1 value >= minimum."""
    effective_min = max(int(minimum), 5)
    # Smallest 4k+1 >= effective_min.
    k_min = max(1, -(-(effective_min - 1) // 4))
    floor_valid = 4 * k_min + 1
    if frames <= floor_valid:
        return floor_valid
    k = round((frames - 1) / 4)
    snapped = 4 * max(k, k_min) + 1
    return snapped


def _rebalance_wan_counts(counts: list[int], total: int, n: int, effective_min: int,
                          share_boundary_frames: bool) -> list[int]:
    """Nudge Wan-aligned segment lengths so the stitched length matches the target."""
    step = 4

    def stitched(frames: list[int]) -> int:
        return sum(frames) - (n - 1) if share_boundary_frames else sum(frames)

    adjustment = int(round((total - stitched(counts)) / step))
    guard = 0
    index = 0
    while adjustment > 0 and guard < 4 * n:
        counts[index % n] += step
        adjustment -= 1
        index += 1
        guard += 1
    index = 0
    while adjustment < 0 and guard < 8 * n:
        target = n - 1 - (index % n)
        if counts[target] - step >= effective_min:
            counts[target] -= step
            adjustment += 1
        index += 1
        guard += 1
    return counts


def plan_video_segments(
    total_frames: int,
    num_segments: int = 0,
    min_frames_per_segment: int = 5,
    align_to_wan: bool = False,
    share_boundary_frames: bool = True,
    num_participants: int | None = None,
) -> dict:
    """Plan adaptive video segments for parallel rendering.

    ``num_segments == 0`` means auto: use ``num_participants`` (enabled
    workers plus the master when it participates). The plan is
    deterministic so every participant computes identical boundaries.

    When ``share_boundary_frames`` is true (FLF2V style), each segment's
    end frame is the next segment's start frame, and the stitched video
    drops the duplicated first frame of segments 2..N. When false,
    segments partition the timeline with no overlap.

    When ``align_to_wan`` is true, every segment's frame count is snapped
    to a Wan-valid 4k+1 value. The stitched length may then differ from
    ``total_frames`` by a few frames (exact alignment is only possible
    when ``total_frames % 4 == 1``); the plan reports the true stitched
    length in ``stitched_frames``.
    """
    total = int(total_frames)
    if total <= 0:
        raise ValueError("total_frames must be positive")
    minimum = int(min_frames_per_segment)
    if minimum < 1:
        raise ValueError("min_frames_per_segment must be at least 1")
    requested = int(num_segments)
    if requested < 0:
        raise ValueError("num_segments must be 0 (auto) or positive")

    warnings: list[str] = []
    if requested == 0:
        if num_participants is None:
            num_participants = 1
        n = max(1, int(num_participants))
        auto = True
    else:
        n = requested
        auto = False

    # Conservative cap: every segment needs at least `minimum` frames.
    max_by_frames = max(1, total // minimum)
    capped = False
    if n > max_by_frames:
        warnings.append(
            f"Requested {n} segments but only {max_by_frames} fit "
            f"({total} frames / {minimum} min frames); capped to {max_by_frames}."
        )
        n = max_by_frames
        capped = True
    if n > MAX_VIDEO_SEGMENTS:
        warnings.append(
            f"Requested {n} segments exceeds the {MAX_VIDEO_SEGMENTS}-segment ceiling; "
            f"capped to {MAX_VIDEO_SEGMENTS}."
        )
        n = MAX_VIDEO_SEGMENTS
        capped = True

    if align_to_wan:
        effective_min = max(minimum, 5)
        ideal = total / n
        counts = [_snap_to_wan_frames(ideal, effective_min) for _ in range(n)]
        # Snapping each segment independently accumulates rounding error, which
        # would shorten a many-account run (8 accounts: 18 -> 17 frames each
        # loses 15 stitched frames). Correct it in whole 4-frame steps.
        counts = _rebalance_wan_counts(counts, total, n, effective_min, share_boundary_frames)
    elif share_boundary_frames:
        # Rendered frames = stitched frames + one shared frame per join.
        rendered = total + (n - 1)
        counts = [end - start for start, end in _chunk_bounds(rendered, n)]
    else:
        counts = [end - start for start, end in _chunk_bounds(total, n)]

    segments: list[dict] = []
    cursor = 0
    for index, count in enumerate(counts):
        start_frame = cursor
        end_frame = cursor + count - 1
        if share_boundary_frames:
            cursor = end_frame  # next segment starts on this frame
        else:
            cursor = end_frame + 1
        segments.append({
            "index": index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": count,
        })

    rendered_frames = sum(counts)
    if share_boundary_frames:
        stitched_frames = rendered_frames - (n - 1)
    else:
        stitched_frames = rendered_frames

    # One keyframe per segment start (each is also the previous segment's end
    # frame) plus the exclusive end boundary so the last key is the video end.
    keyframe_frames = [seg["start_frame"] for seg in segments]
    keyframe_frames.append(segments[-1]["end_frame"] + 1)

    return {
        "total_frames": total,
        "num_segments": n,
        "auto": auto,
        "capped": capped,
        "min_frames_per_segment": minimum,
        "align_to_wan": bool(align_to_wan),
        "share_boundary_frames": bool(share_boundary_frames),
        "rendered_frames": rendered_frames,
        "stitched_frames": stitched_frames,
        "keyframe_frames": keyframe_frames,
        "segments": segments,
        "warnings": warnings,
    }


def _parse_worker_index(worker_id: str) -> int | None:
    """Parse ``worker_N`` (or a bare index) into an int, else None."""
    text = str(worker_id or "").strip()
    if not text:
        return None
    if text.startswith("worker_"):
        text = text[len("worker_"):]
    try:
        return int(text)
    except ValueError:
        return None


def _count_participants(enabled_worker_ids, delegate_only: bool) -> int:
    """Count participants from the injected enabled-worker list."""
    try:
        ids = json.loads(enabled_worker_ids) if isinstance(enabled_worker_ids, str) else enabled_worker_ids
    except (TypeError, ValueError):
        ids = []
    if not isinstance(ids, (list, tuple)):
        ids = []
    return len(ids) + (0 if delegate_only else 1)
