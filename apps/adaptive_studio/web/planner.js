/*
 * Browser port of utils/video_plan.py + the Adaptive Studio plan response.
 *
 * The Python planner is the reference implementation. `apps/adaptive_studio/
 * parity_vectors.json` is generated from it and asserted by both
 * `tests/test_adaptive_studio.py` (Python side) and
 * `apps/adaptive_studio/tests/planner.test.js` (browser side), so the two
 * cannot drift apart silently.
 *
 * Classic script (no modules) so index.html also works from file://.
 */
(function (global) {
    "use strict";

    const MAX_DIVIDER_OUTPUTS = 16;
    const MAX_VIDEO_SEGMENTS = 32;
    const FPS_CHOICES = [8, 12, 16, 24, 30];

    /** Round .5 away from zero, matching Python's _round_half_up. */
    function roundHalfUp(value) {
        return value >= 0 ? Math.floor(value + 0.5) : -Math.floor(-value + 0.5);
    }

    function chunkBounds(totalItems, nSplits) {
        const splitCount = Math.max(1, Math.trunc(nSplits));
        const total = Math.max(0, Math.trunc(totalItems));
        const base = Math.floor(total / splitCount);
        const remainder = total % splitCount;
        const bounds = [];
        let start = 0;
        for (let index = 0; index < splitCount; index += 1) {
            const size = base + (index < remainder ? 1 : 0);
            const end = start + size;
            bounds.push([start, end]);
            start = end;
        }
        return bounds;
    }

    /** Nearest Wan-valid 4k+1 frame count at or above the minimum. */
    function snapToWanFrames(frames, minimum) {
        const effectiveMin = Math.max(Math.trunc(minimum), 5);
        const kMin = Math.max(1, Math.ceil((effectiveMin - 1) / 4));
        const floorValid = 4 * kMin + 1;
        if (frames <= floorValid) return floorValid;
        const k = roundHalfUp((frames - 1) / 4);
        return 4 * Math.max(k, kMin) + 1;
    }

    function rebalanceWanCounts(counts, total, n, effectiveMin, shareBoundaryFrames) {
        const step = 4;
        const stitched = (frames) => (
            shareBoundaryFrames ? frames.reduce((a, b) => a + b, 0) - (n - 1)
                : frames.reduce((a, b) => a + b, 0)
        );
        let adjustment = roundHalfUp((total - stitched(counts)) / step);
        let guard = 0;
        let index = 0;
        while (adjustment > 0 && guard < 4 * n) {
            counts[index % n] += step;
            adjustment -= 1;
            index += 1;
            guard += 1;
        }
        index = 0;
        while (adjustment < 0 && guard < 8 * n) {
            const target = n - 1 - (index % n);
            if (counts[target] - step >= effectiveMin) {
                counts[target] -= step;
                adjustment += 1;
            }
            index += 1;
            guard += 1;
        }
        return counts;
    }

    function planVideoSegments(options) {
        const {
            totalFrames,
            numSegments = 0,
            minFramesPerSegment = 5,
            alignToWan = false,
            shareBoundaryFrames = true,
            numParticipants = 1,
        } = options || {};

        const total = Math.trunc(totalFrames);
        if (!(total > 0)) throw new Error("total_frames must be positive");
        const minimum = Math.trunc(minFramesPerSegment);
        if (minimum < 1) throw new Error("min_frames_per_segment must be at least 1");
        const requested = Math.trunc(numSegments);
        if (requested < 0) throw new Error("num_segments must be 0 (auto) or positive");

        const warnings = [];
        let n;
        let auto;
        if (requested === 0) {
            n = Math.max(1, Math.trunc(numParticipants ?? 1));
            auto = true;
        } else {
            n = requested;
            auto = false;
        }

        const maxByFrames = Math.max(1, Math.floor(total / minimum));
        let capped = false;
        if (n > maxByFrames) {
            warnings.push(
                `Requested ${n} segments but only ${maxByFrames} fit `
                + `(${total} frames / ${minimum} min frames); capped to ${maxByFrames}.`);
            n = maxByFrames;
            capped = true;
        }
        if (n > MAX_VIDEO_SEGMENTS) {
            warnings.push(
                `Requested ${n} segments exceeds the ${MAX_VIDEO_SEGMENTS}-segment ceiling; `
                + `capped to ${MAX_VIDEO_SEGMENTS}.`);
            n = MAX_VIDEO_SEGMENTS;
            capped = true;
        }

        let counts;
        if (alignToWan) {
            const effectiveMin = Math.max(minimum, 5);
            const ideal = total / n;
            counts = Array.from({ length: n }, () => snapToWanFrames(ideal, effectiveMin));
            counts = rebalanceWanCounts(counts, total, n, effectiveMin, shareBoundaryFrames);
        } else if (shareBoundaryFrames) {
            const rendered = total + (n - 1);
            counts = chunkBounds(rendered, n).map(([start, end]) => end - start);
        } else {
            counts = chunkBounds(total, n).map(([start, end]) => end - start);
        }

        const segments = [];
        let cursor = 0;
        counts.forEach((count, index) => {
            const startFrame = cursor;
            const endFrame = cursor + count - 1;
            cursor = shareBoundaryFrames ? endFrame : endFrame + 1;
            segments.push({ index, start_frame: startFrame, end_frame: endFrame, frames: count });
        });

        const renderedFrames = counts.reduce((a, b) => a + b, 0);
        const stitchedFrames = shareBoundaryFrames ? renderedFrames - (n - 1) : renderedFrames;
        const keyframeFrames = segments.map((segment) => segment.start_frame);
        keyframeFrames.push(segments[segments.length - 1].end_frame + 1);

        return {
            total_frames: total,
            num_segments: n,
            auto,
            capped,
            min_frames_per_segment: minimum,
            align_to_wan: Boolean(alignToWan),
            share_boundary_frames: Boolean(shareBoundaryFrames),
            rendered_frames: renderedFrames,
            stitched_frames: stitchedFrames,
            keyframe_frames: keyframeFrames,
            segments,
            warnings,
        };
    }

    // --- plan response (mirrors apps/adaptive_studio/server.py) ------------

    function asNumber(value, fallback, minimum, maximum) {
        if (value === null || value === undefined || value === "") return fallback;
        const number = Number(value);
        if (!Number.isFinite(number)) return fallback;
        return Math.max(minimum, Math.min(maximum, number));
    }

    function asBool(value, fallback = false) {
        if (value === null || value === undefined) return fallback;
        if (typeof value === "boolean") return value;
        return ["1", "true", "yes", "on"].includes(String(value).trim().toLowerCase());
    }

    function buildPlanResponse(payload, fleet) {
        const body = payload || {};
        const config = fleet || { workers: [] };
        const workers = Array.isArray(config.workers) ? config.workers : [];

        const duration = asNumber(body.duration_seconds, 6, 0.04, 3600);
        const fps = Math.trunc(asNumber(body.fps, 24, 1, 240));
        const minFrames = Math.trunc(asNumber(body.min_frames_per_segment, 5, 1, 1000));
        const masterParticipates = asBool(body.master_participates, true);
        const alignToWan = asBool(body.align_to_wan, true);
        const shareBoundary = asBool(body.share_boundary_frames, true);

        let enabled = Math.trunc(asNumber(body.accounts, 0, 0, 10000));
        const autoAccounts = enabled <= 0;
        if (autoAccounts) enabled = workers.length;

        const participantCount = enabled + (masterParticipates ? 1 : 0);
        const totalFrames = Math.max(1, roundHalfUp(duration * fps));
        const plan = planVideoSegments({
            totalFrames,
            numSegments: 0,
            minFramesPerSegment: minFrames,
            alignToWan,
            shareBoundaryFrames: shareBoundary,
            numParticipants: participantCount,
        });

        const accounts = [];
        for (let position = 0; position < plan.num_segments; position += 1) {
            const isMaster = masterParticipates && position === 0;
            const workerIndex = position - (masterParticipates ? 1 : 0);
            let label;
            let kind;
            if (isMaster) {
                label = "master";
                kind = "master";
            } else if (workerIndex >= 0 && workerIndex < workers.length) {
                const worker = workers[workerIndex];
                label = (worker.name || worker.id || `worker_${workerIndex + 1}`);
                kind = "worker";
            } else {
                label = `worker_${workerIndex + 1}`;
                kind = "worker";
            }
            const segment = plan.segments[position];
            accounts.push({
                slot: isMaster ? "master" : `worker_${workerIndex + 1}`,
                label,
                kind,
                index: position,
                start_frame: segment.start_frame,
                end_frame: segment.end_frame,
                frames: segment.frames,
                start_seconds: Math.round((segment.start_frame / fps) * 1000) / 1000,
                end_seconds: Math.round(((segment.end_frame + 1) / fps) * 1000) / 1000,
                seconds: Math.round((segment.frames / fps) * 1000) / 1000,
            });
        }

        const warnings = [...plan.warnings];
        const idle = enabled - plan.num_segments;
        if (idle > 0) {
            warnings.push(
                `${idle} of ${enabled} accounts have no segment in this plan; a longer video `
                + "or fewer accounts keeps every GPU busy.");
        }
        if (plan.num_segments < participantCount) {
            warnings.push(
                "The plan has fewer segments than accounts, so some accounts will idle.");
        }

        return {
            fleet: config,
            accounts,
            plan,
            warnings,
            summary: {
                duration_seconds: duration,
                fps,
                total_frames: totalFrames,
                enabled_workers: enabled,
                auto_accounts: autoAccounts,
                master_participates: masterParticipates,
                segments: plan.num_segments,
                keyframes: plan.keyframe_frames.length,
                keyframe_times: plan.keyframe_frames.map((frame) => Math.round((frame / fps) * 1000) / 1000),
                stitched_frames: plan.stitched_frames,
                stitched_seconds: Math.round((plan.stitched_frames / fps) * 1000) / 1000,
                frames_per_account: [...new Set(plan.segments.map((segment) => segment.frames))].sort((a, b) => a - b),
                align_to_wan: alignToWan,
                share_boundary_frames: shareBoundary,
                max_segments: MAX_VIDEO_SEGMENTS,
            },
        };
    }

    function ffmpegScript(result) {
        const { summary, accounts } = result;
        const { fps, share_boundary_frames: share } = summary;
        const lines = ["# Save every account's clip as segment_<slot>.mp4, then run:"];
        let files;
        if (share && accounts.length > 1) {
            lines.push("", "# Drop the duplicated boundary frame from every segment after the first:");
            for (const account of accounts.slice(1)) {
                lines.push(
                    `ffmpeg -y -i segment_${account.slot}.mp4 `
                    + `-vf "select='gte(n\\,1)',setpts=N/FRAME_RATE/TB" `
                    + `-r ${fps} trim_${account.slot}.mp4`);
            }
            files = [`segment_${accounts[0].slot}.mp4`,
                ...accounts.slice(1).map((account) => `trim_${account.slot}.mp4`)];
        } else {
            files = accounts.map((account) => `segment_${account.slot}.mp4`);
        }
        lines.push("", "# Concatenate in plan order:");
        lines.push(`printf "file '%s'\\n" ${files.join(" ")} > list.txt`);
        lines.push("ffmpeg -y -f concat -safe 0 -i list.txt -c copy -movflags +faststart final.mp4");
        return lines.join("\n");
    }

    const api = {
        MAX_DIVIDER_OUTPUTS,
        MAX_VIDEO_SEGMENTS,
        FPS_CHOICES,
        roundHalfUp,
        chunkBounds,
        snapToWanFrames,
        rebalanceWanCounts,
        planVideoSegments,
        buildPlanResponse,
        ffmpegScript,
    };

    global.AdaptivePlanner = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof globalThis !== "undefined" ? globalThis : this));
