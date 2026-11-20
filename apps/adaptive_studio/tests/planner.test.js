import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it } from "vitest";

const appDir = new URL("../", import.meta.url);
const source = readFileSync(new URL("web/planner.js", appDir), "utf8");
const vectors = JSON.parse(readFileSync(new URL("parity_vectors.json", appDir), "utf8"));

const context = vm.createContext({});
vm.runInContext(source, context);
const planner = context.AdaptivePlanner;

function fingerprint(plan) {
    return {
        num_segments: plan.num_segments,
        frames: plan.segments.map((segment) => segment.frames),
        starts: plan.segments.map((segment) => segment.start_frame),
        ends: plan.segments.map((segment) => segment.end_frame),
        keyframe_frames: plan.keyframe_frames,
        rendered_frames: plan.rendered_frames,
        stitched_frames: plan.stitched_frames,
        capped: plan.capped,
        warnings: plan.warnings,
    };
}

describe("Browser planner parity with utils/video_plan.py", () => {
    it("exposes the planner and constants", () => {
        expect(typeof planner.planVideoSegments).toBe("function");
        expect(planner.MAX_VIDEO_SEGMENTS).toBe(32);
        expect(planner.MAX_DIVIDER_OUTPUTS).toBe(16);
    });

    it("matches every Python plan vector", () => {
        expect(vectors.plan_cases.length).toBeGreaterThan(150);
        for (const testCase of vectors.plan_cases) {
            const plan = planner.planVideoSegments(testCase.input);
            expect(fingerprint(plan), JSON.stringify(testCase.input)).toEqual(testCase.expected);
        }
    });

    it("matches every Python plan-response vector", () => {
        expect(vectors.response_cases.length).toBeGreaterThan(8);
        for (const testCase of vectors.response_cases) {
            const result = planner.buildPlanResponse(testCase.input.payload, testCase.input.fleet);
            expect(result.accounts, JSON.stringify(testCase.input.payload)).toEqual(testCase.expected.accounts);
            expect(result.summary, JSON.stringify(testCase.input.payload)).toEqual(testCase.expected.summary);
            expect(result.warnings, JSON.stringify(testCase.input.payload)).toEqual(testCase.expected.warnings);
            expect(planner.ffmpegScript(result), JSON.stringify(testCase.input.payload))
                .toBe(testCase.ffmpeg);
        }
    });
});

describe("Browser planner behaviour", () => {
    it("plans one segment per account", () => {
        const plan = planner.planVideoSegments({
            totalFrames: 144, numSegments: 0, alignToWan: false,
            shareBoundaryFrames: true, numParticipants: 6,
        });
        expect(plan.num_segments).toBe(6);
        expect(plan.stitched_frames).toBe(144);
    });

    it("snaps Wan-aligned segments to 4k+1", () => {
        const plan = planner.planVideoSegments({
            totalFrames: 144, alignToWan: true, numParticipants: 5,
        });
        for (const segment of plan.segments) expect(segment.frames % 4).toBe(1);
    });

    it("keeps the stitched length near the target for any fleet size", () => {
        for (const accounts of [1, 2, 3, 4, 6, 8, 12, 16, 20]) {
            const plan = planner.planVideoSegments({ totalFrames: 144, alignToWan: true, numParticipants: accounts });
            expect(Math.abs(plan.stitched_frames - 144), `${accounts} accounts`).toBeLessThanOrEqual(1);
        }
    });

    it("caps the segment count and explains why", () => {
        const plan = planner.planVideoSegments({
            totalFrames: 48, minFramesPerSegment: 5, alignToWan: false, numParticipants: 40,
        });
        expect(plan.num_segments).toBe(9);
        expect(plan.capped).toBe(true);
        expect(plan.warnings.join(" ")).toMatch(/only 9 fit/);
    });

    it("uses the 32-segment ceiling on long videos", () => {
        const plan = planner.planVideoSegments({ totalFrames: 1200, alignToWan: true, numParticipants: 80 });
        expect(plan.num_segments).toBe(32);
        expect(plan.warnings.join(" ")).toMatch(/ceiling/);
    });

    it("rejects nonsense input", () => {
        expect(() => planner.planVideoSegments({ totalFrames: 0 })).toThrow(/positive/);
        expect(() => planner.planVideoSegments({ totalFrames: 10, minFramesPerSegment: 0 }))
            .toThrow(/at least 1/);
    });

    it("labels accounts from the imported fleet and beyond it", () => {
        const fleet = {
            workers: [
                { id: "a", name: "Colab A" },
                { id: "b", name: "Colab B" },
            ],
        };
        const result = planner.buildPlanResponse(
            { duration_seconds: 6, fps: 24, accounts: 4 }, fleet);
        expect(result.accounts.map((account) => account.label)).toEqual(
            ["master", "Colab A", "Colab B", "worker_3", "worker_4"]);
        expect(result.accounts[0].slot).toBe("master");
        expect(result.accounts[4].slot).toBe("worker_4");
    });

    it("auto accounts follow the imported fleet", () => {
        const fleet = { workers: [{ id: "a" }, { id: "b" }, { id: "c" }] };
        const result = planner.buildPlanResponse({ duration_seconds: 6, fps: 24, accounts: 0 }, fleet);
        expect(result.summary.auto_accounts).toBe(true);
        expect(result.summary.segments).toBe(4);
    });

    it("works without a fleet at all", () => {
        const result = planner.buildPlanResponse({ duration_seconds: 6, fps: 24 }, null);
        expect(result.summary.segments).toBe(1);
        expect(result.accounts[0].slot).toBe("master");
    });
});
