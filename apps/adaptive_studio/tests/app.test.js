// End-to-end check of the browser app: loads index.html into jsdom and boots
// app.js against both a static host (no backend) and the Python server API.
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { describe, expect, it } from "vitest";

const appDir = new URL("../", import.meta.url);
const html = readFileSync(new URL("web/index.html", appDir), "utf8");
const plannerSource = readFileSync(new URL("web/planner.js", appDir), "utf8");
const appSource = readFileSync(new URL("web/app.js", appDir), "utf8");

const WORKERS = [
    { id: "a", name: "Colab A", type: "remote", host: "https://a.trycloudflare.com", port: 443 },
    { id: "b", name: "Colab B", type: "remote", host: "https://b.trycloudflare.com", port: 443 },
    { id: "c", name: "Colab C", type: "remote", host: "https://c.trycloudflare.com", port: 443 },
];

function json(payload, ok = true) {
    return Promise.resolve({ ok, status: ok ? 200 : 500, json: async () => payload });
}

/** Boot the page. `fetchImpl(url, options)` decides how the backend behaves. */
async function boot(fetchImpl) {
    const dom = new JSDOM(html, {
        url: "https://studio.test/",
        runScripts: "outside-only",
        pretendToBeVisual: true,
    });
    dom.window.fetch = fetchImpl;
    dom.window.eval(plannerSource);
    dom.window.eval(appSource);
    await new Promise((resolve) => setTimeout(resolve, 40));
    return dom;
}

const staticFetch = () => Promise.reject(new TypeError("Failed to fetch"));

describe("Adaptive Studio in the browser", () => {
    it("boots without a backend and plans locally", async () => {
        const dom = await boot(staticFetch);
        const { document } = dom.window;

        expect(document.getElementById("mode-badge").textContent).toContain("browser-only");
        expect(document.getElementById("mode-badge").className).toContain("static");

        // No fleet imported: falls back to a manual 3-account plan (+ master).
        expect(document.getElementById("stat-accounts").textContent).toBe("4");
        expect(document.getElementById("stat-segments").textContent).toBe("4");
        expect(document.querySelectorAll("#accounts-body tr")).toHaveLength(4);
        expect(document.querySelectorAll("#timeline .seg")).toHaveLength(4);
        expect(document.querySelectorAll("#keyframes .kf")).toHaveLength(5);
        expect(document.getElementById("import-card").hidden).toBe(false);

        dom.window.close();
    });

    it("renders the master as the first segment", async () => {
        const dom = await boot(staticFetch);
        const { document } = dom.window;
        const firstRow = document.querySelector("#accounts-body tr");
        expect(firstRow.className).toBe("master-row");
        expect(firstRow.textContent).toContain("master");
        dom.window.close();
    });

    it("loads the demo fleet and uses it for auto accounts", async () => {
        const dom = await boot(staticFetch);
        const { document } = dom.window;

        document.getElementById("fleet-demo").click();
        await new Promise((resolve) => setTimeout(resolve, 40));

        expect(document.getElementById("fleet-text").textContent).toContain("5 workers");
        expect(document.getElementById("stat-accounts").textContent).toBe("6");
        expect(document.getElementById("accounts-body").textContent).toContain("Colab A (T4)");
        expect(dom.window.localStorage.getItem("adaptiveStudio.fleet.v1")).toContain("colab-a");
        dom.window.close();
    });

    it("re-plans when the duration changes", async () => {
        const dom = await boot(staticFetch);
        const { document, Event } = dom.window;

        const duration = document.getElementById("duration-num");
        const before = document.getElementById("stat-stitched").textContent;
        duration.value = "12";
        duration.dispatchEvent(new Event("input"));
        await new Promise((resolve) => setTimeout(resolve, 220)); // re-plan is debounced

        expect(document.getElementById("stat-stitched").textContent).not.toBe(before);
        expect(Number.parseFloat(document.getElementById("stat-stitched").textContent)).toBeGreaterThan(11);
        dom.window.close();
    });

    it("uses the server API when one is available", async () => {
        const planBody = {
            accounts: [{
                slot: "master", label: "master", kind: "master", index: 0,
                start_frame: 0, end_frame: 47, frames: 48,
                start_seconds: 0, end_seconds: 2, seconds: 2,
            }],
            summary: {
                duration_seconds: 6, fps: 24, total_frames: 144, enabled_workers: 0,
                auto_accounts: true, master_participates: true, segments: 1, keyframes: 2,
                keyframe_times: [0, 6], stitched_frames: 48, stitched_seconds: 2,
                frames_per_account: [48], align_to_wan: true, share_boundary_frames: true,
                max_segments: 32,
            },
            plan: { rendered_frames: 48, stitched_frames: 48 },
            warnings: [],
            fleet: { workers: WORKERS },
            ffmpeg: "# from server",
        };
        const calls = [];
        const dom = await boot((url, options) => {
            calls.push(`${options?.method || "GET"} ${url}`);
            if (url === "/api/health") return json({ ok: true });
            if (url === "/api/fleet") return json({ workers: WORKERS, source: "comfyui" });
            if (url === "/api/plan") return json(planBody);
            return json({ error: "not found" }, false);
        });
        const { document } = dom.window;

        expect(document.getElementById("mode-badge").textContent).toContain("server");
        expect(document.getElementById("fleet-text").textContent).toContain("3 workers");
        expect(document.getElementById("import-card").hidden).toBe(true);
        expect(document.getElementById("stat-segments").textContent).toBe("1");
        expect(calls).toContain("POST /api/plan");
        dom.window.close();
    });

    it("survives the server rejecting a plan", async () => {
        const dom = await boot((url) => {
            if (url === "/api/health") return json({ ok: true });
            if (url === "/api/fleet") return json({ workers: [] });
            if (url === "/api/plan") return json({ error: "boom" }, false);
            return json({}, false);
        });
        const { document } = dom.window;

        expect(document.getElementById("warnings").textContent).toContain("boom");
        dom.window.close();
    });
});
