// Adaptive Studio UI.
//
// Runs in two modes with one code path:
//   * server  – served by apps/adaptive_studio/server.py: live fleet from
//               gpu_config.json and the Python planner.
//   * static  – opened from disk or any static host: the fleet is imported from
//               a gpu_config.json file and planner.js plans locally. No network
//               calls leave the page.

const state = {
    mode: "starting", // "server" | "static"
    fleet: null,
    result: null,
    requestId: 0,
};

const SETTINGS_KEY = "adaptiveStudio.settings.v1";
const FLEET_KEY = "adaptiveStudio.fleet.v1";

// Inlined so "Load demo fleet" works even from file://, where fetching a
// sibling JSON file is blocked. apps/adaptive_studio/demo_fleet.json stays the
// canonical copy for the Python server (--config).
const DEMO_FLEET = {
    workers: [
        { id: "colab-a", name: "Colab A (T4)", type: "remote", host: "https://colab-a.trycloudflare.com", port: 443 },
        { id: "colab-b", name: "Colab B (T4)", type: "remote", host: "https://colab-b.trycloudflare.com", port: 443 },
        { id: "colab-c", name: "Colab C (L4)", type: "remote", host: "https://colab-c.trycloudflare.com", port: 443 },
        { id: "colab-d", name: "Colab D (T4)", type: "remote", host: "https://colab-d.trycloudflare.com", port: 443 },
        { id: "colab-e", name: "Colab E (L4)", type: "remote", host: "https://colab-e.trycloudflare.com", port: 443 },
    ],
};

const el = (id) => document.getElementById(id);

const PALETTE = [
    ["#7c3aed", "#a78bfa"],
    ["#0e7490", "#22d3ee"],
    ["#1d4ed8", "#60a5fa"],
    ["#4d7c0f", "#a3e635"],
    ["#b45309", "#fbbf24"],
    ["#9d174d", "#f472b6"],
    ["#065f46", "#34d399"],
    ["#4338ca", "#818cf8"],
];

function segmentColor(index) {
    const [from, to] = PALETTE[index % PALETTE.length];
    return `linear-gradient(160deg, ${from}, ${to})`;
}

function segmentSolid(index) {
    return PALETTE[index % PALETTE.length][0];
}

// --- settings ---------------------------------------------------------------

function readSettings() {
    try {
        return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    } catch {
        return {};
    }
}

function saveSettings() {
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(formState()));
    } catch {
        /* private mode / file:// with storage disabled: not fatal */
    }
}

function formState() {
    return {
        duration: Number(el("duration-num").value),
        fps: Number(el("fps").value),
        mode: el("accounts-mode").value,
        accounts: Number(el("accounts").value),
        minFrames: Number(el("min-frames").value),
        master: el("master").checked,
        align: el("align").checked,
        share: el("share").checked,
    };
}

function applySettings(settings) {
    if (Number.isFinite(settings.duration)) el("duration-num").value = settings.duration;
    if (Number.isFinite(settings.fps)) el("fps").value = String(settings.fps);
    if (settings.mode === "manual" || settings.mode === "auto") el("accounts-mode").value = settings.mode;
    if (Number.isFinite(settings.accounts)) el("accounts").value = settings.accounts;
    if (Number.isFinite(settings.minFrames)) el("min-frames").value = settings.minFrames;
    if (typeof settings.master === "boolean") el("master").checked = settings.master;
    if (typeof settings.align === "boolean") el("align").checked = settings.align;
    if (typeof settings.share === "boolean") el("share").checked = settings.share;
    syncDuration();
    syncAccountsInput();
}

function syncDuration() {
    const value = Number(el("duration-num").value) || 6;
    el("duration").value = String(Math.min(60, Math.max(1, value)));
    el("duration-out").textContent = `${value.toFixed(2)} s`;
}

function syncAccountsInput() {
    el("accounts").disabled = el("accounts-mode").value !== "manual";
}

function formPayload() {
    const settings = formState();
    return {
        duration_seconds: settings.duration,
        fps: settings.fps,
        accounts: settings.mode === "manual" ? settings.accounts : 0,
        min_frames_per_segment: settings.minFrames,
        master_participates: settings.master,
        align_to_wan: settings.align,
        share_boundary_frames: settings.share,
    };
}

// --- fleet ------------------------------------------------------------------

function normaliseFleet(raw) {
    const config = Array.isArray(raw) ? { workers: raw } : (raw || {});
    const workers = (Array.isArray(config.workers) ? config.workers : [])
        .filter((worker) => worker && typeof worker === "object")
        .map((worker) => ({
            id: String(worker.id || ""),
            name: String(worker.name || worker.id || "worker"),
            type: String(worker.type || "local"),
            host: String(worker.host || ""),
            port: worker.port,
        }));
    return {
        source: "imported",
        config_path: "",
        master: {
            host: String((config.master || {}).host || ""),
            delegate_only: Boolean((config.settings || {}).master_delegate_only),
        },
        workers,
    };
}

function storedFleet() {
    try {
        const raw = localStorage.getItem(FLEET_KEY);
        return raw ? normaliseFleet(JSON.parse(raw)) : null;
    } catch {
        return null;
    }
}

function storeFleet(fleet) {
    try {
        if (fleet) localStorage.setItem(FLEET_KEY, JSON.stringify(fleet));
        else localStorage.removeItem(FLEET_KEY);
    } catch {
        /* ignore quota/private-mode errors */
    }
}

async function loadFleet() {
    if (state.mode === "server") {
        try {
            const response = await fetch("/api/fleet", { cache: "no-store" });
            if (response.ok) {
                state.fleet = await response.json();
            }
        } catch (error) {
            console.warn("fleet unavailable", error);
        }
    } else {
        state.fleet = storedFleet();
    }
    renderFleet();
}

function renderFleet() {
    const pill = el("fleet-pill");
    const workers = (state.fleet && state.fleet.workers) || [];
    if (workers.length === 0) {
        pill.className = "fleet warn";
        el("fleet-text").textContent = state.mode === "server"
            ? "no workers in gpu_config.json — planning manually"
            : "no fleet imported — planning manually";
    } else {
        pill.className = "fleet live";
        const names = workers.slice(0, 3).map((worker) => worker.name).join(", ");
        el("fleet-text").textContent =
            `${workers.length} worker${workers.length === 1 ? "" : "s"} · ${names}${workers.length > 3 ? "…" : ""}`;
    }
    el("accounts-hint").textContent = workers.length
        ? `0 = all ${workers.length} imported workers (master adds one more when it participates)`
        : "0 = the imported fleet; use Manual to plan for accounts you have not configured";

    el("import-card").hidden = state.mode === "server";
}

// --- planning ---------------------------------------------------------------

let pending = null;

function schedulePlan() {
    if (pending) clearTimeout(pending);
    pending = setTimeout(plan, 120);
}

async function plan() {
    const body = formPayload();
    const requestId = ++state.requestId;
    document.body.classList.add("loading");
    try {
        let result;
        if (state.mode === "server") {
            const response = await fetch("/api/plan", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            result = await response.json();
            if (!response.ok) throw new Error(result.error || "plan failed");
        } else {
            result = window.AdaptivePlanner.buildPlanResponse(body, state.fleet);
            result.ffmpeg = window.AdaptivePlanner.ffmpegScript(result);
        }
        if (requestId !== state.requestId) return;
        state.result = result;
        render(result);
        saveSettings();
    } catch (error) {
        el("warnings").innerHTML = `<li>Could not build a plan: ${escapeHtml(error.message)}</li>`;
    } finally {
        if (requestId === state.requestId) document.body.classList.remove("loading");
    }
}

// --- rendering --------------------------------------------------------------

function render(result) {
    const { summary, accounts, plan: planData,
        warnings } = result;

    el("stat-accounts").textContent = String(accounts.length);
    el("stat-accounts-note").textContent =
        `${summary.enabled_workers} worker${summary.enabled_workers === 1 ? "" : "s"}`
        + (summary.master_participates ? " + master" : " · master delegating");

    el("stat-segments").textContent = String(summary.segments);
    el("stat-keyframes").textContent = `${summary.keyframes} keyframes`;

    const frames = summary.frames_per_account;
    el("stat-frames").textContent =
        frames.length === 1 ? String(frames[0]) : `${frames[0]}–${frames[frames.length - 1]}`;
    el("stat-seconds").textContent =
        `${(frames[0] / summary.fps).toFixed(2)}–${(frames[frames.length - 1] / summary.fps).toFixed(2)} s each`;

    el("stat-stitched").textContent = `${summary.stitched_seconds.toFixed(2)} s`;
    el("stat-stitched-frames").textContent =
        `${summary.stitched_frames} frames · target ${summary.duration_seconds.toFixed(2)} s`;

    el("timeline-tag").textContent =
        `${planData.rendered_frames} rendered → ${planData.stitched_frames} stitched`;
    el("table-tag").textContent = `${accounts.length} jobs`;

    renderTimeline(result);
    renderKeyframes(summary.keyframe_times, summary.fps);
    renderAccounts(accounts);
    renderWarnings(warnings);

    el("meta").textContent =
        `${planData.rendered_frames} rendered frames · ${summary.fps} fps · `
        + (summary.align_to_wan ? "Wan-aligned 4k+1" : "free frame counts");
}

function renderTimeline(result) {
    const { accounts, summary, plan: planData } = result;
    const share = summary.share_boundary_frames;
    const timeline = el("timeline");
    timeline.innerHTML = "";

    for (const account of accounts) {
        // Each segment contributes frames - 1 to the final timeline when it
        // shares its first frame with the previous segment's last frame.
        const shown = share && account.index > 0 ? account.frames - 1 : account.frames;
        const segment = document.createElement("div");
        segment.className = `seg${account.kind === "master" ? " master" : ""}`;
        segment.style.flexGrow = String(shown);
        segment.style.flexBasis = "0";
        segment.style.background = segmentColor(account.index);
        segment.title =
            `${account.slot} · segment ${account.index + 1} · frames ${account.start_frame}–${account.end_frame} `
            + `(${account.frames} rendered, ${shown} in final video)`;
        segment.innerHTML = `<b>${escapeHtml(account.label)}</b><i>${account.seconds.toFixed(2)}s</i>`;
        timeline.appendChild(segment);
    }

    const axis = el("axis");
    axis.innerHTML = "";
    const stitched = Math.max(1, planData.stitched_frames);
    let cursor = 0;
    const marks = [];
    for (const account of accounts) {
        marks.push({ frame: cursor, label: (cursor / summary.fps).toFixed(2) });
        cursor += share && account.index > 0 ? account.frames - 1 : account.frames;
    }
    marks.push({ frame: stitched, label: (stitched / summary.fps).toFixed(2) });
    for (const mark of marks) {
        const tick = document.createElement("span");
        tick.className = "tick";
        tick.style.left = `${(mark.frame / stitched) * 100}%`;
        tick.textContent = `${mark.label}s`;
        axis.appendChild(tick);
    }
}

function renderKeyframes(times, fps) {
    const container = el("keyframes");
    container.innerHTML = "";
    times.forEach((seconds) => {
        const chip = document.createElement("span");
        chip.className = "kf";
        chip.innerHTML = `<b>${seconds.toFixed(2)}s</b><span>frame ${Math.round(seconds * fps)}</span>`;
        container.appendChild(chip);
    });
}

function renderAccounts(accounts) {
    const body = el("accounts-body");
    body.innerHTML = "";
    for (const account of accounts) {
        const row = document.createElement("tr");
        if (account.kind === "master") row.className = "master-row";
        row.innerHTML = `
            <td class="swatch"><i style="background:${segmentSolid(account.index)}"></i></td>
            <td>${escapeHtml(account.slot)}</td>
            <td>${escapeHtml(account.label)}</td>
            <td>${account.index + 1}</td>
            <td>${account.start_frame}–${account.end_frame}</td>
            <td>${account.seconds.toFixed(2)}s</td>`;
        body.appendChild(row);
    }
}

function renderWarnings(warnings) {
    const list = el("warnings");
    list.innerHTML = "";
    if (!warnings || warnings.length === 0) return;
    for (const warning of warnings) {
        const item = document.createElement("li");
        item.textContent = warning;
        list.appendChild(item);
    }
}

function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (character) => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]
    ));
}

// --- exports ----------------------------------------------------------------

async function copyText(text, status) {
    try {
        await navigator.clipboard.writeText(text);
        el("copy-status").textContent = status;
    } catch {
        // Clipboard needs a secure context; fall back to a temporary textarea.
        const area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
        el("copy-status").textContent = status;
    }
    setTimeout(() => { el("copy-status").textContent = ""; }, 2200);
}

function download(name, text, type = "application/json") {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(url);
}

function wireExports() {
    document.querySelectorAll("[data-copy]").forEach((button) => {
        button.addEventListener("click", () => {
            const result = state.result;
            if (!result) return;
            const kind = button.dataset.copy;
            if (kind === "plan") {
                copyText(JSON.stringify(result.plan, null, 2), "Plan JSON copied");
            } else if (kind === "keyframes") {
                copyText(result.summary.keyframe_times.map((value) => value.toFixed(3)).join(","),
                    "Keyframe times copied");
            } else if (kind === "ffmpeg") {
                copyText(result.ffmpeg || "", "FFmpeg script copied");
            } else if (kind === "download") {
                download("adaptive-plan.json", JSON.stringify(result.plan, null, 2));
            }
        });
    });
}

// --- fleet import (static mode) ---------------------------------------------

async function importFleetFile(file) {
    const text = await file.text();
    const fleet = normaliseFleet(JSON.parse(text));
    state.fleet = fleet;
    storeFleet(fleet);
    renderFleet();
    if (el("accounts-mode").value === "auto") el("accounts").value = String(fleet.workers.length);
    await plan();
}

async function loadDemoFleet() {
    const fleet = normaliseFleet(DEMO_FLEET);
    state.fleet = fleet;
    storeFleet(fleet);
    el("accounts-mode").value = "auto";
    syncAccountsInput();
    el("accounts").value = String(fleet.workers.length);
    renderFleet();
    await plan();
}

function clearFleet() {
    state.fleet = { workers: [] };
    storeFleet(null);
    renderFleet();
    plan();
}

function wireFleetImport() {
    el("fleet-file").addEventListener("change", async (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        try {
            await importFleetFile(file);
        } catch (error) {
            el("copy-status").textContent = `Could not read that file: ${error.message}`;
        }
    });
    el("fleet-demo").addEventListener("click", loadDemoFleet);
    el("fleet-clear").addEventListener("click", clearFleet);
}

// --- inputs -----------------------------------------------------------------

function wireInputs() {
    const durationRange = el("duration");
    const durationNumber = el("duration-num");

    durationRange.addEventListener("input", () => {
        durationNumber.value = durationRange.value;
        syncDuration();
        schedulePlan();
    });
    durationNumber.addEventListener("input", () => {
        syncDuration();
        schedulePlan();
    });

    el("accounts-mode").addEventListener("change", () => {
        syncAccountsInput();
        if (el("accounts-mode").value === "auto" && state.fleet) {
            el("accounts").value = String((state.fleet.workers || []).length);
        }
        schedulePlan();
    });

    for (const id of ["fps", "accounts", "min-frames"]) {
        el(id).addEventListener("input", schedulePlan);
        el(id).addEventListener("change", schedulePlan);
    }
    for (const id of ["master", "align", "share"]) {
        el(id).addEventListener("change", schedulePlan);
    }
}

// --- boot -------------------------------------------------------------------

async function detectMode() {
    try {
        const response = await fetch("/api/health", { cache: "no-store" });
        if (response.ok) {
            const payload = await response.json();
            if (payload && payload.ok) return "server";
        }
    } catch {
        /* file:// or a plain static host: no backend */
    }
    return "static";
}

function renderMode() {
    const badge = el("mode-badge");
    if (state.mode === "server") {
        badge.className = "mode server";
        badge.textContent = "server · live fleet";
        badge.title = "Served by apps/adaptive_studio/server.py — reads gpu_config.json directly";
    } else {
        badge.className = "mode static";
        badge.textContent = "browser-only · no server";
        badge.title = "Runs entirely in this tab; import gpu_config.json to plan for your fleet";
    }
}

async function boot() {
    applySettings(readSettings());
    wireInputs();
    wireExports();
    wireFleetImport();

    state.mode = await detectMode();
    renderMode();

    await loadFleet();
    if (state.mode === "static" && (state.fleet?.workers || []).length === 0) {
        el("accounts-mode").value = "manual";
        syncAccountsInput();
        if (!Number(el("accounts").value)) el("accounts").value = "3";
    }
    await plan();

    // Pick up workers added to the panel without a reload (server mode only).
    if (state.mode === "server") setInterval(loadFleet, 30000);
}

boot();
