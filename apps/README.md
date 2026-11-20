# The app: Adaptive Studio + Colab worker notebook

Two pieces that sit around the adaptive video nodes:

| Piece | Runs on | What it does |
|---|---|---|
| **Adaptive Studio** (`apps/adaptive_studio/`) | Master machine, in a browser | Plans the split from your fleet, shows the timeline/keyframes/per-account jobs, exports the plan, keyframe times and an FFmpeg recipe |
| **Colab worker notebook** (`apps/colab_worker.ipynb`) | Each Google account | Installs ComfyUI + this extension, starts a worker, opens a Cloudflare tunnel and registers itself with the master |

Both use the same planner (`utils/video_plan.py`) as the **Adaptive Video
Segments** node, so what the app shows is exactly what each account renders.

## Adaptive Studio

```bash
python3 apps/adaptive_studio/server.py            # http://localhost:8765
python3 apps/adaptive_studio/server.py --port 9000 --open
python3 apps/adaptive_studio/server.py --host 0.0.0.0   # reachable from other machines
```

Standard library only — no new dependencies, nothing to build.

- **Accounts** defaults to *Auto*, which reads `gpu_config.json` (the same file the
  Distributed panel writes) so the plan follows the workers you enabled. Switch to
  *Manual* to plan for a fleet you have not configured yet.
- **Master participates** adds the master's own segment; turn it off to mirror
  *orchestrator-only* mode.
- **Align to Wan (4k+1)** snaps every segment to a Wan-valid frame count. Turn it off
  for perfectly even segments with an arbitrary frame count.
- **Share boundary frames** is FLF2V continuity: neighbouring segments share the
  frame they meet on, and stitching drops the duplicate.
- Exports: plan JSON, keyframe times (comma separated), and an FFmpeg concat script
  that trims the duplicated boundary frames.

Endpoints (all local JSON):

| Route | Purpose |
|---|---|
| `GET /api/health` | liveness |
| `GET /api/fleet` | master + workers from `gpu_config.json` |
| `POST /api/plan` | `{duration_seconds, fps, accounts, min_frames_per_segment, master_participates, align_to_wan, share_boundary_frames}` → plan, per-account jobs, warnings |

`accounts: 0` means auto (every configured worker).

## Colab worker notebook

Open `apps/colab_worker.ipynb` in each Google account (File → Upload notebook, or
copy it into Drive) and run the cells top to bottom:

1. **Parameters** — set a unique `WORKER_NAME` (`colab-a`, `colab-b`, …) and, if the
   master is reachable, `MASTER_URL`.
2. **GPU check** — confirms the Colab GPU and free disk.
3. **Install** — clones ComfyUI and this extension, installs ComfyUI's requirements
   without touching Colab's torch.
4. **Start the worker** — launches ComfyUI with `COMFYUI_IS_WORKER=1 --listen`.
5. **Tunnel** — runs `cloudflared` and prints the public `https://…trycloudflare.com` URL.
6. **Register** — posts the tunnel URL to the master's
   `/distributed/config/update_worker`, which is exactly what the panel does when you
   add a Remote worker. Re-running updates the same worker instead of duplicating it.
7. **Verify** — checks the worker responds locally and that the master lists it.

If the master is not publicly reachable, skip registration and add the worker in
the panel manually: **Add Worker → Remote → host = tunnel URL, port = 443**.

## Typical loop

1. Start the master with `--enable-cors-header`.
2. Run the notebook on each Colab account (one per account, each with its own name).
3. Open Adaptive Studio, set duration/fps, confirm the segments match the number of
   accounts, and copy the keyframe times.
4. In ComfyUI, add **Adaptive Video Segments** → **Adaptive Video Stitcher**, without
   changing `num_segments` (leave it at 0 = auto).
5. Queue the workflow. Every enabled account renders one segment; the stitcher joins
   them on the master.

Details, limits and troubleshooting: [adaptive-video-segments.md](../docs/adaptive-video-segments.md)
and the [compatibility report](../docs/multi-account-test-report.md).
