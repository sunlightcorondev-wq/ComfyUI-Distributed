# Adaptive segment rendering (any number of accounts)

The multi-account video workflow used to be a fixed plan: 6 seconds split into
3 x 2-second jobs because there were exactly 3 Colab accounts. Adding a fourth
account meant re-cutting the workflow by hand, and shrinking to two accounts
wasted a machine.

Adaptive segment rendering derives everything from **whichever accounts are
enabled when the run is dispatched**:

| Derived value | Source |
|---|---|
| Number of segments | Number of participants (master, when it participates, plus every enabled worker) |
| Keyframe count | Segments + 1 (each keyframe is a segment boundary) |
| Frame window per account | The plan's split of the timeline, snapped to a Wan-valid `4k + 1` count |
| Stitched length | Rendered frames minus the duplicated boundary frames |

Nothing is hard-coded: enable 2 Colab accounts and you get two segments; enable
12 and you get twelve. `num_segments` still accepts a fixed number when you want
to pin the split (for example, two segments per account on a small fleet).

## Nodes

### Adaptive Video Segments (`DistributedVideoPlanner`)

Plans the split and reports what *this* account should render. The orchestrator
injects the enabled-worker list into every dispatched copy (master and each
worker), so the same saved workflow is correct on every machine.

| Input | Meaning |
|---|---|
| `duration_seconds` | Target length of the finished video (e.g. `6.0`) |
| `fps` | Target frame rate (e.g. `24`) |
| `num_segments` | `0` = auto (one segment per participant). Any positive value pins the split |
| `min_frames_per_segment` | Floor for segment length; the plan caps the segment count rather than going below it |
| `align_to_wan` | Snap every segment to a Wan-valid `4k + 1` frame count |
| `share_boundary_frames` | First/last-frame continuity: each segment ends on the frame the next one starts from |

| Output | Meaning |
|---|---|
| `plan` | JSON of the whole plan: every segment's start/end frames, the per-segment frame counts, keyframes, warnings |
| `num_segments` | Resolved segment count |
| `segment_index` | This account's segment (`0` for the master, `N` for `worker_N`) |
| `start_frame` / `end_frame` | This account's window inside the target timeline |
| `frames` | Frames this account renders (feed it into the sampler's length) |
| `keyframe_times` | Comma-separated boundary times in seconds, e.g. `0.000,2.000,4.000,6.000` |
| `stitched_frames` | Frame count of the final joined video |
| `first_keyframe` / `last_keyframe` | Indexes into a keyframe strip: this segment renders from boundary `k` to boundary `k+1` |

### Adaptive Video Stitcher (`DistributedVideoStitcher`)

Joins the rendered segments back into one timeline.

| Input | Meaning |
|---|---|
| `segment_1` … `segment_16` | One socket per account slot (`master` = first segment) |
| `num_segments` | `0` = stitch every connected, non-empty segment |
| `share_boundary_frames` | Drop each later segment's first frame, because it repeats the previous segment's last frame |

Unconnected or empty sockets are skipped, so the same node stitches a 2-account
run and a 16-account run. A resolution mismatch between accounts raises a clear
error instead of producing a corrupt video.

### Image / Audio Batch Divider auto mode

`divide_by = 0` now means **auto**: the orchestrator rewrites it to the number
of participants (master included unless it is delegating) right before
dispatch. A positive value keeps the old fixed behaviour, and a `divide_by`
that was converted into a link is never touched. Both dividers expose up to
16 outputs and both dividers plus the collectors' per-account previews now
cover 16 accounts (previously 10 outputs / 8 workers).

## What the plan looks like

Target: 6 seconds at 24 fps = 144 frames, `share_boundary_frames` on.

| Accounts | Segments | Frames rendered per segment | Stitched frames |
|---|---|---|---|
| 1 | 1 | 145 | 145 |
| 2 | 2 | 73, 73 | 145 |
| 3 | 3 | 49, 49, 49 | 145 |
| 4 | 4 | 37 x 4 | 145 |
| 6 | 6 | 25 x 6 | 145 |
| 8 | 8 | 21 x 4 + 17 x 4 | 145 |
| 12 | 12 | 13 x 12 | 145 |
| 16 | 16 | 13 x 4 + 9 x 12 | 145 |

With `align_to_wan` off (any frame count allowed), the split is even and exact:
3 accounts give `49, 49, 48` and 6 accounts give `25, 25, 25, 25, 25, 24`, all
stitching back to exactly 144 frames.

The old fixed plan was `48, 48, 48` (48 is not `4k + 1`, which is why 2-second
segments were awkward for Wan in the first place). Note that a segment of
`frames` frames contributes `frames - 1` to the final timeline when boundaries
are shared: 3 accounts x 49 frames - 2 shared frames = 145.

## Wiring

### A. Variation mode (existing WAN workflow)

Every account renders a full video with its own seed, and the divider hands each
account's clip to its own save node.

1. WAN workflow through **Distributed Collector** → **Image Batch Divider**
2. Set `divide_by = 0`
3. Wire one `batch_N` output per save node you want (extra outputs stay empty
   and are pruned by the run)

### B. Segment mode (one long video, split across accounts)

1. Build a **keyframe strip** for the target video: `segments + 1` keyframes.
   The planner's `keyframe_times` output gives you the times to generate
   (`0, 2, 4, 6` seconds for a 3-account 6-second plan).
2. **Adaptive Video Segments** → set duration and fps, leave `num_segments = 0`.
3. Per account, pick its two keyframes from the strip with `first_keyframe` and
   `last_keyframe` (for example `ImageFromBatch` on a batched keyframe strip, or
   a `DistributedValue` lookup), and feed `frames` into the FLF2V sampler length.
4. Render, VAE-decode, and send the frames to **Distributed Collector**.
5. **Adaptive Video Stitcher** → wire the collector's combined `images` into the
   stitcher's `images`, the planner's `plan` into `plan`, then send the
   stitcher's `images` to your video combine / save node. The collector orders
   the combined batch master first and then the enabled workers, which is
   exactly the plan's segment order, so the stitcher can slice out each segment
   and drop the duplicated boundary frames.

Wiring the collector's per-account preview sockets (`segment_1` ← `master`,
`segment_2` ← `worker_1`, …) also works and is handy for previewing, but the
collector emits a *silent blocker* on slots for accounts that did not take part,
and a blocked input skips the whole stitcher. Only use that wiring when every
wired account is guaranteed to run; otherwise use the combined `images` + `plan`
wiring above, which skips missing accounts by checking the plan against the
received frame count.

### C. Stitch outside ComfyUI

`plan` gives you the boundaries, so an FFmpeg concat works too:

```bash
# Each account writes segment_1.mp4 ... segment_N.mp4 with identical settings.
printf "file '%s'\n" segment_*.mp4 > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy -movflags +faststart final.mp4
```

Because neighbouring segments share a boundary frame, drop the duplicate first
frame of segments 2..N before concatenating:

```bash
ffmpeg -i segment_2.mp4 -vf "select='gte(n\,1)',setpts=N/FRAME_RATE/TB" -r 24 seg2.mp4
```

The stitcher node does this automatically when `share_boundary_frames` is on,
using either the combined batch plus the plan or the per-account sockets.

## Limits and caveats

- **16 segment slots.** Dividers and the stitcher expose 16 segments, and the
  collector exposes a preview slot for the master plus 16 workers. A run with
  more participants still renders correctly, but only the first 16 fit the
  per-account sockets — the rest appear only in the collector's combined
  `images` output. Raise `MAX_DIVIDER_OUTPUTS` in `nodes/utilities.py` and
  `MAX_WORKER_OUTPUTS` in `nodes/collector.py` together if you need more.
- **32 segments maximum.** Above that a single Wan segment would be shorter than
  a couple of frames, so the planner caps the count and records a warning in
  `plan`.
- **Wan alignment trades evenness for exact length.** `4k + 1` lengths rarely
  divide a target evenly, so a few segments are up to 4 frames longer than
  others and the longest one sets the wall-clock time. Turn `align_to_wan` off
  for a perfectly even split, or lower `num_segments` so accounts take turns.
- **Static workers only.** Serverless pools still reject the dividers and
  adaptive nodes, because a pool cannot be told which segment it owns. Adaptive
  video targets the master + worker fleet (including Colab-hosted workers).
- **Every account needs the same resolution.** The stitcher refuses to join
  mismatched sizes; keep width/height identical across the workflow copies.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Everything renders as one video | `divide_by` is still a fixed number, or `num_segments` is pinned above 0 |
| Video is slightly shorter or longer than requested | Wan alignment forced `4k + 1` lengths; check `plan["stitched_frames"]` and `warnings` |
| A segment is missing from the final video | That account produced nothing (crashed, skipped, or toggled off). With the combined `images` + `plan` wiring the stitcher logs a warning and joins every frame it received instead of cutting at stale offsets |
| The stitcher never runs when using per-account sockets | An idle account's preview slot carries a silent blocker, which skips the node; use the combined `images` + `plan` wiring instead |
| Stitcher refuses to join | Segments were rendered at different resolutions |
| Master renders segment 1 twice | The master is participating *and* `worker_1` was wired into `segment_1`; wire `master` into `segment_1` |
