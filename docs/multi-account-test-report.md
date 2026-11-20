# Multi-account compatibility test report

Scope: does adaptive segment rendering hold up when the number of Colab
accounts changes? Tested on 2026-11-19 on branch
`arena/4bebe2b5-comfyui-distributed`.

## How it was tested

No GPU or Colab session is available in the test environment, so compatibility
is proven by driving the **production code paths** with a simulated fleet:

| Stage | Production code exercised |
|---|---|
| Dispatch wiring | `api/orchestration/prompt_transform.py` (`apply_participant_overrides`, `prune_prompt_for_worker`) |
| Per-account planning | `nodes/utilities.py` `DistributedVideoPlanner.plan` |
| Result gathering | `nodes/collector.py` (`_reorder_and_combine_tensors`, `_worker_output_slots`) |
| Stitching | `nodes/utilities.py` `DistributedVideoStitcher.stitch` |
| UI sockets | `web/image_batch_divider.js` |

Every simulated account renders a synthetic clip whose pixels encode the global
frame index, so the final assertion is exact: the stitched video must be frames
`0..N-1`, in order, with no duplicated or missing boundary frame.

Test files:

```bash
python3 -m pytest tests/test_multi_account_compatibility.py tests/test_adaptive_video.py -q
npx vitest run web/tests/imageBatchDivider.test.js
```

## Fleet sizes covered

Fleets of **1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 17 and 20 accounts**, each run
twice: master participating and master in delegate-only mode.

For a 6-second, 24 fps target (144 frames) with Wan alignment on:

| Accounts | Segments | Frames per account | Stitched frames | Keyframes |
|---|---|---|---|---|
| 1 | 1 | 145 | 145 | 2 |
| 2 | 2 | 73, 73 | 145 | 3 |
| 3 | 3 | 49, 49, 49 | 145 | 4 |
| 4 | 4 | 37 x 4 | 145 | 5 |
| 5 | 5 | 33, 29 x 4 | 145 | 6 |
| 6 | 6 | 25 x 6 | 145 | 7 |
| 8 | 8 | 21 x 4, 17 x 4 | 145 | 9 |
| 10 | 10 | 17 x 6, 13 x 4 | 145 | 11 |
| 12 | 12 | 13 x 12 | 145 | 13 |
| 15 | 15 | 13 x 6, 9 x 9 | 145 | 16 |
| 16 | 16 | 13 x 4, 9 x 12 | 145 | 17 |
| 17 | 17 | 13 x 2, 9 x 15 | 145 | 18 |
| 20 | 20 | 9 x 16, 5 x 4 | 145 | 21 |

The old fixed plan was 3 x 48 frames, which is not a Wan-valid `4k + 1` length.

## Invariants that passed

1. **One segment per account, exactly once.** For every fleet size and mode, the
   segment indexes reported across all accounts equal `range(account_count)` with
   no duplicates, and every account computes a byte-identical plan.
2. **Seamless reconstruction.** Concatenating the accounts' clips in collector
   order (master first, then enabled workers) and stitching with the plan yields
   frames `0..stitched-1`, in order, no duplicate, no gap.
3. **Auto divider matches the fleet.** `divide_by = 0` resolves to
   `enabled workers + 1` (or just the workers when the master delegates), on both
   the image and audio divider, and the stitcher receives the same count.
4. **Segment identity per account.** The master plans segment 0; worker *i*
   plans segment *i + 1* when the master participates and segment *i* when it
   delegates. That keeps collector preview slot `worker_N` aligned with
   `segment_(N+1)`.
5. **Collector ordering and slots.** With 16 configured workers the collector
   maps all 16 to `worker_1..worker_16` in config order; extra accounts get no
   dedicated slot but still contribute frames to the combined batch. Combined
   order is master then workers, which is exactly the plan order.
6. **Over 16 accounts.** 17 and 20 account fleets stitch exactly through the
   combined `images` + `plan` path even though there are no per-account sockets.
7. **Dropped account.** If one account never reports, the stitcher keeps every
   frame it received instead of cutting at stale offsets (logged warning), so the
   run degrades to an untrimmed join rather than a corrupt video.
8. **Worker prompts stay valid.** Pruning keeps the planner on workers (it is
   upstream of the sampler) and drops the stitcher (master-only), so workers run
   a legal graph.
9. **Ceilings behave.** 40 accounts on a 6-second video is capped to 28 segments
   by the 5-frame minimum; on a 20-second video it is capped to 32 segments. Both
   record a warning in `plan`. Segments under 0.5 s also raise an advisory
   warning.
10. **UI sockets.** `divide_by = 0` exposes all 16 sockets so any fleet size can
    be pre-wired; a manual value above 16 is clamped; changing the widget grows
    and shrinks sockets again.

## Results

| Suite | Result |
|---|---|
| `tests/test_multi_account_compatibility.py` | 93 passed, 1 skipped |
| `tests/test_adaptive_video.py` + `tests/test_batch_dividers.py` | 44 passed |
| Full Python suite | 1978 passed, 82 skipped (25 pre-existing failures, all `PyAV`/ffmpeg missing in this sandbox) |
| Web suite (`npx vitest run`) | 137 passed, 14 files |

## Not covered here

These need real hardware and are the checks to run on an actual Colab fleet:

- **Real Wan FLF2V runs.** Frame-count validity (`4k + 1`) is proven
  mathematically, not by sampling on a GPU.
- **Wall-clock speedup.** The plan makes every account render one segment; the
  observed speedup depends on Colab GPU speed and transfer time.
- **Real network transfer of 4k + 1 frame PNG/WebP chunks** at 720p — covered by
  the existing transfer codec suite, but not with this exact frame count.
- **VHS/VideoCombine encoders** around the stitcher output.

### Manual acceptance checklist (3 → 5 accounts)

1. Run the workflow with 3 accounts and confirm the plan node reports
   `num_segments = 3` (or 4 when the master participates) and `keyframe_times`
   spans `0..6`.
2. Enable 2 more Colab workers and re-queue the **same saved workflow**; the plan
   should now report a larger `num_segments` without any workflow edit.
3. Confirm the stitched video length matches `plan["stitched_frames"]` and has no
   visible stutter or repeated frame at the joins.
4. Toggle one worker off mid-run and confirm the stitcher logs the frame-count
   warning instead of producing a truncated video.
