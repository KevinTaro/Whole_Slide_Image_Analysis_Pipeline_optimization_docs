# 38 — Round 12: the cross-tile multiprocessing scaling ceiling (load balancing / pipelined stitching / workers≥5) — plan

> Follows up directly on the question the user asked: "`workers=1` vs `workers=4` only gets about
> 2x — can load balancing, pipelined stitching, or opening more workers get closer to using the
> hardware fully?" This document does exactly one thing: it assembles numbers already scattered
> across docs 8/20/21/27/29/32/35/37 into the **first** complete Amdahl account of where 2.216x
> is actually being lost, then sequences the next step per
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)'s
> Discover → Analyze → Plan → Choose. **Planning document only — no pipeline code changed here.**
> §1 is entirely a citation of existing measurements (every number is sourced); the Amdahl
> decomposition in §2 is the new work this round does — no prior document has ever added "the
> composition-driven ceiling" and "Phase D's serial tax" into the same table, and that sum is
> exactly what the user is asking about.

## 0. Two stale statements corrected while writing this (found along the way)

Re-reading `measurement/bottleneck-list.md` and `19-open-backlog.md`'s live status before writing
this doc turned up something both said: `expandable_segments:True` is "recommended but not yet
flipped as the default." **That is now false.** This repo's HEAD commit (`b3fa47d`, the current
tip as of this document) already changed `config_example.py`'s `cuda_alloc_conf` default to
`"expandable_segments:True"`, and the commit message is itself a full restatement of "round 11's
decision, see `docs/hybrid-pipeline/37-round-11-backlog-implementation.md` §3/§9." §3 below
circles back and fixes those two files' stale line (this document does not re-litigate the
underlying measurement, only updates the one stale sentence).

**Why this matters for this document specifically**: VRAM headroom did **not** widen because of
this knob — round 11's own sweep found peak framebuffer rising from 18.2 GB to 30.1 GB (92.2% of
the 32,607 MB card), because the knob eliminates an allocator OOM failure mode, it does not
actually save memory. In other words: for "can we open more workers," the answer under the
**setting that's live today** is a tighter margin than round 8's 93.3% / ~2.2 GB headroom
figure, not a looser one — this is the first fact to put on the table when answering "can we run
`workers=5`," not an assumption.

Separately, and unrelated to this round's topic but worth recording while reading the code: the
top-level `run_batch(..., workers: int = 4, ...)` signature at `hybrid_pipeline.py:128` currently
defaults to `4`, while every document in this folder (`CLAUDE.md`, README, docs 19/20/21…)
repeatedly states "`workers=1` is the default." `git log -S"workers: int = 4"` traces this to
commit `a64b92e` (2026-07-25, "adjust worker count in run_batch"), which flipped that top-level
default from 1 to 4 — but **neither real call site is affected**: `backend/api/hybrid.py:89`
explicitly passes `workers=1`, and the CLI's `--workers` argparse default is still `1`
(`hybrid_pipeline.py:435`). So no production path currently inherits the `4` — this is a latent
footgun (a future caller that omits `workers=` would silently get 4), not a live bug, and it's
out of scope for this round (no performance consequence, no measurement attached). Recorded here
for the next documentation-maintenance pass.

## 1. Discover — this question already has a lot of existing measurement; line it up first

### 1.1 Which round's number does the user's observation match?

The user says "`workers=1` vs `workers=4` only gets about 2x." That's exactly this project's
only real full-slide (27,565-tile) `workers=4` measurement — **round 8**
(`27-remaining-work-implementation.md` §6.5):

| configuration | wall | speedup | source |
|---|--:|--:|---|
| `workers=1` (round 8 baseline) | 13,762.5 s = 3.82 h | 1.00x | [27 §6.5](./27-remaining-work-implementation.md) |
| `workers=4` (round 8) | 6,211 s = 1.73 h | **2.216x** | same |

**This number has never been re-run at `workers=4` since round 8 (2026-07-27).** The two big
rounds 9–11 wins — `gc.freeze()` re-freezing (19.5 s vs 2,218.4 s) and tile-read prefetch settling
at 4.21% of wall (not the earlier 17.2%) — were both only confirmed at full-slide scale on
`workers=1` (`measurement/bottleneck-list.md`'s global wall-clock table). This is the reason §4's
first action item exists: **the table above may itself already be stale** — if `workers=1`'s
absolute time has dropped substantially since round 8 (round 11 measured 10,666.1 s, a 1.290x
improvement) while `workers=4` was never re-run, then 2.216x's numerator and denominator aren't
even measured on the same code, and every Amdahl calculation in this document below must be read
as **"derived from the round-8 baseline,"** not as today's real number.

### 1.2 Why crop-scale speedup (3.09x–3.51x) beats full-slide speedup — this was already measured, twice

`21-cross-tile-multiprocessing-implementation.md` §4.1 measured, on the **tissue-dense**
large/441-tile anchor (only 14.1% background — not the real slide's composition):

| workers | wall | speedup | efficiency |
|--:|--:|--:|--:|
| 1 | 482.8 s | 1.00 | 100% |
| 2 | 208.9 s | 2.31 | 116% |
| 3 | 156.1 s | 3.09 | 103% |
| 4 | 137.4 s | **3.51** | 88% |

But this doc series later re-measured on the match24/576-tile anchor, **composition-matched to
the real slide** (55.9% background, essentially identical to the real slide's measured 55.8%) —
`measurement/bottleneck-list.md`'s "Current anchors" table:

| anchor | background share | `workers=1` | `workers=4` | speedup |
|---|--:|--:|--:|--:|
| large/441 (tissue-dense, not real composition) | 14.1% | 302.7 s | 128.8 s | 2.351x |
| match24 (composition-matched to real slide) | 55.9% (measured real: 55.8%) | 188.8 s | 88.3 s | **2.138x** |
| full-WSI (real 27,565 tiles, round 8) | 55.8% | 13,762.5 s | 6,211 s | **2.216x** |

**Once composition is matched, the speedup drops from 3.51x to 2.14x — and this happens before
Phase D stitching even enters the picture** (a 576-tile crop's stitch cost is on the order of a
few seconds — negligible enough not to move this ratio). This means what the user is seeing as
"only 2x" is **mostly not caused by Phase D**, but by the real slide's composition itself: doc 21
§4.2 already attributed the 3.51x figure (far above doc 20's 1.23x–1.7x prediction) to "recovered
GIL contention" and "overlapping forwards across processes" — and both mechanisms' payoff scales
with **how much work each tile actually does**. The real slide is 55.8% background, and
background tiles short-circuit outright when the UNet++ core mask comes back empty
(`measurement/bottleneck-list.md` ①) — there's no GIL contention left to recover on a tile that
does almost nothing. This is exactly why the tissue-dense crop overstates the payoff. **This is
not a new inference — it is directly confirmed by the composition-matched measurement itself.**

### 1.3 Phase D (`_stitch_overlay_slide`) is currently the only fully serial phase, and it does not scale with worker count

Reading `hybrid_pipeline.py` confirms the architecture: whether `workers=1` or `workers>1`, both
paths only call `_finish_batch()` once (`hybrid_pipeline.py:245` / `:257` / `:342`) **after** the
entire worker pool (or single-process loop) has fully drained, and `_finish_batch()` does the
global renumbering + table export + `_stitch_overlay_slide(output_dir, geometry)`
(`hybrid_pipeline.py:379`) — this is a hard barrier: no matter which of the 4 workers finishes
first, stitching cannot start until the **last** worker's **last** tile lands, and stitching
itself runs entirely in the parent process, on a single CPU/IO path, unaffected by `workers`.

- Round 8's measured absolute cost: **~1,185 s** ("real stitch took 1,185 s," `19-open-backlog.md`
  1b), which is **8.6%** of the `workers=1` 13,762.5 s wall; the same absolute number (since it
  doesn't move with `workers`) becomes **19.3%** of the `workers=4` 6,211 s wall — the denominator
  shrank, the numerator didn't, so the share more than doubles.
- Round 11's latest `workers=1` measurement (wall already down to 10,666.1 s) shows Phase D
  itself quietly grew to **1,239.2 s (11.6% of wall)** — `measurement/bottleneck-list.md` ⑤b
  states directly: "Phase D's share of wall keeps growing anyway, purely because everything
  around it got faster." That means: even if Phase D's own absolute cost never gets worse, as
  long as anything else gets optimized (or worker count goes up), its share on `workers=4` will
  keep rising — eventually it becomes the dominant term. §2 quantifies this.
- **This exact line has already been fully investigated twice, and both times judged "the GPU
  port isn't worth it"**: doc 29 derived a corrected ceiling of ≈1.05–1.09x → doc 32's in-memory
  slab spike briefly measured 1.40x/2.15x → doc 35 wired the real read/join half back in at real
  disk scale and **the result reversed entirely** — both candidates came out slower than the
  shipped pyvips path (0.788x/0.884x), because the read itself is 48.7% of Phase D's own wall,
  and pyvips already overlaps that read with encoding for free — the streamed candidates instead
  pay it serially. **The one lever identified but never built**: pipeline the "streamed read"
  half. doc 35 §3.3 already computed its ceiling: "(Phase D = 1,200.8 s of a 6,211 s wall) gives
  ~1.094× end-to-end." **This is exactly what the user means by "pipelined stitching," and its
  ceiling has already been measured — ~1.09x, not something that turns 2.2x into 4x.** This
  document has to say that plainly up front rather than repackage it as a bigger lever than it is.

### 1.4 Mapping the user's three directions onto what's already known

| user's direction | current state | evidence |
|---|---|---|
| **load balancing** | **Already a dynamic work queue, not static round-robin.** Round 5 (doc 20 §2 Candidate B/D) chose this specifically because "per-tile cost varies enormously and is unknown upfront; static partitioning would leave a worker idle while a sibling is stuck in a dense cluster." `_run_tiles_multiprocess` (`m0_multiprocess.py:250`) uses `ctx.Queue()` + a feeder thread — each worker pulls the next tile only once it's free, which is self-balancing by construction. **No measurement anywhere in this project has attributed the speedup gap to this mechanism** — every round's shortfall has been traced to composition (§1.2) or Phase D (§1.3), never to "a worker sitting idle." | [20 §2 Candidate B](./20-cross-tile-multiprocessing-plan.md), `m0_multiprocess.py:250` |
| **Pipelined Stitching** | **The one genuinely open lever, and the user's intuition is correct** — but its ceiling has already been measured: **~1.094x** (doc 35 §3.3), not the "push to the limit" scale the phrase suggests. Sequenced in §3. | [35 §3.3](./35-round-10-backlog-implementation.md) |
| **more workers (`workers≥5`)** | **A VRAM hardware ceiling, not an algorithm problem.** `workers=4` measured 30,439/32,607 MB (93.3%, ~2.2 GB headroom, round 8) on the real full slide; once round 11's allocator OOM fix was rolled into the default (`expandable_segments:True`, §0), peak framebuffer rose to 92.2% — **headroom got tighter, not looser**. Even ignoring VRAM, the crop-scale marginal-gain curve is already converging: efficiency drops from 103% at `workers=3` to 88% at `workers=4`, and doc 21 §2 states directly, "the knee is N=3." **On this 32 GB card, with this model set (3 CUDA contexts per worker), `workers≥5` is not recommended until per-worker VRAM footprint is actually reduced — not merely until the OOM failure mode is suppressed.** | [27 §6.6](./27-remaining-work-implementation.md), [37 §3](./37-round-11-backlog-implementation.md), [21 §2/§4.1](./21-cross-tile-multiprocessing-implementation.md) |

## 2. Analyze — assembling §1's numbers into one Amdahl account, for the first time

> This section is new arithmetic done in this document, not a citation of an existing document.
> It uses the round-8 `workers=1`/`workers=4` pair (the only matched set on the same code, on the
> same real slide, at both worker counts — rounds 9–11 only re-ran `workers=1`, as flagged in
> §1.1).

**Inputs** (all sourced, all cited in §1):
- `workers=1` wall: 13,762.5 s
- `workers=4` wall: 6,211 s (measured speedup 2.216x)
- Phase D absolute cost: ~1,185–1,199 s, essentially unchanged across worker counts (§1.3)
- match24 (composition-matched crop, Phase D cost negligible): independently measured "pure tile
  parallelism" speedup of 2.138x

**Calculation 1: if Phase D were fully hidden and tile parallelism scaled perfectly 4x, what
would the theoretical ceiling be?**

```
workers=1 parallel portion   = 13,762.5 − 1,185      = 12,577.5 s
ideal workers=4               = 12,577.5 / 4 + 1,185  = 4,329.4 s
ideal speedup                 = 13,762.5 / 4,329.4    = 3.179x
```

**Even with perfect 4x linear scaling of the tile-parallel portion, Phase D alone caps the
ceiling at ~3.18x — it never reaches 4x.**

**Calculation 2: working backward from the measured 6,211 s, how much did the tile-parallel
portion actually scale?**

```
workers=4 parallel portion (backed out) = 6,211 − 1,199         = 5,012 s
actual parallel speedup                  = 12,577.5 / 5,012     = 2.509x
```

This 2.509x is in the same range as the **2.138x** independently measured on match24 (also "pure
tile-parallel speedup with Phase D excluded") — two independent sources (one backed out, one
directly measured) corroborate each other: **the real slide's composition alone caps achievable
tile parallelism at ~2.1x–2.5x, far below the 3.51x measured on the tissue-dense crop.** The
remaining gap between the two (2.51x vs 2.14x) is within the order of magnitude of the VRAM /
allocator pressure difference between full-slide and crop scale (full-slide `workers=4` sits at
93.3% VRAM occupancy vs. the crop's 63%, see the table in §1.4) — worth confirming the next time
`workers=4` is genuinely re-run, but it doesn't change the conclusion's direction.

**Calculation 3: if the one known, already-sized lever (Phase D pipelining, ~1.094x, doc 35
§3.3) were stacked on top of today's measured 2.216x, what does the next realistic step buy?**

```
2.216 × 1.094 ≈ 2.42x
```

**Conclusion (the honest version, and what to tell the user directly)**: on this card, with this
model set, at this real slide's composition, the achievable ceiling for this pipeline is roughly
**~2.5x–2.8x, not 4x** — and the gap isn't "not tuned yet," it's the sum of two already-measured,
independent effects: (a) the real slide is 55.8% background, which caps tile-parallel speedup
itself at ~2.1x–2.5x (the same mechanism that produced 3.51x on the crop, just proportional to
how much work each tile actually does), and (b) Phase D stitching is fully serial, doesn't scale
with `workers`, and even "fully pipelined" only buys ~1.09x more. **Load balancing has no
attributable gap** (§1.4). **`workers≥5` is blocked by VRAM**, not by the scaling curve.

## 3. Plan — next steps, cheapest-first per the playbook

| # | item | cost | known ceiling | why this ordering |
|---|---|---|---|---|
| 0 | Fix the stale "`expandable_segments:True` not yet set as default" line in `bottleneck-list.md` / `19-open-backlog.md` | near zero | n/a (doc maintenance) | HEAD already changed it (§0) — keeping the live-state docs accurate is this folder's own stated discipline (19's "Update discipline" section). Done alongside this document (see the diff below). |
| 1 | **Re-run a clean full-slide `workers=4`** | high (~1.7h+ real full-slide run, needs to run unattended) | n/a — this is a measurement gap, not an optimization | This document's entire Amdahl account in §2 rests on the round-8 (2026-07-27) `workers=4` number; rounds 9–11's three substantial wins (gc re-freeze, tile-read prefetch settling) were only re-confirmed on `workers=1`. **Before investing any new engineering, the current `workers=4` baseline needs to be known.** If the gc/prefetch gains transfer proportionally to `workers=4` (there's no obvious reason they wouldn't — neither is worker-count-dependent), the 2.216x ratio itself may not move much even though both numerator and denominator shrink — but Phase D's share of the new, smaller wall would rise further (§1.3 already shows this trend), which would raise item 2's priority. |
| 2 | **Pipeline Phase D's read** (§1.3's one never-built lever, ceiling already known at ~1.094x) | medium (doc 35 §3.3 already has a spike script to extend from; doesn't require rewriting the stitch logic itself — just overlapping `_stitch_overlay_slide`'s file-open/read of `_stitch_scratch/*.tiff` with the tail of the still-running tile analysis, or at minimum moving that read onto a background thread the way `prefetch_tile_reads` already does elsewhere in this codebase) | **~1.094x end-to-end** (`workers=4`), already sized by doc 35 §3.3, not a new estimate | The only lever with a real, known, unbuilt ceiling; only worth pursuing once step 1 confirms Phase D still accounts for a meaningful share of `workers=4` wall on the current baseline (likely, given the growth trend in §1.3). Before touching pipeline code, run a cheap spike at `scripts/stitch_probe.py` scale first, per this folder's established discipline. |
| 3 | **Cheap synthetic microbenchmark of batch-claiming the dynamic queue** | low (a synthetic microbenchmark in the style of `mp_concurrency_probe.py`, no pipeline code touched) | unknown (never measured — §1.4 notes the real slide is 55.8% background, and each background tile short-circuits fast, so the relative IPC cost of a per-tile queue round-trip at this composition has never been measured) | Cheap, low-risk, worth one measurement to confirm or rule out. **If it comes back at the noise floor (likely — the mechanism is already self-balancing, and no round's shortfall analysis has ever pointed at it), close it out without building anything** — this is the playbook's "measure before optimizing" taken literally: raising it in a conversation isn't license to skip straight to a code change. |
| 4 | **Shrink per-worker VRAM footprint to fit `workers=5`** | high, with real correctness risk (more aggressive quantization/precision changes; this pipeline's own Cellpose checkpoint swap is still pending pathologist sign-off, `19-open-backlog.md` §2) | low (§1.4: the crop-scale marginal-gain curve is already converging between `workers=3` and `workers=4`, 103%→88%; `workers=5` is likely lower still, and the actual blocker is VRAM, not the curve) | **Not recommended without new evidence** — the one item in this document explicitly marked "do not do this." Written down to close the door on it, the same way doc 20 §2 Candidate E records "fork-based reuse" — as a record, not a to-do. |

## 4. Decision gates / stop-loss

- **If step 1's re-run (`workers=4`) comes back materially outside a reasonable noise band around
  2.216x**: stop and find out why before proceeding (most likely explanation: the gc/prefetch
  gains transferred proportionally, which is expected; but if the direction or magnitude is off,
  §2's Amdahl arithmetic needs to be redone with the new number, not carried forward from this
  document as-is).
- **If step 2's cheap spike doesn't come back close to the ~1.09x signal**: apply the same
  discipline as doc 35 §3 — decide by ablation, and if it doesn't clear the bar, close it out and
  record it, rather than building it anyway because the user raised the direction.
- **Any correctness veto**: this document does not change this folder's standing discipline — no
  per-cell correctness vote, no verified fail-fast/sibling-termination behavior, no ship. Every
  lever in §1 already went through this same gate; nothing here is exempt.

## 5. What this document is not

- **Not a record of completed work** — no pipeline code was changed. §3 is a sequenced list of
  next steps, not an announcement.
- **Not reopening already-closed items** — CUDA MPS, the CPU-back-end-only pool, fork-based model
  reuse, cross-tile Cellpose/UNet++ batching, the CUDA-stream/depth-2 redesign, and Phase D's
  GPU-port route all remain closed (§1's table cites each disposition); this document does not
  re-argue any of them.
- **Not a promise of 4x** — §2's conclusion is an honest account of why 4x isn't reachable at this
  composition and this architecture, and what the next step is realistically worth within the
  ceilings already known.
