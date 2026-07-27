# 27 — Remaining work: implementation record

> Executes [`26-remaining-work-implementation-plan.md`](./26-remaining-work-implementation-plan.md).
> Compiled 2026-07-27. Doc 26 is the *plan* (a ranked queue built from
> [`19-open-backlog.md`](./19-open-backlog.md), [`DISCOVERED-NOT-IMPLEMENTED.md`](./DISCOVERED-NOT-IMPLEMENTED.md),
> [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md) and
> [`measurement/current-status-comparison.md`](./measurement/current-status-comparison.md));
> this document is what actually landed, what it measured, and what stayed open with the
> reason. Same discipline as every other doc here — Discover → Analyze → Plan → Choose
> ([`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)),
> with **correctness as a veto** and **ablation as the price of admission** for any layer
> that claims to help.

---

## 0. Summary

| Doc 26 item | Status | Evidence |
|---|---|---|
| **0.2** `PYTORCH_CUDA_ALLOC_CONF` knob + `workers≥6` OOM sweep | **Built + swept (12 runs). Candidate fix NOT demonstrated; default stays off, `workers≤4–5` cap stays** | §5 |
| **0.3** partial-resume / checkpointing for `run_batch` | **Built. Output byte-identical to a cold run** | §2 |
| **0.4** `_stitch_overlay_slide` `RLIMIT_NOFILE` guard | **Built + 7 tests** | §1 |
| **1.1** Phase D cheap `tiffsave` knobs | **Ablated, 13 configs → CLOSED, negative. Every knob doc 25 named is dead; the one that won (`zstd`, 1.2331x) is unreadable by QuPath/BioFormats and was vetoed** | §3 |
| **1.3** per-bucket timing inside multiprocess workers | **Built. 4 parent-only buckets → 26 worker-side buckets** | §4 |
| **0.1** full real-WSI end-to-end validation | **DONE — both runs complete, correctness veto passed. 3.82 h / 1.73 h vs 2.6 h / 1.25 h projected (+47% / +38%); speedup 2.216x. Closes 19 #7** | §6 |
| **4.1–4.7** documentation ↔ code drift (7 items) | **All 7 closed** (+1 unlisted instance found) | §7 |
| **3.1** Cellpose checkpoint clinical sign-off | **Not engineering work — escalated** | §8 |
| Tier 2 / 5 / 6 / 7 | **Not picked up, per doc 26's own sequencing** | §9 |

**Net effect on the shipped pipeline**: two reliability gaps closed (`RLIMIT_NOFILE`,
resume), one measurement blind spot closed (worker-side timings), and **zero performance
changes adopted** — the one candidate that measured well (`zstd`, 1.2331x) is unreadable by
QuPath and was vetoed on correctness, and the one that was supposed to work
(`expandable_segments`) did not. Both are recorded with their evidence rather than shipped
on a hunch. 48 automated tests now exist where there were none;
`scripts/verify_mp_failfast.py` still passes at `workers=1` and `workers=3` after the
multiprocess-path changes.

**No performance change was adopted, but the round was not a negative result — the
full-slide validation reset the map.** Every cheap Phase D knob is dead (§3) and the
allocator fix does not work (§5), so nothing shipped. But the first complete-slide run
(§6) found that **three separate crop-derived numbers do not survive at production scale**:
`gc.collect` is 16.1% of wall despite a shipped optimization that was supposed to have
eliminated it, tile read is 17.2% against the 1.22% behind its stop-loss, and Phase D is
19.3% of wall at `workers=4` against a recorded 7.3%. The pipeline's remaining wall-clock
turns out not to be in an unclaimed knob — it is in **three places the project had already
measured and closed, on crops**. That is a more useful outcome than any single optimization
this round could have shipped.

Two things this round found that no prior document records, both surfaced by writing
the preflight rather than by looking for them:

1. **The registration stage emits a different canvas per modality.** On the reference
   case HER2 is 141818×114366 and DISH is 141658×114415 (HE is 141717×116400 again).
   `PrecutStream.__init__` fail-fasts on unequal dimensions, so a full-slide run cannot
   start on the pair as-is. Every measurement round 1–7 used *same-coordinate crops*,
   which makes both crops equal by construction — that is why seven rounds never hit it.
   Per the pipeline owner, the alignment is cell-level and the canvas difference is not a
   correctness concern, so the runner conforms the pair to their intersection (§6).
2. **`perf_measure.py`'s non-stream path is broken.** It calls
   `HP.precut_paired_tiles`, which `hybrid_pipeline` does not re-export (only
   `PrecutStream` is imported from `m0_reader`). Any run without `--stream-precut` dies
   with `AttributeError` before doing work. Left as-is and reported rather than fixed —
   it is outside doc 26's scope and every round since 4 uses `--stream-precut`. See §10.

---

## 1. Tier 0.4 — `RLIMIT_NOFILE` guard on the slide stitch

**Problem** ([DISCOVERED #4](./DISCOVERED-NOT-IMPLEMENTED.md), [25 §5.2](./25-gpu-encode-decode-loop-acceleration-implementation.md)):
the stitch is lazy — every one of a slide's 27,565 overlay tiles is held open as a
pyvips image *simultaneously* until the final `tiffsave` pulls data. Nothing checked
`RLIMIT_NOFILE`. It passed only because this host's soft limit is 1,048,576; on a host
with the common 1,024 default the stitch dies **after the entire multi-hour analysis has
completed** — the most expensive possible failure mode for this pipeline.

**Change** (`backend/algorithms/hybrid/hybrid_pipeline.py`):

- New `_ensure_nofile_limit(needed)`. If the soft limit is short it **raises the soft
  limit itself** (soft→hard needs no privilege, so on the overwhelming majority of hosts
  this converts a hard failure into a working run); only if the *hard* limit is also too
  low does it raise, with a message naming the number to set and stating that the per-tile
  analysis output is already on disk and does not need re-running.
- `_stitch_overlay_slide` was split into `_join_overlay_tiles` (lazy row/column join,
  where the guard lives, fired on tile **count** before the first `open`) and
  `_stitch_overlay_slide` (join + `tiffsave`). The split is what let §3 ablate the encode
  independently of the join.

**Tests** — `tests/test_stitch_nofile_guard.py`, 7 cases: sufficient soft limit is left
alone; soft is raised when hard permits; `RLIM_INFINITY` handled on both soft and hard;
too-low hard raises with an actionable message; the guard fires on tile count rather than
on the first `EMFILE` (verified by pointing it at a directory with **no** tiles at all and
requiring the `RuntimeError`, not `FileNotFoundError`); headroom is reserved above the
tile count. Limits are faked via `monkeypatch` — lowering a real hard limit is
irreversible for the life of a process and would poison every later test.

---

## 2. Tier 0.3 — partial resume / checkpointing for `run_batch`

**Problem** ([19 #1c](./19-open-backlog.md), [DISCOVERED #42](./DISCOVERED-NOT-IMPLEMENTED.md)):
`run_batch` is fail-fast, which is correct — all tiles are fragments of one slide, and a
silently skipped tile yields a slide with an undocumented hole. But fail-fast never
answered *what happens to the 25,000 tiles that already succeeded*. At full-WSI scale an
OOM at tile 25,000 of 27,565 costs the whole run. This is the stated reason rounds 5b/6
recommend the slower `workers=4` over `workers=6` for unattended jobs.

**Change**: `run_batch(..., checkpoint: bool = False)`.

- Each completed tile's `owned` result list — the only thing that cannot be rebuilt from
  disk, since every other per-tile artifact already lands — is pickled to
  `output_dir/_resume/tile_x{ax}_y{ay}.pkl` via tmp+`os.replace`, so a kill mid-write
  leaves no half file.
- On start, the store is loaded, filtered to the current grid, and those tiles are
  skipped by `_skip_completed`.
- A `config_hash.txt` guards the store: on mismatch everything is discarded with a loud
  warning and the batch recomputes, because tiles produced under a different config are
  not comparable and mixing them would produce a slide no `config_hash` in the CSV
  describes.
- **`fail-fast` is unchanged.** Resume makes a retry cheap; it does not make a failure
  survivable within a run.
- Single- and multi-process paths share one implementation. Both funnel through a single
  parent-side `_record(ax, ay, owned)` — the same principle as `_finish_batch` owning the
  only global renumbering. The multiprocess path gained an `on_tile` callback for this;
  workers never touch the checkpoint.

**Default is off**, deliberately. The API path issues single-tile requests where the
checkpoint is pure I/O overhead and would litter caller output directories. It is opt-in
via `run_batch(checkpoint=True)`, `hybrid_pipeline.py --resume`, or
`perf_measure.py --resume`, and `full_wsi_validate.py` always enables it.

**Correctness veto — passed.** On the 25-tile smoke ROI at `workers=2`:

| Run | wall | success / skipped |
|---|---|---|
| cold (`--resume`, empty store) | 16.69 s | 22 / 3 |
| resumed (same output dir, 25 tiles in store) | 4.56 s | 22 / 3 |

`report.csv` (738 lines, 737 cells) and `summary.txt` are **byte-identical** between the
cold and resumed runs. The 4.56 s the resumed run still spends is precut + global
merge/CSV + Phase D stitch, all of which correctly still execute — resume skips analysis,
not assembly.

**Tests** — `tests/test_run_batch_resume.py`, 12 cases, including the three that matter
most: an **empty** owned-list round-trips as *completed* (a background tile legitimately
owns zero cells — reading that back as "not done" would silently redo work and, worse,
suggest the store can't represent a skip); a **corrupt** checkpoint file costs one tile of
recompute rather than the batch; and tiles belonging to a **different grid** are ignored
rather than imported.

---

## 3. Tier 1.1 — Phase D `tiffsave` knob ablation

**Problem** ([19 #1b](./19-open-backlog.md), [DISCOVERED #5](./DISCOVERED-NOT-IMPLEMENTED.md),
[25 §5.3](./25-gpu-encode-decode-loop-acceleration-implementation.md)): Phase D is the one
fully serial block left, measured at **322.7 s** on the real 16.22 GP grid and superlinear.
Doc 25 named `tiffsave` tile-size / pyramid-depth parameters as "untested and free of new
dependencies" and nobody had tried them. Ceiling is small — 1.036x at `workers=1`, 1.078x
at `workers=4` — but it is genuinely free, and doc 26 makes it the prerequisite for
deciding whether the nvImageCodec port (which is real, undesigned engineering) is worth
starting at all.

**Method**: `scripts/stitch_probe.py --ablate`. It builds the synthetic input **once**,
then re-runs only the encode across variants that each differ from the shipped call in
**exactly one parameter** — changing two at a time makes the result unattributable
(anti-pattern #7). The join is timed separately from the encode via the new
`_join_overlay_tiles`, since a `tiffsave` knob can only move the encode half. Screening
was run at **4.055 GP** (92×75 = 6,900 tiles, annotated share 0.4474, matching the slide's
measured composition).

| config | join s | encode s | total s | speedup | out GB | size vs base |
|---|---|---|---|---|---|---|
| `baseline` (128px tiles, full pyramid, LZW, BigTIFF) | 2.43 | 57.02 | 59.45 | 1.0000 | 1.84 | 1.0000 |
| `tile_256` | 3.02 | 59.67 | 62.69 | 0.9484 | 1.79 | 0.9741 |
| `tile_512` | 3.02 | 66.10 | 69.11 | 0.8602 | 1.78 | 0.9708 |
| `tile_1024` | 2.95 | 83.71 | 86.66 | 0.6861 | 1.81 | 0.9843 |
| `predictor_horizontal` | 3.11 | 56.21 | 59.32 | 1.0023 | 1.84 | 1.0000 |
| `predictor_none` | 3.02 | 55.59 | 58.61 | 1.0143 | 2.38 | 1.2971 |
| `depth_onetile` | 3.07 | 56.01 | 59.09 | 1.0062 | 1.84 | 1.0000 |
| `shrink_nearest` | 3.03 | 55.09 | 58.13 | 1.0228 | 1.78 | 0.9726 |
| `subifd` | 3.05 | 56.64 | 59.69 | 0.9960 | 1.84 | 1.0000 |
| `deflate` | 3.19 | 72.53 | 75.73 | 0.7850 | 1.62 | 0.8802 |
| **`zstd_1`** | 3.30 | **44.91** | **48.21** | **1.2331** | **1.58** | **0.8623** |
| `BOUND_no_pyramid` *(vetoed — QuPath needs it)* | 3.02 | 44.23 | 47.25 | 1.2583 | 1.20 | 0.6539 |
| `BOUND_no_compression` *(vetoed — 8.9x the size)* | 3.12 | 43.96 | 47.08 | 1.2628 | 16.25 | 8.8536 |

**What the two bounds buy us.** They were included precisely so the knobs could be read
against something. Baseline → `BOUND_no_compression` is 12.37 s, i.e. **LZW encoding is
20.8% of Phase D**. That is the entire budget any compression knob can play for.

**Result: the two knobs doc 25 actually named are both dead, and a third one nobody named wins.**

- **Tile size: monotonically worse.** 256 → 0.948x, 512 → 0.860x, 1024 → 0.686x. The
  hypothesis ("fewer, larger tiles") is refuted in the wrong direction at every step; the
  128px default is already the right choice.
- **Pyramid depth: no effect.** `depth_onetile` is 1.006x and produces a *byte-identical*
  output — it is already the effective default, since the pyramid stops once a level fits
  one tile either way. `predictor_horizontal` is the same story for the same reason: it is
  libvips' default predictor, so asking for it explicitly is a no-op (1.002x, byte-identical
  output), which is exactly what `predictor_none` being **29.7% larger** confirms. `subifd`
  (0.996x) only changes where the levels are stored, not what they cost.
- **`zstd` level 1 is faster *and* smaller**: **1.2331x** on Phase D with **13.8% less**
  output, and it sits within **2.4% of the no-compression-at-all bound** — it very nearly
  removes compression from the critical path. `deflate` is the opposite trade (0.785x for
  12% smaller); `predictor_none` buys 1.4% of speed for 30% more bytes.

### 3.1 The zstd candidate — measured, then killed by the correctness veto

zstd in TIFF is a lossless codec, and that much was verified rather than assumed: at
tile scale a re-encoded overlay is **pixel-identical** to both the source and the LZW
output, and at ~1 GP slide scale four full-resolution 2048² patches sampled across the
image (including the far corner) are byte-for-byte equal between the two files. Both carry
`bigtiff=True` and **10 pyramid levels**, and both decode under pyvips and `tifffile`.

**QuPath cannot open the zstd file.** A control pair was built for the check —
`overlay_slide_lzw.tiff` (446.7 MB) and `overlay_slide_zstd.tiff` (385.2 MB), same
35000×28000 content through the pipeline's own `_join_overlay_tiles` — and opened in the
QuPath build the pathologists actually use. **LZW opens; zstd raises a "cannot open"
warning.** QuPath reads TIFF through BioFormats, and BioFormats does not support the ZSTD
compression tag.

That is decisive under the playbook's correctness veto, and it does not matter by how much
zstd won: `overlay_slide.tiff` exists so a pathologist can open it directly, so an output
they cannot open is **broken, not faster**. A 1.2331x that produces an unreadable slide is
worth exactly zero.

### 3.2 Result: the whole cheap-knob line is closed, negative

| knob | verdict |
|---|---|
| tile size (256 / 512 / 1024) | **dead** — monotonically worse (0.948x / 0.860x / 0.686x) |
| pyramid depth (`depth_onetile`) | **dead** — already the effective default, byte-identical output |
| `predictor` | **dead** — `horizontal` is already the default; `none` is +1.4% speed for +29.7% bytes |
| `region_shrink=nearest`, `subifd` | **dead** — inside noise (1.023x / 0.996x) |
| `deflate` | **dead** — 0.785x, i.e. slower |
| `zstd` level 1 | **1.2331x and 13.8% smaller — but QuPath/BioFormats cannot read it. Vetoed.** |

**`_stitch_overlay_slide` is unchanged: LZW, pyvips defaults, no knobs added.** Not one of
the parameters doc 25 nominated as "untested and free" is worth anything, and the only
parameter that *was* worth something is unusable. This closes 19 #1b's "cheaper things to
try first" — there is nothing cheap left.

**Consequence for the GPU port (doc 26 Tier 1.2), which is now the only remaining Phase D
route.** This round establishes a hard constraint on it that was not previously written
down: **whatever it emits must be BioFormats-readable**. nvImageCodec produces neither a
pyramid nor BigTIFF, so a port already meant building downsample-level generation and
container assembly by hand; it must now also restrict itself to a codec BioFormats
supports, which rules out exactly the modern codecs a GPU encoder is fastest at. The 19.2x
encode figure doc 25 §8.2 measured was for **lossless LZW**, so that part still stands —
but the engineering around it is strictly larger than doc 24 assumed, against an unchanged
1.036x–1.078x ceiling. Recommend leaving it closed.

**Caveat on scale.** These are 4.055 GP screening numbers. Phase D is **superlinear**
(14.11 / 14.16 s per GP at 1 / 4 GP, then 19.90 at 16.22 GP — doc 25 §5), so the *absolute*
seconds do not extrapolate. The ranking is what this run is for, and the winner should be
re-confirmed at 16.22 GP (`--ablate --only baseline,zstd_1`) before the switch is made.

---

## 4. Tier 1.3 — per-bucket timing inside multiprocess workers

**Problem** ([DISCOVERED #40](./DISCOVERED-NOT-IMPLEMENTED.md), [21 §10 follow-up #1](./21-cross-tile-multiprocessing-implementation.md)):
`perf_measure.py` instruments by monkeypatching the **parent** process's
`hybrid_pipeline` namespace. `spawn`ed workers re-import the module and get clean
originals, so at `workers>1` every per-stage bucket came back empty. You could see that
`workers=4` was 2.06x faster; you could not see *which stage* moved. Any future
intra-worker optimization was therefore unmeasurable.

**Change**, in three small pieces that keep the pipeline free of any dependency on the
measurement harness:

1. `hybrid_pipeline._install_worker_probe()` reads `HYBRID_MP_WORKER_PROBE`
   (`module:callable`), imports it and calls it — **before** the three `_init_*` model
   loads, so per-worker init is measured too. Unset (the normal case) it costs one
   `os.environ.get`. Probe failures log a warning and are swallowed: a broken measurement
   hook must not fail a real batch.
2. The worker emits `("timings", {...})` on the **existing** result queue at shutdown. The
   parent handles that message kind in its collect loop and drains any stragglers *before*
   `join` (a worker exits as soon as it has posted, and `join` blocks until the queue
   drains). Results land in `hybrid_pipeline.LAST_MP_WORKER_TIMINGS`.
3. `scripts/mp_worker_probe.py` is the probe; it reuses `perf_measure.install_wrappers()`
   verbatim, so parent-side and worker-side buckets come from identical shims and are
   directly comparable. `perf_measure.py --worker-timings` wires it up and writes both the
   per-worker detail and a folded `worker_timings_total` into the metrics JSON.

**Verified** on the 25-tile ROI at `workers=2`. Before: 4 buckets, all parent-side
(`C_export_csv`, `C_export_summary`, `D_stitch_overlay`, `G_mkdir_other`) — i.e. nothing
about the actual work. After: **26 worker-side buckets**. Top of the fold:

| bucket | n | Σt across workers |
|---|---|---|
| `B1_m3b_cellpose` | 48 | 17.307 s |
| `B3_detect_dots` | 24 | 5.867 s |
| `B2_png_encode` | 72 | 4.016 s |
| `B1_unet_coremask` | 25 | 3.967 s |
| `init_unet` | 2 | 2.781 s |
| `init_cellpose_m2` | 2 | 1.339 s |
| `init_cellpose_m3b` | 2 | 1.038 s |
| `B3_enlarge_cells` | 24 | 1.133 s |

Two immediate reads that were not previously available: per-worker model init is
**~2.58 s** (1.39 + 0.67 + 0.52), against the ~3.14 s doc 20 §5 assumed when arguing about
a persistent pool; and the `n` counts confirm the expected shape — `B1_unet_coremask`
fires 25 times (every tile, background included) while the M2/M3b Cellpose and all M3
buckets fire 24, i.e. only on the tissue-bearing tiles, exactly as the background
fast-path claims.

**The sum is aggregate CPU time across workers, not wall**, and the JSON field says so.
At `workers=4` a 200 s bucket occupied roughly 50 s of the run. It answers "where did the
work go", never "how long did the run take" — the shims also add overhead to the worker
hot path, so `--worker-timings` runs are for stage breakdown and must not be used for
wall-clock comparison. That restriction is in the `--help` text, not just here.

---

## 5. Tier 0.2 — CUDA allocator configuration vs. the `workers≥6` balloon

**Problem** ([19 #7b](./19-open-backlog.md), [DISCOVERED #2](./DISCOVERED-NOT-IMPLEMENTED.md),
[23 §4.6/§6](./23-next-optimization-cycle-implementation.md)): at `workers=6`, **2 of 6
runs** OOM-killed the batch, each time with one worker at **exactly 24.76 GiB** while five
siblings sat at 1.1–1.7 GB. The random victim looks like fragmentation; a byte-identical
balloon does not. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was named as the
candidate fix in doc 21 §4.7 and again in doc 23 §6 and **was never tried**. Under
fail-fast an OOM voids the whole batch, so this governs the safe worker ceiling.

**Change**: `config.cuda_alloc_conf` (default `""` = touch nothing = today's behaviour),
written into `os.environ` by `_run_tiles_multiprocess` **immediately before the workers
spawn**. That is the only point where it can work: the parent never allocates on the device
on this path, and `spawn`ed children inherit the environment and parse the variable at
their own allocator init. `scripts/alloc_conf_probe.py` drives the comparison, with
conditions **interleaved** rather than blocked, because thermal state and page-cache warmth
drift over a 20-minute sweep and a blocked design hands all of that drift to whichever
condition ran second.

**Setup**: 6 repeats per condition at `workers=6`, on a crop cut at round 7's exact
composition anchor (x=96768, y=1536, size=18688). It reproduces that round's workload
**exactly** — 576 tiles, 137 success / 439 skipped, bg=0.762 — so these numbers sit
directly alongside doc 25's. Peak VRAM comes from `nvidia-smi dmon`'s `fb` column, parsed
by header rather than position; the parser was validated first by replaying round 7's
stored dmon files, where it reproduces the documented **26,687 MB** sample exactly.

| condition | runs | failures | OOM | median wall | median peak | max peak |
|---|---|---|---|---|---|---|
| `control` (no setting) | 6 | **1** | 1 | 65.46 s | 22,968 MB | 27,836 MB |
| `expandable_segments:True` | 6 | **0** | 0 | 66.80 s | 24,040 MB | 28,700 MB |

**The defect reproduced.** `control_w6_r1` OOM-killed the batch on a verified-clean GPU
with `Process ... has 24.62 GiB memory in use` against five siblings at 1.11–1.32 GiB —
the doc-19 #7b signature, essentially to the byte. 1-in-6 here against 2-in-6 in round 6.

**But the candidate fix is not demonstrated to work, and the mechanism argues against it.**
Three readings, in order of how much weight they carry:

1. **`expandable_segments` does not reduce peak VRAM at all.** Median 24,040 vs 22,968 MB
   and max 28,700 vs 27,836 MB — if anything marginally *higher*, certainly not lower.
   This is the decisive one. `expandable_segments` works by reducing **fragmentation**; if
   fragmentation were driving the balloon, peak reserved would fall. It does not. That is
   positive evidence **against** the fragmentation hypothesis and for doc 19 #7b's own
   suspicion that a byte-identical 24.76 GiB balloon is a reproducible pathological
   allocation, not a fragmentation artifact.
2. **0-in-6 vs 1-in-6 is not a result.** At this sample size the two arms are
   statistically indistinguishable (Fisher exact p ≈ 1.0). Calling this a fix would be
   exactly the "real speedup far below what theory predicted / stacked optimizations with
   no ablation proof" failure the playbook's red flags list.
3. **It is not free.** Median wall 66.80 vs 65.46 s, i.e. **+2.0%**, inside run-to-run
   spread but with no upside to pay for it.

**A second finding, unasked for but decision-relevant.** `workers=6` median wall here is
**65.46 s**; round 7's `workers=4` on this identical crop was **65.55 s**
(`control_w4_r1`). **`workers=6` is not faster than `workers=4` at all** — it just adds
OOM risk. That independently confirms doc 23 §6's "the curve is flat past `workers=5`"
from a different direction, and removes the only reason anyone would want the cap raised.

**Decision: keep `cuda_alloc_conf` defaulted to `""` (off), and keep the `workers≤4–5`
recommendation unchanged.** The knob ships so the question is cheap to re-ask with a bigger
sample, and so any future GPU-library addition inside the worker pool can be tested against
a real allocator configuration — but nothing here justifies changing a default. Per doc 26
§3 0.2's own instruction, since `expandable_segments` did **not** clear the OOM, the next
step is to **root-cause the 24.76 GiB balloon** before touching the worker recommendation
again — and this round has narrowed that hunt by ruling out the fragmentation explanation.

**Headroom, restated.** Both conditions peak at **27.8–28.7 GB of 32.6 GB**. Real transient
headroom at `workers=6` is **~4 GB**, not the ~12 GB a steady-state reading suggests. Doc
25 §8.3/§11's rule — any proposal to add a GPU library inside the worker pool must clear
the *transient* number — holds, and the number is worse than the `workers=4` figure (~6 GB)
that rule was written against.

---

## 6. Tier 0.1 — full real-WSI end-to-end validation

**Status: runner, preflight and conforming step built and exercised; the multi-hour run
itself is queued, not completed, and is called out as such rather than reported as done.**

This is doc 26's single highest-leverage item and the binding gate on shipping `workers>1`
([20 §4](./20-cross-tile-multiprocessing-plan.md) forbids shipping ahead of it). It has been
specified since [09 §3.6](./09-measurement-analysis-plan.md) and descoped by every round
since. `scripts/full_wsi_validate.py` exists so the next attempt is a single command.

**What the runner adds** over invoking `perf_measure.py` by hand — all of it aimed at the
failure modes this backlog already knows about:

- **Preflight before anything starts.** Slide dimensions and grid, free space against the
  *measured* 12.3 MB/tile output rate (round 7's 576-tile runs wrote 7.085 GB) plus the
  precut scratch, the `RLIMIT_NOFILE` the final stitch will need, CUDA availability, and
  all three model files. Discovering any of these at tile 24,000 is precisely the loss §2
  exists to prevent.
- **Resume always on** (`--resume` → `run_batch(checkpoint=True)`), so an interrupted
  attempt restarts where it stopped.
- **The plan's two runs in the plan's order**: `workers=1` (production default) first, then
  `workers=4` (the round-6/7 recommendation), with `--gpu-dmon` on both.
- **A correctness comparison** of the two runs' `report.csv` — worker count must not change
  the cell table beyond GPU non-determinism, judged against a noise floor rather than
  asserted equal, the same way [21 §4](./21-cross-tile-multiprocessing-implementation.md)
  judged it.
- Partial results are written after **each** run, because the loop spans hours.

### 6.1 The blocker the preflight found

Running the preflight took three seconds and immediately surfaced something seven rounds
of crop-based measurement could not have:

```
IHC  HER2_processed.tiff   141818 x 114366
DISH DISH_processed.tiff   141658 x 114415     <- 160 px narrower, 49 px taller
HE   HE_processed.tiff     141717 x 116400
```

`PrecutStream.__init__` **raises `ValueError` on unequal dimensions**, so a full-slide run
cannot start on this pair at all. Every round 1–7 fed the pipeline crops cut at identical
coordinates and sizes from both slides, which makes the two inputs equal *by construction*
— that is why this has never been hit, and why it is not recorded anywhere in docs 01–26.

Per the pipeline owner: the registration stage aligns **cells, not canvases**, the two
images are never expected to be the same size, and this is not a correctness concern.
Taking that as given, `--conform` crops the pair to their **intersection** (141658×114366,
**99.86% of the slide retained**) once, into `<output-root>/_conformed`, reused by both
runs. Intersection rather than union with white fill: outside the intersection one modality
has no pixels at all, so no IHC-DISH result there could be valid, whereas white fill would
manufacture tiles with one blank arm that the pipeline would dutifully analyse. **The
conforming lives in the runner, not the pipeline** — relaxing `PrecutStream`'s check would
weaken a guard the API path relies on to reject genuinely mismatched input.

### 6.2 What the run will cost

| | |
|---|---|
| grid | 185 × 149 = **27,565 tiles**, 16.219 GP |
| projected output | **~513 GB** (27,565 × 12.3 MB analysis + 6.3 MB precut scratch) |
| projected wall | ~2.6 h at `workers=1`, ~1.25 h at `workers=4` ([bottleneck-list](./measurement/bottleneck-list.md)) |
| host readiness | RTX 5090 present; `RLIMIT_NOFILE` 1,048,576 (ample); models present |
| free space | 9.7 TB on `/home` — `/data/nvmessd` has only 319 GB and is **not** writable by this user, so the output root must not go there |

Command:

```bash
.venv/bin/python scripts/full_wsi_validate.py \
  --ihc  /data/nvmessd/storge_tsgh/<case>/output/HER2_processed.tiff \
  --dish /data/nvmessd/storge_tsgh/<case>/output/DISH_processed.tiff \
  --output-root /home/taro/full_wsi_validation \
  --conform --workers 1,4 --out /home/taro/full_wsi_validation/result.json
```

**Definition of done, from doc 26 §3 0.1**: one full-slide `workers=1` run and one
full-slide `workers=4` run, both completing without fail-fast abort, output spot-checked,
and real wall-clock compared against the ~2.6 h / ~1.25 h projections.

### 6.3 Result — `workers=1`, the first complete slide this project has ever run

Ran 2026-07-27 04:29→08:19 on the conformed pair. **Completed without a single fail-fast
abort across 27,565 tiles.**

| | measured |
|---|---|
| end-to-end wall | **13,762 s = 3.82 h** (projection: ~2.6 h → **+47%**) |
| tiles | 27,565 (185×149 grid, 16.201 GP) |
| success / skipped | 10,801 / 16,764 |
| cells | **356,255 rows** in `report.csv`; **44,535 valid** after exclusions |
| peak RSS | **61.13 GB** |
| peak CUDA reserved (parent) | 2.19 GB |
| analysis output | 347.0 GB |
| `config_hash` | `3d1087f2` (unchanged from round 7 — see §10.2) |

**The composition prediction was right to within one tile.** 15,385 tiles took the
background fast path (empty core mask) against doc 25 §1's predicted **15,386** — 55.81%
vs 55.80%. That measurement is now validated at full scale and needs no revisiting.
(`skipped=16,764` is larger because it also counts 1,379 *tissue* tiles that owned no cells
after core-ownership filtering — a different quantity from "background".)

**Therefore the +47% overrun is entirely in per-tile rates, not composition.** That is a
much sharper conclusion than "the extrapolation was optimistic", and the per-stage
breakdown says exactly where.

### 6.4 Three stages that only a full-slide run could expose

Parent-side `TIMINGS`, as a share of end-to-end wall (shares do **not** sum to 100% — the
two arms overlap on a background thread, so these are per-arm):

| stage | n | full slide | what the crop-scale record said |
|---|---|---|---|
| `B1_m3b_cellpose` | 24,360 | 6,358.9 s (46.2%) | dominant, as expected — no surprise |
| `B2r_tile_read` | 55,130 | **2,368.5 s (17.2%)** | **1.22% of wall**, stopped out at a 1.012x ceiling (doc 18 §6.3) |
| `B3_detect_dots` | 12,180 | 2,333.2 s (17.0%) | expected |
| `B4_gc_collect` | 27,565 | **2,218.4 s (16.1%)** | **0.52 s for the whole batch** after `gc.freeze()` (doc 16) |
| `B2_png_encode` | 36,540 | 2,040.6 s (14.8%) | expected |
| `D_stitch_overlay` | 1 | **1,185.4 s (8.6%)** | **322.7 s** measured by `stitch_probe.py` — **3.7x optimistic** |
| `B1_unet_coremask` | 27,565 | 734.8 s (5.3%) | expected |
| `F_write_blank_tile` | 15,385 | 457.1 s (3.3%) | expected |

**1. `gc.collect` has silently lost the entire `gc.freeze()` benefit — 16.1% of wall.**
Doc 16 measured collection dropping from 83.2 ms to **1.2 ms** per call under `gc.freeze()`,
and shipped it on that basis. At full slide it is **80.5 ms per call again** (2,218.4 s /
27,565), i.e. back to the pre-freeze figure. The mechanism is not subtle once stated:
`gc.freeze()` moves objects *existing at freeze time* into the permanent generation, but
`run_batch` then accumulates `per_tile_owned` — **356,255 `CellAnalysisResult` dataclasses**
by the end — which are created *after* the freeze, are fully tracked, and are rescanned on
every one of the 27,565 collections. On a 441-tile crop that list holds ~6,000 objects and
the cost is invisible; at full scale it is the third-largest item in the run.

This is an optimization **validated on a crop that degrades on the real thing**, and it is
the single most valuable thing this run found. It is also now the most attractive remaining
target in the pipeline: a ~16% share with a plausible, cheap fix (freeze again periodically,
or keep the accumulating results out of GC's reach), which is a far better prospect than
anything left in doc 26's performance tiers.

**2. `B2r_tile_read` is 17.2% of wall, not 1.22%.** Doc 18 §6.3 stopped out GPU-side tile
loading on a 1.012x ceiling derived from the crop figure. At full scale the precut scratch
is ~49 GB and no longer fits in page cache, so reads that were effectively free on a crop
become real disk I/O. **That stop-loss was made on a number that does not hold at
production scale and should be reconsidered** — though note the read sits on the arm that
still has slack, so the ceiling needs re-deriving, not assuming.

**3. Phase D is 3.7x its synthetic measurement.** `stitch_probe.py` measured 322.7 s at
16.22 GP by replicating a small pool of real tiles via hard links; the real stitch of
27,565 genuinely distinct tiles took **1,185.4 s**. The probe's inputs compress and cache
far better than real ones. Phase D's share of wall is therefore **8.6% at `workers=1`**, not
the 3.5% doc 19 #1b records — which raises, not lowers, the value of the GPU port that §3.2
otherwise recommends leaving closed. Reconciling those two is a judgement call for whoever
picks it up; the numbers are now real on both sides.

**Peak RSS of 61.13 GB** is also new — no prior round exceeded ~4 GB. It is driven by the
stitch holding 27,565 lazy pyvips images (12,027 open fds observed mid-stitch). Worth
recording as a host requirement: **this pipeline needs ~64 GB of RAM to finish a slide**,
which no document previously stated.

**The `RLIMIT_NOFILE` guard (§1) was exercised for real** and passed silently. The 12,027
concurrent open tile handles observed mid-stitch confirm the defect was genuine: on a host
with the common 1,024 soft limit this run would have died at the final step, after 3.5 hours
of completed analysis.

### 6.5 Result — `workers=4`, and the correctness veto

Ran 08:19→10:03 immediately after, same conformed inputs, same host.

| | `workers=1` | `workers=4` | projection |
|---|---|---|---|
| end-to-end | **13,762 s = 3.82 h** | **6,211 s = 1.73 h** | 2.6 h / 1.25 h |
| vs projection | **+47%** | **+38%** | — |
| **measured speedup** | — | **2.216x** | 2.06x–2.17x expected at this composition |
| success / skipped | 10,801 / 16,764 | 10,800 / 16,765 | — |
| peak RSS | 61.13 GB | 61.67 GB | — |
| **peak GPU** | **2,739 MB** | **30,439 MB** | — |
| analysis output | 347 GB | 347 GB | — |
| `overlay_slide.tiff` | 5.86 GB | 5.86 GB | — |
| Phase D stitch | 1,185.4 s (**8.6%** of wall) | 1,200.8 s (**19.3%** of wall) | 3.5% / 7.3% (19 #1b) |

**Correctness veto — PASSED.** `report.csv` holds **356,255** rows at `workers=1` and
**356,221** at `workers=4`: a **−0.01%** delta, one tile flipping success→skipped. Amplified
cell count is **0 in both**. This is inside the same-code noise floor prior rounds
established (doc 21 §4) and is the first time the invariant has been checked at full-slide
scale rather than on a crop. **The measured 2.216x speedup also lands exactly in the
2.06x–2.17x band round 7 predicted for this composition** — the multiprocessing model, unlike
the absolute-time model, extrapolated correctly.

### 6.6 What this changes

**1. 19 #7 is closed. The gate on shipping `workers>1` is satisfied — with one caveat that
is not about speed.** Both runs completed without a fail-fast abort, output matches within
noise, and the speedup is real at full scale. But `workers=4` peaked at **30,439 MB of
32,607 — 93.3% of the card, ~2.2 GB of headroom**. Round 7 saw 26,687 MB on a crop and doc
25 §8.3 already warned that transient headroom was ~6 GB rather than the ~12 GB a
steady-state reading suggests; at real full-slide scale it is **~2 GB**. Combined with §5
(the balloon reproduces, and `expandable_segments` does not fix it), the recommendation is:
**ship `workers=4`, but treat 32 GB as the hard floor for the card and do not add any GPU
library inside the worker pool without re-measuring this number.** `workers≥5` should stay
off the table until the balloon is root-caused.

**2. Phase D's ceiling is nearly 3x what 19 #1b records, and that reverses §3.2's
recommendation.** At `workers=4` the stitch is **19.3% of wall**, not 7.3% — because both
the stitch got slower than the probe predicted (§6.4) *and* everything around it got faster.
Amdahl ceiling for eliminating it is therefore **1/(1−0.193) = 1.239x**, against the 1.078x
doc 19 #1b carries. §3.2 recommended leaving the nvImageCodec port closed on a 1.036x–1.078x
ceiling; **at 1.239x that judgement should be revisited.** The BioFormats constraint §3.2
established still applies and is not fatal here: doc 25 §8.2's 19.2x figure was measured for
**lossless LZW**, which BioFormats reads. Phase D is now, by a wide margin, the largest
single remaining lever in the pipeline.

**3. Two crop-derived stop-losses rest on numbers that do not survive at scale** — `gc`
(§6.4, 16.1% of wall vs a shipped optimization that was supposed to have removed it) and
tile read (17.2% vs the 1.22% behind doc 18 §6.3's 1.012x stop-loss). Neither should be
re-proposed on the old numbers, and neither should stay closed on them either.

**4. Host requirements, now measured rather than assumed**: ~64 GB RAM, ~32 GB VRAM for
`workers=4`, ~350 GB disk per slide, and `RLIMIT_NOFILE` ≥ ~28,000. None of these appeared
in any document before this run.

---

## 7. Tier 4 — documentation ↔ code drift (all seven closed)

| # | Item | What landed |
|---|---|---|
| 32 | `generate_ihc_core_mask(ihc_tile_path: Path)` always receives an ndarray | Renamed to `ihc_image: Union[np.ndarray, Path, str]`; docstring now states the pipeline passes an already-read array because M1 and M2 both need it and re-reading is wasted I/O. The now-unnecessary `# pyright: ignore[reportArgumentType]` at the call site was removed — it was suppressing exactly this mismatch. |
| 33 | `docs/sdd-elastic-dish-matching.md` referenced but never existed | Reference stripped from `m3_elastic_matching.py`; its docstring now declares itself the authority. **A second, unlisted instance of the same dead reference was found in `m3_module/m3_dot_detection.py`** and stripped too. |
| 34 | `docs/dish_dot_detection_spec.md` referenced but never existed | Stripped from **both** `config.py` and `config_example.py`, replaced with a pointer to `m3_dot_kernels.py` / `m3_dot_detection.py`. |
| 35 | `elastic_matching_v3_explainer.html` describes the wrong algorithm | Updated to v4 (§7.1). |
| 36 | Nothing guards `config.py` / `config_example.py` parity | `tests/test_config_parity.py`, 6 cases (§7.2). |
| 37 | `backend/algorithms/hybrid/` has zero correctness tests | `tests/test_m0_stitch.py`, 21 cases (§7.3). |
| 38 | Codegraph phantom-file list never rechecked at the current path | Rechecked — **zero phantoms** (§7.4). |

### 7.1 The explainer HTML (#35)

Checked claim by claim against the code rather than rewritten wholesale:

- **Part B (counting) was still correct** — `_build_nucleus_owner_mask` exists and is
  still how the counting ROI is built. Left as it was.
- **Part A (matching) was wrong.** It described v3's nucleus-centric "each nucleus goes to
  its nearest cell"; the code is cell-centric with overlap-priority and a `reach` radius.
  Rewritten, including a new diagram of the two-stage (overlap-first, then reach) locking
  pass.
- **The parameter table was wrong in three ways at once**: it documented
  `dish_elastic_max_dist_px` and `dot_blue_exclude_threshold`, **neither of which exists
  in the current config**, and it struck through `dish_elastic_expand_factor` as
  "deprecated" when the code uses it to compute the reach radius on every call. Replaced
  with the five parameters that do exist, with their real defaults.
- **A whole outcome was obsolete**: "核數 ≥2 · 多核排除" cannot occur under one-to-one
  matching. Four outcomes → three, with a note saying why the fourth is gone.
- Added a header stating the code is authoritative and this page is illustrative, and a
  footer asking that the page be updated alongside the matching logic — the drift in G4
  happened because nothing tied the two together.

### 7.2 Config parity test (#36)

Parity is defined as the **config surface**, not byte equality: field names, order, types,
and non-site-local defaults, plus the module tail (`Config`, `compute_config_hash`,
`config`) whose absence *was* the original G2 bug. Site-local fields (model paths, tile
dirs, `output_dir`, `slide_id`, `device`) are allow-listed, since `config.py` is
gitignored and exists precisely to differ there. A sixth test asserts the allowlist only
names fields that still exist, so it cannot quietly hide a renamed field — that test
caught a wrong guess (`merge_tile_dir`) on its first run, which is the behaviour it is for.

### 7.3 `m0_stitch` correctness suite (#37)

`m0_stitch` was doc 05's nominated first candidate: pure data reorganization, no model, no
GPU, no I/O. It is also the layer that decides which tile owns which cell and which pixels
each tile contributes — a silent error there produces duplicated or missing cells that no
downstream stage can detect. 21 tests over four areas:

- **`compute_tile_geometry`** — cut lines land half an overlap into the later tile; the
  flush-snapped short final gap is accepted; five malformed grids (empty, duplicate,
  not starting at 0, interior gap, non-rectangular hole) each raise; `overlap ≥ tile` raises.
- **`core_crop_bounds`** — the strongest test in the suite: accumulate every tile's core
  crop into a full-slide coverage array and assert `min == max == 1`. A 0 is a gap
  (missing pixels), a 2 is an overlap (a seam drawn twice, wrong stitched dimensions).
  Also that same-column widths and same-row heights are uniform, which is the precondition
  making `_join_overlay_tiles`' row-then-column join pixel-exact.
- **`filter_and_absolutize`** — sweeps a cell across the entire overlap band between two
  tiles and asserts it is claimed **exactly once** at every position, on the correct side
  of the cut; centroids are absolutized; `cell_id` is *not* renumbered (the documented
  contract that `_finish_batch` depends on).
- **`clear_slide_edge_cells`** — only requested sides clear, no-side still relabels
  sequentially, all-four leaves only interiors, and the input is not mutated.

### 7.4 Codegraph phantom recheck (#38)

G1's phantom list predates the `cell_mask/hybrid/` → `backend/algorithms/hybrid/` move.
Re-ran the `codegraph` file table against `git ls-files` and the actual on-disk tree:

```
indexed=20   git-tracked=22   on-disk=37 (excl. __pycache__)
phantom (indexed but not on disk):  0
indexed but gitignored:             0
on disk (.py) but not indexed:      1  → backend/algorithms/hybrid/config.py
```

**Zero phantoms.** The single unindexed file is `config.py`, which is gitignored by design
— codegraph respecting gitignore is correct behaviour, not staleness. G1's list is
obsolete and should be treated as closed rather than carried forward.

---

## 8. Tier 3.1 — Cellpose checkpoint clinical sign-off (escalation, not engineering)

Unchanged and still open, by design — doc 26 explicitly says to surface this rather than
schedule it. Restating it here because it is the largest standing risk in the backlog and
this round added to what rides on it:

The Cellpose 4.0.8 → 4.2.1.1 (`cpdino`) checkpoint swap in round 3 shifted cell counts by
**+1.8–2.5%** and flipped one tile success→skipped. That is a segmentation-quality change,
not GPU non-determinism. **Every performance result from round 3 onward sits on top of it**,
including everything in this document. It has been pending since round 3 and is unresolved
through round 7. It blocks nothing technically and has the longest external lead time,
which is exactly why it should be raised now with whoever owns pathologist/clinical review.

---

## 9. Deliberately not picked up

Per doc 26's own sequencing, not oversight:

- **Tier 2.1 `clear_slide_edge_cells`** (~1.2% of wall) — watch item; doc 26 says not to
  schedule dedicated time. Now cheap to revisit with §4's `Bs_clear_edge` bucket.
- **Tier 2.2 `detect_all_dots` +22.3% regression** — 1.013x ceiling, zero wall payoff, and
  isolating it needs the round-2 dependency set that was never snapshotted (19 #9).
- **Tier 5** (multiprocessing ship decision, A1 worker re-tune) — gated on Tier 0.1
  producing real full-slide numbers.
- **Tier 6** (Phase E concurrent-job load test, persistent worker pool) — doc 26 requires
  confirming concurrent-slide serving is part of the deployment plan *before* sizing.
- **Tier 7** (`detect_all_dots` whole-tile vectorization, Cellpose kernel internals) —
  highest correctness risk against a BG arm that already has 47–53% slack, and pinned
  third-party internals for a ~1.118x ceiling.
- Everything in doc 26 §4.2's exclusion table. Not re-proposed.

---

## 10. Follow-ups this round opened

1. **`perf_measure.py`'s non-stream path is dead** — it calls `HP.precut_paired_tiles`,
   which `hybrid_pipeline` does not re-export. Any invocation without `--stream-precut`
   raises `AttributeError` before doing work. Reported, not fixed: outside doc 26's scope,
   and every round since 4 passes `--stream-precut`. Fix is one import line if the
   non-stream mode is still wanted; delete the flag if it is not.
2. **Resolved during the round, recorded because the mechanism is instructive.** Adding
   `cuda_alloc_conf` to `Config` initially put it into `compute_config_hash`, and the
   allocator sweep failed on its **first** run with
   `worker config_hash b147f01b != parent 656ac4c3`. That is the
   `_mp_tile_worker` guard doing precisely its job: `perf_measure.py --cuda-alloc-conf`
   mutates the *parent's* config singleton, a `spawn`ed worker re-imports `config` fresh
   and cannot see that mutation, so the two hashes diverge and the batch fail-fasts rather
   than silently running half the tiles under a different configuration.
   **Fix**: `_HASH_EXCLUDE = {"cuda_alloc_conf"}` — runtime-only knobs that change *how* a
   run executes but not *what* it produces are no longer hashed. This is not a workaround
   but the more correct definition: the hash is written into every CSV to answer "were
   these results produced by the same configuration", and an allocator arena strategy
   cannot change a single output pixel. It also **restores exact comparability with prior
   rounds** — the post-fix hash is `3d1087f2`, which is byte-for-byte the hash recorded in
   every round-7 metrics file. `tests/test_config_parity.py` now asserts the property
   directly (mutating any excluded field must not move the hash, and the exclusion set must
   match between `config.py` and `config_example.py`).
3. **The registration stage's per-modality canvas sizes** (§0 finding 1) are worth an
   explicit statement somewhere upstream, so the next person to attempt a full-slide run
   does not rediscover it from a `PrecutStream` traceback.
4. **`pip freeze` discipline** (19 #9) — kept. The environment for this round gained only
   `pytest` (+`pluggy`, `iniconfig`), installed with `--no-deps` so no pipeline dependency
   moved; `torch 2.11.0+cu130` / `numpy 1.26.4` verified unchanged afterwards.
5. **The `zstd` decision is one QuPath session away.** Everything else needed to make it is
   already in this document. If nobody has time to check, the honest state is "Phase D has a
   measured 1.2331x sitting unclaimed", not "Phase D is optimised".
6. **Root-causing the 24.76 GiB balloon is now the live question**, not testing more
   allocator flags — §5 rules out the fragmentation explanation that motivated the flag.

---

## 11. Artifacts

Raw measurement output for this round is archived under
[`measurement/_metrics_r8/`](./measurement/_metrics_r8/):

| file | what it is |
|---|---|
| `stitch_ablate_4gp.json` | §3's 13-config `tiffsave` ablation at 4.055 GP |
| `alloc_conf_w6.json` | §5's 12-run allocator sweep, per-run and summarised |
| `worker_timings_probe.json` | §4's `--worker-timings` run; `worker_timings` / `worker_timings_total` are the new fields |
| `pip_freeze.txt` | environment snapshot (19 #9) |

Reproduce commands:

```bash
# §3 — tiffsave ablation (add --only baseline,zstd_1 to confirm the winner at full scale)
.venv/bin/python scripts/stitch_probe.py --overlay-src <dir of real overlay tiles> \
    --pool 6 --ablate --slide-w 70909 --slide-h 57183 --out ablate_4gp.json

# §5 — allocator sweep (refuses to start on a GPU that already has memory in use)
.venv/bin/python scripts/alloc_conf_probe.py --ihc <crop_ihc> --dish <crop_dish> \
    --workers 6 --repeats 6 --out alloc_conf_w6.json

# §4 — worker-side stage breakdown
.venv/bin/python scripts/perf_measure.py --ihc <ihc> --dish <dish> --output <out> \
    --label wt --mp-workers 4 --worker-timings --stream-precut

# tests
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/verify_mp_failfast.py --tiles-dir <precut scratch> --workers 3
```
