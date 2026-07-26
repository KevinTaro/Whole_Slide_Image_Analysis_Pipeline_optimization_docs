# 24 — GPU/CUDA acceleration candidates: encode/decode + independent per-item loops (survey plan, no code changed)

> **Scope of this document.** Requested survey: re-read
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md),
> [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md) and
> [`measurement/current-status-comparison.md`](./measurement/current-status-comparison.md), then
> find what is **still CPU-bound and could plausibly move to CUDA / a `cu*`-series library** —
> with emphasis on **encode/decode** work and **independent, per-item loops that repeat a lot**
> (per-tile, per-cell). **This document proposes nothing to build yet and no code in the
> pipeline was touched to produce it** — it is Discover-phase output per the playbook's own
> discipline, cross-checked against every later round (4, 5, 5b, 6) and the two docs that already
> asked this exact question — [`13-next-optimization-plan.md`](./13-next-optimization-plan.md) §"未來方向"
> and [`22-next-optimization-cycle-plan.md`](./22-next-optimization-cycle-plan.md) §3 — so this
> does not re-open what they already closed.
>
> **Revision note 1.** The first version of this doc used the **441-tile "large" crop** (302.7 s)
> as its primary yardstick, the same crop every round 1–6 measurement used. That crop is ~85–89%
> tissue-dense; **a real slide is only ~61% tissue-bearing**
> ([`23-...-implementation.md`](./23-next-optimization-cycle-implementation.md) §1). This revision
> makes **full-WSI scale (27,565 tiles, not 441) the primary baseline**, and re-opens the question
> of which candidates were dismissed *because* their effect was too small on a 441-tile crop but
> could matter at 10,000+ tiles — either fixed per-call overhead amortizing differently at that
> repetition count, or the crop's tissue density hiding costs proportional to *background*-tile
> count, which is far higher on a real slide than in any crop tested so far. §0 lays out both
> mechanisms and which candidates each applies to — not every "closed" item reopens.
>
> **Revision note 2.** Upstream commit `a64b92ebf46b8e1c660f3f4ec3946ac811425f62` (coworker,
> 2026-07-25) removed dead code from `m0_reader.py` (`precut_paired_tiles()`, `read_size()`, the
> module's `__main__` CLI block) and changed `run_batch`'s bare default from `workers=1` to
> `workers=4`. Noted for the record; per project direction this is settled and out of scope here —
> this doc doesn't revisit it and doesn't propose undoing either change.
>
> **Revision note 3 (this pass) — target is shortest total wall-clock, not a process-count
> preference.** There is no house rule favoring single- or multi-process execution — the goal is
> whichever configuration finishes fastest while staying correct and stable, and `workers>1`
> multiprocessing (§0.5, [`21-...-implementation.md`](./21-cross-tile-multiprocessing-implementation.md))
> is already the single largest lever this project has measured (up to **3.09–3.51×** at
> `workers=3–4`, dwarfing every candidate in this document). Consequently every candidate below is
> now evaluated against two questions, not one: *(a)* does it help at all, and *(b)* does it still
> help — or does it net-lose — once `workers>1` is in the picture. §0.5 works out why most
> single-process verdicts transfer unchanged, and why the one thing that genuinely changes under
> multiprocessing is **shared VRAM headroom**, which becomes the dominant new gate for any candidate
> that would add its own GPU library and run under `workers>1`.
>
> **No pipeline code was changed to produce this document or any revision of it.**

---

## 0. Rebaselining to full-WSI scale

### 0.1 The full-WSI grid, and why it isn't just "441 tiles × 62"

Per [`23-...-implementation.md`](./23-next-optimization-cycle-implementation.md) §1 (measured
directly from the aligned slide's dimensions, not assumed): the full slide is
141,818 × 114,366 px; at `default_tile_size=1024`, `window_overlap_px=256` (stride 768) the grid is
**185 × 149 = 27,565 tiles** — **62.5× the tile count of the 441-tile crop**, not the ~35× a naive
"large is representative, just smaller" reading would suggest (that 35× figure is a *pixel-count*
ratio, relevant to Candidate A below, not a tile-count ratio).

The same doc also corrected a long-standing assumption every prior full-WSI projection in this doc
set was built on: **"a real slide is mostly white background" does not hold for this input.**
Measured over a stride-768 thumbnail, only ~39% of grid cells sit at/above the background
brightness level — i.e. **~61% of the full slide is tissue-bearing, ~39% is background**
([`19-open-backlog.md`](./19-open-backlog.md) item 7). Compare that to what every crop tested so
far actually contains, from `bottleneck-list.md`'s own anchor table:

| scale | tiles | tissue (success) | background (skipped) | background share |
|---|--:|--:|--:|--:|
| small | 25 | 22 | 3 | 12.0% |
| medium | 121 | 103 | 18 | 14.9% |
| large | 441 | 379 | 62 | 14.1% |

**Every crop this project has ever measured is ~2.8× less background-heavy than the real slide**
(14% vs. 39%). That gap matters a lot for what follows, because two different mechanisms in this
pipeline scale with *tile count* and two others scale with *tissue-tile count* — and every existing
full-WSI projection (the official "~5.3h at `workers=1`" figure included) was produced by fitting a
straight line to *blended* per-tile costs measured on an 85–89%-tissue crop, then multiplying by
total tile count. That linear fit silently assumes the full slide has the same tissue density as
the crop. It doesn't.

### 0.2 What actually short-circuits on a background tile — verified against the code, not assumed

Read directly from `hybrid_pipeline.py`:

- `_process_one_chunk_gpu` (`hybrid_pipeline.py:546-614`) runs the UNet++ core-mask forward
  (`generate_ihc_core_mask`, line 571) **unconditionally on every tile**, then
  **`if core_mask.sum() == 0: return None`** (line 576-577) — background tiles bail out **before**
  the two Cellpose forwards (M2, M3b) ever run. **The expensive GPU forwards are already skipped
  for background tiles; this is existing, working code, not a gap.**
- `_process_precut_tile_cpu` (`hybrid_pipeline.py:361-459`) checks `tg.chunk is None` (line 384) and
  calls `_write_blank_tile` (line 386-389) instead of `_finish_chunk_cpu` for background tiles —
  so `detect_all_dots`, `enlarge_cell_instances`, `build_all_positive_results`, and per-cell crop
  export **also never run** on background tiles. The entire BG-arm workload for a background tile
  collapses to one function call.
- **But `_write_blank_tile` (`hybrid_pipeline.py:468-509`) writes the exact same six files a real
  tile writes** — `core_mask` (PNG), `masked_ihc` (PNG), `dish_mask_overlay` (PNG), `instance_mask`
  (TIFF), `dish_nucleus_mask` (TIFF), `overlay_annotated` (TIFF) — just filled with a constant
  `config.background_fill_value` instead of real content. It runs on the **BG thread** (reached from
  inside `_process_precut_tile_cpu`, which every call site submits to the background
  `ThreadPoolExecutor` — `hybrid_pipeline.py:1093-1095` single-process, `:802-804` per
  multiprocess-worker).

So the accurate picture is **not** "background tiles skip GPU work and could skip more" — the
expensive skip already exists and is correctly placed. The accurate, previously-unexamined finding
is: **six file-encode calls run per background tile, every one of them via the same
`_save_tile_array`/`skimage.io.imsave` path a real tile uses, and this cost has never been
separately measured because it's invisible at 14% background share.** At 39% background share and
27,565 tiles (~10,750 background tiles), that's **~64,500 file-encode calls** whose per-call cost
this doc set has never isolated. See Candidate F (§2.5).

### 0.3 Two different reasons scale could change a verdict — and which candidates each applies to

1. **Composition mismatch** (§0.1/§0.2): a cost proportional to *background*-tile count is
   underrepresented by 2.8× in every crop measured. Applies to: Candidate F (new). Does **not**
   apply to anything gated on the *tissue*-tile-only Cellpose/`detect_all_dots` path, because those
   are already *correctly* skipped for background tiles — more background tiles at full scale makes
   those costs **smaller** as a share of wall, not larger.
2. **Fixed-per-call-overhead amortization**: if a candidate's rejection was "the fixed dispatch/
   launch/transfer cost per call doesn't pay off at the batch sizes tested," more total repetitions
   (27,565 vs. 441) doesn't change that verdict **unless** the original test's batch size was itself
   the limiting factor rather than the per-unit cost being genuinely proportional. Matters for §1's
   one reopened item (cross-tile Cellpose batching, cheap re-check only).

Neither mechanism changes ceilings that are **percentages of wall** — it changes **which
absolute-second buckets those percentages apply to**, because the tissue/background mix shifts what
MAIN and BG actually contain at full scale. §0.4 redoes the arm math on that basis; §0.5 then adds a
**third** mechanism this revision introduces — what changes, and what doesn't, once `workers>1` is
in the picture.

### 0.4 Arm model, corrected for tissue/background composition — back-of-envelope, not a new official number

Round 6's official full-WSI projection (`23-...-implementation.md` §4.4: **~5.3h at `workers=1`**)
comes from fitting `wall ≈ intercept + 0.686 s/tile` on the blended (85–89%-tissue) crop anchors,
then multiplying by 27,565. Below, the same round-6 per-bucket numbers
(`23-...-implementation.md` §4.4's table, large/441, `workers=1`, `dot_detect_n_jobs=1`) are instead
split by **which population each bucket actually runs on** — tissue-only buckets divided by the
crop's 379 successful tiles, all-tile buckets divided by all 441 — and then re-multiplied by the
**real** slide's population (16,815 tissue tiles, 27,565 total), not the crop's blended rate:

| bucket | arm | runs on | crop cost | crop rate | full-WSI (corrected) |
|---|---|---|--:|--:|--:|
| `B1_m3b_cellpose` (2× Cellpose fwd) | MAIN | tissue tiles only | 232.0 s | 0.6122 s/tissue-tile | 0.6122 × 16,815 ≈ **171.6 min** |
| `B1_unet_coremask` (UNet++ fwd) | MAIN | **all** tiles | 13.6 s | 0.0308 s/tile | 0.0308 × 27,565 ≈ **14.2 min** |
| `B3_detect_dots` | BG | tissue tiles only | 100.2 s | 0.2644 s/tissue-tile | 0.2644 × 16,815 ≈ **74.1 min** |
| `B2_png_encode` (real content) | BG | tissue tiles only* | 70.6 s | 0.1863 s/tissue-tile | 0.1863 × 16,815 ≈ **52.2 min** |
| `B3_enlarge_cells`+`B3_build_results` | BG | tissue tiles only | 29.7 s | 0.0784 s/tissue-tile | 0.0784 × 16,815 ≈ **22.0 min** |
| Phase D (stitch) | outside | once, scales with pixels not tiles | 5.05 s | — (§2.1) | ≈ **3.0 min** |
| Candidate F (blank-tile writes, §2.5) | BG | background tiles only | **unmeasured** | **unmeasured** | **unknown — flagged, not estimated** |

*assumes background-tile writes inside `B2_png_encode`'s crop-era measurement were a small,
fast-compressing share of that 70.6 s (constant-fill arrays compress trivially under PNG/LZW) —
reasonable but **unverified**; this is exactly why Candidate F calls for a direct measurement.

- **MAIN, corrected (single-process)** ≈ 171.6 + 14.2 ≈ **~186 min ≈ 3.1 h**
- **BG, corrected (single-process)** ≈ 74.1 + 52.2 + 22.0 ≈ **~148 min ≈ 2.5 h**, plus Candidate F's unknown
- **single-process wall ≈ max(MAIN, BG) + outside ≈ roughly 3.1–3.4 h**, materially below the
  official blended-rate **~5.3 h** — **if** Candidate F's unmeasured term stays small.

This is a back-of-envelope correction, not a replacement for the official number — it compounds
measurement uncertainty and can't account for Candidate F. It's included because it changes this
whole document's practical takeaway: **the project's own full-WSI single-process time estimate is
probably itself inflated by the same composition mismatch this doc is about**, and ranking GPU
candidates by "minutes saved" needs the *right* baseline or it will systematically overrate BG-arm
candidates and underrate anything that runs on every tile regardless of content (Candidate F). §0.5
carries this same correction into the `workers>1` case, which is the one that actually matters for
"shortest total time" (Revision note 3).

**The actionable step, not just the caveat**: get one real, composition-matched measurement before
trusting any ranking here. A crop sampled to hit the real ~61/39 tissue/background ratio (a few
hundred to ~1,000 tiles) through the existing `scripts/perf_measure.py` harness is far cheaper than
backlog item 7's full 27,565-tile run and answers this specific question directly — listed as
priority 1 in §4, run at **both** `--mp-workers 1` and the currently-recommended `workers=4` in the
same pass (§0.5), so the composition correction and the multiprocess question get answered together.

### 0.5 What changes, and what doesn't, under multiprocessing — and why VRAM is the real new gate

Backlog item 1 / [`21-...-implementation.md`](./21-cross-tile-multiprocessing-implementation.md)
already measured `workers>1` at **3.09× (workers=3) to 3.51× (workers=4)** on the large/441 crop —
far larger than anything in this document, and the reason is GIL contention recovery, not just more
compute (doc 20 §5, doc 21). Since the target is shortest total time regardless of process model
(Revision note 3), every candidate below has to be checked against this baseline, not just the
`workers=1` one. Two questions, answered by reading the multiprocess code path directly rather than
assuming:

**1. Does the per-candidate ceiling analysis (§1, §2) even transfer to `workers>1`?** Yes, and
cleanly. `_mp_tile_worker` (`hybrid_pipeline.py:732-812`) runs the **same** two-stage
GPU-front/CPU-back overlap as single-process `run_batch` inside *each* worker process — same
`_process_precut_tile_gpu`/`_process_precut_tile_cpu` split, same `ThreadPoolExecutor(max_workers=1)`
background arm, same depth-1 pipelining. Each worker is an independent copy of the exact structure
§0.4's arm model analyzes, operating on its own slice of the dynamic tile queue. So a candidate's
MAIN/BG ceiling is **per-worker-invariant**: if GPU-porting `detect_all_dots` doesn't move the
needle inside one worker's own arm balance, adding more workers doesn't change that — it's still
governed by the same slack-arm ceiling, just multiplied by however many workers are running it in
parallel. This is why §1/§2's "do not build" verdicts for Candidates B/D/E are **not** reopened by
`workers>1` — the mechanism that makes them low-value (arm placement, not raw speed) is identical in
every worker.

**2. What genuinely is different: shared VRAM headroom, and it's tighter than it looks.**
[`21-...-implementation.md`](./21-cross-tile-multiprocessing-implementation.md)/round 5 measured
**per-process** VRAM growing **superlinearly** with worker count, not flat: 2,787 / 3,117 / 4,118 /
5,167 MB at `workers=1/2/3/4`. At the currently-recommended `workers=4` (`23-...-implementation.md`
§6), that's **~20.7 GB of the 32 GB card already spent**, leaving **~11.3 GB nominal headroom** —
and that nominal number is optimistic: `workers≥6` already shows a reproducible allocator-fragmentation
failure mode where **one worker alone balloons to exactly 24.76 GiB** while its siblings sit at
1–2 GB (`19-open-backlog.md` item 7b, 2 failures in 6 runs at `workers=6`). **Any new GPU library
this document proposes (nvTIFF, nvImageCodec, CuPy, cuCIM) adds its own CUDA context and working
buffers to every process that imports it.** Under `workers=1` that cost is free (32 GB minus ~2.8 GB
in use). Under `workers=4` — the configuration that actually delivers the largest measured
speedup — it competes directly with the headroom that's already the thing capping safe worker count.
Losing even one worker from the safe ceiling costs far more wall-clock than any single-tile encode
optimization in this document: round 6's crop sweep (`23-...-implementation.md` §6) shows
`workers=3→4` alone buys ~11% (142.9 s → 128.8 s), i.e. dropping one worker to make room for a new
library's VRAM footprint could cost more time than that library saves.

**Consequence for every candidate in §2**: none of them are disqualified by this — Candidate A
(§2.1) runs once, in the parent process, entirely outside the worker pool, so it carries **zero**
extra per-worker VRAM cost and is unaffected by this gate. Candidates B/D/E/F, if ever built, would
run *inside* each worker and must be checked against remaining VRAM headroom at the target worker
count as part of §3's environment spike — not assumed safe because they passed a `workers=1`
microbenchmark. This is now folded into §3 and §4 explicitly.

---

## 1. Already GPU, or already evaluated and explicitly closed — do not re-propose

Cross-referenced against [`19-open-backlog.md`](./19-open-backlog.md) §1's "already closed" list
and [`23-...-implementation.md`](./23-next-optimization-cycle-implementation.md) §2/§3/§5/§7. The
**"full-WSI reframing"** column records whether §0's rebaselining changes the verdict, not just the
absolute size — per §0.5, it also holds under `workers>1` since these verdicts are per-worker-invariant.

| item | status | why it's closed | full-WSI reframing |
|---|---|---|---|
| GPU forwards (UNet++, 2× Cellpose) | Already CUDA (`torch`) | Internals traced and stop-lossed at **1.118x** ceiling — third-party-patch risk too high (doc 23 §7). | **Ceiling unchanged** (it's a ratio). Absolute value at the corrected ~3.1h MAIN estimate: saves ≈ 186min×(1−1/1.118) ≈ **20 min** single-process (proportionally the same under `workers>1`, since every worker carries the same internal ceiling). Verdict unchanged: risk, not size, blocks this. |
| Cross-tile Cellpose batching (`(N,H,W,3)` stacked `eval`) | **Measured negative**, tested G=1,2,4,8 | Per-patch cost genuinely proportional (`run_net`-only ms/tile flat-to-worse across G), not fixed-per-call — doc 23 §2 explicitly separated the two and found batching doesn't amortize anything. | One item worth a cheap, optional re-check: VRAM headroom under `workers=1` (2.79 GB steady-state, 32 GB card) allows G up to ~16-20, untested range. But the found mechanism (per-patch-proportional, not launch-bound) predicts more of the same — flat-to-worse. **Also now doubly capped under `workers>1`** by §0.5's VRAM gate: even if G=16 helped at `workers=1`, its linear VRAM cost (1.17→8.01 GB at G=8) would eat directly into the headroom that funds `workers=4`, working against the dominant lever rather than with it. Cheap to settle (`scripts/cellpose_batch_probe.py`, add G=16); not worth a pipeline change speculatively. |
| Cross-tile UNet++ batching | **Measured negative** | `predict_batch` is a serial loop, not batched; strictly worse at every G (doc 23 §3). | Already ~0.4% of wall before testing; under the corrected full-WSI baseline that's under a minute. Not worth even the cheap re-check. |
| `detect_all_dots` → CuPy/cuCIM GPU port (doc 22's "B2") | **Explicitly evaluated, not built** | Ceiling 1.013x (BG/slack arm); the real fix was a one-line `n_jobs` config change, 1.60x, no GPU (doc 23 §4–§5). | **Verdict strengthens, doesn't reopen.** §0.4 shows BG's real-world tissue-tile population is *smaller* relative to MAIN than the crop suggested — more slack, not less, at full scale. §0.5 confirms this holds per-worker under `workers>1` too, and adds that building it would cost VRAM headroom the dominant multiprocessing lever needs more. Revisit only after §0.4's composition-matched measurement, run at both worker counts, actually shows otherwise. |
| `enlarge_cell_instances`/`build_all_positive_results` → GPU (doc 22's "B3") | **Explicitly evaluated, not built** | Bundled behind B2's gate, never opened (doc 23 §5). | Same reasoning as the row above. |
| Precut A (WSI tile decode) | **Resolved without GPU** | Streamed, cost ~0.004s/tile already (doc 18 §4). | Runs on every tile; 27,565 × 0.004s ≈ 1.8 min total. Not worth touching at either worker count. |
| CUDA MPS | **Tested, flat end-to-end** | Real pipeline isn't serialization-limited at its knee (doc 21 §5). | No population- or worker-count-dependent mechanism in the negative result. Unchanged; also the closest thing to a direct multiprocess-vs-GPU-sharing test this project has already run, and it still came back flat. |
| GPU-side tile/transform loading | **Stopped out** | 1.22% of wall, ceiling 1.012x (doc 18 §6.3). | ≈ 2.5 min ceiling-if-zero at the corrected full-WSI total. Not worth building a pipeline that doesn't exist for a couple of minutes, at any worker count. |

**Net effect of §0 on §1**: one item (cross-tile Cellpose batching) gets a cheap, optional
confirmatory re-check, now doubly disfavored once its VRAM cost under `workers>1` is considered.
Nothing else reopens; the `detect_all_dots`/`enlarge_cell_instances` GPU-port rows get **more**
confidently closed once both the composition correction and the multiprocess VRAM gate are applied.

---

## 2. Survey — encode/decode + independent-loop candidates, ranked for full-WSI scale and process-agnostic total time

### 2.1 Candidate A — final slide-level overlay stitch (`_stitch_overlay_slide`, Phase D) — TIFF **encode**

**Where**: `hybrid_pipeline.py:1152` (`_stitch_overlay_slide`). Manual row-then-column
`pyvips.Image.join()` (pure metadata, lazy) feeding a single `tiffsave(..., compression="lzw",
tile=True, pyramid=True)` call — one big serial CPU LZW encode + pyramid downsample generation, at
the very end of `run_batch`, after all tile analysis is done, **in the parent process only,
regardless of `workers`** — `_finish_batch` (`hybrid_pipeline.py:1106-1149`) is the single code path
both the single-process and multiprocess branches of `run_batch` converge on before calling it.

**Current measured cost**: 5.05–5.11 s at the 441-tile crop (~21,504×21,504 px ≈ 0.46 gigapixel),
flat across three library rounds including a pyvips 3.1.1→2.2.3 downgrade — compression-bound, not
library-version-sensitive. Closed once already on the *overlap* axis only (doc 18 §4: structurally
non-overlappable) — nobody has evaluated making the encode itself faster (playbook option (c)),
which is a different lever from overlap (option (b)).

**Full-WSI sizing**: the full slide is 141,818×114,366 px ≈ **16.2 gigapixels**, **~35× the pixel
count** of the 441-tile crop. At a flat compression-bound rate, that extrapolates to **~3 minutes of
single-threaded, fully serial wall-clock**.

**Why this candidate is unaffected by, and grows relatively more valuable under, multiprocessing**:
unlike every other candidate here, its cost doesn't depend on tile count, tissue fraction, or
`workers`, and it carries **zero incremental VRAM cost per worker** (§0.5) since it never runs inside
the worker pool. As `workers>1` shrinks total wall-clock (single-process ~3.1–3.4h corrected →
multiprocess sub-hour territory at `workers=4`), this fixed ~3-minute serial block becomes a
**larger, not smaller, share of what's left** — the opposite trend from Candidates B/D/E/F, whose
share shrinks as more of the pipeline gets parallelized away. That makes it the strongest single
candidate in this document once the goal is shortest *total* time under whatever process
configuration is actually deployed.

**Candidate libraries** (none currently in the venv — new dependencies):
- **NVIDIA nvTIFF** — GPU TIFF encode, Volta+, CUDA Toolkit ≥12.0, LZW compression, planar
  contiguous mode, up to 32 bits/sample, strip-parallel encode. Matches this pipeline's exact output
  format (LZW, 8-bit RGB). **Caveat**: no confirmed first-class pyramid-generation feature — getting
  a pyramidal output means building downsample levels separately (e.g. `cupyx.scipy.ndimage`/`torch`
  on GPU) and feeding each level's strips to nvTIFF; real engineering, not a drop-in call. Python
  bindings specifically unconfirmed (see §3).
- **nvCOMP** (GPU compression: LZ4/GDeflate/Zstd/ANS) — could replace the LZW compress step paired
  with a hand-rolled TIFF container writer, but LZW itself isn't an nvCOMP codec — this means
  **changing the compression scheme**, a correctness-adjacent decision needing sign-off.
- **cuCIM's `cuslide2` backend** — GPU TIFF I/O via `nvImageCodec`, purpose-built for pathology.
  Most of cuCIM's published strength is *read*/decode; whether it exposes a write path at all needs
  a direct API check, not an assumption.

**Recommendation**: still the strongest candidate in this doc. Do not build yet — first get the
standalone measurement (`_stitch_overlay_slide` against a synthetic 16-gigapixel input, no model
inference needed, cheap, and — because it runs in the parent process — this measurement doesn't even
need to be repeated per worker count) to convert "~3 minutes" from an extrapolation into a fact. If
it clears a meaningful bar, next step is a standalone nvTIFF throughput spike (§3) before any
pyramid-integration engineering. Because it has no VRAM interaction with the worker pool, it's also
the one candidate in this document that a future `workers=4`/`workers=6` deployment doesn't make
riskier to build.

### 2.2 Candidate B — per-tile debug array PNG/TIFF encode + per-cell crop PNG encode (tissue tiles only)

**Where**: `_save_tile_array` (`hybrid_pipeline.py:462-465`) writes 5 arrays per **tissue** tile
(`core_mask`, `masked_ihc`, `dish_mask_overlay` as PNG; `instance_mask`, `dish_nucleus_mask` as
int32 TIFF), plus `overlay_annotated/{tile}.tiff` (`hybrid_pipeline.py:431-434`). Separately,
`export_per_cell_images` (`m4_module/cell_crops.py:27-132`) calls `cv2.imwrite` once per cell (line
128) — an independent per-item loop over ~20–98 cells/tile at these crop densities. Runs inside
`_finish_chunk_cpu`'s BG-thread call, which under `workers>1` runs inside every worker process
independently (§0.5) — total work is the same, spread N ways, each worker's own arm-balance ceiling
unchanged.

**Full-WSI sizing (corrected for tissue-only population, §0.4)**: ≈ **52.2 min** (single-process
total across all tissue tiles; under `workers=4` this workload is divided ~4 ways alongside
everything else on the BG arm, same as the MAIN arm's Cellpose forwards).

**Arm/ceiling**: BG (slack) arm, ceiling ~1.01–1.05x, per-worker-invariant (§0.5). §0.4's composition
correction shows the BG arm likely has *more* headroom at full-WSI scale than the crop suggested,
not less — background tiles (39% of the real slide, only 14% of every crop tested) contribute ~0 to
BG but still cost MAIN its (cheap) UNet++ forward. (The first pass of this doc predicted the opposite
trend from round-over-round crop data alone; this composition analysis corrects that.)

**Recommendation**: still **do not build now**. If ever reconsidered, it would run inside the worker
pool under `workers>1` and would need to clear §0.5's VRAM-headroom gate on top of its already-thin
ceiling — a strictly higher bar than the `workers=1` case alone. The re-exposure trigger from the
first pass ("BG/MAIN crossing ~0.9") is suspended pending the composition-matched measurement (§4
item 1) — don't act on crop-only data going forward.

### 2.3 Candidate C — `instance_mask` / `dish_nucleus_mask` int32 label TIFF writes

Same `_save_tile_array` path as Candidate B, historically <0.5% of wall. At full-WSI scale that's
still on the order of a minute or two even before any composition correction, at any worker count.
Not worth a separate investigation; fold into Candidate B if that's ever prototyped.

### 2.4 Candidates D/E — `detect_all_dots` and `enlarge_cell_instances`/`build_all_positive_results` as batched GPU kernels

Already evaluated this round under a different name (doc 22's "B2"/"B3", §1 above) and explicitly
not built — ceiling 1.013x, BG/slack arm, per-worker-invariant (§0.5). The distinction between the
joblib fan-out doc 23 §4 killed (dispatch parallelism over an unchanged per-cell loop, loses because
each ~4ms task is GIL/dispatch-dominated) and a genuine CuPy vectorized batch (rewriting the
algorithm to process a tile's cells as one GPU call, no per-cell Python loop) is real and still
worth recording — but §0.4's composition correction and §0.5's VRAM gate both make the case for
revisiting it **weaker**, not stronger: smaller real-world footprint than the crop implied, and any
GPU-resident implementation would compete with the worker pool's VRAM budget. **Explicitly not
next-round work**, contingent on the composition-matched measurement (§4 item 1).

### 2.5 Candidate F — background-tile placeholder write path

**Where**: `_write_blank_tile` (`hybrid_pipeline.py:468-509`), reached from
`_process_precut_tile_cpu:384-391` for every tile whose `core_mask` is entirely empty. Writes the
same six files (§0.2) a real tile writes, filled with `config.background_fill_value`, on the BG
thread — inside each worker process under `workers>1`, same as Candidate B.

**Why this is the one genuinely new finding in this doc, not a reframing of something already
measured**: nothing in `bottleneck-list.md`, `current-status-comparison.md`, or docs 13–23
separately measures this path — it can't show up in `B2_png_encode` or any other named bucket the
way the harness currently instruments things, and at 14% background share in every tested crop it
was never large enough to be visible in aggregate wall-clock. At **39% background share and 27,565
tiles (~10,750 background tiles, ~64,500 file-encode calls)**, it's large enough in *call count* that
its per-call cost — currently completely unknown — could matter.

**What's actually unknown, stated plainly**: whether each blank-tile encode call is cheap (constant
arrays typically compress trivially fast under PNG/LZW) or whether fixed per-call overhead
(file-open/close, directory I/O, six `_save_tile_array` dispatches) dominates regardless of content,
in which case ~64,500 calls could add up to something real. This doc does not know which.

**Not necessarily a GPU/CUDA problem, worth saying directly**: if fixed per-call overhead dominates,
the fix isn't nvImageCodec or any CUDA library — it's the cheaper move of recognizing that every
background tile's six output files are byte-identical for a given tile-position class and **writing
one blank-tile file set once, then copying/hard-linking it**, instead of re-encoding ~10,750 times.
Pure I/O fix, zero GPU dependency, and — if the fixed-overhead hypothesis holds — plausibly the
highest-value-per-hour-of-engineering item in this document. See also Candidate G, a closely related,
already-confirmed finding.

**Multiprocessing interaction**: under `workers>1`, background tiles are distributed across the
dynamic work queue same as tissue tiles, so this workload's *compute* time divides ~N ways like
everything else. What does **not** automatically divide favorably is any **shared-resource**
contention the writes generate — which is exactly what Candidate G below identifies directly in the
code, rather than leaving as a hypothesis.

**Recommendation**: measure specifically — timer around `_write_blank_tile`, run on a
background-tile-heavy sample (§4 item 1), at both `workers=1` and the target worker count so any
filesystem-contention effect under concurrency shows up rather than being hidden by a single-process
test.

### 2.6 Candidate G — redundant per-call `mkdir(exist_ok=True)` syscalls (NEW — not a GPU candidate, highest-confidence item in this doc)

**Where, verified directly against the code**: `_save_tile_array` (`hybrid_pipeline.py:462-465`)
runs `path.parent.mkdir(parents=True, exist_ok=True)` **before every single write** — a real
`mkdir()` syscall each time, not a no-op, even though the target (one of six fixed, module-level
directories: `core_mask/`, `masked_ihc/`, `dish_mask_overlay/`, `instance_mask/`,
`dish_nucleus_mask/`, `overlay_annotated/`) already exists after the first tile. `_save_tile_array`
is called 6× per tissue tile (§2.2) and 6× per background tile via `_write_blank_tile` (§2.5) — so
**every one of the 27,565 tiles at full-WSI scale issues 6 mkdir syscalls against the same six
directories**, ≈**165,390 redundant syscalls** total. (`export_per_cell_images`'s
`cells_dir.mkdir(...)` at `m4_module/cell_crops.py:44` is different in kind and **not** part of this
finding — that directory is genuinely unique per tile (`cell_crops/{tile_name}/cells`), so its
per-tile `mkdir` call is necessary, not redundant.)

**Why this is worth flagging even though it's small per-call**: it is exactly the shape the original
brief asked about — an independent operation repeating a great many times — and it is the one item
in this whole survey that plausibly gets **worse, not just proportionally distributed, under
multiprocessing**: `workers=4` means 4 processes issuing `mkdir()` against the *same* directory
inodes concurrently, which is a real (if usually small) source of filesystem-level contention that a
single-process measurement cannot reveal and that doesn't show up in any GPU-vs-CPU framing at all.
Given this project's explicit goal — fastest, reliable, *and stable* total time — a source of
concurrent contention that scales with worker count is worth eliminating on stability grounds even
before its speed contribution is measured.

**The fix, stated plainly, is not a GPU candidate**: create the six fixed output directories once,
before the tile loop starts (in `run_batch` or `_finish_batch`'s setup, or once per worker in
`_mp_tile_worker`), and drop the per-call `mkdir` from `_save_tile_array`'s hot path entirely — it
would only need to remain for genuinely tile-unique paths (`cell_crops/{tile_name}`, already
mkdir'd once per tile in `export_per_cell_images`, which is correct as-is). This is a small, surgical
change with an obvious correctness argument (the six directories are fixed and known before the loop
starts) and zero new dependencies.

**Recommendation**: **measure, then very likely build** — this is the highest-confidence, lowest-risk
item in this survey, specifically because it doesn't require a new GPU dependency, doesn't touch
correctness-sensitive numerics, and directly serves the "reliable and stable under multiprocessing"
half of the target, not just "fast." Time it first (wrap `_save_tile_array`'s `mkdir` call, measure
saved syscalls/time at both worker counts) as part of §4 item 1's measurement pass, since it's nearly
free to add to that same run.

---

## 3. Environment/resource gate — resolve before building anything above

This project's own quickref lists *"ignoring hardware/platform quirks"* as anti-pattern #8 — same
caution applies before any candidate above gets built. Two independent gates, both must clear:

**Hardware/software compatibility**:
- RTX 5090, Blackwell, compute capability **sm_120**. CuPy added CUDA 13.x wheel support at v13.6.0,
  but confirmation that `cupyx.scipy.ndimage`, nvImageCodec, nvTIFF, nvCOMP, or cuCIM are verified
  against sm_120 specifically was **not** found in research for this doc — several of these
  ecosystems were still catching up to Blackwell. `torch==2.11.0+cu130` runs correctly on this
  machine; that does not imply the others do — each has its own kernel/JIT compilation path.
- **No dependency currently installed** — confirmed by grep over `pyproject.toml`/`uv.lock`; every
  GPU-library candidate above is 100% new dependency surface.
- **Confound risk, from this project's own history**: round 3's `uv sync` bundled a Cellpose upgrade
  with numpy/scikit-image/pyvips/opencv downgrades in one step, and the resulting `detect_all_dots`
  +22.3% regression (item ⑨) still has no isolated cause two rounds later. Any new GPU codec
  dependency must **not** be bundled with anything else — its own `uv sync`, its own `pip freeze`
  snapshot, standalone benchmark before touching the pipeline.

**Shared VRAM budget under multiprocessing (§0.5 — new, and the more binding gate for anything that
would run inside the worker pool)**:
- Current headroom at `workers=4`: ~11.3 GB nominal (20.7 of 32 GB in use), less in practice given
  the fragmentation failure mode already observed at `workers≥6` (item 7b).
- Any candidate touching Candidates B/D/E/F must have its per-process VRAM footprint measured
  **standalone** and then checked against remaining headroom **at the worker count it would actually
  ship at**, not just proven fast at `workers=1`.
- Candidate A (§2.1) is exempt — it never runs inside the worker pool.

**Concretely, before Candidate A gets a line of pipeline code**: a throwaway script that imports the
candidate library, allocates a representative buffer, and times one encode call against the current
CPU path — on this exact GPU, this exact CUDA 13.0/driver — same "prove the mechanism in isolation
first" step this project already uses everywhere else. Before Candidates B/D/E/F get *any*
consideration: the same spike, plus a VRAM measurement at the target worker count.

---

## 4. Ranked priority for the next measurement round (full-WSI baseline, process-agnostic)

1. **One composition-matched measurement, run at both `workers=1` and `workers=4`.** Build a crop
   sampled to hit the real ~61% tissue / ~39% background ratio (a few hundred to ~1,000 tiles),
   run it through `scripts/perf_measure.py` at both worker counts in the same pass. This replaces
   §0.4's back-of-envelope arithmetic with real numbers, directly measures Candidate F (§2.5) and
   Candidate G's mkdir overhead (§2.6, cheap to add to the same run) for the first time, and
   confirms whether §0.5's per-worker-invariance reasoning actually holds for BG/MAIN under
   multiprocessing — the prerequisite for deciding whether Candidates B/D/E are worth anything at
   any worker count. Cheaper than backlog item 7's full-slide run.
2. **Size Candidate A (stitch D) at real scale** — independent of step 1 (pixel-count-driven, not
   composition- or worker-count-dependent), cheap, no model inference needed.
3. **Prototype and measure Candidate G (mkdir hoist)** — near-zero risk, no new dependency, directly
   addresses the "reliable and stable under multiprocessing" half of the goal. Can piggyback on
   step 1's run.
4. **Environment + VRAM spike (§3)**, in parallel with the above — prerequisite for any GPU-library
   candidate (A, or B/D/E/F if step 1 ever reopens them), with the VRAM half specifically required
   before considering anything that would run inside the `workers>1` pool.
5. **Candidates B, D/E**: **do not build**, contingent on step 1's real numbers at both worker
   counts. If step 1 shows Candidate F dominates BG's real-world cost, that (or Candidate G) becomes
   the priority instead — and per §2.5/§2.6, the likely fix for either is not GPU work.
6. **Cross-tile Cellpose batching at G=16** (§1's one reopened item): optional, cheap, low
   expectation given the per-patch-proportional mechanism already found, and now additionally
   disfavored by its VRAM cost under `workers>1` — reuse `scripts/cellpose_batch_probe.py`, don't
   build a pipeline change speculatively.

## 5. What this document is not

A build plan. Every recommendation above is "measure this specific thing next," not "implement
this," with the possible exception of Candidate G (§2.6), which is a small, low-risk, no-new-dependency
change this doc considers close to ready once timed. Every full-WSI absolute-minute figure in §0.4
and §2 is a **back-of-envelope correction**, explicitly flagged as such, not a new official project
number — it exists to fix a systematic bias (crop tissue-density ≠ real slide tissue-density) in how
this doc set has projected full-WSI cost, and should be replaced by §4 item 1's direct measurement
as soon as that's run. Per the playbook: no number here should be trusted until measured with the
same rigor as the rest of this doc set (`--gpu-dmon`, GPU-idle-before-launch, interleaved control vs.
candidate, correctness veto against the same-code noise floor, ablation before adoption). No code in
`backend/algorithms/hybrid/` was changed to produce this survey or any revision of it.
