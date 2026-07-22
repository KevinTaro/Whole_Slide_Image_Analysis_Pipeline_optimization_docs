# 17 — GPU starvation prerequisites: what's left before multiprocess / bigger batch / GPU-side transforms

> Solution-design follow-up to [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md) and
> [`13-next-optimization-plan.md`](./13-next-optimization-plan.md), scoped by a direct request: **before**
> reaching for the three "big lever" architecture changes — cross-tile multiprocessing, extending Cellpose's
> batch size, or moving tile/transform loading onto the GPU — what is still measurably leaving the GPU
> under-fed on *this* architecture? Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md): Discover →
> Analyze → Plan → Choose, cheapest lever first, ablation-proof every layer, correctness is a veto.
> **Planning document — no pipeline code changed here.** Grounded by reading current HEAD
> (`backend/algorithms/hybrid/hybrid_pipeline.py`, confirmed line numbers below) — not by re-deriving state
> from the docs alone, since docs 13/16 predate the most recent local run referenced in §5.

## 0. Why this document exists

The live observation that triggered it: **GPU usage is intermittent and starvation is still a big problem**,
even after round 3's Cellpose swap. That is not a new complaint contradicted by the measured record — it is
*confirmed* by it. `bottleneck-list.md`'s own round-3 table shows GPU **idle_frac went up, not down**, between
the overlap round and round 3, despite wall-clock falling:

| large/441 | overlap round | round 3 | Δ |
|---|--:|--:|--:|
| GPU idle_frac | 0.190 | **0.370** | **+0.18** |
| GPU mean SM % | 32.9 | **16.6** | **−16.3 pt** |
| end-to-end wall | 707.4 s | 573.7 s | −18.9% |

Cellpose got faster **per call** (bfloat16 + DINOv3 backbone), so the GPU spends less wall-clock time actually
computing — but nothing shortened the CPU-side gaps *between* calls, so those gaps are now a bigger share of a
smaller total. Faster kernels made the pipeline more, not less, bursty. This document's job is to find and
rank what's closeable about that specifically, before spending effort on levers (multiprocess / batch size /
GPU transforms) that assume the GPU is the saturated resource and won't pay off proportionally while it isn't.

## 1. Recap — what's already known and already ranked (don't re-litigate)

`13-next-optimization-plan.md` §4 already ranks, by measured ceiling:

1. GPU forwards (444.6 s / 77.5% wall, ceiling 1.382x, **bounded to 34% headroom** before the BG arm becomes
   critical) — done to the extent a model swap could do it; further shrinkage is out of this project's hands.
2. `gc.collect` frequency — **done**, shipped as `gc.freeze()`, confirmed 1.069–1.077x (doc 16).
3. ⑧ stranded CPU prep (`enlarge_cell_instances` + `build_all_positive_results`, 28.4 s / 4.96%, **on the
   MAIN/critical arm**) — **still unbuilt**, confirmed by reading current HEAD (§2 below).
4. ⑤ precut A + stitch D overlap (25.6 s / 4.5%, wholly serial, wholly outside the two-arm overlap) — **still
   unbuilt**.
5. Wire `cellpose_batch_size` into `Config`, then sweep — **still unbuilt** (§2 below); this doc's §4 explains
   why it should stay ordered *after* items 3/4/6, not before.
6. Isolate the `detect_all_dots` +22.3% regression — optional, no wall-clock payoff (BG/slack arm, ceiling
   1.013x).

This document does not reopen or re-rank 1/2/6. It adds **one new candidate (§3)** that the existing ranking
doesn't cover, re-confirms 3/4/5's status against current source, and gives all of them a combined framing
under "what must close before the three big levers are worth pulling" (§4).

## 2. Confirmed against current HEAD — nothing in the plan has silently landed

Read directly (not assumed from doc age):

- **⑧ still on the MAIN arm.** `_process_one_chunk_gpu` (`hybrid_pipeline.py:547-620`) calls
  `build_all_positive_results` (line 597) and `enlarge_cell_instances` (line 601) **between** the M2 and M3b
  Cellpose forwards, entirely inside the GPU-front function. Priority 2 of doc 13 is unbuilt.
- **`cellpose_batch_size` still not wired.** `_init_cellpose_segmenter` / `_init_dish_cellpose_segmenter`
  (`hybrid_pipeline.py:207,219`) still call `getattr(config, "cellpose_batch_size", 16)`; grepping
  `config_example.py` finds no such field. Priority 4 of doc 13 (step 1, the no-op wiring) is unbuilt.
- **`gc.freeze()` confirmed live** (`hybrid_pipeline.py:224-248`, `_frozen_gc_generation`, wrapped around the
  whole `run_batch` loop at line 794) — Priority 1 of doc 13 really is done, cross-checked against the fresh
  local run in §5 (`B4_gc_collect` total 0.085 s over 121 calls, i.e. ~0.7 ms/call — matches doc 16's 1.2 ms).
- **No multiprocessing anywhere in this file.** `run_batch` is single-process, single background thread
  (`ThreadPoolExecutor(max_workers=1, ...)`, line 795), pipeline depth 1. The module's own `CLAUDE.md` already
  states why: *"the 3 GPU models are loaded once in the main process and share one CUDA context, so cross-tile
  process parallelism is unsafe (fork-under-CUDA)"* — this constraint is architectural, not a measurement
  finding, and any multiprocess plan has to design around it (§4.4), not assume it away.
- **No GPU-resident transform/decode path exists today.** Tile reading is `_read_rgb` over already-precut PNG
  files on disk (M0's `precut_paired_tiles`, CPU/pyvips, 8-thread I/O pool per the module doc) — there is no
  CPU→GPU tensor pipeline to move "onto the GPU" yet; it would have to be built, not flipped on.

## 3. New finding — intra-tile GPU bubbles, not just inter-tile arm imbalance

`13-next-optimization-plan.md` §2's arm model (MAIN vs BG, `wall ≈ max(MAIN, BG) + outside`) explains the
**cross-tile** overlap — how tile *N*'s GPU front hides behind tile *N-1*'s CPU back. It does not model what
happens **inside** a single tile's own GPU front, and reading `_process_one_chunk_gpu` shows that front is not
one continuous GPU-busy stretch — it's three GPU forwards with synchronous, GPU-idle CPU work threaded between
every one of them, all on the same main thread with nothing else scheduled to fill the gaps:

```
generate_ihc_core_mask   [GPU: UNet++ forward]
  ↓
_run_m1_overlay_stage    [CPU only: apply_mask / overlay_dish / fuse — GPU idle]
  ↓
segment_windowed (M2)    [GPU: Cellpose forward #1]
  ↓
clear_slide_edge_cells   [CPU only — GPU idle]
build_all_positive_results  [CPU only — GPU idle]   ← item ⑧, doc 13 Priority 2
enlarge_cell_instances       [CPU only — GPU idle]   ← item ⑧, doc 13 Priority 2
  ↓
segment_windowed (M3b)   [GPU: Cellpose forward #2]
```
(`hybrid_pipeline.py:568-608`)

The background thread (`_finish_chunk_cpu`) never touches `torch` and only ever handles the **previous**
tile's back end — it cannot fill these gaps, because they belong to the tile currently occupying the GPU.
Every one of the four CPU-only segments above forces a real GPU idle window, once per tile, three times per
tile's forward sequence. This is a direct, mechanical explanation for "GPU usage is intermittent": it isn't
only the documented 34%/36.8% cross-tile margin — the GPU is *designed* to stop and restart multiple times
within every single tile, by the shape of `_process_one_chunk_gpu` itself.

Doc 13's item ⑧ (Priority 2) already covers two of these four segments (`build_all_positive_results` +
`enlarge_cell_instances`). It was scoped as an **arm-balance** fix (move CPU work from the busier MAIN arm to
the slacker BG arm, shrinking wall). It was never scoped as a **bubble-elimination** fix, and moving those two
calls to the BG thread does not, by itself, remove the GPU-idle window they cause today — it only decides
which arm absorbs the wait. The `_run_m1_overlay_stage` CPU glue and `clear_slide_edge_cells` are not on
doc 13's radar as movable items at all (they weren't priced into ⑧'s 28.4 s, and are smaller — see §5's
measured breakdown, ~6 s/121 tiles combined at medium).

**What would actually close bubbles, not just relabel which arm absorbs them:** overlap this CPU glue with the
*next* GPU forward's kernel launch, not the *previous* tile's CPU back end. That needs either (a) a second
background thread/queue specifically prefetching the next tile's read + glue while the current forward runs on
a CUDA stream, or (b) restructuring `run_batch`'s pipeline depth from 1 to ≥2 so more than one tile's GPU-front
state can be in flight. Both are real architecture changes with correctness risk (ordering, thread-safety of
the three shared model objects, CUDA-stream synchronization) — not something to improvise inline. This
document is not proposing either as ready-to-build; it is naming the mechanism and asking for it to be sized
(§4, priority 2) before anyone assumes multiprocessing is the only way left to raise GPU occupancy.

## 4. Why this must be closed before the three named "big levers"

Each of the three levers the request asked about implicitly assumes the GPU itself is the saturated,
compute-bound resource and that more parallel work thrown at it will convert into wall-clock time saved. §0
and §3 show that assumption doesn't hold cleanly today — the GPU is idle 22–37% of the time (scale-dependent,
§5) for reasons that have nothing to do with insufficient batch size or insufficient concurrent tiles. Sizing
those levers against the wrong bottleneck risks the exact failure mode `PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`
warns about — real speedup far below what theory predicted, because the thing that was actually starving the
GPU (serial CPU glue, not insufficient batch depth) is still there afterward:

- **Cross-tile multiprocessing** — architecturally blocked today (fork-under-CUDA, one shared CUDA context
  across three resident models, §2). Making it work needs either per-process model reloading (VRAM ×N, model
  init cost ×N — both currently "negligible because amortized once," §…⑥ in `bottleneck-list.md`, would stop
  being negligible) or a multi-context/MPS design. That is a large, correctness-risky rewrite. It should not
  be scoped until it's known how much of today's idle time survives after §3's bubbles and doc 13's ⑤/⑧ are
  closed — some of what multiprocessing would "fix" may already close for a fraction of the engineering cost.
- **Extending Cellpose's batch size** — doc 13 Priority 4 already says this explicitly: the margin it would be
  sized against (currently stale at 34.0%/36.8%, pre-`gc.freeze()`) must be **re-measured**, not assumed, and
  §5 below does exactly that for the first time. A batch-size win bigger than the real current margin doesn't
  move wall further, it just relocates the bottleneck to the BG arm (②③) — the same warning doc 13 already
  gives, now with a number to check it against. Batch size also does nothing about the intra-tile bubbles in
  §3 — bigger batches make each forward call cover more tiles at once, which does not by itself remove the
  CPU-glue gaps *between* forwards; it could even make the bubbles proportionally worse if forward time per
  call grows while glue time doesn't shrink.
- **Moving tile/transform loading to the GPU** — measured evidence today says this specific lever has a small
  ceiling: `B2r_tile_read` (the file-read/decode step this would replace) is **1.16% of wall** at medium scale
  (§5) — an Amdahl ceiling of ~1.01x even at 100% elimination. There is also no existing CPU→GPU transform
  pipeline to "move" (§2) — this would be new construction, not a flip of an existing stage's device. Nothing
  in the current measurement record supports this as a high-value next step; it is not ranked in §4 below.

**Recommended reading of "before":** items 1–4 in §4's ranking are cheaper, lower-risk, and directly address
the measured idle time. Multiprocessing (the only one of the three big levers with a plausible large ceiling)
should be re-scoped, not abandoned — but sized against whatever idle_frac remains *after* 1–4 land, not
against today's 22–37%, some of which those four items already claim.

### Ranked priority for this stage

1. **Re-measure the MAIN/BG margin with `gc.freeze()` in place.** Doc 13 flagged this as required before
   sizing Priority 2 or 4 and it was never done formally. §5 does a first pass from the most recent local run
   (medium anchor only) — **re-run at the large anchor with the standard `--gpu-dmon` + `pip freeze` protocol
   (§0 of doc 13) before trusting it.**
2. **Land doc 13's Priority 2 (⑧ off the MAIN arm) — still the cheapest known lever.** Unchanged from doc 13;
   restated here because it's also partial cover for §3's bubbles (2 of 4 CPU-glue segments). Needs the
   thread-safety verification doc 13 already calls for.
3. **Size and, if it survives measurement, prototype §3's bubble-overlap fix.** New this document. Before
   building: instrument `_process_one_chunk_gpu` with `torch.cuda.Event` timestamps around each forward to
   measure the actual idle gap per CPU-glue segment (wall-clock proxies like `B1_unet_coremask`'s Python timer
   include Python-side overhead, not just true GPU-idle time — an Event-based measurement is the honest number
   here, and doesn't currently exist in `perf_measure.py`'s instrumentation). If the measured gap is small
   relative to kernel-launch/sync overhead already priced into ①'s 1.382x ceiling, this item may itself be
   near the Amdahl floor and not worth building — **measure before committing to the CUDA-stream/depth-2
   redesign**, per the same discipline that stopped out the GIL-contention avenue in `gil-contention-diag.md`.
4. **Land doc 13's Priority 3 (precut A + stitch D overlap).** Unchanged; the only other 100%-critical-path
   item with a real, if small, ceiling.
5. **Then, and only then, size multiprocessing / batch-size / GPU-transform against the residual idle_frac.**
   Re-run the full `--gpu-dmon` measurement after 1–4 land; whatever idle_frac remains is the honest ceiling
   those three levers are competing for. If it's still double digits, multiprocessing (with its correctness
   cost) becomes the only lever left that scales past what CPU-side rebalancing can buy — worth scoping
   properly at that point, not before.

## 5. First re-measurement of the MAIN/BG margin (medium anchor, provisional)

A local run completed today (not yet a recorded round in `bottleneck-list.md` — see caveat below) at the
medium/121-tile anchor: `docs/hybrid-pipeline/measurement/_metrics_local_20260722_1234/medium_121tile_*`.
`gc.freeze()` is confirmed active in this run (`B4_gc_collect` = 0.085 s / 121 calls ≈ 0.7 ms/call). Breaking
its `timings.json` down by arm, the way doc 13 §2 defines them:

| MAIN arm component | self-time | % wall |
|---|--:|--:|
| GPU forwards (`B1_unet_coremask` + `B1_m3b_cellpose`) | 130.35 s | 82.7% |
| `_read_rgb` (`B2r_tile_read`) | 1.83 s | 1.16% |
| M1 overlay CPU glue (`BM1_apply_mask`+`overlay_dish`+`fuse`) | 3.88 s | 2.46% |
| `clear_slide_edge_cells` (`Bs_clear_edge`) | 1.62 s | 1.03% |
| ⑧ `build_all_positive_results` + `enlarge_cell_instances` | 8.01 s | 5.08% |
| `empty_cache` + `gc.collect` | 0.50 s | 0.31% |
| **MAIN total** | **146.2 s** | **92.7%** |

| BG arm component | self-time | % wall |
|---|--:|--:|
| `detect_all_dots` + merge + filter | 77.49 s | 49.2% |
| PNG/TIFF encode + overlay render + per-cell crops | 27.72 s | 17.6% |
| **BG total** | **105.2 s** | **66.7%** |

`outside` (precut A + stitch D + model init) = 5.73 + 1.49 + 2.75 = 9.97 s. Arm-model prediction:
`max(146.2, 105.2) + 9.97 = 156.2 s` vs measured end-to-end **157.7 s** — within 1%, same validation quality
doc 13's own arm model reports.

**Margin: BG/MAIN = 105.2 / 146.2 = 0.719 → MAIN must shed 28.0% before BG becomes critical.** This is the
first actual re-measurement of the stale figure both doc 13 §2 and §4 Priority 2 flagged as needing
re-verification after `gc.freeze()` landed. It moved from the pre-fix **36.8%** (medium) down to **28.0%** —
tighter, as doc 13 predicted, but the pre-`gc.freeze()` number was overstating the room by only ~9 points, not
collapsing it. There is still real headroom for ⑧ (Priority 2/§4-2 above) before BG is re-exposed; there is
markedly less than the number currently printed in `bottleneck-list.md`.

**Caveats — do not treat this as a finished round-4 entry:**
- This is the **medium** anchor only. The corresponding large/441-tile run
  (`large_441tile_*` in the same directory) was still in progress at last check (log stopped at tile 336/441)
  — do not extrapolate this 28.0% to the large anchor without finishing that run.
- No `nvidia-smi` idle-check or `pip freeze` was captured alongside this run (doc 13 §0's discipline). The
  wall-clock (157.7 s) is *faster* than the recorded round-3 medium anchor (166.6 s) by ~5.4%, which is either
  normal run-to-run noise or an unrecorded environment difference — **don't fold this number into
  `bottleneck-list.md`'s official record until it's re-run under §0's protocol** (idle-GPU check, `--gpu-dmon`,
  `pip freeze` next to the metrics dir).
- `idle_frac` computed directly from `medium_121tile_gpu_dmon.txt` (1 Hz `nvidia-smi dmon`, 163 samples):
  **0.227**, mean SM **22.75%** — both between the overlap-round and round-3 large-anchor figures quoted in
  §0, consistent with the same intermittent pattern, not a contradiction of it.

**Action, not part of this plan's scope:** finish the large-anchor local run (or re-launch cleanly per §0),
capture `pip freeze` + a pre-launch `nvidia-smi` check, and fold both anchors into `bottleneck-list.md` as a
proper round-4 entry once §4's priority-1 item is done for real.

## 6. What not to do yet

Consistent with `13-next-optimization-plan.md` §1's "already closed" list — don't re-litigate these without
new evidence:

- Don't scope cross-tile multiprocessing until §4 items 1–4 are measured; the fork-under-CUDA constraint means
  it's a large redesign, and part of what it would buy may already be claimed by cheaper fixes.
- Don't sweep `cellpose_batch_size` against the stale 34.0%/36.8% margin — use the re-measured 28.0% (§5,
  medium-only, provisional) or a properly re-run large-anchor number, not the number currently printed in
  `bottleneck-list.md`.
- Don't build a GPU-side transform/decode path — `B2r_tile_read` is 1.16% of wall today (§5); the ceiling
  doesn't justify new construction ahead of items with measured double-digit ceilings.
- Don't build §3's bubble-overlap redesign before instrumenting the actual per-segment idle gap with
  `torch.cuda.Event` — this document names the mechanism, it does not yet have a measured ceiling for it.
