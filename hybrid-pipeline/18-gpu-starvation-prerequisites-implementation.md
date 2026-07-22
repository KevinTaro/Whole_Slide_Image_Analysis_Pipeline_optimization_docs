# 18 — GPU starvation prerequisites: implementation & measurement record (round 4)

> Executes [`17-gpu-starvation-prerequisites-plan.md`](./17-gpu-starvation-prerequisites-plan.md) §4,
> items 1–5, in order. Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose; cheapest lever first; every layer ablation-proved; correctness
> is a veto. **This document changes pipeline code** (unlike docs 13/17, which were planning-only) —
> see §8 for the complete diff surface.
>
> Round-4 anchors, git `2f89fea` + the changes in §8, RTX 5090 / driver 580.159.03, cellpose 4.2.1.1,
> torch 2.11.0+cu130, same `test_picture/_roi_crops/{med,large}` crops as rounds 1–3.
> Raw artifacts: `measurement/_metrics_r4/` (incl. `env_stamp_p0.txt`, `env_stamp_p2.txt`,
> `pip_freeze.txt`), per-run `report.csv`/`summary.txt` in `measurement/runs_r4/`.

## 0. Protocol

Per doc 13 §0, and this time actually recorded (the gap doc 17 §5 flagged in its own provisional run):

- GPU confirmed idle before every launch — **89 MiB / 0%, no compute processes**.
- `--gpu-dmon --workers 8` on every run; `pip freeze` + env stamp (incl. checkpoint SHA-256) beside
  the metrics. Model checkpoint hashes are **identical to round 3's**, so this round is comparable to
  round 3 on the correctness side.
- **n=3 at the large anchor, n=2 at medium**, per doc 16 follow-up #1 ("a single run at large is not
  interpretable; B1 varies ±3%"). Measured baseline spread this round: large **1.0%** (536.4–541.9 s),
  medium **1.6%**.

Configurations measured (each stacked on the previous, so each comparison isolates one layer):

| tag | code state |
|---|---|
| `p0` | HEAD `2f89fea` — `gc.freeze()` in, nothing from doc 17 built |
| `p2` | p0 + item 2 (⑧ moved off the MAIN arm) |
| `p3` | p2 + item 4 (precut A overlapped with the analysis loop) |
| `p0ev` / `p2ev` | `--cuda-events` instrumentation runs (item 3) |

## 1. Item 1 — re-measure the MAIN/BG margin with `gc.freeze()` in place

Doc 17 §4-1 required this before anything else could be sized, and flagged its own §5 pass as
medium-only and unstamped. Done properly here, at both anchors.

`scripts/arm_report.py` (new, §8) computes the doc-13 §2 arm model. It was validated first by
reproducing `bottleneck-list.md`'s **recorded round-3 numbers exactly** from the archived
`_metrics_cellpose421/` — MAIN 538.3 / BG 387.3 / outside 28.0 / predicted 566.3 vs 573.7 measured,
BG/MAIN 0.719, idle_frac 0.370 — so the decomposition is the project's existing method, not a new one.

**Round-4 baseline (`p0`):**

| | large/441 (n=3) | medium/121 (n=2) |
|---|--:|--:|
| end-to-end wall | **538.5 s** (536.4–541.9) | **154.0 s** (152.8–155.3) |
| MAIN arm | 503.4 s | 142.7 s |
| BG arm | 374.6 s | 105.1 s |
| outside (A + D + init) | 28.0 s | 9.6 s |
| arm-model prediction | 530.3 s (**−1.3%**) | 152.4 s (**−1.1%**) |
| **BG/MAIN** | **0.744** | **0.736** |
| **MAIN must shed** | **25.6%** | **26.4%** |

**The margin is 25.6% / 26.4%, not the 34.0% / 36.8% still printed in `bottleneck-list.md`.**
Doc 17 §5's provisional medium-only estimate (28.0%) was close but ~1.6 pt optimistic.

Cross-check that this baseline is sane: round 3 measured 573.7 s with a 36.4 s `gc.collect` cost;
`gc.freeze()` landed after round 3 and cut that to 0.21 s. 573.7 − 36.4 ≈ **537 s**, versus **538.5 s**
measured here. Doc 16's `gc.freeze()` claim is therefore independently confirmed end-to-end at the
large anchor by this round's baseline, which is a stronger check than doc 16's own bucket-decomposed one.

**Correctness reference for this round** (used as the bar for everything below): large **13152–13153**
cells / 378 success / 63 skipped; medium **3647–3649** / 103 / 18. Consistent with round 3's 13,150 and
3,647.

## 2. Item 2 — move ⑧ off the MAIN arm (doc 13 Priority 2) — **BUILT, ADOPTED**

`build_all_positive_results` + `enlarge_cell_instances` moved from `_process_one_chunk_gpu`
(main/GPU thread, between the M2 and M3b forwards) into `_finish_chunk_cpu` (background thread).

**Why it is computation-preserving** — established by reading, not assumed:
- Both are pure. `build_all_positive_results` only reads (`np.unique` + `ndimage.center_of_mass`);
  `enlarge_cell_instances` returns `expand_labels(...)`, a new array, and returns its input untouched
  when the factor is disabled. Neither mutates `instance_mask`.
- `matching_mask` / `results_pre` are referenced **nowhere outside `hybrid_pipeline.py`** and consumed
  only by `_finish_chunk_cpu`; nothing on MAIN reads them after the handoff, so no race is introduced.
  Both fields were deleted from `_ChunkGpuState` accordingly.
- `instance_mask` was already handed to the BG thread before this change. MAIN does not touch it after
  `_process_precut_tile_gpu` returns (the loop body only does gc / `empty_cache` / submit).
- Call order relative to `detect_all_dots` is unchanged, and the M3b forward never depended on either
  value — so the GPU work and its ordering are untouched.

**Result:**

| | large/441 (n=3) | medium/121 (n=2) |
|---|--:|--:|
| wall | **495.5 s** (488.1–499.6) | **146.3 s** (144.5–148.1) |
| vs baseline | **−8.0% (1.087x)** | **−5.0% (1.053x)** |
| MAIN | 503.4 → **458.5 s** | 142.7 → **134.5 s** |
| BG | 374.6 → **382.6 s** | 105.1 → **112.7 s** |
| **MAIN must shed** | 25.6% → **16.6%** | 26.4% → **16.2%** |

Doc 13 projected `~501 s` for P1+P2 combined at large. Measured **495.5 s** — the projection is
confirmed and slightly beaten.

**It beat its own arm-model projection, and the reason matters.** The model predicted MAIN −28.4 s.
Measured MAIN −44.9 s, because two *unmodified* buckets also got faster:

| large/441 | baseline | p2 | Δ |
|---|--:|--:|--:|
| B1 GPU forwards | 444.4 s | **431.6 s** | −12.8 s |
| `detect_all_dots` | 279.9 s | **257.6 s** | −22.3 s |
| ⑧ itself | 28.6 s | 29.8 s | +1.2 s (now competing on BG) |

Neither function changed. The mechanism is GIL contention, the same one
[`gil-contention-diag.md`](./measurement/gil-contention-diag.md) demonstrated in the opposite
direction: previously ⑧ ran on MAIN *interleaved* with `detect_all_dots` on BG, and the two fought for
the GIL; now they are serialized on one thread and MAIN is pure GPU-driving Python. **The arm model
does not capture this** — it treats arms as independent — which is why it under-predicted. Recorded so
future arm-model projections are read as lower bounds on placement changes, not point estimates.

**Correctness veto — passed.** Cell counts moved slightly (large 13146/13150/13148 vs baseline
13152/13153/13153), so counts alone were not sufficient; the per-cell veto
(`gc_ablation_report.py`, nearest-centroid matched) was run in all pairings:

| comparison | reddot max\|Δ\| | blackdot max\|Δ\| | score max\|Δ\| | X-flips | differing cells |
|---|--:|--:|--:|--:|--:|
| baseline vs baseline (**same code**) | 1 | 1 | 2 | 18 | 1 |
| **p2 vs p2 (same code)** | **2** | **8** | **4** | 15–21 | 1 |
| p2 vs baseline | 1–2 | 1–8 | 2–4 | 15–30 | 1–2 |

The decisive point: the `blackdot Δ=8` signature appears **between two runs of identical p2 code**
(r1 vs r3). Cross-config deltas never exceed within-config deltas, and only 1–2 of ~13,140 matched
cells differ in any pairing. This is GPU nondeterminism in the forwards, not a computation change —
consistent with the change being provably pure above.

**Memory invariant — holds.** VRAM (`dmon fb`) peak **2787 MB in all six large runs**, identical
across configs. Peak RSS 3.86–3.99 GB (baseline 3.92–3.95). See §7.2 on the `torch_reserved` counter.

## 3. Item 3 — size the intra-tile GPU bubbles with `torch.cuda.Event`

Doc 17 §4-3 required an Event-based measurement before any CUDA-stream/depth-2 redesign, and noted the
instrumentation "doesn't currently exist in `perf_measure.py`". It does now (`--cuda-events`, §8).

**Method.** Each GPU entry point records a timing Event immediately before the call and immediately
after it returns, on the default stream. Between two consecutive forwards no kernels are enqueued, so
the device time from forward *k*'s trailing event to forward *k+1*'s leading event **is** the idle
window. A Python `perf_counter` cannot answer this — it also counts time the GPU spent draining work
already queued.

**Baseline bubble map (`p0ev`, medium, 157.1 s wall):**

| segment | device-side idle | % wall | per tile |
|---|--:|--:|--:|
| M2 → M3b gap (`clear_slide_edge_cells` + ⑧) | **10.06 s** | 6.40% | 84.5 ms |
| UNet → M2 gap (M1 overlay glue) | **4.53 s** | 2.88% | 38.0 ms |
| tile boundary (`_read_rgb` + gc + `empty_cache` + BG join) | **3.07 s** | 1.96% | 25.6 ms |
| **total inter-forward idle** | **17.66 s** | **11.24%** | |

Two findings:

1. **The gaps match the Python timers to within 0.01 s** (Event 10.06 s vs `clear_edge` 1.65 +
   `build` 2.65 + `enlarge` 5.75 = 10.05 s; Event 4.53 s vs M1 glue 4.41 s). So the CPU glue is
   **100% GPU-idle** — there is no hidden asynchrony absorbing it. Doc 17 §3's mechanism is confirmed
   exactly as described.
2. **But the bubbles are a minority of total device idle.** Total device idle at this anchor is ~40 s;
   the §3 bubbles are 17.7 s of it (**~44%**). The remaining ~56% is *inside* the forward calls —
   the kernel-launch-bound Cellpose/SAM Python loops that `gil-contention-diag.md` already traced and
   **stop-lossed**. No pipeline-level restructuring reaches that portion.

**After item 2 (`p2ev`, medium, 143.1 s wall):**

| segment | baseline | p2 | Δ |
|---|--:|--:|--:|
| M2 → M3b gap | 10.06 s | **1.66 s** | **−8.40 s** |
| UNet → M2 gap | 4.53 s | 3.94 s | −0.59 s |
| tile boundary | 3.07 s | 3.09 s | +0.02 s |
| **total inter-forward idle** | **17.66 s** | **8.69 s** | **−8.97 s** |

The M2→M3b bubble closed to **1.66 s**, which is precisely `clear_slide_edge_cells` (1.62 s by Python
timer) — the only CPU work left between those two forwards. Item 2 removed **8.97 s of provable GPU
idle** at medium, i.e. **half of all inter-forward idle**, exactly as much as ⑧'s own cost.

### Decision: do **not** build the CUDA-stream / depth-2 redesign

Doc 17 §4-3 asked for this to be sized before committing, and made the call conditional. The measured
ceiling does not justify it:

- Remaining inter-forward idle after item 2 is **8.69 s / 6.07% of wall** at medium (≈1.065x if driven
  to *zero*, which no realistic design achieves).
- It is now spread thin across three segments (3.94 / 3.09 / 1.66 s) rather than concentrated in one,
  so a redesign must attack all three to collect it.
- Doc 17 itself prices the risk: ordering, thread-safety of three shared model objects, CUDA-stream
  synchronization. That is a large correctness-risky change against a ≤1.065x ceiling, while the
  cheapest remaining item (§4) is a ~30-line change.
- Item 2 already collected the single largest bubble for a 12-line move.

Per the playbook's stop-loss discipline and the precedent of `gil-contention-diag.md`, this avenue is
**recorded as sized and stopped-out at ~1.065x**, not carried as an open backlog item. Reopen only if
the intra-forward launch-bound idle (the other 56%) is ever fixed upstream, which would raise this
item's relative share.

## 4. Item 4 — overlap precut A and stitch D (doc 13 Priority 3)

Doc 13 P3 noted no design work existed. Two stages, two different outcomes.

### 4.1 Stitch D — not overlappable, recorded as a structural negative

D is `_stitch_overlay_slide`: `pyvips` row-then-column `join()` followed by one `tiffsave`. The joins
are **lazy** — they build a pipeline and do no work; all the cost is inside the single `tiffsave` C
call, which is why `bottleneck-list.md` ⑤ already records it as "read+join+compress fused in one C
call, not separable by Python timing". D also runs at the very end of `run_batch`, with no remaining
pipeline work to overlap it *with* (only the ~10 ms CSV export follows).

Overlapping it would require incremental row-wise encoding, which a pyramidal TIFF cannot do — the
pyramid needs the whole image. **D's overlappable content is ~0 s.** At 5.10 s / 1.0% of wall it is
not worth restructuring; recorded and closed, not carried as backlog.

### 4.2 Precut A — streamed, overlapped with the analysis loop

The enabling observation: the tile grid is derivable from the **image header alone**
(`m0_reader.read_size` decodes no pixels, `chunk_offsets` is pure arithmetic). So the geometry
`run_batch` needs upfront does **not** require the cutting to have happened. New `PrecutStream`
(§8) hands over `positions` immediately and yields `(ihc_tile, dish_tile, pos)` as each pair lands,
cutting on the same 8-thread pool via the same `_crop_to_tile` + deflate path.

**Why processing order is safe to change:** `run_batch` flattens every tile's cells and sorts by
`(abs_y, abs_x, cell_id)` before the single global renumbering, and `_stitch_overlay_slide` reads tiles
by coordinate. Both are order-independent, so the stream may emit in completion order. Verified
directly: `PrecutStream` produces an identical grid and **byte-identical tile files** to
`precut_paired_tiles` (sha256 over all 242 medium tiles, 0 differing).

**Result — adopted.** `phaseA_precut_s` drops to **0.004 s** (header read only), and `outside` falls
from 27.9 s to **7.64 s** (1.6% of wall — only stitch D and model init remain there).

| | large/441 (n=3) | medium/121 (n=2) |
|---|--:|--:|
| wall | **480.3 s** (479.8–481.0) | **140.8 s** (140.2–141.4) |
| vs `p2` | **−3.1%** | **−3.8%** |
| precut A recovered | 20.31 s → hidden; net −15.2 s (**75%**) | 5.6 s → net −5.5 s (**~98%**) |

Not all of A is recovered, as expected: the cutting threads now contend with the BG arm for CPU. The
75% recovery at large is the honest figure; the contention cost is the other 25%. The large-anchor
spread also tightened to **0.25%** (479.8–481.0), the tightest of any configuration this round.

Correctness veto passed with the same signature as the within-config noise floor (1 differing cell of
~13,140 matched; reddot ≤2, blackdot ≤8, score ≤4, X-flips 12–24 — all inside the baseline-vs-baseline
band). Cells 13148 in all three large runs, 3648 in both medium runs. Tiles 378/63 and 103/18, correct.

VRAM steady state 2785 MB, identical to every other config. (`p3_large_r2` shows a single 4219 MB
sample out of 480, at startup — a transient, not a level shift; steady state is 384 samples at 2785 MB.)

## 5. Consolidated results

| config | large/441 (n=3) | Δ vs prev | medium/121 (n=2) | Δ vs prev |
|---|--:|--:|--:|--:|
| round-3 record | 573.7 s | — | 166.6 s | — |
| `p0` baseline (`gc.freeze()` in) | 538.5 s | −6.1% | 154.0 s | −7.6% |
| `p2` (item 2: ⑧ off MAIN) | 495.5 s | **−8.0%** | 146.3 s | **−5.0%** |
| `p3` (item 4: precut streamed) | **480.3 s** | **−3.1%** | **140.8 s** | **−3.8%** |

**Cumulative this round: 538.5 → 480.3 s = −10.8% (1.121x) at large; 154.0 → 140.8 s = −8.6%
(1.094x) at medium.** Against round 3's recorded 573.7 s the cumulative improvement including
`gc.freeze()` is **−16.3%**.

Full-WSI linear refit on the two `p3` anchors: `wall ≈ 12.5 s + 1.0608 s/tile` ⇒ **~10.5 h** for
35,700 tiles (was ~12.6 h at round 3), same tissue-density upper-bound caveat as every previous round.

Arm state after `p3` (large): MAIN **453.9 s**, BG **381.7 s**, outside **7.6 s** — **BG/MAIN 0.841,
so MAIN has only 15.9% left to shed** before the background CPU arm becomes the critical path.

## 6. Item 5 — sizing the three big levers against the residual

This is doc 17 §4-5: the three levers may only be sized against what is *left* after items 1–4, not
against the pre-round idle. What is left, measured:

- **Provable inter-forward GPU idle: 6.07% of wall** (Event-based, §3) — the rest of the device idle
  is intra-forward and launch-bound, already stop-lossed.
- **MAIN-arm headroom before BG becomes critical: 15.9%** (large). Any MAIN-side win larger than this
  buys nothing further until BG also shrinks — the floor for all single-process levers is
  `BG + outside = 381.7 + 7.6 = ` **389.3 s**, i.e. **1.23x** from today's 480.3 s.

### 6.1 Extending Cellpose's batch size — **now the top-ranked open item**

Still unwired at HEAD: `hybrid_pipeline.py:206,218` call `getattr(config, "cellpose_batch_size", 16)`
and no such field exists on `Config` (the real `batch_size=4` at `config.py:184` feeds UNet++ only), so
16 has always been the only value ever used. Two facts found this round make it the best remaining bet:

1. At these tile sizes `segment_windowed` runs **exactly one window per call** (tile 1024 px ==
   `default_tile_size`), so `batch_size` is not about cross-tile batching at all — it governs
   **Cellpose's internal patch batching within one 1024² image**. Cellpose 4.x tiles that image into
   `bsize` patches with overlap (order 25–36 patches), so at 16 it takes 2–3 internal batches per
   forward. 32 or 64 would make it one.
2. That directly targets the *intra-forward launch-bound* idle which is the larger half of all device
   idle (§3) and which no pipeline-level restructuring can reach — fewer, larger internal batches means
   fewer Python-side launch iterations per unit of work.

VRAM headroom is ample: steady state 2785 MB of 32 GB. Ceiling bounded by the 15.9% margin ⇒ **≤1.19x**.
Cost: one `Config` field plus a sweep. **Do step 1 as a bit-exact no-op first** (default 16, flat
wall, `report.csv` within this round's noise floor), then sweep 16 → 32 → 64 at medium before large.

### 6.2 Cross-tile multiprocessing — scope only after 6.1, and size it honestly

Still architecturally blocked exactly as doc 17 §2 states (fork-under-CUDA; three models in one CUDA
context). Cost of unblocking: per-process model reload — VRAM 2.79 GB × N (fits ~10 processes in
32 GB) and model init 2.4 s × N, both currently negligible only because they amortize once.

Unlike every single-process lever, multiprocessing is **not** bounded by the 389.3 s BG floor, because
each process carries its own BG thread — so BG parallelizes too. Its real bound is the GPU's
serialized device-busy time. Mean SM is only **~20%** even now, so there is genuine room for
concurrent forwards to interleave. Plausible ceiling is therefore **between 1.23x** (if the BG side
somehow stayed serial) **and roughly 1.7x** (if both arms scale and the device fills) — the widest
ceiling of anything left, and also the largest correctness risk. **Scope it only if 6.1's sweep
disappoints**, and re-measure the margin again first; 6.1 may claim part of the same idle.

### 6.3 Moving tile/transform loading onto the GPU — **stopped out**

`B2r_tile_read` is **5.87 s = 1.22% of wall** at the large anchor post-`p3` (Amdahl ceiling **1.012x**
at 100% elimination), and there is still no CPU→GPU transform pipeline to move — it would be new
construction. Item 4 additionally removed the *other* I/O cost (precut A) by overlap rather than by
device change. Nothing in the measured record supports building this. Recorded as stopped-out, not
backlog.

**Ranked recommendation:** 6.1 (cheap, targets the larger idle half, ≤1.19x) → re-measure the margin →
6.2 only if 6.1 disappoints. 6.3 closed.

## 7. Methodology corrections found while doing this

Three measurement pitfalls surfaced; all three affect how earlier rounds should be read.

### 7.1 `idle_frac` (SM==0) is knife-edge and should not be compared across configurations

Every round so far has recorded GPU idle as *the fraction of 1 Hz `dmon` samples reading exactly
SM==0*. That definition is unstable: a sample reading SM=1% is idle in substance but not by that count.
Between p0 and p2 a cluster of SM=1–3% samples collapsed onto 0, so `idle_frac` **rose** 0.32 → 0.43
while wall **fell** 8% — reading it naively suggests item 2 starved the GPU, which is false.

Using `SM<=3` instead, near-idle **fell 0.50 → 0.45** (large) and 0.51 → 0.43 (medium), which is
consistent with the Event data and the wall-clock. `arm_report.py` now reports both. **Prefer the
cuda-Event gaps over either** when the question is "how long was the device provably doing nothing".

### 7.2 `peak_cuda_reserved_gb` is unreliable; read `dmon fb`

p2_large_r2 reported `torch_reserved` = **25.967 GB** while its `dmon fb` peak was 2787 MB — identical
to all five other large runs. This is the same sampling artifact doc 13 §4-P4 warned about (the "22.2 GB
`cuda_alloc_peak`" case). The VRAM invariant is intact; the counter is not trustworthy for peaks.

### 7.3 Arm membership is a property of the code version, not of the bucket name

After item 2, `B3_build_results`/`B3_enlarge_cells` are on the BG thread. Any analysis keyed on a
static bucket→arm map silently mis-attributes them (it inflated the arm-model error from −1.5% to
+4.2% until corrected). `arm_report.py` therefore takes `--moved`/`--moved-labels` explicitly rather
than guessing. A more robust fix — record the executing thread name per bucket in `perf_measure.py`,
making arm membership *measured* — is listed in §10.

## 8. Code changed

| file | change |
|---|---|
| `backend/algorithms/hybrid/hybrid_pipeline.py` | item 2: ⑧ moved `_process_one_chunk_gpu` → `_finish_chunk_cpu`; `matching_mask`/`results_pre` dropped from `_ChunkGpuState`. item 4: `run_batch(..., tile_stream=None)`; loop consumes an iterator; `_run_single_tile_cli` uses `PrecutStream` |
| `backend/algorithms/hybrid/m0_reader.py` | item 4: new `PrecutStream` (grid upfront, tiles yielded as cut) |
| `backend/api/hybrid.py` | item 4: `/api/hybrid/tile` uses `PrecutStream` |
| `scripts/perf_measure.py` | item 3: `--cuda-events` GPU-timeline instrumentation. item 4: `--stream-precut` |
| `scripts/arm_report.py` | **new** — two-arm decomposition, margin, idle attribution, Event buckets |

No config fields added; `config.py` / `config_example.py` untouched (hash unchanged).

## 9. Reproducing

```bash
# baseline / p2 / p3 anchors (label picks the config; code state is git history)
.venv/bin/python scripts/perf_measure.py \
  --ihc  backend/algorithms/hybrid/test_picture/_roi_crops/large_ihc.tiff \
  --dish backend/algorithms/hybrid/test_picture/_roi_crops/large_dish.tiff \
  --output <out> --label <tag> --workers 8 --gpu-dmon --metrics-dir <m> [--stream-precut]

# GPU-timeline bubble map (adds a per-tile synchronize -- size bubbles, don't compare wall)
... --label <tag>ev --cuda-events

# arm model + margin + idle attribution
.venv/bin/python scripts/arm_report.py --metrics-dir <m> --group --detail \
  --moved B3_build_results,B3_enlarge_cells --moved-labels p2

# correctness veto
.venv/bin/python scripts/gc_ablation_report.py --metrics-dir <m> \
  --runs-dir <runs> --reference <baseline report.csv>
```

## 10. Follow-ups this round raised

1. **Record the executing thread per bucket in `perf_measure.py`.** Arm membership is currently a
   hand-maintained map in `arm_report.py` (§7.3). `threading.current_thread().name` inside `_rec`
   would make it *measured*, and would have caught the mis-attribution automatically instead of via a
   +4.2% model error.
2. **`bottleneck-list.md`'s ①/⑧ entries and the 34.0%/36.8% margin are superseded** by §1 and §2 here;
   a round-4 entry has been added there pointing at this document.
3. **The two-arm model needs a third term now.** With precut streamed (§4.2), `wall ≈ max(MAIN, BG) +
   outside` under-predicts by 2–4% at medium because the cutting threads are a third concurrent
   producer. Fine for ranking, but stop quoting its prediction error as a validation of the model at
   `p3` and beyond.
4. **`detect_all_dots`'s ⑨ regression is partly self-resolving.** It measured 279.9 s at the `p0`
   baseline and **257.6 s** at `p2` — without being touched (§2). Any future attempt to isolate the
   round-3 +22.3% regression must control for thread placement, or it will chase a number that moves
   for unrelated reasons.
5. **`clear_slide_edge_cells` is now the only CPU work left between the M2 and M3b forwards** (1.66 s
   / 14.0 ms per tile of provable GPU idle, §3). It is the natural next candidate if anyone revisits
   bubble-closing — but at 1.2% of wall it is below any reasonable floor on its own.
