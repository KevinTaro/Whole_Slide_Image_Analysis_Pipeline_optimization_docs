# 23 — Next optimization cycle: implementation & measurement record (round 6)

> Executes [`22-next-optimization-cycle-plan.md`](./22-next-optimization-cycle-plan.md), in its own
> sequencing order, under
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)'s
> Discover → Analyze → Plan → Choose discipline: cheapest signal first, every layer
> ablation-proved, **correctness is a veto**. Doc 22 was planning-only; **this document changes
> pipeline code** — the complete diff surface is §8.
>
> Round-6 anchors, git `a4a6254` + the changes in §8, RTX 5090 / driver 580.173.02 / Compute Mode
> `Default`, 20 CPU cores, cellpose 4.2.1.1 (`cpdino` / `dino_vitb` backbone), torch 2.11.0+cu130,
> numpy 1.26.4, scikit-image 0.24.0, joblib 1.5.3. Config hash **`ad41c42f` → `3d1087f2`** (one
> field added this round, §8). Same `test_picture/_roi_crops/{small,med,large}` crops as rounds 1–5;
> checkpoint SHA-256s unchanged from round 5. Raw artifacts: `measurement/_metrics_r6/`
> (incl. `env_stamp_r6.txt`, `pip_freeze_r6.txt`).
>
> **Headline:** the two GPU-batching tracks doc 22 ranked highest (B1, B1b) are **both stop-lossed
> on measurement** — the traced root cause was real, but activating it does not pay. The audit doc
> 22 ranked third (A2) instead found a **different and larger defect than the one it was written to
> look for**: `detect_all_dots`'s `joblib` fan-out is not an over-subscription risk under
> multiprocessing, it is a **net loss at every process count, including one** — 20 threads are
> **2.77x slower** than plain serial for identical output. Fixing it is a one-line config change.

## 0. Protocol

Same protocol as doc 18 §0 / doc 21 §0, so this round stays comparable to rounds 3–5:

- GPU confirmed idle before **every** launch; the run loops block on `memory.used < 200 MiB`
  between runs rather than assuming it.
- `--gpu-dmon --workers 8 --stream-precut` on every end-to-end run; `pip freeze` + env stamp
  (incl. checkpoint SHA-256) beside the metrics (`19-open-backlog.md` §1 item 9).
- **n=2 per configuration** at the large/441 anchor, control and candidate **interleaved**
  (`w1 nj-1, w1 nj1, w6 nj-1, w6 nj1`, then the same again) so machine drift hits both arms.
- Microbenchmarks: 3 reps, **min** reported (min is the least noise-contaminated estimator for a
  latency floor), with the full rep spread recorded so a difference smaller than the spread is
  never read as a result.
- Correctness veto on every candidate, judged against the **same-code noise floor** rounds 4–5
  established, not against exact equality.

## 1. Scope decisions taken before any measurement

Doc 22's §5 sequencing opens with the full-WSI validation (its step 0) and derives A1 from it.
**The project owner scoped that out for this round**, and the reasoning is recorded here because it
changes what several items below could be measured against:

| doc 22 item | decision | reason given |
|---|---|---|
| **step 0** — formalize full-WSI validation | **Not run this round.** Item 7 in [`19-open-backlog.md`](./19-open-backlog.md) stays open. | Tiles are processed independently, so the small/medium/large crops are sufficient to measure throughput, memory stability and correctness; an ~11 h run + ~270 GB of transient output was judged not worth the displacement of core optimization work; a teammate has separately confirmed a complete slide runs end-to-end without crashing. |
| **A1** — worker re-tune vs real tissue density | **Re-scoped to the crops** (§6). The real-density question stays open with step 0. | Same decision. Round 5b had already swept `workers` 1–12 on the crops; what was still worth measuring was whether the §4 fix moves that recommendation. |
| **A3** — multi-request load test | **Logged, not built.** | Concurrent slide submission is not confirmed as part of the deployment plan; doc 22 §2 gates A3 on exactly that confirmation. |

Two facts found while sizing step 0 are worth recording anyway, because both contradict assumptions
that this document set has been carrying since round 2:

1. **The full slide is 27,565 tiles, not 35,700.** `HER2_processed.tiff` / `DISH_processed.tiff` are
   141,818 × 114,366; at `default_tile_size=1024` with `window_overlap_px=256` (stride 768) the grid
   is 185 × 149 = **27,565**. Every "35,700-tile" projection in docs 19–22 is therefore ~30% too
   pessimistic in tile count.
2. **"A real slide is mostly white background" does not hold for this input.** The aligned output is
   already cropped to the tissue ROI: over a stride-768 thumbnail the background level is ~213 (not
   255) and only **~39%** of grid cells sit above it. The load-balance and "background tiles
   short-circuit cheaply" arguments in doc 22 §1/§A1 were written for a much emptier slide and need
   re-deriving against this number whenever step 0 is actually run.

---

## 2. B1 — Cellpose cross-tile batching: root cause confirmed, then stop-lossed on measurement

Doc 22 §3 traced the flat round-4 `cellpose_batch_size` sweep to the **call site**, not the GPU, and
predicted that a genuine stacked `(Lz,H,W,C)` array would reach `cellpose.core.run_net`'s
multi-image batching path. **Step 1 was to verify that reading, then benchmark it before building
anything.** Both were done.

### 2.1 The traced mechanism is real (verified against the pinned source, not assumed)

Confirmed by reading `.venv/lib/python3.11/site-packages/cellpose/{models,core}.py` at the pinned
4.2.1.1:

- `models.py:231` — `if isinstance(x, list) or x.squeeze().ndim == 5:` → loops one image at a time.
  **A Python list of tiles buys exactly nothing**, as doc 22 said.
- A genuine 4-D ndarray instead falls through to `_run_net` → `core.run_net`, where
  `nimgs = max(1, batch_size // ntiles)` packs **multiple images'** patches into one `IMGa` buffer
  (`core.py:207-224`) and `_forward` runs them together.
- At 1024px with the `cpdino` backbone's `bsize=384`, `ntiles = 4×4 = 16` per image — exactly the
  `cellpose_batch_size=16` already in use, which is why the round-4 sweep had nothing to group.
  This reproduces round 4's finding from the source rather than re-running the sweep.
- Per-tile identity survives: with `do_3D=False` and `stitch_threshold=0` cellpose logs
  *"3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only"* and
  returns one mask per input plane.

**So the wrapper limitation doc 22 identified is confirmed, and the batching machinery does exist
in the pinned dependency, unmodified.** The remaining question was the one doc 22 could not answer
from source: does using it actually cost less per tile?

### 2.2 Microbenchmark — it does not

`scripts/cellpose_batch_probe.py`, 8 real tissue tiles from the large crop, **real per-model
inputs** (M2 gets the M1 overlay produced by the actual UNet++/M1 stage; M3b gets the raw DISH
tile), warm cache, 3 reps, min reported. `G` = tiles stacked into one `eval` call, with
`batch_size = 16*G` so `run_net` actually groups them.

| model | G | full `eval` ms/tile | Δ vs G=1 | `run_net`-only ms/tile | rep spread | peak alloc |
|---|--:|--:|--:|--:|--:|--:|
| **M2** (cell) | 1 | **253.1** | — | 193.6 | 1.4% | 1,167 MB |
| | 2 | 238.6 | −5.7% | 184.3 | 7.1% | 2,146 MB |
| | 4 | 254.6 | +0.6% | 198.5 | 6.1% | 4,102 MB |
| | 8 | 259.4 | +2.5% | 204.1 | 16.3% | 8,014 MB |
| **M3b** (DISH nucleus) | 1 | **245.2** | — | 196.0 | 2.6% | 1,165 MB |
| | 2 | 242.1 | −1.3% | 188.1 | 1.7% | 2,143 MB |
| | 4 | 253.9 | +3.5% | 202.2 | 0.7% | 4,100 MB |
| | 8 | 265.2 | +8.2% | 211.1 | 1.7% | 8,012 MB |

**Read against doc 22's own stop rule** ("If grouping doesn't reduce per-tile time, the
fixed-per-call-overhead hypothesis is wrong — stop here, cheaply"):

- The only non-negative cell is **G=2**, at −5.7% (M2) / −1.3% (M3b) — and M2's is **inside its own
  7.1% rep spread**, i.e. not distinguishable from noise. At G=4 and G=8 both models get
  monotonically **worse**.
- **The hypothesis is refuted, and the split says why.** Splitting each call into the batchable part
  (`run_net`, measured with `compute_masks=False`) and the non-batchable per-plane part shows the
  forward is **76%** of the call and does not amortize: 193.6 → 184.3 → 198.5 → 204.1 ms/tile. Cost
  is genuinely **per-patch proportional** inside the DINOv3 backbone, not fixed-per-call, so
  grouping tiles does not reduce total patch work. The ~50–60 ms/tile of `_compute_masks`
  (dynamics) is per-plane by construction and cannot batch at all.
- **The price is the resource that actually caps this pipeline.** Peak allocation scales linearly
  with G (1.17 → 8.01 GB). VRAM per process is exactly what caps worker count (round 5:
  2787/3117/4118/5167 MB at N=1..4; round 5b: `workers≥12` cannot even load). Spending 2–8 GB to buy
  0–5% per tile is a strictly worse trade than spending that VRAM on another worker, which round 5b
  measured at ~10–20% each.
- **Correctness (recorded, though moot given the above):** masks were **bit-identical** at every
  group size — cell counts identical per tile and **0.0000%** differing pixels vs G=1. The
  float-ordering risk doc 22 flagged did not materialize for cellpose at this granularity.

**Disposition: stop-lossed, do not build the N-tile gather stage.** Doc 22 §3's cost list (pipeline
restructure, overlap-model re-validation, interaction with `workers=N`) is all still accurate — it
simply never gets paid for. This closes the reopened cross-tile-batching question with a measured
answer rather than an architectural one; `cellpose_batch_size` stays at 16 and stays closed.

---

## 3. B1b — UNet++ cross-tile batching: the research lead is a dead end, and batching loses anyway

Doc 22 §3 flagged a "concrete research lead, not yet followed up": `cell_mask/unet_mask/inference.py`
has a `predict_batch` method the hybrid pipeline does not use, which "may already contain a working
multi-image batching implementation … that can be ported."

**It does not.** `predict_batch` (`inference.py:279-337`) is a `for img_path in tqdm(image_paths)`
loop that calls `predict_single` **one image at a time** and writes mask/proba/overlay PNGs per
image. It is a file-level convenience wrapper, not a batched forward — there is nothing to port.
(Same shape of finding as cellpose's `isinstance(x, list)` path in §2.1: a method whose name says
"batch" while the body is a serial loop.)

Measured anyway, since the alternative was designing it from scratch — `scripts/unet_batch_probe.py`,
same 8 tiles, stacking G tiles into one `(G,3,1024,1024)` forward:

| G | ms/tile | Δ vs G=1 | peak alloc | max px diff vs G=1 |
|--:|--:|--:|--:|--:|
| 1 | **23.78** | — | 1,171 MB | — |
| 2 | 25.11 | +5.6% | 2,258 MB | 0.0024% |
| 4 | 27.93 | +17.5% | 4,428 MB | 0.0058% |
| 8 | 27.81 | +16.9% | 8,769 MB | 0.0052% |

Cross-tile batching is **strictly worse at every group size**, and unlike cellpose it is not even
numerically identical (tiny argmax flips at class boundaries from reduction-order changes — real,
but negligible). Note also that at the pipeline's 1024px tile the tile *equals* `unet_image_size`,
so `predict_single` takes `_predict_direct`; `_predict_sliding_window`'s `self.batch_size` grouping
is never reached at all.

Cross-check on the harness: 23.78 ms/tile × 441 = 10.5 s, consistent with the 13.68 s / 441 calls
recorded at the round-3 anchor (which also carries per-call Python overhead this probe excludes).

**Disposition: stop-lossed.** It was Amdahl-capped at ~0.4% of wall before it started (doc 22 said
so); it is also negative on its own terms. Doc 22 §4's `torch.compile` / CUDA-graph back-pocket
items for this same forward inherit the same ~0.4% cap and stay unscheduled.

---

## 4. A2 — CPU-core contention audit: the hypothesis was wrong, and what replaced it is bigger

Doc 22 §2's hypothesis: `detect_all_dots` runs `Parallel(n_jobs=-1, prefer='threads')` **inside**
each worker process (`m3_dot_detection.py:194`), so at `workers=N` the machine runs N processes each
fanning out over all cores, over-subscribing N-fold and silently eating into the measured speedup.
Doc 22 called the **audit** the research, with "cap `n_jobs` per worker" as the one-line fix if a
problem showed up. The audit was run exactly as scoped. **The over-subscription hypothesis is
refuted; a larger defect was found underneath it.**

### 4.1 Method

`scripts/cpu_contention_probe.py`, in two phases so the CPU measurement never shares the box with
GPU work:

- `--prepare` runs the **real** GPU front (M1 → M2 → M3b) over 8 real tissue tiles once and pickles
  the resulting `_ChunkGpuState`s — i.e. the exact inputs the BG arm consumes (22–98 cells/tile,
  in line with the ~35 cells/tile average of the 441-tile crop).
- `--run` replays the whole BG arm (`build_all_positive_results` + `enlarge_cell_instances` +
  `detect_all_dots` + merge) from those pickles in **P concurrent processes**, barrier-synchronised
  so the timed window is genuinely concurrent, and reports per-call wall time.

This is a *lower* bound on real contention: in the pipeline each process also runs a GPU/MAIN thread
that holds a core, which a CPU-only replay does not reproduce. Machine: 20 cores, `n_jobs=-1` → 20
threads per process.

### 4.2 What the audit found

| P (processes) | `n_jobs` | threads total | `detect_all_dots` median | BG arm / tile | throughput |
|--:|--:|--:|--:|--:|--:|
| 1 | −1 (today) | 20 | **694.8 ms** | 771.0 ms | 1.19 calls/s |
| 3 | −1 (today) | 60 | **329.8 ms** | 410.1 ms | 6.13 calls/s |
| 6 | −1 (today) | 120 | **364.2 ms** | 459.8 ms | 11.11 calls/s |

Per-call time going **down** as concurrency goes up is the opposite of over-subscription, and it is
not explainable by contention at all — so the audit was widened into an `n_jobs` sweep:

| `n_jobs` per proc | P=1 | P=3 | P=6 |
|--:|--:|--:|--:|
| **1 (serial)** | **252.4 ms** | **274.2 ms** | **287.9 ms** |
| 2 | 288.6 | 304.4 | 313.2 |
| 4 | 315.5 | 323.5 | 346.0 |
| 8 | 343.0 | 329.7 | 357.2 |
| −1 (all 20) | 680.5 | 334.3 | 358.9 |

Confirmed independently in a single process, with the dot counts checked on every setting:

| `n_jobs` | 1 | 2 | 3 | 4 | 6 | 8 | 12 | −1 (20) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ms/tile | **260.3** | 296.7 | 309.6 | 319.7 | 335.4 | 343.2 | 527.0 | **721.5** |
| red/black dot counts | ref | same | same | same | same | same | same | same |

**The fan-out is a negative optimization, monotonically, at every process count.** 20 threads are
**2.77x slower** than plain serial (721.5 vs 260.3 ms/tile) for **identical** output. The reason
per-call time *improved* from P=1 to P=3 in the first table is now obvious: at P=3 the OS cannot
actually give each process 20 runnable cores, so each process is forced closer to the serial
behaviour that was faster all along.

Mechanism: each `_detect_one_cell` task is a small, largely Python-level LAB/morphology job (~4 ms;
only a few dozen per tile), so joblib dispatch + GIL hand-off dominate the work being distributed.
This is **quickref anti-pattern #10 — "mistaking 'looks parallel' for 'is parallel'"** — and it also
corrects `measurement/bottleneck-list.md` item ②, which has described this stage as "already
CPU-parallel" since the control round. It is parallel in shape only.

### 4.3 The change

`detect_all_dots` already exposed an `n_jobs` parameter (default `None` → `-1`); nothing but the
call site had to change.

- `config.py` / `config_example.py`: new field `dot_detect_n_jobs: int = 1` (config hash
  `ad41c42f` → `3d1087f2`).
- `hybrid_pipeline.py:636` (`_finish_chunk_cpu`): passes `n_jobs=config.dot_detect_n_jobs`.

### 4.4 End-to-end ablation — 1.60x at `workers=1`, nothing at `workers=6`

large/441 anchor, n=2 per configuration, control and candidate interleaved,
`--gpu-dmon --workers 8 --stream-precut`, GPU verified idle before each launch.

| config | `dot_detect_n_jobs=-1` (today) | `=1` (candidate) | Δ |
|---|--:|--:|--:|
| `workers=1` (**production default**) | 471.9 / 497.5 s (mean **484.7**) | 302.7 / 302.8 s (mean **302.7**) | **−37.5% = 1.60x** |
| `workers=6` (round-5b recommendation) | 121.2 / 120.4 s (mean **120.8**) | 120.7 s (n=1 valid, §4.6) | **~0%** |

The controls reproduce the existing record — `workers=1` 484.7 vs round 5's 482.8 s, `workers=6`
120.8 vs round 5b's 123.3 s — so both arms are measured against real anchors, not strawmen.

**Where the 1.60x comes from — and it is not where the change was made.** Per-bucket, at
`workers=1`:

| bucket | arm | `n_jobs=-1` | `n_jobs=1` | Δ |
|---|---|--:|--:|--:|
| `B3_detect_dots` | BG | 267.1 s | 100.2 s | **−62.5%** |
| **`B1_m3b_cellpose`** (both Cellpose forwards) | **MAIN** | **410.2 s** | **232.0 s** | **−43.4%** |
| `B2_png_encode` | BG | 78.4 s | 70.6 s | −10.0% |
| `B1_unet_coremask` | MAIN | 13.3 s | 13.6 s | +2.3% |
| `B3_enlarge_cells` / `B3_build_results` | BG | 19.0 / 11.1 s | 18.7 / 11.0 s | ~0 |

**The GPU forwards got 178.2 s faster without one line of GPU code changing.** Only the number of
`joblib` threads in the *background* arm changed. Those 19 surplus threads were competing for the
GIL with the main thread that drives the Cellpose forwards — and the main thread's own Python
overhead (cellpose's `dynamics.py` kernel-launch loops) is exactly the thing
[`gil-contention-diag.md`](./measurement/gil-contention-diag.md) measured as holding the GIL 81.4%
of the time. Starve that thread of the GIL and the whole MAIN arm dilates.

**This closes an open item in the measurement record.** `bottleneck-list.md` ① recorded that B1
grew **+192.9 s in absolute seconds** when the two-arm overlap landed, and logged the cause as
*"consistent with — but not yet directly isolated as — GIL contention from the background CPU
stage … not proven, high-confidence hypothesis"*, with the isolating measurement left as item 2 of
the re-sorted list. **That isolation now exists**: removing the background arm's surplus threads
returns **−178.2 s** to the same bucket — same magnitude, same direction, same mechanism. The
hypothesis is confirmed and the item can be closed.

**Why `workers=6` gains nothing, and why that is consistent rather than contradictory.** Round 5
attributed multiprocessing's surprising 3.09x (vs a 1.23–1.7x estimate) to recovering exactly this
GIL contention, which "only separate processes can recover". That reading was right about the
mechanism and wrong only about the exclusivity: **`workers=6` had already recovered this loss, so
there is nothing left for the `n_jobs` fix to take.** The two changes are alternative routes to the
same recovery and **do not stack**. At `workers=6` the pipeline is now GPU-bound (~120 s at both
settings, and 6 processes cannot get 20 cores each anyway).

**The practical consequence is the opposite of "no gain", though.** `run_batch(workers=1)` is
**still the production default** — the API path never passes `workers`, and round 5's
multiprocessing is *not cleared to ship* pending the full-WSI gate (item 7, still open per §1).
So the 1.60x lands on the configuration production actually runs, with **no** new process model, no
VRAM multiplication, and no dependency on that gate:

- single-process large/441: **484.7 → 302.7 s**;
- single-process per-tile: 1.0608 → **0.686 s/tile**;
- full-WSI single-process projection at the corrected 27,565-tile grid (§1): **~8.1 h → ~5.3 h**
  (same linear-extrapolation caveat as every other projection in this document set);
- and it narrows the gap to multiprocessing from 4.0x (484.7/120.8) to **2.5x** (302.7/120.7),
  which changes the cost/benefit of shipping `workers>1` at all.

### 4.5 Correctness veto — passed

Same method and same reference standard as doc 18 §2 / doc 21 §4.3: per-cell reddot / blackdot /
score matched by nearest centroid against a `workers=1` reference run, judged against the
**same-code noise floor**, not exact equality.

| run | cells (ref 13145) | reddot max\|Δ\| | blackdot max\|Δ\| | score max\|Δ\| | X-flips |
|---|--:|--:|--:|--:|--:|
| `w1_njm1_r2` (**same-code control**) | 13152 | 0 | 0 | 0 | 15 |
| `w1_nj1_r1` (candidate) | 13147 | 1 | 1 | 2 | 12 |
| `w1_nj1_r2` (candidate) | 13144 | 2 | 8 | 4 | 21 |
| `w6_nj1_r2` (candidate) | 13151 | 0 | 0 | 0 | 3 |

The candidate's worst case is **reddot 2 / blackdot 8 / score 4** — bit-for-bit the fingerprint doc
21 §4.3 recorded between two runs of *identical* code — and its 21 X-flips equal the same-code
control's recorded ceiling of 21, with this round's own same-code control landing at 15 flips and
+7 cells on its own. Only 1 cell in ~13,140 differs in any pairing. **Veto passed.**

This is also corroborated from the other direction: the §4.2 probe compared red/black dot counts
across every `n_jobs` setting on identical inputs and found them **exactly equal** (`n_jobs` cannot
change results — it only changes how the per-cell tasks are dispatched). The residue above is
ordinary GPU non-determinism upstream in segmentation, not an effect of this change.

### 4.6 A `workers=6` run failed — the known fragmentation OOM, not this change

`w6_nj1_r1` aborted (fail-fast, as designed) with `torch.OutOfMemoryError` at
`tile_x1536_y0`. The allocator dump shows one worker holding **24.76 GiB** while its five siblings
held 1.11–1.69 GiB each, against a ~2.8 GB steady state. That is precisely the non-deterministic
**CUDA allocator fragmentation** signature doc 21 §4.7 documented at `workers=7` ("某個 worker 單獨
吃到 7.7–9.4 GB，是穩態 ~2.8 GB 的 3–4 倍"), here observed at `workers=6`.

`dot_detect_n_jobs` cannot cause it — it changes CPU thread counts only and allocates no CUDA
memory; the control arm at the same worker count ran the same tiles without incident, and the
failure is on the candidate arm of an interleaved pair.

**It is still worth recording as evidence about `workers=6`, not about this change.** Doc 21 §4.7
explicitly cautioned that its 6 trials were "enough to distinguish `workers=6` from `workers=7`'s
already-visible ~25% failure rate, not enough to bound `workers=6`'s true failure rate". This round
adds **2 failures in 6 `workers=6` runs** (this one, plus `a1_w6_nj1_r2` in the §6 sweep) — a
materially higher rate than doc 21's 0/6, and enough to change the recommendation. §6 re-walks the
curve and §6's closing note records what the two failures have in common.

---

## 5. B2 / B3 — the gate did not open, and the reason it stays shut has changed

Doc 22 §3 made both GPU-port items **conditional on A2**: B2 (`detect_all_dots` → CuPy/cuCIM) had
"no standalone wall-clock case at ceiling 1.013x", and its only live rationale was *"if A2's audit
finds CPU-core oversubscription, the fix might be 'do less CPU work per process', and a GPU port
directly relieves CPU-core pressure"*. B3 (`enlarge_cell_instances` / `build_all_positive_results`)
was explicitly bundle-only behind B2.

**A2 found no over-subscription** (§4.2) — and the CPU-pressure problem it *did* find is solved by
setting one integer, which relieves far more CPU pressure than a GPU port would: each worker process
now uses **1 core instead of 20** for dot detection, at 2.77x less CPU time per tile, with no CuPy
dependency, no new numerics path, and no correctness risk (§4.5).

**Disposition: do not build B2 or B3.** Doc 22's own stop-loss applies verbatim ("if A2 finds no CPU
contention … do not build this"), and the CPU-core-freeing argument that was its only remaining
rationale has been satisfied more cheaply. Both stay logged, not built. Note the §7 trace
independently confirms the BG arm is *not* the constraint: `detect_all_dots`'s own leaves
(`_binary_erosion`, `xyz2lab`, `rgb2xyz`) are 1.2–2.2% of sampled activity on an arm that has no
cellpose work on it at all.

---

## 6. A1 — worker-count re-tune (crop scope): the recommendation moves down, from 6 to 4–5

Doc 22 §2's A1 was written to re-tune `workers` against **real full-WSI tissue density**; that half
is out of scope this round (§1) and stays open. What was still worth measuring on the crops is
whether §4's change moves the curve round 5b established — it does, and in the useful direction.

large/441, `dot_detect_n_jobs=1`, n=2 per point, same protocol as §4.4:

| `workers` | round 5/5b (`n_jobs=-1`) | **round 6 (`n_jobs=1`)** | Δ | vs the w=8 floor | VRAM (round 5) |
|--:|--:|--:|--:|--:|--:|
| 1 | 482.8 s | **302.7 s** | −37.3% | 2.54x off | 2,787 MB |
| 3 | 156.1 s | **142.9 s** (142.6 / 143.1) | −8.5% | +19.9% | 12,354 MB |
| 4 | 137.4 s | **128.8 s** (129.6 / 128.0) | −6.3% | +8.0% | 20,667 MB |
| 5 | — | **122.7 s** (123.2 / 122.1) | — | +2.9% | — |
| 6 | 123.3 s | **119.9 s** (n=1 valid; 1 OOM) | −2.8% | +0.6% | — |
| 8 | — | **119.2 s** (119.3 / 119.0) | — | — | — |

**The curve is flat past `workers=5`.** Going 5 → 6 buys **2.3%** and 6 → 8 buys **0.6%**: the
pipeline is GPU-bound at ~119 s on this anchor, and additional workers buy almost nothing while
each one adds VRAM and OOM exposure. That is a materially different trade than round 5b faced,
where 5 → 6 was still worth 10.3% over `workers=4`.

**And `workers=6` failed twice more this session** (§4.6): `w6_nj1_r1` and `a1_w6_nj1_r2`, i.e.
**2 failures in 6 `workers=6` runs**, against round 5b's 0/6. Both show the *same* signature — one
worker holding **exactly 24.76 GiB** while its five siblings sit at 1.1–1.7 GB — but with
**different victim tiles** (`tile_x1536_y0`, `tile_x5376_y0`) and different requested sizes (80 vs
48 MiB). The victim is random, as fragmentation would predict; the balloon repeating at exactly
24.76 GiB is not, and suggests a reproducible pathological allocation rather than gradual drift.
**This is not caused by the §4 change** (it allocates no CUDA memory, and one failure is on each
side of the same-tile workload), but it is now the dominant risk at `workers≥6` and it deserves its
own investigation — logged as a new backlog item rather than chased here.

**Recommendation: `workers=4` for unattended jobs, `workers=5` when a restart is cheap.** This
supersedes round 5b's `workers=6` and, notably, *agrees* with doc 21 §4.7's own advice for
unattended full-slide runs. `workers=4` is now within **8%** of the achievable floor; the 2.3%
that `workers=6` adds does not justify a failure mode that voids an entire batch under fail-fast.

**Caveat, unchanged from §1:** this is the crop curve. Whether it holds at full-slide scale — where
tissue density, queue balance and per-process VRAM growth over 27,565 tiles all differ — is still
item 7's question, and item 7 is still open.

---

## 7. B4 — the 4.0.8 kernel-launch trace re-run against 4.2.1.1: one function is gone, the ceiling shrank

Doc 22 §3's B4 was administrative: `19-open-backlog.md` item 5's "launch-bound, not placement-bound,
~1.23x ceiling" diagnosis was traced against `cellpose==4.0.8` with the SAM ViT backbone, and round
3 replaced that with `cpdino`/DINOv3 — so the trace needed re-confirming before anyone reopens or
re-closes the item. Re-run with `py-spy record` (`--gil` and all-thread), 100 Hz, over the medium
crop at `workers=1`, aggregated by `scripts/gil_trace_report.py`.

**Attribution note:** `run_batch` runs two arms concurrently, so a share of *total samples* is not
a share of wall-clock. The table below therefore splits the trace by arm (the stack root separates
them: `<module>` = MAIN/GPU thread, `_bootstrap` = the background thread pool) and reads the
Amdahl ceiling off the **MAIN arm only**, which is the critical path.

| the five functions the 4.0.8 trace named | 4.0.8 record | round-6 MAIN-arm self (wall trace) |
|---|---|--:|
| `_extend_centers_gpu` (`dynamics.py:23`) | named, launch-bound | **5.46%** |
| `steps_interp` (`dynamics.py:325`) | named, launch-bound | **2.09%** |
| `fill_holes_and_remove_small_masks` (`utils.py:621`) | the one true CPU-only item | **1.68%** |
| `get_masks_torch` (`dynamics.py:505`) | named, launch-bound | **1.35%** |
| `get_rel_pos` (`segment_anything/.../image_encoder.py`) | named, launch-bound | **0.00% — not on the path at all** |
| **combined** | ~19% of wall → **1.23x** ceiling | **10.58% → 1.118x ceiling** |

**Three things changed, and all three strengthen the existing stop-loss:**

1. **`get_rel_pos` is gone.** The model is now `CPDINO` / `dino_vitb`; `segment_anything` is still
   installed but the backbone never calls it. **Zero** samples in either trace. One fifth of the
   original item-5 target list no longer exists.
2. **The ceiling fell from ~1.23x to ~1.118x** on the critical arm. The other four functions do
   still exist (at shifted line numbers: `dynamics.py` 21→23, 311→325, 488→505; `utils.py`
   619→621), and are still launch-bound Python loops as described — the diagnosis is *qualitatively*
   confirmed — but they are worth less than they were. Patching pinned third-party internals for
   ≤1.12x is a worse trade than it was when the item was stop-lossed at 1.23x.
3. **The MAIN arm's Python time has moved somewhere the 4.0.8 trace never flagged.** The two
   largest leaves are now `_from_device` (`cellpose/core.py:141`) at **24.2%** of MAIN samples —
   moving results off the GPU — and `_quantile` (`numpy`, called from cellpose's image
   normalization) at **10.1%**. Neither is a kernel-launch loop; both are cellpose-internal. Any
   future attempt on cellpose internals should target these first, not the four functions item 5
   lists.

**Disposition: item 5 stays stop-lossed**, now with evidence against the version actually running.
Its supporting trace reference should be updated from "4.0.8, needs re-confirming" to "re-confirmed
round 6 at 1.118x, and `get_rel_pos` no longer applies".

---

## 8. Code changes this round (complete diff surface)

| file | change | why |
|---|---|---|
| `backend/algorithms/hybrid/config.py`, `config_example.py` | **new field** `dot_detect_n_jobs: int = 1` (執行參數 block, next to `batch_size`), with the measured justification in a comment | §4. Both files edited together so they cannot drift (`19-open-backlog.md` §3 flags the absence of a parity test). **Config hash `ad41c42f` → `3d1087f2`.** |
| `backend/algorithms/hybrid/hybrid_pipeline.py:636` | `_finish_chunk_cpu` passes `n_jobs=config.dot_detect_n_jobs` to `detect_all_dots` | §4.3. `detect_all_dots` already had the parameter (`m3_dot_detection.py:103`); only the call site was missing it. |

**Nothing else in the pipeline was touched.** No geometry field (`default_tile_size`,
`window_overlap_px`, `window_dedup_iomin`) and no GPU-dispatch field (`cellpose_batch_size`,
`batch_size`) changed, per doc 22's constraint. `run_batch`'s default stays `workers=1`.

New measurement tooling (standalone; imports the pipeline, never modifies it):

| script | purpose |
|---|---|
| `scripts/cellpose_batch_probe.py` | §2 — per-tile cost of stacked vs per-tile `model.eval`, both Cellpose models, with the `run_net`/`_compute_masks` split and a mask-equality check |
| `scripts/unet_batch_probe.py` | §3 — same question for the UNet++ forward |
| `scripts/cpu_contention_probe.py` | §4 — two-phase (GPU prepare / CPU replay) BG-arm audit at P processes × `n_jobs` |
| `scripts/gil_trace_report.py` | §7 — aggregates a py-spy raw profile into the buckets the 4.0.8 trace used |

## 9. What this leaves open

| item | state after this round |
|---|---|
| **Full-WSI validation (backlog item 7)** | **Still open, unchanged.** Descoped this round (§1). Note two of its inputs are now known to be wrong: the grid is 27,565 tiles (not 35,700) and this slide is ~61% tissue-bearing (not "mostly background"). |
| **`workers≥6` allocator balloon** | **New item.** 2 failures in 6 runs, byte-identical 24.76 GiB balloon, random victim tile. Governs the safe worker ceiling under fail-fast. First cheap thing to try: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (doc 21 §4.7 already listed it as untried). |
| **A3 multi-request load test** | Logged, not built — still gated on confirming concurrent slide submission is in the deployment plan. |
| **Cellpose internals (item 5)** | Stop-lossed, re-confirmed at **1.118x** (§7). If ever reopened, target `_from_device` / `_quantile`, not the four named loops. |
| **B1 / B1b cross-tile batching** | **Closed on measurement** (§2, §3). Do not re-propose without a change to tile size or backbone. |
| **B2 / B3 GPU ports** | **Not built**, gate never opened (§5). |
| **⑨ `detect_all_dots` +22.3% regression** | Overtaken: the stage is 2.77x cheaper now, so isolating the old regression has no remaining payoff. |
| **Clinical sign-off on the round-3 checkpoint retrain** | **Still pending and still blocking** — unchanged by this round, but note every performance number here rides on top of it. |

## 10. Reproducing this round

```bash
# §2 / §3 -- batching microbenchmarks (GPU idle; a few minutes each)
.venv/bin/python scripts/cellpose_batch_probe.py --tiles 8 --groups 1,2,4,8 --reps 3 \
    --out docs/hybrid-pipeline/measurement/_metrics_r6/b1_cellpose_batch_probe.json
.venv/bin/python scripts/unet_batch_probe.py --tiles 8 --groups 1,2,4,8 --reps 3 \
    --out docs/hybrid-pipeline/measurement/_metrics_r6/b1b_unet_batch_probe.json

# §4.1/4.2 -- CPU audit (prepare once on GPU, then replay CPU-only)
.venv/bin/python scripts/cpu_contention_probe.py --prepare --tiles 8 --state-dir <dir>
.venv/bin/python scripts/cpu_contention_probe.py --run --state-dir <dir> --procs 1,3,6 \
    "--n-jobs=-1,1,2,4,8" --passes 3 \
    --out docs/hybrid-pipeline/measurement/_metrics_r6/a2_contention_njobs_sweep.json

# §4.4 / §6 -- end-to-end ablation and worker sweep. NOTE: set dot_detect_n_jobs in config.py
#              between runs -- spawn workers re-read config from disk, not from the parent,
#              so perf_measure.py's in-process overrides do not reach them.
.venv/bin/python scripts/perf_measure.py --ihc <roi>/large_ihc.tiff --dish <roi>/large_dish.tiff \
    --output <out> --label w6_nj1_r1 --workers 8 --gpu-dmon --stream-precut --mp-workers 6 \
    --metrics-dir docs/hybrid-pipeline/measurement/_metrics_r6

# §4.5 -- correctness veto + ablation table (existing tooling, unchanged)
.venv/bin/python scripts/gc_ablation_report.py --metrics-dir <m> --runs-dir <runs> \
    --reference <runs>/w1_njm1_r1/report.csv --baseline w1_njm1_r1

# §7 -- B4 trace against the current cellpose (two passes over the same workload)
.venv/bin/py-spy record --gil --format raw --rate 100 --nonblocking -o b4_gil.raw -- \
    .venv/bin/python scripts/perf_measure.py --ihc <roi>/med_ihc.tiff --dish <roi>/med_dish.tiff \
    --output <out> --label b4_gil --workers 8 --stream-precut --mp-workers 1 --metrics-dir <m>
.venv/bin/python scripts/gil_trace_report.py --raw b4_gil.raw --raw b4_wall.raw --out b4.json
```
