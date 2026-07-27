# Current status vs. original baseline — measured comparison

> **Purpose.** Re-measure the hybrid pipeline on the **current HEAD** and compare
> it, apples-to-apples, against the preserved **serial "dumb-version" control
> baseline**, then cross-reference [`bottleneck-list.md`](./bottleneck-list.md)
> to state — per item — **what is done**, **what became a non-issue**, and **what
> is still worth optimizing**. Measurement-only, per plan §1.2 / playbook
> discipline; no new fixes proposed inside this doc.
>
> **Original (control):** git `96a28ba`, serial `run_batch` (one tile at a time),
> `_metrics/` (2026-07-07). Preserved, not overwritten.
> **Current:** git `0e27b20`, config_hash `db2b7e6a`, `_metrics_current/`
> (2026-07-11). Same machine (RTX 5090 / CUDA 13.0 / torch 2.10.0+cu130), same
> non-invasive `scripts/perf_measure.py` harness, same real-WSI crops
> (`test_picture/_roi_crops/{med,large}`), `--gpu-dmon`, `--workers 8`, no py-spy.
> Scales: **medium 121 tiles (11×11)**, **large 441 tiles (21×21)**.
>
> **A third round was measured on 2026-07-22** (Cellpose 4.2.1.1 / `cpdino` swap, git
> `f95a573`). §1–§7 below are preserved unchanged as the 2026-07-11 record; **for current
> status jump to [§8](#8-round-3-2026-07-22--cellpose-4211--cpdino-swap)**. Headline:
> large/441 wall **707.4 → 573.7 s (−18.9%)**, full-WSI projection **~15.5 h → ~12.6 h**.

## What landed between the two commits (code, `backend/algorithms/hybrid/`)

| commit | change | bottleneck item |
|---|---|---|
| `010308f` | **① 方案 (b)**: split per-tile work into a **GPU front (main thread)** + **CPU back (background thread)**, depth-1 overlap in `run_batch` | ① |
| `feedfbd` | sliding-window seam stitching for segmentation (+ VALIS docs) | — |
| `119ad73` | `draw_tile_seam_edges` (visual QA helper) | — |
| `9e618d3` | **②③ 元件級**: `@lru_cache` morphology `disk()` footprint (read-only, shared) | ② (a) |

Full landing records: [`pipeline-overlap-result.md`](./pipeline-overlap-result.md) (①),
[`detect-all-dots-result.md`](./detect-all-dots-result.md) (②③).

---

## 1. Headline — end-to-end anchors (control vs current)

| scale | tiles | **original wall** | **current wall** | Δ wall | orig s/tile | cur s/tile |
|---|--:|--:|--:|--:|--:|--:|
| medium | 121 | 243.3 s | **208.2 s** | **−14.5%** | 2.011 | 1.720 |
| large  | 441 | 848.0 s | **707.4 s** | **−16.6%** | 1.923 | 1.604 |

No negative optimization at any scale; the improvement grows slightly with scale
(one-time init amortizes). Current large (707 s) is marginally below the
`00f2c91` intermediate measurement (724.7 s) — consistent, within thermal noise,
no regression from the later commits.

### GPU utilisation — the mechanism (nvidia-smi dmon, per-second)

| scale | metric | original | current | change |
|---|---|--:|--:|---|
| large  | **idle_frac** (sm==0) | 0.459 | **0.190** | **idle cut ~59%** |
| large  | mean SM % | 28.3 | **32.9** | +4.6 pt |
| large  | busy≥50% frac | 0.252 | 0.292 | +0.04 |
| large  | mem-ctrl mean % | 18.1 | 21.5 | +3.4 pt |
| medium | **idle_frac** | 0.477 | **0.163** | **idle cut ~66%** |
| medium | mean SM % | 28.4 | **35.9** | +7.5 pt |

**This is the whole story of ①.** The serial pipeline left the GPU idle ~46–48% of
wall while the main thread did CPU work (dot detection, PNG encode, GC) between
forwards. The two-stage overlap now runs that CPU back on a background thread
*behind* the next tile's GPU front, so GPU idle collapses to ~16–19% and wall
drops ~15–17%.

---

## 2. Where the time went — phase shift (large 441, % of that run's wall)

| bucket | original s (% wall) | current s (% wall) | reading |
|---|--:|--:|---|
| **GPU front** (UNet + 2× Cellpose fwd) | 386.0 (45.5%) | **578.9 (81.8%)** | **now the critical path** |
| `detect_all_dots` (B3) | 260.1 (30.7%) | 239.4 (33.8%) | same abs. cost, **now overlapped/hidden** |
| PNG encode+write (B2) | 76.8 (9.1%) | 78.5 (11.1%) | same abs. cost, **now overlapped/hidden** |
| `gc.collect` (B4) | 36.3 (4.3%) | 36.3 (5.1%) | unchanged (now on background stage) |
| precut A | 20.6 (2.4%) | 20.4 (2.9%) | unchanged (separate phase, not overlapped) |
| stitch D | 5.1 (0.6%) | 5.1 (0.7%) | unchanged (serial, post-analysis) |

The "% of wall" for the CPU items **rose** even though their absolute seconds are
flat — because the denominator (wall) shrank. Under overlap, "self-time ÷ wall" is
**no longer** a critical-path share: these stages run concurrently with the GPU
front and their *effective* end-to-end Amdahl ceiling ≈ 1.0.

> Harness caveat (unchanged from `detect-all-dots-result.md` §5): after the ①
> refactor M2 and M3b both call `segment_windowed`, so the `B1_m3b_cellpose`
> bucket now sums **both** Cellpose forwards; "GPU front" above =
> `B1_unet_coremask` + `B1_m3b_cellpose` (+ legacy `B1_m2_cellpose` where present).
> `B_process_precut_tile_TOTAL` reads 0 because that function was split into
> gpu/cpu chunk functions the old wrapper no longer spans. End-to-end wall and
> GPU idle_frac (the primary comparison metrics) are unaffected by this.

---

## 3. Per-bottleneck status vs [`bottleneck-list.md`](./bottleneck-list.md)

| # | item | class | original | **current status** |
|---|---|---|---|---|
| ① | GPU under-utilisation / serial pipeline | 3+6 | idle 45.9%, GPU front 45.5% wall | **DONE (方案 b).** idle→19.0%, wall −16.6%; GPU front is now the standing critical path (81.8% wall). |
| ② | `detect_all_dots` (M3 dots) | 1+3 | 30.7% wall, ceiling 1.44 | **RESOLVED "for free".** Runs on background stage, **fully hidden** behind GPU front (per-tile 0.58 s ≪ GPU front 1.33 s). Effective ceiling ≈ 1.0. `disk()`-hoist landed bit-exact; process/regionprops rewrites **stopped-out**. |
| ③ | PNG encode+write | 5 | 9.1% wall, ceiling 1.10 | **HIDDEN.** Now on the background CPU stage, overlapped with GPU front; ~11% self-time but ~0 critical-path contribution. Not independently actioned. |
| ④ | per-tile `gc.collect` | 4+6 | 4.3%, ceiling 1.04 | **UNCHANGED** (5.1% self-time). Now on the background stage → partially hidden. Below Amdahl floor; logged only. |
| ⑤ | precut A / stitch D | 5(+3) | 2.4% / 0.6% | **UNCHANGED.** Both are separate serial phases *outside* the overlapped B loop; still below floor. D is the one remaining serial-but-overlappable candidate. |
| ⑥ | model init (one-time) | 6 | 0.37% @441 | **UNCHANGED**, amortizes to negligible at scale. |
| ⑦ | API / job layer | 6 | ~10⁻⁷ | **UNCHANGED**, negligible. |

**Net:** the three deep-record candidates (①②③) are all addressed — ① by direct
fix, ②③ by being overlap-hidden. The critical path has **moved** from "GPU idle +
scattered CPU work" to "**the GPU forwards themselves**".

---

## 4. Memory (bounded — claim still holds)

| scale | peak RSS orig → cur | VRAM (dmon fb) orig → cur |
|---|---|---|
| medium | 3.07 → **3.06 GB** | 5159 → **5159 MB** |
| large | 4.04 → **3.94 GB** | 5159 → **5159 MB** |

VRAM physical peak is **flat at 5.16 GB / 32 GB** at both scales and both commits —
the double-buffered overlap keeps only two tiles in flight (RSS actually a hair
*lower* on current large). The "memory bounded, not linear in tiles" invariant
survives the refactor.

> One anomaly, **not** a real regression: the resource sampler logged
> `cuda_alloc_peak 22.2 GB` on the current medium run, but the driver's `dmon`
> framebuffer peak for that same run is **5159 MB** — physically impossible to
> exceed. The 22 GB is an uncorroborated `torch.cuda.memory_allocated()` sampling
> artifact (the CPROFILE baseline showed the same class of spike). Physical VRAM
> stayed bounded; treat the torch-allocated column as unreliable and read VRAM
> from `dmon fb`.

---

## 5. Full-WSI reprojection (35,700 tiles @ 1024px)

Linear fit on the two current anchors: `wall ≈ 19.4 s + 1.560 s/tile`.

| | s/tile slope | full-WSI (35,700) | note |
|---|--:|--:|---|
| original (control) | 1.903 | ~18.9 h | upper bound |
| **current** | **1.560** | **~15.5 h** | upper bound, **−18% at scale** |

Still an **upper bound** for the same reason as before: the crops are tissue-dense
(~85%); a real slide is mostly white background whose empty-core tiles
short-circuit cheaply. The relative −18% is the meaningful figure.

---

## 6. What is still worth optimizing (ranked, measurement-only)

Ranked by current critical-path share; **classification only, no fix designed here**.

1. **The GPU front — M1 UNet + 2× Cellpose forwards (81.8% of wall @441). PRIMARY.**
   Now that overlap hid the CPU work, this *is* the wall. This is **①'s next
   stage** (CuPy / GPU-kernel level), scoped in
   [`../11-gpu-pipeline-stage2-plan.md`](../11-gpu-pipeline-stage2-plan.md) and
   flagged in [`gil-contention-diag.md`](./gil-contention-diag.md) (Cellpose
   `dynamics.py` / SAM `get_rel_pos` are kernel-launch-bound, already GPU-resident).
   Class 1 (algorithm) + 2 (hardware/kernel).

2. **Residual GPU idle ~16–19%.** The overlap is depth-1 and `detect_all_dots`
   uses joblib `prefer='threads'`, so its CPU work contends with main-thread torch
   on the GIL — capping overlap below the theoretical ~50%. Levers: deeper
   pipeline, or a process backend for the CPU stage. **But** limited end-to-end
   value while the CPU stage is already hidden — only pays off *after* the GPU
   front (lever 1) shrinks and re-exposes it. Class 3 (concurrency).

3. **`cellpose_batch_size` — dead config (Class 7). The untapped structural lever.**
   VRAM sits at **5.16 / 32 GB** — ~27 GB idle headroom. The field is still not
   wired into `Config` (`getattr(..., 16)` fallback), so no cross-tile / larger
   GPU batch is possible. Wiring it is a **prerequisite** before any batch-size
   sweep can even attempt to cut lever 1. Must be fixed for that measurement to be
   meaningful. (Config correctness, not itself a fix recommendation for wall.)

4. **`gc.collect` every tile (5.1%)** and **serial stitch D (0.7%)** — both below
   the Amdahl floor; recorded, not deep-analyzed. D is the only remaining
   serial-but-in-principle-overlappable stage, but <1% of a ~15 h run.

**Stop-loss note (unchanged):** ②③④ are hidden or below floor — chasing them
cannot move wall until lever 1 shrinks the GPU front. The disciplined next round
is lever 1 only.

---

## 7. Reproduce

```bash
cd /data/taro_Projects/tsgh
ROI="$PWD/backend/algorithms/hybrid/test_picture/_roi_crops"
MC=docs/hybrid-pipeline/measurement/_metrics_current
for s in "med:medium_121tile:medium" "large:large_441tile:large"; do
  IFS=: read pre lab out <<<"$s"
  .venv/bin/python scripts/perf_measure.py \
    --ihc "$ROI/${pre}_ihc.tiff" --dish "$ROI/${pre}_dish.tiff" \
    --output docs/hybrid-pipeline/measurement/runs_current/$out \
    --label $lab --workers 8 --gpu-dmon --metrics-dir "$MC"
done
.venv/bin/python scripts/aggregate_report.py "$MC"
.venv/bin/python scripts/resource_analyze.py "$MC"
# compare against preserved control baseline in docs/hybrid-pipeline/measurement/_metrics/
```

Raw artifacts (preserved): control in `_metrics/`, current in `_metrics_current/`
(`*_timings.json`, `*_agg.json`, `*_resource_summary.json`, `*_gpu_dmon.txt`,
`*_resource.csv`, `*_stdout.log`). Pipeline outputs in `runs/` (control) and
`runs_current/` (current).

---

# 8. Round 3 (2026-07-22) — Cellpose 4.2.1.1 / `cpdino` swap

> **§1–§7 above are preserved verbatim as the 2026-07-11 record.** This section adds
> the third measured round and re-states the comparison three ways.
> **Round 3:** git `f95a573`, config_hash `db2b7e6a` (**unchanged** — same parameters),
> `_metrics_cellpose421/` (2026-07-22). Same machine (RTX 5090, driver 580.159.03), same
> `scripts/perf_measure.py`, same crops (`test_picture/_roi_crops/{med,large}`),
> `--gpu-dmon`, `--workers 8`, no py-spy. GPU verified idle before launch (89 MiB, 0%,
> no other compute processes) — the shared-server discipline from doc 13 §0 was followed.
> Environment stamp + `pip freeze` + model-weight SHA256s: `_metrics_cellpose421/env_stamp.txt`.

## 8.1 What changed between round 2 and round 3

**Pipeline code: essentially nothing.** `git diff 0e27b20..f95a573 -- backend/algorithms/hybrid/`
is two hunks: deletion of the now-unused `_process_one_chunk` sync wrapper, and dropping
explicit `tile_width/tile_height=256` from `_stitch_overlay_slide`'s `tiffsave`. The
two-arm overlap, `run_batch`, M0–M4 and all thresholds are untouched.

**The models and the environment: a lot.** A coworker moved Cellpose to the faster
DINOv3-backbone line and the venv was rebuilt from `uv.lock`:

| | round 1–2 (`_metrics`, 2026-07-07) | round 3 (2026-07-22) |
|---|---|---|
| cellpose | 4.0.8 (`cpsam`, fp32) | **4.2.1.1** (`cpdino` / DINOv3, `use_bfloat16=True` default) |
| M2 + M3b checkpoints | previous training | **retrained** (both 343 MB; SHA256 in `env_stamp.txt`) |
| torch / torchvision | 2.10.0+cu130 / 0.25.0 | 2.11.0+cu130 / 0.26.0 |
| numpy | 2.2.6 | **1.26.4** ↓ |
| scikit-image | 0.25.2 | **0.24.0** ↓ |
| pyvips | 3.1.1 | **2.2.3** ↓ |
| opencv | 4.9/4.10/4.12 mix | **4.8.1.78** ↓ |
| timm / scipy | 1.0.22 / 1.16.3 | 1.0.26 / 1.17.1 |

> **This is a bundled change, not a controlled single-variable swap.** Attribution below is
> done per timing bucket, which is as far as the data honestly goes. Note also that
> `_metrics_current/` (round 2) has **no `pip freeze`** — so round 2's exact package set is
> unrecorded and it is assumed, not proven, to match round 1's. Every future round should
> drop a `pip freeze` next to its timings; round 3 does.

## 8.2 Headline — three-way anchors

| scale | tiles | control (r1) | overlap (r2) | **round 3** | Δ r2→r3 | Δ r1→r3 |
|---|--:|--:|--:|--:|--:|--:|
| medium | 121 | 243.3 s | 208.2 s | **166.6 s** | **−20.0%** | **−31.6%** |
| large  | 441 | 848.0 s | 707.4 s | **573.7 s** | **−18.9%** | **−32.3%** |
| large s/tile | | 1.923 | 1.604 | **1.301** | −18.9% | −32.3% |

No negative optimization at any scale; the two scales agree to within 1.1 pt, so the
result is regime-stable, not a small-sample artifact.

### Where the gain came from — it is localised to the Cellpose forwards

| bucket (large/441) | r1 control | r2 overlap | **r3** | Δ r2→r3 |
|---|--:|--:|--:|--:|
| **Cellpose forwards** (824 calls) | 372.6 s | 564.5 s | **430.9 s** | **−23.7%** |
| — **per call** | 0.4521 s | 0.6850 s | **0.5229 s** | **−23.7%** |
| UNet++ forward (441 calls) | 13.45 s | 14.41 s | **13.68 s** | −5.1% (flat) |
| **VRAM peak** (dmon fb) | 5159 MB | 5159 MB | **2787 MB** | **−46.0%** |
| `cuda_reserved` peak | 4.68 GB | 4.68 GB | **2.19 GB** | −53% |

UNet++ is flat and every non-Cellpose bucket is flat or *worse*, so the −18.9% wall is
attributable to the Cellpose forwards specifically. The −46% VRAM is consistent with
4.2's documented `use_bfloat16=True` default.

> **The B1 "absolute growth" anomaly from §2/①  is largely resolved.** B1 total went
> 386.0 (r1) → 578.9 (r2) → **444.6 s** (r3). Most of the unexplained +192.9 s reversed with
> the library upgrade, which retires the open question §2 was written to chase. The harness
> relabeling caveat is unchanged: `B1_m3b_cellpose` still sums **both** Cellpose forwards
> (n=824 = 2 × 412), and `B_process_precut_tile_TOTAL` still reads 0.

## 8.3 The GPU got *less* busy — and why that is the real headline

| metric (dmon, per-second) | r1 control | r2 overlap | **r3** |
|---|--:|--:|--:|
| large **idle_frac** (sm==0) | 0.459 | 0.190 | **0.370** |
| large mean SM % | 28.3 | 32.9 | **16.6** |
| large busy≥50 frac | 0.252 | 0.292 | **0.130** |
| medium **idle_frac** | 0.477 | 0.163 | **0.293** |
| medium mean SM % | 28.4 | 35.9 | **17.4** |

Idle_frac roughly doubled and mean SM halved. **This is not a regression** — the wall
dropped 18.9% at the same time. It means the GPU now finishes its share of each tile
sooner and spends longer waiting on the CPU arm. The pipeline is drifting from
GPU-bound toward CPU-bound.

### The arm model (what actually determines wall now)

`run_batch` (`hybrid_pipeline.py:766-810`) runs two concurrent arms — membership confirmed
by reading the source, not inferred from timings:

- **MAIN** (main thread): 3 GPU forwards, `_read_rgb`, M1 overlay, `clear_slide_edge_cells`,
  `build_all_positive_results`, `enlarge_cell_instances`, **`gc.collect` + `empty_cache`**.
- **BG** (one background thread): `detect_all_dots`, merge, PNG/TIFF encode,
  `render_overlay_image`, per-cell crops, `filter_and_absolutize`.

`wall ≈ max(MAIN, BG) + outside` (outside = precut A + stitch D + init):

| large/441 | r1 control | r2 overlap | **r3** |
|---|--:|--:|--:|
| MAIN arm | 467.9 s | 672.0 s | **538.3 s** |
| BG arm | 350.6 s | 333.2 s | **387.3 s** |
| **BG / MAIN** | 0.749 | **0.496** | **0.719** |
| outside overlap | — | — | 28.0 s |
| model check: `max(arm)+outside` | — | — | 566.3 s vs **573.7 s measured** (−1.3%) |

**The single most important number in this round: the MAIN arm must shed only
151.0 s — 34.0% of the GPU forwards — before the background CPU arm becomes the new
critical path.** At medium the margin is 36.8%. Round 2 had ~100% of margin
(BG/MAIN 0.496); round 3 has 39%. One more Cellpose-sized improvement re-exposes ② and ③.

## 8.4 Amdahl ceilings under the arm model (large/441, round-3 anchor 573.7 s)

Self-time ÷ wall is **not** a critical-path share under overlap. Ceilings computed as
`wall / (max(MAIN', BG') + outside)`:

| lever | self-time | % wall | arm | **ceiling if →0** |
|---|--:|--:|---|--:|
| GPU forwards → 0 | 444.6 s | 77.5% | MAIN | **1.382x** |
| `gc.collect` → 0 | 36.4 s | 6.3% | MAIN | **1.083x** |
| ⑧ CPU prep off MAIN | 28.4 s | 5.0% | MAIN | ~1.05x |
| precut A + stitch D → 0 | 25.6 s | 4.5% | outside | ~1.05x |
| `detect_all_dots` → 0 | 292.9 s | 51.1% | BG | **1.013x** |
| PNG encode → 0 | 78.7 s | 13.7% | BG | **1.013x** |
| both arms → 0 (theoretical) | — | — | — | 4.69x |

**Read this table before believing any "% of wall" figure.** `detect_all_dots` shows 51.1%
of wall and is worth **1.3%**. `gc.collect` shows 6.3% and is worth **8.3%** — six times
more, despite looking eight times smaller. This is the plain reason the flat "<10% ⇒ drop
it" floor is suspended for critical-arm items (see `bottleneck-list.md`, revised stop-loss):
at a ~12.6 h full-WSI run, `gc.collect` alone is **~44 min** of wall.

## 8.5 What regressed

| bucket (large/441) | r2 | **r3** | Δ |
|---|--:|--:|--:|
| `detect_all_dots` | 239.4 s | **292.9 s** | **+22.3%** |
| — per cell | 18.5 ms | **22.3 ms** | +20.2% |
| `enlarge_cell_instances` | 18.29 s | 19.60 s | +7.2% |
| `build_all_positive_results` | 7.23 s | 8.81 s | +21.9% |
| PNG encode | 78.54 s | 78.72 s | flat |
| `gc.collect` | 36.33 s | 36.39 s | flat |
| precut A / stitch D | 20.39 / 5.08 s | 20.51 / 5.05 s | flat |

Cell count rose only **+1.8%** (12,922 → 13,150), so `detect_all_dots` is ~20% slower
*per cell*. Leading hypothesis is the **scikit-image 0.25.2→0.24.0 / numpy 2.2.6→1.26.4 /
opencv→4.8.1.78 downgrades** (that path is LAB + H-morphology over exactly those libraries);
the competing explanation is different cell geometry from the retrained checkpoints. The two
changed together and **the cause is not isolated** — logged as ⑨ in `bottleneck-list.md`.
None of this moves wall today (BG arm, ceiling 1.013x); it matters because it eats the
39% margin computed in §8.3.

Notably flat: precut A and stitch D did **not** regress despite **pyvips 3.1.1 → 2.2.3**.

## 8.6 Correctness — not held constant this round

Plan §1.3 puts accuracy out of scope, but the previous rounds were bit-comparable and this
one is not, so it must be flagged before −18.9% is read as free:

| | r1 control | r2 overlap | **r3** |
|---|--:|--:|--:|
| cells, medium | 3559 | 3558 | **3647** (+2.5%) |
| cells, large | 12919 | 12922 | **13150** (+1.8%) |
| tiles success/skipped, large | 379 / 62 | 379 / 62 | **378 / 63** |

r1 vs r2 agreed to ±3 cells (GPU-nondeterminism noise floor). r3 does not: retrained
checkpoints produce different segmentations, and one tile flipped success → skipped.
**This is a model-quality change that needs clinical//pathologist validation on its own
terms — the performance result does not speak to whether the new masks are better.**

## 8.7 Memory

| scale | peak RSS r2 → r3 | VRAM (dmon fb) r2 → r3 |
|---|---|---|
| medium | 3.06 → **3.09 GB** | 5159 → **2785 MB** |
| large | 3.94 → **3.90 GB** | 5159 → **2787 MB** |

The bounded-memory invariant survives. VRAM headroom is now **~29.8 GB of 32 GB idle**,
which strengthens (does not change) the standing `cellpose_batch_size` dead-config finding:
`hybrid_pipeline.py:206,218` still call `getattr(config, "cellpose_batch_size", 16)` against
a `Config` with no such field.

## 8.8 Full-WSI reprojection (35,700 tiles @ 1024px)

Linear fit on the two round-3 anchors: `wall ≈ 12.6 s + 1.2722 s/tile`.

| round | s/tile slope | full-WSI (35,700) | Δ vs control |
|---|--:|--:|--:|
| r1 control | 1.903 | ~18.9 h | — |
| r2 overlap | 1.560 | ~15.5 h | −18% |
| **r3 cpdino** | **1.2722** | **~12.6 h** | **−33%** |

Still an **upper bound** for the unchanged reason (crops are ~85% tissue-dense; a real
slide is mostly white background whose empty-core tiles short-circuit cheaply). The
relative −33% is the meaningful figure.

## 8.9 Answering the question this round was run for

- **How much did the Cellpose swap buy?** **−18.9% wall at large/441, −20.0% at medium/121**
  — 707.4 → 573.7 s, and ~15.5 h → ~12.6 h projected for a full WSI. The mechanism is a
  **−23.7% per-call Cellpose forward** (0.6850 → 0.5229 s) plus **−46% VRAM**.
- **Versus the original control baseline**, cumulative improvement is now **−32.3%**
  (848.0 → 573.7 s), of which the ① overlap contributed −16.6% and this swap −18.9%.
- **Is the GPU still the bottleneck?** Yes, but barely — 34% of margin left. The GPU idles
  37% of the time now, and the CPU back-stage is 72% of the critical arm.
- **What is now worth optimizing that was not before?** `gc.collect` (ceiling 1.083x,
  ~44 min at full WSI), the stranded CPU prep on the MAIN arm (⑧), and precut A + stitch D —
  all "sub-10%" by self-time, all on the critical path, all reducible toward zero. See
  `bottleneck-list.md` → "Re-sorted priority after round 3".

## 8.10 Reproduce round 3

```bash
cd /data/taro_Projects/tsgh
uv sync                                  # env must match uv.lock; `uv sync --frozen --dry-run` should report no changes
ROI="$PWD/backend/algorithms/hybrid/test_picture/_roi_crops"
MC=docs/hybrid-pipeline/measurement/_metrics_cellpose421
nvidia-smi                               # shared server: confirm the GPU is idle first (doc 13 §0)
for s in "med:medium_121tile:medium" "large:large_441tile:large"; do
  IFS=: read pre lab out <<<"$s"
  .venv/bin/python scripts/perf_measure.py \
    --ihc "$ROI/${pre}_ihc.tiff" --dish "$ROI/${pre}_dish.tiff" \
    --output docs/hybrid-pipeline/measurement/runs_cellpose421/$out \
    --label $lab --workers 8 --gpu-dmon --metrics-dir "$MC"
done
.venv/bin/python scripts/aggregate_report.py "$MC"
.venv/bin/python scripts/resource_analyze.py "$MC"
uv pip freeze > "$MC/pip_freeze_actual.txt"   # do this every round
```

Raw artifacts (preserved, three rounds side by side): `_metrics/` (r1 control),
`_metrics_current/` (r2 overlap), `_metrics_cellpose421/` (r3). Pipeline outputs in
`runs/`, `runs_current/`, `runs_cellpose421/`.

---

# 9. Rounds 4–6 (2026-07-22 to 2026-07-25) — condensed chain to round 6

> **This document's detailed round-by-round record stops at round 3 above.** Rounds 4–7 are
> recorded in full in [`bottleneck-list.md`](./bottleneck-list.md) (its "Round-4/5/6/7 anchors"
> sections) — this section only condenses the wall-clock chain so this document stays a usable
> single point of reference, and does not reproduce the per-bucket detail already recorded there.

| round | change | large/441 wall | Δ vs previous | source |
|---|---|--:|--:|---|
| r3 (above) | Cellpose 4.2.1.1 `cpdino` swap | 573.7 s | −18.9% | §8 above |
| r4 | ⑧ CPU prep off MAIN arm + precut streamed | **480.3 s** | −16.3% | [bottleneck-list "Round-4 anchors"](./bottleneck-list.md) |
| r5 | cross-tile multiprocessing built (`workers=3` adopted) | **156.1 s** | −67.5% | [bottleneck-list "Round-5 anchors"](./bottleneck-list.md) |
| r5b | worker-count ceiling found; `workers=6` recommended | **123.3 s** | −21.0% | [21-implementation §4.7](../21-cross-tile-multiprocessing-implementation.md) |
| r6 | `detect_all_dots` joblib fan-out removed (`dot_detect_n_jobs=1`) — `workers=1` gets **1.60×** for free; recommendation revised down to `workers=4`/`5` (allocator OOM risk at ≥6) | `workers=1`: **302.7 s**; `workers=4`: **128.8 s** | −37.5% (`workers=1`) | [bottleneck-list "Round-6 anchors"](./bottleneck-list.md), [23-implementation](../23-next-optimization-cycle-implementation.md) |

Cumulative, `workers=1`: **848.0 s → 302.7 s (−64.3%)** across six rounds, with none of it shipped
to multi-worker production yet (`19-open-backlog.md` item 7 — full real-WSI validation — is still
the gate). Correctness caveat from §8.6 still applies to every round from r3 onward: the retrained
Cellpose checkpoints have not received clinical/pathologist sign-off.

Rounds 4–6 also each measured and stop-lossed several candidates (`cellpose_batch_size` sweep,
cross-tile Cellpose/UNet++ batching, CUDA MPS, deeper CPU pipelining) — see
[`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md) for the complete list of
what was tried and rejected versus what is still open, rather than duplicating that ledger here.

---

# 10. Round 7 (2026-07-26) — the composition premise was wrong, and current status is re-based on it

> Executes [`24-gpu-encode-decode-loop-acceleration-plan.md`](../24-gpu-encode-decode-loop-acceleration-plan.md)'s
> survey against measurement. Full record:
> [`25-gpu-encode-decode-loop-acceleration-implementation.md`](../25-gpu-encode-decode-loop-acceleration-implementation.md).
> Git `025f9a5`, config hash **`3d1087f2` unchanged**, same RTX 5090, torch 2.11.0+cu130.

## 10.1 The headline is a correction, not a new optimization

Every full-WSI projection since round 6 — including the "~5.3h at `workers=1`" figure this
document's own §9 table cites — assumed the slide is **39% background / 61% tissue-bearing**, a
number derived from a brightness thumbnail whose answer swings from 2.4% to 55.1% depending on an
unstated grey-level threshold. Round 7 measured the pipeline's **own** definition of background
(UNet++ core-mask forward returns empty) over **all 27,565 tiles of the real grid** — not sampled,
not thumbnailed — and found:

| | assumed (rounds 6–7-planning) | **measured, round 7** |
|---|--:|--:|
| background share | 39% | **55.82%** |
| tissue share | 61% | **44.18%** |
| background tiles | ~10,750 | **15,386** |
| tissue tiles | 16,815 | **12,179** |

This flips the direction of every composition-dependent conclusion in this document set: MAIN-arm
costs (gated on tissue tiles) are smaller in absolute terms than every prior projection assumed, and
the one BG-arm cost that scales with background-tile *count* (blank-tile placeholder writes) is
larger. See §10.2.

## 10.2 Composition-matched anchors and the arm model

Two new 576-tile crops were cut at full resolution and measured at `workers=1`/`workers=4`
(`--gpu-dmon`, GPU idle-verified before launch): `comp24` (73.4% background, brightness-proxy
selected) and `match24` (55.9% background, selected from the exact measured map — matching the real
slide's 55.8% almost exactly).

| anchor | `workers=1` | `workers=4` | speedup | BG/MAIN | MAIN must shed |
|---|--:|--:|--:|--:|--:|
| large/441 (round 6, 14.1% background — **not** representative of the real slide) | 302.7 s | 128.8 s | 2.35× | 0.719 (round 3 tissue-dense figure; see bottleneck-list) | 28% |
| comp24 (73.4% background) | 134.9 s | 65.5 s | 2.06× | 0.527 | 47.3% |
| **match24 (55.9% background — matches the real slide)** | **188.8 s** | **88.3 s** | **2.14×** | **0.470** | **53.0%** |

**Every tissue-density crop this project has measured before round 7 (12–14% background) badly
understated the real slide's background share (55.8%), and therefore badly understated how much
slack the BG arm actually has.** At the slide's real composition, `detect_all_dots`, PNG encode, and
every other BG-arm candidate this document's §6/§3 previously flagged as "worth re-checking if the
margin tightens" are now **further from being re-exposed than at any prior round**, because more
tissue tiles load the GPU-forward-heavy MAIN arm faster than they load BG — the opposite of what a
brightness-proxy-based "mostly background" story would have predicted.

## 10.3 Per-bottleneck status, updated

| # | item | round-3 status (§3 above) | **round-7 status** |
|---|---|---|---|
| ① GPU forwards | Fixed via Cellpose swap, 34% margin before BG re-exposed | **Margin now 47–53% at real composition** (was measured at 15.9%–28% on tissue-dense crops) — further from re-exposure, not closer. Round 6's `dot_detect_n_jobs=1` (see §9) also made MAIN itself 43.4% faster by removing GIL contention with the BG arm's surplus threads. |
| ②③ `detect_all_dots` / PNG encode | Hidden, ceiling ~1.01–1.03× | **Ceiling confirmed 1.00× at real composition** — BG-arm candidates have *more* headroom than tissue-dense crops implied, not less. |
| **Phase D slide stitch (new, round 7)** | Not separately sized before round 7 | **The one item that got *worse* than estimated.** Measured 322.7 s at 16.2 gigapixels — 1.8× doc 24's crop-based extrapolation, and superlinear (+40%/gigapixel at full scale vs. the 1–4 GP range). Runs once outside the worker pool, so its wall-clock **share doubles from `workers=1` to `workers=4`** (3.5% → 7.3%) as everything else shrinks. Ceiling 1.036×–1.078×, still below the actionable bar, but the strongest remaining candidate. |
| **Background-tile placeholder writes (new, round 7 — Candidate F)** | Never separately measured (invisible at 14% background) | **Measured: 24 ms/tile, 7.5% of wall — zero wall-clock payoff** (BG arm has 47–53% slack). Real cost is disk, not time: ~157 GB/slide of identical uncompressed TIFF; `os.link` alternative is 272×–407× cheaper if ever wanted for storage reasons. |
| `cellpose_batch_size` / cross-tile batching | Wired, sweep flat at existing tile size | Cross-tile batching re-tested at G=16 (round 7): still worse (+5.9–6.6%), 15.8 GB peak. Confirmed closed. |
| GPU codec dependencies (new, round 7) | n/a | **Environment gate resolved.** nvImageCodec works here (19.2× faster lossless TIFF encode) and is the only viable path — for Phase D only. nvTIFF has no Python binding, cuCIM can't write, and CuPy (needed for every BG-arm GPU-port candidate) cannot run on this host without a `numpy<2` pin violation or a system CUDA toolkit install. |

## 10.4 Full-WSI reprojection, rebuilt from measured rates and measured composition

| round | figure | basis |
|---|--:|---|
| r1 control | ~18.9 h | 3-tile extrapolation, upper bound |
| r2 overlap | ~15.5 h | — |
| r3 cpdino | ~12.6 h | — |
| r6 (§9) | ~5.3 h (`workers=1`) / ~2.2 h (`workers=3`) | blended per-tile rate, **assumed 39% background** |
| **r7, measured composition** | **~2.6 h (`workers=1`) / ~1.25 h (`workers=4`)** | composition-matched crop rates × the slide's **measured** 55.8%/44.2% split, plus measured Phase D (322.7 s) |

Still a rate-based extrapolation, not the real full-WSI run this document set has needed since
[`09-measurement-analysis-plan.md`](../09-measurement-analysis-plan.md) §3.6 — see
[`19-open-backlog.md`](../19-open-backlog.md) item 7, still open.

## 10.5 What is still worth optimizing (round 7, ranked)

1. **Phase D slide stitch — cheaper non-GPU knobs first** (`tiffsave` tile-size/pyramid-depth
   parameters, not re-encoding constant background regions). Untested, free of new dependencies,
   and the one candidate whose relative importance *grows* as multiprocessing shrinks everything
   else. GPU port (nvImageCodec) is real but needs its own pyramid/container engineering — do the
   cheap knobs first.
2. **Full real-WSI validation** (`19-open-backlog.md` item 7) — still the single highest-leverage
   item in this whole document set: it both closes the gate on shipping `workers>1` to production
   and would replace every rate-based projection above with a real number.
3. **`workers≥6` allocator-fragmentation OOM** — reliability defect, not sized as speed;
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` still untried.
4. Everything else this project has surveyed on the BG arm (`detect_all_dots`/CPU-prep GPU ports,
   background-tile write dedup) is **confirmed closed, not weakened** by the composition
   correction — see §10.3.

**For the complete list of every candidate this project has discovered but not shipped** — what's
still open, what was measured and rejected, and what's gated on something else — see
[`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md), compiled 2026-07-26 by
reading every document in this folder.

## 10.6 Reproduce round 7

```bash
cd /data/taro_Projects/tsgh
M=docs/hybrid-pipeline/measurement/_metrics_r7
SLIDE=/data/nvmessd/storge_tsgh/<case>/output

# the pipeline's own background rule, over the whole 27,565-tile grid (~25 min)
.venv/bin/python scripts/core_mask_map.py --ihc $SLIDE/HER2_processed.tiff --out $M/core_mask_map.npz

# cut the composition-matched crop from the exact map
.venv/bin/python scripts/composition_crop.py --ihc $SLIDE/HER2_processed.tiff \
    --dish $SLIDE/DISH_processed.tiff --grid 24 --map $M/core_mask_map.npz \
    --out-ihc test_picture/_roi_crops/match24_ihc.tiff \
    --out-dish test_picture/_roi_crops/match24_dish.tiff --report $M/match24_crop.json

# Phase D at real scale (no inference needed)
.venv/bin/python scripts/stitch_probe.py --overlay-src <run>/overlay_annotated \
    --slide-w 141818 --slide-h 114366 --out $M/stitch_probe_full.json

# full-WSI projection from measured rates + measured composition
.venv/bin/python scripts/wsi_projection.py --timings $M/match_w1_r1_timings.json \
    --background-share 0.5582 --stitch-s 322.7 --out $M/wsi_projection.json
```

Raw artifacts: `_metrics_r7/` (incl. `env_stamp_r7.txt`, `pip_freeze_r7.txt`,
`core_mask_map.npz`). Full reproduction command set:
[`25-gpu-encode-decode-loop-acceleration-implementation.md`](../25-gpu-encode-decode-loop-acceleration-implementation.md) §12.

---

# 11. Round 8 (2026-07-27) — the real full-slide run, and three "closed" crop-scale numbers reopen

> Executes [`26-remaining-work-implementation-plan.md`](../26-remaining-work-implementation-plan.md).
> Config hash **`3d1087f2` unchanged**. Full record:
> [`27-remaining-work-implementation.md`](../27-remaining-work-implementation.md).

## 11.1 Headline

Every round-7 projection through §10 above was a **rate-based extrapolation from crops**. Round 8
ran the real thing for the first time — both `workers=1` and `workers=4`, on the conformed
HER2/DISH pair (the registration stage emits a different canvas per modality, 141818×114366 vs
141658×114415, which `PrecutStream` fail-fasts on; `scripts/full_wsi_validate.py --conform` crops
both to their intersection, 99.86% retained — a blocker no crop-based round could ever hit).

| | `workers=1` | `workers=4` | round-7 projection |
|---|--:|--:|--:|
| end-to-end wall | **13,762 s = 3.82 h** | **6,211 s = 1.73 h** | 2.6 h / 1.25 h |
| Δ vs projection | **+47%** | **+38%** | — |
| measured speedup | — | **2.216x** | 2.06x–2.17x predicted |
| `report.csv` rows | 356,255 | 356,221 (**−0.01%**, veto passed) | — |
| peak RSS | 61.13 GB | 61.67 GB | ~4 GB at crop scale |
| peak GPU | 2,739 MB | 30,439 MB (93.3% of 32,607) | — |

The composition prediction was right to within one tile (15,385 vs predicted 15,386 background
tiles) — so the +38–47% miss is entirely in per-tile rates, not the tissue/background mix.

## 11.2 Three crop-scale numbers this project had already measured and closed do not survive at scale

| stage | crop-scale record | **full-slide, round 8** |
|---|--:|--:|
| `gc.collect` (§9 r6 note; doc 16) | ~0 (`gc.freeze()`, 1.083x ceiling) | **16.1% of wall** (2,218.4 s, back to 80.5 ms/call) |
| tile read (doc 18 §6.3) | 1.22% of wall, ceiling 1.012x | **17.2% of wall** (2,368.5 s) |
| Phase D stitch (§10.3 above; doc 25) | 3.5%/7.3% of wall, ceiling 1.036x/1.078x | **8.6%/19.3% of wall** (1,185.4 s — 3.7× the synthetic probe's 322.7 s) |

`gc.freeze()` only exempts objects live *at freeze time*; `run_batch` accumulates `per_tile_owned`
(356,255 `CellAnalysisResult`s by the end of a slide) *after* the freeze, fully tracked, rescanned
on all 27,565 collections — invisible on the 441-tile crop this document's §2 measured (~6,000
objects). Tile read was free on a crop because the ~49 GB precut scratch fit page cache; at full
scale it doesn't. Phase D's synthetic probe (§10.3) replicated a small pool of real tiles via hard
links, which compress and cache far better than 27,565 genuinely distinct ones.

## 11.3 Phase D `tiffsave` knob ablation — CLOSED, negative

13 single-knob configs at 4.055 GP screening scale: tile size monotonically **worse**
(256/512/1024 → 0.948x/0.860x/0.686x); pyramid depth and `predictor=horizontal` already the
effective defaults (byte-identical output); `deflate` 0.785x. The one winner, **`zstd` level 1 —
1.2331x, 13.8% smaller, verified lossless** — is **vetoed on correctness**: QuPath/BioFormats
cannot open a zstd-compressed TIFF. `_stitch_overlay_slide` stays on LZW. This closes §10.5 item
1's "cheaper knobs first" recommendation — nothing cheap is left, and the GPU port now carries a
hard new constraint: its output must be BioFormats-readable.

## 11.4 Reliability and measurement infrastructure built

- **`RLIMIT_NOFILE` guard** — `_ensure_nofile_limit()` raises the soft limit itself when the hard
  limit permits, else fails loudly before opening anything. Exercised for real on the full-slide
  run (12,027 open fds observed mid-stitch) and passed silently.
- **Partial resume** — `run_batch(checkpoint=True)`, opt-in, config-hash-guarded; cold vs. resumed
  output byte-identical. Fail-fast unchanged.
- **Per-worker timing** — `perf_measure.py --worker-timings` now reports 26 worker-side buckets
  (was parent-process-only, 4 buckets, at `workers>1`).
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — swept 12 runs at `workers=6` on §9's
  exact defect-reproduction anchor. Does **not** reduce peak VRAM (median 24,040 vs 22,968 MB
  control — evidence against the fragmentation hypothesis), costs +2.0% wall, and 0-in-6 vs 1-in-6
  OOM is statistically indistinguishable. Default stays off. Also: `workers=6` is not faster than
  `workers=4` on this crop (65.46 vs 65.55 s).

## 11.5 What this changes

1. **§10.5 item 2 (full real-WSI validation) is closed.** The `workers>1` production gate is
   satisfied. Recommendation: **ship `workers=4`**, with a VRAM caveat — 93.3% of the reference
   card at peak, ~2.2 GB headroom — not the speed caveat every prior round expected.
2. **§10.5 item 1 (Phase D cheap knobs) is closed, negative** — see §11.3. The GPU port is now the
   only remaining Phase D route, and its ceiling is **~3x higher than §10.3 recorded** (19.3% of
   wall at `workers=4`, not 7.3%) because both the stitch got slower than the probe predicted *and*
   everything else got faster.
3. **§10.5 item 3 (`workers≥6` allocator OOM) — candidate fix tried, did not clear it.** See §11.4.
   Root-causing the 24.76 GiB balloon directly is now the live question, not more allocator flags.
4. **Two of §10.3's "confirmed closed" BG-arm items don't apply here** — `gc.collect` and tile read
   were never BG-arm/GPU-port candidates in the round-7 sense; they are MAIN-arm/outside costs that
   this round found were mismeasured at crop scale, not re-litigated composition conclusions. See
   §11.2.

For the complete list of every item this project has discovered but not shipped — including what
round 8 closed and what it reopened — see
[`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md).

## 11.6 Reproduce round 8

```bash
.venv/bin/python scripts/full_wsi_validate.py \
  --ihc  /data/nvmessd/storge_tsgh/<case>/output/HER2_processed.tiff \
  --dish /data/nvmessd/storge_tsgh/<case>/output/DISH_processed.tiff \
  --output-root /home/taro/full_wsi_validation \
  --conform --workers 1,4 --out /home/taro/full_wsi_validation/result.json

# Phase D tiffsave ablation (add --only baseline,zstd_1 to confirm the winner at full scale)
.venv/bin/python scripts/stitch_probe.py --overlay-src <dir of real overlay tiles> \
    --pool 6 --ablate --slide-w 70909 --slide-h 57183 --out ablate_4gp.json

# allocator sweep (refuses to start on a GPU that already has memory in use)
.venv/bin/python scripts/alloc_conf_probe.py --ihc <crop_ihc> --dish <crop_dish> \
    --workers 6 --repeats 6 --out alloc_conf_w6.json

# tests
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/verify_mp_failfast.py --tiles-dir <precut scratch> --workers 3
```

Raw artifacts: `_metrics_r8/` (`stitch_ablate_4gp.json`, `alloc_conf_w6.json`,
`worker_timings_probe.json`, `pip_freeze.txt`). Full record and reproduce commands:
[`27-remaining-work-implementation.md`](../27-remaining-work-implementation.md) §11.
