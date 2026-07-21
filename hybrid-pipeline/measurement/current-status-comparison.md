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
