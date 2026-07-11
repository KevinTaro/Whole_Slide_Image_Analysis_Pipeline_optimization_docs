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
