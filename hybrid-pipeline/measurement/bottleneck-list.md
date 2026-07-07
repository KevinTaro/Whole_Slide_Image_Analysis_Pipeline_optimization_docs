# Bottleneck list — hybrid pipeline (measured)

> Companion to `perf_report.html`. Executes plan §5.2/§5.3 (per-candidate record +
> Amdahl stop-loss) and §6 (classification only — **no fixes proposed**).
> Measured on git `96a28ba`, config_hash `db2b7e6a`, RTX 5090 / CUDA 13.0 /
> torch 2.10.0+cu130, {small 25 · medium 121 · large 441}-tile real WSI crops.
> Primary numbers below are the **large (441-tile)** anchor = **848.0 s**.

## Anchors (control / "dumb-version" baselines — preserve, do not overwrite)

| scale | tiles | grid | end-to-end | s/tile | precut A | cells | bg tiles | peak RSS | peak VRAM |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| small  | 25  | 5×5   | 54.3 s  | 2.17 | 1.45 s  | 22  | 3  | 2.82 GB | 4.68 GB |
| medium | 121 | 11×11 | 243.3 s | 2.01 | 5.57 s  | 103 | 18 | 3.07 GB | 4.68 GB |
| large  | 441 | 21×21 | 848.0 s | 1.92 | 20.58 s | 379 | 62 | 4.04 GB | 4.68 GB |

Full-WSI (35,700 tiles @ 1024px) linear projection: `total ≈ 9.4 s + 1.903 s/tile`
⇒ **~18.9 h**, treated as an **upper bound** (crops are tissue-dense ~85%; a real
slide is mostly white background whose empty-core tiles short-circuit cheaply).

## Amdahl stop-loss (plan §5.2)

Ranking is by **% of anchor, not absolute seconds**. Items **< ~10%** (Amdahl
ceiling < 1.11) are recorded but **not** deep-analyzed this round. Deep-record
candidates are therefore only #1–#3; #4–#7 are logged for completeness.

---

## Deep-record candidates (share not negligible)

### ① GPU under-utilisation from a fully serial pipeline — **PRIMARY**
| field | value |
|---|---|
| 現象 | GPU idle **~46–49%** of wall (large: idle_frac 0.459), mean SM **~28%**, even as the only GPU job. B1 forward = **45.5%** of wall = three sequential forwards/tile (M2 Cellpose + M3b Cellpose + M1 UNet) sharing one CUDA context. VRAM peak **5.16 / 32 GB** (batch_size hard-wired 16). |
| 佔總時間 | **45.5%** (B1 phase); the idle itself ≈ half of wall |
| Amdahl 天花板 | 1.84 (B1 phase) — but the real lever is the idle, not B1 compute |
| 所在 Phase | B1 (× cross-cutting §4.1/§4.7) |
| 量測規模 | 25/121/441 tiles — regime-stable (idle 0.49→0.48→0.46) |
| 新現象 | Old 03/04 *suspected* GPU starvation but never verified; **now measured** |
| 分類方向 | **Class 3 (parallel/concurrency) + Class 6 (architecture)**: serial `run_batch` loop, no cross-tile GPU batching, models never overlapped with CPU work |
| 信心 | 實測直接量到 (measured directly) |

### ② `detect_all_dots` — M3 HER2/CEP17 dot detection
| field | value |
|---|---|
| 現象 | **260.1 s = 30.7%** of wall — the single largest sub-item, larger than any one GPU forward. Pure-CPU LAB + H-morphology dot detection per cell, dispatched via joblib `n_jobs=-1`; the main thread blocks on subprocess workers (cProfile main-thread shows ~11.5 s `time.sleep`). Grows with **cell count** (26.4% at 121 tiles → 30.7% at the denser 441-tile crop). |
| 佔總時間 | **30.7%** |
| Amdahl 天花板 | 1.44 |
| 所在 Phase | B3 |
| 量測規模 | 121/441 tiles; scales with tissue/cell density |
| 新現象 | Old 03-doc had it at 17.1%; **now higher** (dedup refactor + denser input) |
| 分類方向 | **Class 1 (algorithm complexity)** + Class 3 (already CPU-parallel, but runs while GPU idles) |
| 信心 | 實測直接量到 |

### ③ PNG encode + write of per-tile artifacts
| field | value |
|---|---|
| 現象 | **76.8 s = 9.05%** of wall. Every tile unconditionally writes `core_mask` / `masked_ihc` / `dish_mask_overlay` as lossless PNG via `skimage.io.imsave`. (int32 TIFF label writes are separately < 0.5%; per-cell crops ~0.5%.) |
| 佔總時間 | **9.05%** |
| Amdahl 天花板 | 1.10 |
| 所在 Phase | B2 |
| 量測規模 | 25/121/441 — stable ~9–10% |
| 新現象 | New breakdown; old doc lumped "13.4% file I/O" differently and pre-int32-TIFF |
| 分類方向 | **Class 5 (I/O & storage layout)**: lossless PNG encode of debug arrays on the critical path |
| 信心 | 實測直接量到 |

---

## Logged below the Amdahl floor (recorded, not deep-analyzed — plan §5.2)

### ④ Per-tile `gc.collect()` — tile-boundary cleanup
- **36.3 s = 4.28%**, ceiling 1.04. Entirely the explicit `gc.collect()` once per tile in `run_batch` (`torch.cuda.empty_cache` is separate < 0.3%). A full Python GC sweep every tile is fixed overhead, linear in tile count. **Never measured before.** Class 4 (memory lifecycle) + 6 (framework). 信心: 實測.

### ⑤ Phase A precut (2.43%) & Phase D overlay stitch (0.60%) — new stages
- Both brand-new stages the old perf_report never covered. Precut A ~2.4% at every scale (real 8-thread parallel I/O, pyvips releases GIL). Stitch D ~0.6% — a single serial `pyvips` join+lzw+`tiffsave` after all analysis (read+join+compress fused in one C call, not separable by Python timing). At full WSI, D becomes a single serial ~minutes block but still < 1% of a ~19 h run. Class 5 (I/O); D also Class 3 (serial, could overlap analysis). 信心: 實測 + 外推.

### ⑥ Model init (one-time)
- 7.3% at 25 tiles → 1.3% at 121 → **0.37%** at 441. Pure one-time load, amortizes to negligible at WSI scale. Class 6. Informational. 信心: 實測.

### ⑦ API / job layer (Phase E)
- `submit_job` enqueue **2.3 µs** mean; BackgroundTask dispatch **~3.8 ms** — ~10⁻⁷ of a multi-hour run. Negligible. Only caveat: concurrent analysis requests each hold a threadpool worker but serialize on the single GPU/CUDA context (future multi-request risk, not a current hotspot). Class 6. 信心: 實測.

---

## Memory-growth claim verification (plan §3.6 / §4.3)
- **VRAM**: flat **5.16 GB / 32 GB** at all scales — does **not** grow with tile count. ✔ bounded.
- **RSS**: 2.82 → 3.07 → 4.04 GB over 25 → 121 → 441 tiles (17.6× tiles, +43% RSS) — clearly **sub-linear in tiles**, but not perfectly flat: it tracks **total accumulated cell results** (`per_tile_owned` held in RAM until the final global merge), i.e. grows with cell count, not tile count. New-architecture "memory bounded, not linear in tiles" claim **holds**; the residual growth is cell-count-driven, not the deleted 400 GB canvas.

## Classification summary (plan §6)
| class | bottlenecks |
|---|---|
| 1 algorithm/model complexity | ② detect_all_dots; the Cellpose SAM `get_rel_pos` forward cost inside ① |
| 3 parallel/concurrency | ① serial pipeline / GPU idle; D serial stitch |
| 4 memory lifecycle | ④ per-tile gc; RSS cell-result accumulation |
| 5 I/O & storage | ③ PNG encode; ⑤ precut & stitch |
| 6 architecture/framework | ① no cross-tile batching; ⑥ init; ⑦ API layer |
| 7 config/dead-code | `cellpose_batch_size` dead (G-B) — must fix before any batch-size sweep is meaningful |

---

## Next stage (solution design, out of scope here)

Per plan §5.2/§6, this document classifies only. The solution-design follow-up for
**① (PRIMARY)** — the largest lever — lives in
[`../10-gpu-serial-pipeline-plan.md`](../10-gpu-serial-pipeline-plan.md). It also explains why
② and ③ are not independently actioned this round (their wall-clock cost overlaps almost exactly
with ①'s measured idle window and may be resolved for free if ①'s fix lands).
