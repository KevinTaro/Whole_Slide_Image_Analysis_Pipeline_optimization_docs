# Bottleneck list — hybrid pipeline (measured)

> Companion to `perf_report.html`. Executes plan §5.2/§5.3 (per-candidate record +
> Amdahl stop-loss) and §6 (classification only — **no fixes proposed**).
> Measured on git `96a28ba`, config_hash `db2b7e6a`, RTX 5090 / CUDA 13.0 /
> torch 2.10.0+cu130, {small 25 · medium 121 · large 441}-tile real WSI crops.
> Primary numbers below are the **large (441-tile)** anchor = **848.0 s**.
>
> **Reading order (3 measured rounds — all preserved, none overwritten):** the ①–⑦ item
> bodies below are the **original control-era record**; each carries dated `Update:` notes
> for later rounds. For *current* status start at **"Round-3 anchors (2026-07-22)"** and
> **"Current ranking"** immediately below, then **"Re-sorted priority after round 3"** at
> the end. Round 3's anchor is **573.7 s** (large/441) — where a % below is quoted without
> a round, it is against that round's own anchor, per plan §5.1.

## Anchors (control / "dumb-version" baselines — preserve, do not overwrite)

| scale | tiles | grid | end-to-end | s/tile | precut A | cells | bg tiles | peak RSS | peak VRAM |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| small  | 25  | 5×5   | 54.3 s  | 2.17 | 1.45 s  | 22  | 3  | 2.82 GB | 4.68 GB |
| medium | 121 | 11×11 | 243.3 s | 2.01 | 5.57 s  | 103 | 18 | 3.07 GB | 4.68 GB |
| large  | 441 | 21×21 | 848.0 s | 1.92 | 20.58 s | 379 | 62 | 4.04 GB | 4.68 GB |

Full-WSI (35,700 tiles @ 1024px) linear projection: `total ≈ 9.4 s + 1.903 s/tile`
⇒ **~18.9 h**, treated as an **upper bound** (crops are tissue-dense ~85%; a real
slide is mostly white background whose empty-core tiles short-circuit cheaply).

## Round-3 anchors (2026-07-22) — Cellpose 4.2.1.1 / cpdino swap

> **Baselines above are preserved verbatim and must not be overwritten.** This
> section adds the third measured round; full three-way comparison lives in
> [`current-status-comparison.md`](./current-status-comparison.md) §8.
> Measured on git `f95a573`, config_hash `db2b7e6a` (unchanged), same RTX 5090 /
> driver 580.159.03, same crops, same `scripts/perf_measure.py`, `--workers 8`,
> `--gpu-dmon`, GPU otherwise idle (89 MiB / 0% before launch).
> Raw artifacts: `_metrics_cellpose421/` (incl. `env_stamp.txt`, `pip_freeze_actual.txt`).

| scale | tiles | control wall | overlap wall | **round-3 wall** | Δ vs overlap | Δ vs control | s/tile |
|---|--:|--:|--:|--:|--:|--:|--:|
| medium | 121 | 243.3 s | 208.2 s | **166.6 s** | **−20.0%** | −31.6% | 1.377 |
| large  | 441 | 848.0 s | 707.4 s | **573.7 s** | **−18.9%** | −32.3% | 1.301 |

Full-WSI linear refit on the two round-3 anchors: `wall ≈ 12.6 s + 1.2722 s/tile`
⇒ **~12.6 h** (was ~15.5 h, ~18.9 h) — same upper-bound caveat as above.

**What changed** (bundled, not a single-variable swap — see §8 of the comparison doc):
cellpose `4.0.8` → `4.2.1.1` (DINOv3-backbone `cpdino` checkpoints + `use_bfloat16=True`
default), both Cellpose checkpoints **retrained**, and the venv rebuilt from `uv.lock`
(torch 2.10→2.11, numpy 2.2.6→1.26.4, scikit-image 0.25.2→0.24.0, pyvips 3.1.1→2.2.3,
opencv→4.8.1.78). Pipeline code is effectively unchanged vs `0e27b20` (only
`_process_one_chunk` dead-code removal + a `tiffsave` tile_width/height default).

## Amdahl stop-loss (plan §5.2) — **revised for the overlapped pipeline**

Original rule, preserved verbatim (still governs the control-era ranking above):

> Ranking is by **% of anchor, not absolute seconds**. Items **< ~10%** (Amdahl
> ceiling < 1.11) are recorded but **not** deep-analyzed this round. Deep-record
> candidates are therefore only #1–#3; #4–#7 are logged for completeness.

**Revision (2026-07-22).** Since the ① two-stage overlap landed, "self-time ÷ wall" is
no longer a critical-path share, so it can no longer be used to rank. `run_batch` runs
two arms concurrently (confirmed by reading `hybrid_pipeline.py:766-810`):

- **MAIN arm** (main thread, `_process_one_chunk_gpu` + loop body): the 3 GPU forwards,
  `_read_rgb`, M1 overlay ops, `clear_slide_edge_cells`, `build_all_positive_results`,
  `enlarge_cell_instances`, **`gc.collect` + `empty_cache`**.
- **BG arm** (one background thread, `_finish_chunk_cpu`): `detect_all_dots`, merge,
  PNG/TIFF encode, `render_overlay_image`, per-cell crops, `filter_and_absolutize`.

`wall ≈ max(MAIN, BG) + outside`, where `outside` = precut A + stitch D + model init.
Validated at both scales (large: max(538.3, 387.3) + 28.0 = 566.3 s vs measured 573.7 s;
medium: 164.7 vs 166.6 — within 1.3%). **Ranking is therefore by critical-arm
contribution, not self-time %**; an item on the slack arm has ceiling ≈ 1.00 no matter
how large its self-time.

Second revision, per project direction (2026-07-22): the flat "<10% ⇒ drop it" floor is
**suspended** for items that (a) sit on the critical arm and (b) are plausibly reducible
to **near zero**. At a ~12.6 h full-WSI runtime, a 6% item is ~45 min of wall — worth
recording and costing even though `1/(1-p)` looks unexciting. Such items are marked
**"sub-floor but actionable"** below rather than stop-lossed.

## Current ranking (re-sorted by round-3 critical-path share, large/441)

| rank | item | self-time | % wall | **arm** | **ceiling if →0** | status |
|--:|---|--:|--:|---|--:|---|
| 1 | ① GPU forwards (2× Cellpose + UNet++) | 444.6 s | 77.5% | **MAIN (critical)** | **1.382x** | still PRIMARY; −23.2% vs overlap round |
| 2 | ④ `gc.collect` per tile | 36.4 s | 6.3% | **MAIN (critical)** | **1.083x** | **RESOLVED (2026-07-22)** — `gc.freeze()` adopted, cost now 0.52 s; see item ④ update and [doc 16](../16-gc-collect-frequency-result.md) |
| 3 | ⑧ main-arm CPU prep (`enlarge_cell_instances` + `build_all_positive_results`) | 28.4 s | 5.0% | **MAIN (critical)** | ~1.05x | **NEW** — pure CPU sitting on the GPU arm |
| 4 | ⑤ precut A + stitch D | 25.6 s | 4.5% | **outside overlap** | ~1.05x | unchanged; wholly serial |
| 5 | ② `detect_all_dots` | 292.9 s | 51.1% | BG (slack) | **1.013x** | hidden — but **+22.3% vs overlap round** (⑨) |
| 6 | ③ PNG encode+write | 78.7 s | 13.7% | BG (slack) | 1.013x | hidden |
| 7 | ⑥ model init / ⑦ API layer | 2.5 s / µs | 0.4% | outside / n-a | ~1.00 | negligible |

**Arm slack (the number that governs what to do next):** BG arm is **387.3 s vs MAIN
538.3 s** — BG/MAIN = **0.719** (was 0.496 in the overlap round). The MAIN arm must shed
**151.0 s (= 34.0% of the GPU forwards)** before the background CPU arm becomes the new
critical path. At medium the margin is thinner still: **36.8%**. In other words ② and ③
are **one more Cellpose-sized speedup away from being re-exposed**, exactly as this
document predicted ("Still Class 5 backlog if B1 ever shrinks enough to re-expose it").

> **Stale as of 2026-07-22:** this 34.0%/36.8% margin was measured *before* `gc.freeze()`
> landed (item ④, resolved) and shrank MAIN's gc component from 36.4 s to ~0.5 s. MAIN
> itself is a bit smaller now, which *tightens* this margin somewhat — re-measure the
> actual MAIN/BG split before sizing ⑧ or the Priority 4 batch-size sweep against it; don't
> reuse 34%/36.8% as-is.
>
> **SUPERSEDED (round 4, 2026-07-22).** That re-measurement has now been done at both
> anchors under the full §0 protocol — the margin was **25.6% (large) / 26.4% (medium)**,
> not 34.0%/36.8%. After ⑧ was moved off the MAIN arm it is **15.9%**. Full record:
> [`../18-gpu-starvation-prerequisites-implementation.md`](../18-gpu-starvation-prerequisites-implementation.md).

## Round-4 anchors (2026-07-22) — ⑧ moved off MAIN + precut A overlapped

> Adds the fourth measured round; **rounds 1–3 above are preserved verbatim.** Same machine,
> crops, harness and checkpoints as round 3 (checkpoint SHA-256 verified identical), GPU idle
> before every launch, `--gpu-dmon --workers 8`, n=3 at large / n=2 at medium.
> Raw artifacts: `_metrics_r4/` (incl. `env_stamp_p0.txt`, `env_stamp_p2.txt`, `pip_freeze.txt`),
> per-run CSVs in `runs_r4/`. Solution design: [`../17-...-plan.md`](../17-gpu-starvation-prerequisites-plan.md);
> implementation + results: [`../18-...-implementation.md`](../18-gpu-starvation-prerequisites-implementation.md).

| config | large/441 | medium/121 | what changed |
|---|--:|--:|---|
| round-3 record | 573.7 s | 166.6 s | — |
| `p0` round-4 baseline | **538.5 s** | **154.0 s** | `gc.freeze()` (doc 16) now in the baseline |
| `p2` | **495.5 s** | **146.3 s** | ⑧ moved to the BG arm (doc 13 P2) |
| `p3` | **480.3 s** | **140.8 s** | precut A streamed into the analysis loop (doc 13 P3) |

**Cumulative −16.3% vs round 3** (−10.8% from this round's two code changes alone).
Full-WSI refit on the `p3` anchors: `wall ≈ 12.5 s + 1.0608 s/tile` ⇒ **~10.5 h** (was ~12.6 h).

Arm state after `p3` (large): MAIN **453.9 s**, BG **381.7 s**, outside **7.6 s** →
BG/MAIN **0.841**, i.e. MAIN has only **15.9%** left to shed before ②③ are re-exposed.
The single-process floor is therefore `BG + outside = 389.3 s` (**1.23x** from 480.3 s).

**Two measurement caveats this round established — apply them to the tables above:**
1. `idle_frac` counted as *exactly* SM==0 is knife-edge and must not be compared across
   configurations; between `p0` and `p2` it rose 0.32→0.43 while wall **fell** 8%, purely
   because SM=1–3% samples collapsed onto 0. Near-idle (SM≤3) fell 0.50→0.45, consistent with
   the wall-clock. Prefer cuda-Event gaps over either.
2. `peak_cuda_reserved_gb` is unreliable (one run reported 25.97 GB against a `dmon fb` peak of
   2787 MB, identical to every other run) — the same artifact class doc 13 warned about. Read
   VRAM from `dmon fb`.

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

- **Update (方案 (b) landed → single-process two-stage overlap)**: `run_batch` now overlaps each tile's
  GPU front with the previous tile's CPU back (background thread, depth 1). Measured effect: GPU idle_frac
  0.494 → 0.154, 121-tile wall −18.5%; see [`pipeline-overlap-result.md`](./pipeline-overlap-result.md).
  Post-overlap the GPU front (M1 UNet + M2/M3b Cellpose forwards) is now **80.9% of wall @441** and is the
  standing critical path — the next lever is the forwards themselves (① next stage, CuPy/GPU-kernel), out of
  scope here. This is also **why ② became overlap-hidden** (see ②'s cross-check).

- **Re-confirmed (2026-07-11, HEAD `0e27b20`, final control-vs-current comparison)**: full numbers now
  in [`current-status-comparison.md`](./current-status-comparison.md) §1/§3 — idle_frac 0.459→**0.190**
  (large) / 0.477→**0.163** (medium); wall −16.6% (large) / −14.5% (medium). ①'s direct effect is
  confirmed at both scales, not just the earlier 121-tile spot-check.

- **New finding — B1 grew in *absolute* seconds too, not just % of wall (2026-07-11)**: at large/441,
  B1 went **386.0 s → 578.9 s** (+192.9 s), not merely 45.5%→81.8% from a shrinking denominator. Two
  distinct causes, confirmed by reading the code (not assumption):
  1. **Harness relabeling, not a compute change.** `feedfbd` deleted `segment_masked_dish` (the old M2
     entry point) — `hybrid_pipeline.py` now calls `segment_windowed` directly for **both** M2 and M3b,
     and `perf_measure.py`'s monkeypatch times `segment_windowed` under one bucket name
     (`B1_m3b_cellpose`, now `n=824` = 2 calls × 412 tiles). `B1_m2_cellpose` didn't disappear because
     the work vanished — it disappeared because the old wrapped entry point no longer exists.
     Control-equivalent combined cost: 187.0 + 185.5 = **372.55 s** (412 tiles) vs current merged
     **564.48 s** (824 calls) — a real **+51% per-tile** increase remains after accounting for the
     relabeling.
  2. **The remaining +192 s is consistent with — but not yet directly isolated as — GIL contention from
     the background CPU stage**, per the project's own py-spy diagnostic in
     [`gil-contention-diag.md`](./gil-contention-diag.md): that study found the *main thread's own*
     Python overhead (Cellpose's `dynamics.py`/SAM `get_rel_pos` kernel-launch loops, ~19% of wall,
     model-inherent) dominates GIL holding (81.4%) inside the *already-overlapped* pipeline, and its own
     ablation (relocating `gc.collect` to the background stage) showed concurrent background work
     measurably **lengthens** main-thread Python segments (idle_frac 0.183→0.221) — i.e. the mechanism is
     real and demonstrated, just not measured yet for `detect_all_dots`'s much larger (~240–260 s/run)
     background footprint specifically. **Not proven, high-confidence hypothesis** — see item 2 in the
     re-sorted next-stage list below for the isolating measurement that would close this out.
  3. **Practical consequence**: that diagnostic already priced this dynamic into a **~1.18–1.23x**
     residual Amdahl ceiling and formally **stop-lossed** further GIL-contention chasing this round
     (`detect_all_dots` process-backend swap: tested, negative; `gc.collect` relocation: tested,
     negative; CUDA-graph-capture patch to pinned `cellpose==4.0.8`: technically feasible per-function,
     but ceiling too small for the third-party-patch risk). This finding **explains** the absolute-seconds
     anomaly; it does not reopen a lever the project already closed with direct measurement.

- **Round 3 (2026-07-22, HEAD `f95a573`, Cellpose 4.2.1.1 / cpdino) — largest single
  improvement to date, and it came from outside this project's own optimization work.**
  A coworker swapped Cellpose to the DINOv3-backbone `cpdino` line (4.0.8 → 4.2.1.1,
  `use_bfloat16=True` by default) and retrained both checkpoints. Measured effect on the
  standing critical path:
  | metric (large/441) | overlap round | round 3 | Δ |
  |---|--:|--:|--:|
  | end-to-end wall | 707.4 s | **573.7 s** | **−18.9%** |
  | GPU forwards (B1 total) | 578.9 s | **444.6 s** | **−23.2%** |
  | s per Cellpose call (824 calls) | 0.6850 | **0.5229** | **−23.7%** |
  | VRAM peak (dmon fb) | 5159 MB | **2787 MB** | **−46.0%** |
  | GPU idle_frac | 0.190 | **0.370** | +0.18 |
  | GPU mean SM % | 32.9 | **16.6** | −16.3 pt |
  Medium/121 agrees (wall −20.0%, per-call 0.6949 → 0.5181 = −25.4%, VRAM 5159 → 2785 MB).
  **The B1 absolute-seconds anomaly recorded above is partly resolved by this**: B1 went
  386.0 (control) → 578.9 (overlap) → **444.6 s**, i.e. most of the +192.9 s regression is
  gone, while the harness-relabeling caveat (`B1_m3b_cellpose` sums both forwards, n=824)
  still applies and is unchanged. ① remains **rank 1 / PRIMARY**, but its ceiling has
  fallen from 1.84 to **1.382x** and its slack margin over the BG arm is now only 34%.
  信心: 實測直接量到 (both scales, GPU otherwise idle).
- **Caveat — this was not a single-variable change.** The same `uv sync` rebuilt the whole
  venv (torch 2.10→2.11, numpy 2.2.6→**1.26.4**, scikit-image 0.25.2→**0.24.0**, pyvips
  3.1.1→**2.2.3**, opencv→**4.8.1.78**), and the checkpoints were retrained. The GPU-forward
  gain is confidently attributable to Cellpose (it is localised to the two Cellpose buckets;
  UNet++ is flat at 13.45 → 13.68 s), but the CPU-side regression in ⑨ is **not** separable
  from the dependency downgrades with the data on hand. VRAM −46% is consistent with the
  documented bfloat16 default. No pip-freeze snapshot exists for the overlap round
  (`_metrics_current/`), so its exact package set is unrecorded — a gap for future rounds.
- **Correctness is not held constant** (out of scope per plan §1.3, but must be flagged
  before anyone reads −18.9% as free): retrained checkpoints change segmentation output.
  Cells 12,922 → **13,150** (+1.8%) at large, 3,558 → **3,647** (+2.5%) at medium, and one
  tile flipped success → skipped (379 → 378) at large. Previous rounds agreed to within
  ±3 cells (GPU-nondeterminism noise floor); this round does not. **This is a
  model-quality change requiring clinical validation, not a pure performance swap.**

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

- **Cross-check / disposition (2026-07-11, HEAD `00f2c91`, after ① 方案 (b) overlap landed)**: the
  30.7% above is the **serial baseline** (this table). Re-measured on current HEAD (② timer already in
  `perf_measure.py` as `B3_detect_dots`, no py-spy): the two-stage overlap pipeline runs `detect_all_dots`
  on the background thread, and at **both** 121- and 441-tile it is **fully hidden behind the GPU front** —
  per-tile detect ≤ 0.67 s vs per-tile GPU front ~1.33 s (large: detect 238.6 s / **but 0.579 s/tile** <
  B1 1.329 s/tile). So its **effective** end-to-end Amdahl ceiling is ≈ **1.0**, not the serial 1.44, and
  the doc-12 premise "② is now the heavy pole" was **refuted by measurement**. Actioned: the cheap zero-risk
  `disk()`-hoist (doc 12 §3(a)) was landed bit-exact; the process-backend / regionprops_table / whole-tile
  vectorization options (doc 12 §3(b)/(c)/(d)) were **stopped-out** (a hidden stage can't move wall). The
  remaining lever is the GPU front itself (① next stage). Full record:
  [`detect-all-dots-result.md`](./detect-all-dots-result.md). 信心: 實測直接量到.

- **Round 3 (2026-07-22) — still hidden, but the hiding place is shrinking, and it got
  slower in absolute terms.** Self-time **292.9 s = 51.1%** of wall (large), up from
  239.4 s / 33.8%. Effective ceiling is still ≈ **1.013x** because it remains on the BG
  (slack) arm — per-tile 0.711 s vs MAIN-arm 1.221 s/tile — so the "stopped-out" disposition
  above **still holds and should not be reopened on the strength of that 51%**. Two things
  did change and are worth recording:
  1. The BG arm as a whole is now **387.3 s vs MAIN 538.3 s** (ratio 0.496 → **0.719**).
     `detect_all_dots` is 76% of that arm, so it is the thing that will re-expose ② and ③
     the moment the MAIN arm sheds 34%.
  2. Its absolute cost **regressed +22.3%** for only +1.8% more cells → logged separately
     as **⑨** below.
  信心: 實測直接量到.

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

- **Update (2026-07-11, re-measured on HEAD `0e27b20`)**: now runs on the background CPU stage alongside
  ②, overlapped with the GPU front — **78.5 s / 11.1% wall** (large), same absolute cost as the serial
  baseline (76.8 s / 9.05%) but largely hidden behind B1. Not independently actioned this round; see
  [`current-status-comparison.md`](./current-status-comparison.md) §3. Still Class 5 backlog if B1 ever
  shrinks enough to re-expose it.

- **Round 3 (2026-07-22)**: **78.7 s / 13.7%** of wall (large) — absolute cost flat across
  all three rounds (76.8 → 78.5 → 78.7 s), share risen purely from the shrinking denominator.
  Still on the BG (slack) arm, ceiling **1.013x**, still not independently actionable. Same
  re-exposure trigger as ②.

---

## Logged below the Amdahl floor (recorded, not deep-analyzed — plan §5.2)

### ④ Per-tile `gc.collect()` — tile-boundary cleanup
- **36.3 s = 4.28%**, ceiling 1.04. Entirely the explicit `gc.collect()` once per tile in `run_batch` (`torch.cuda.empty_cache` is separate < 0.3%). A full Python GC sweep every tile is fixed overhead, linear in tile count. **Never measured before.** Class 4 (memory lifecycle) + 6 (framework). 信心: 實測.
- **Update (2026-07-11, confirmed unchanged @441)**: still 36.3 s / 5.1% wall (share rose slightly, same
  cause as ①'s denominator-shrink note), now on the background stage → partially hidden. **Escalated by
  py-spy** ([`gil-contention-diag.md`](./gil-contention-diag.md)): within the overlapped pipeline it is the
  **single largest GIL holder (33.6% of GIL-holding samples)** even though it's only 3.7% of wall — a
  sequential main-thread stall where GPU is idle *and* the background thread is blocked. **Ablation already
  run and reverted**: relocating it to the background stage made idle_frac *worse* (0.183→0.221, not
  better) because depth-1 overlap is background-CPU-bound — moving gc there just lengthens the arm the
  main thread already waits on. Only untried lever: **reduce call frequency** (batch every N tiles instead
  of every tile) — lower priority than relocation would have been, since it touches the memory-bounded
  invariant (RSS grows between sweeps) and needs re-validation at 441-tile scale before adoption.
- **Correction (2026-07-22, read from source, supersedes the "now on the background stage"
  wording in the 2026-07-11 note above — kept for history, not deleted):** `gc.collect()` is
  **on the MAIN thread**, called from the `run_batch` loop body at `hybrid_pipeline.py:798`,
  between the tile's GPU front and the `_collect(pending)` join. It is *not* on the background
  stage and never was after the relocation ablation was reverted. It is therefore **fully on
  the critical arm** and **not** "partially hidden" — the earlier wording understated it.
- **Round 3 (2026-07-22) — escalated to rank 2, "sub-floor but actionable".** Self-time
  **36.4 s = 6.34%** of wall (large), essentially identical in absolute seconds across all
  three rounds (36.31 → 36.33 → 36.39 s) — it is a fixed per-tile cost, linear in tile count
  and independent of everything else. Being on the critical arm, its ceiling is a genuine
  **1.083x**, and it is one of the few items plausibly reducible to **near zero** (batch the
  sweep every N tiles instead of every tile). At the round-3 full-WSI projection (~12.6 h)
  that is **~44 min of wall**. Under the revised stop-loss rule above this is **recorded as
  actionable, not stop-lossed** — the "only untried lever" note above stands, with the same
  caveat that it touches the memory-bounded invariant and needs RSS re-validation at 441 scale.
  Class 4 + 6. 信心: 實測直接量到 (three rounds).
- **RESOLVED (2026-07-22) — `gc.freeze()` adopted, not the planned batching lever.** Full
  record: [`15-gc-collect-frequency-implementation.md`](../15-gc-collect-frequency-implementation.md) /
  [`16-gc-collect-frequency-result.md`](../16-gc-collect-frequency-result.md). Measurement found the
  cost driver was **per-call scan volume** (the three resident GPU models being re-walked every
  sweep), not call count — so "batch every N tiles" (the lever this entry called for) was aimed at
  the wrong variable. `gc.freeze()` after model init instead cuts the *price* of each call
  (83.2 ms → 1.2 ms, −98.6%) while leaving cadence untouched (still 441 calls, once per tile — the
  memory-bounded invariant this entry flagged is therefore **not in play**, confirmed by the
  invariant guard `scripts/verify_gc_freeze.py`). Result: gc cost 36.71 s → 0.52 s at the large
  anchor; **1.069x attributable / 1.077x end-to-end**, essentially matching this entry's predicted
  1.083x ceiling. The batching option (N=4/8/16, `gc.freeze()` off) was also built and measured —
  it recovered the *same* price-per-call reduction only when combined with freeze, added nothing on
  top of freeze alone (indistinguishable at large: 535.1 s vs 512.2–540.3 s freeze-alone range), and
  raised peak RSS to the highest of any configuration tested (3.991 GB vs 3.925 GB adopted) — so it
  was implemented, measured, and deleted rather than shipped. No config knob landed; the change is
  unconditional. Correctness veto passed (max|Δ| = 0 for reddot/blackdot/score among matched cells).
  This item is **closed** — see §"Re-sorted priority after round 3" below, item 2.

### ⑤ Phase A precut (2.43%) & Phase D overlay stitch (0.60%) — new stages
- Both brand-new stages the old perf_report never covered. Precut A ~2.4% at every scale (real 8-thread parallel I/O, pyvips releases GIL). Stitch D ~0.6% — a single serial `pyvips` join+lzw+`tiffsave` after all analysis (read+join+compress fused in one C call, not separable by Python timing). At full WSI, D becomes a single serial ~minutes block but still < 1% of a ~19 h run. Class 5 (I/O); D also Class 3 (serial, could overlap analysis). 信心: 實測 + 外推.
- **Update (2026-07-11, confirmed unchanged @441)**: A = 20.4 s/2.9%, D = 5.1 s/0.7% — both flat, both
  still outside the overlapped B loop. D remains the only structurally-overlappable stage nobody has
  touched yet (below floor, but full-WSI it becomes a genuine multi-minute serial block).
- **Round 3 (2026-07-22) — flat in seconds, risen to rank 4 by share; "sub-floor but
  actionable".** A = **20.51 s / 3.58%**, D = **5.05 s / 0.88%** (large); both unchanged in
  absolute terms across all three rounds (A: 20.58/20.39/20.51; D: 5.11/5.08/5.05) despite
  pyvips being **downgraded 3.1.1 → 2.2.3** in this round's venv — so the pyvips version
  change costs nothing measurable here. Combined **25.6 s = 4.5%** of wall, ceiling ~1.05x.
  These are the *only* two stages that are 100% serial **and** entirely outside the overlap
  (`outside` in the arm model), so unlike ②/③ their cost is fully on the critical path and
  both are reducible toward zero by overlapping them with the B loop. A scales linearly with
  tile count → at full-WSI it is the larger of the two by far. Recorded as actionable.
- **Round 4 (2026-07-22) — A resolved, D closed as structurally non-overlappable.**
  **A**: `m0_reader.PrecutStream` hands the tile grid over immediately (derivable from the
  image header alone — `read_size` decodes no pixels) and yields tiles as they are cut, so the
  cutting overlaps the analysis loop. `phaseA_precut_s` 20.31 s → **0.004 s**; net wall
  **−3.1% (large) / −3.8% (medium)**, i.e. ~75% of A recovered at large (the remainder is CPU
  contention with the BG arm). Safe because tile processing order cannot affect output —
  `run_batch` sorts globally by `(abs_y, abs_x, cell_id)` before renumbering and the stitcher
  reads by coordinate; `PrecutStream` was verified to emit a byte-identical tile set.
  **D**: **not overlappable and closed, not backlogged.** The `pyvips` row/column joins are
  lazy — all cost is the single `tiffsave`, which cannot be done incrementally for a pyramidal
  TIFF, and D runs at the very end of `run_batch` with no remaining work to overlap it with.
  Overlappable content ≈ 0 s of its 5.10 s. See
  [`../18-gpu-starvation-prerequisites-implementation.md`](../18-gpu-starvation-prerequisites-implementation.md) §4.

### ⑥ Model init (one-time)
- 7.3% at 25 tiles → 1.3% at 121 → **0.37%** at 441. Pure one-time load, amortizes to negligible at WSI scale. Class 6. Informational. 信心: 實測.
- **Cross-check (2026-07-08, stage-2 GIL diagnosis)**: this same one-time cost is what inflated the "UNet++ Python 14.9%" figure in [`gil-contention-diag.md`](./gil-contention-diag.md)'s original Result 1 table — 93% of that GIL bucket was `_init_unet_inferencer`'s import cascade (`segmentation_models_pytorch`/`torch`/`timm`/`torchvision`/`triton`), not recurring per-tile cost. Confirms this entry's "negligible at scale" and rules out UNet++ as a per-tile GIL contributor; see that doc's "追加深挖" section for the full trace.

### ⑦ API / job layer (Phase E)
- `submit_job` enqueue **2.3 µs** mean; BackgroundTask dispatch **~3.8 ms** — ~10⁻⁷ of a multi-hour run. Negligible. Only caveat: concurrent analysis requests each hold a threadpool worker but serialize on the single GPU/CUDA context (future multi-request risk, not a current hotspot). Class 6. 信心: 實測.

---

## New candidates found in round 3 (2026-07-22)

### ⑧ Pure-CPU prep work stranded on the MAIN (critical) arm — **NEW, rank 3**
| field | value |
|---|---|
| 現象 | `enlarge_cell_instances` (**19.60 s = 3.42%**) and `build_all_positive_results` (**8.81 s = 1.54%**) run inside `_process_one_chunk_gpu` (`hybrid_pipeline.py:569-573`), i.e. on the **main/GPU thread**, between the M2 and M3b Cellpose forwards. Both are pure NumPy/skimage — **neither touches torch or CUDA**. Meanwhile the BG arm has **151.0 s of measured slack**. Combined **28.4 s = 4.96%** of wall (large); 8.36 s = 5.02% (medium) — regime-stable. |
| 佔總時間 | **4.96%** self-time, but **100% of it is on the critical arm** |
| Amdahl 天花板 | ~**1.05x** alone; ~**1.14x** combined with ④ (both are critical-arm items removable without touching the GPU forwards — `max(538.3−36.4−28.4, 387.3+28.4) + 28.0 ≈ 501 s`) |
| 所在 Phase | B3 (placement), × §4.7 arm-balance |
| 量測規模 | 121 + 441 tiles, both round-3 anchors |
| 是否為新現象 | **New.** Invisible before round 3: under the control's serial loop there were no "arms", and in the overlap round the MAIN arm had so much headroom (BG/MAIN 0.496) that arm placement didn't matter. |
| 分類方向 | **Class 3 (parallel/concurrency)** — work placement across an existing two-arm split, not an algorithmic change |
| 信心 | 實測直接量到 (arm membership confirmed by reading the source, not inferred from timings) |

- These grew with the new checkpoints' higher cell yield (enlarge 18.29 → 19.60 s,
  build 7.23 → 8.81 s) and will keep growing with cell count. Recorded only —
  moving them is a code change and is out of scope for this measurement round.
- **RESOLVED (round 4, 2026-07-22).** Both calls moved from `_process_one_chunk_gpu` (MAIN)
  into `_finish_chunk_cpu` (BG). Measured **−8.0% wall at large / −5.0% at medium**, beating
  this entry's ~1.05x estimate because the move also removed GIL contention between ⑧ and
  `detect_all_dots`: the *unmodified* B1 forwards fell 444.4→431.6 s and `detect_all_dots`
  fell 279.9→257.6 s. `torch.cuda.Event` instrumentation confirms the mechanism directly —
  the M2→M3b device-idle gap closed from **10.06 s to 1.66 s** (medium), the residual being
  exactly `clear_slide_edge_cells`. Correctness veto passed (differing cells indistinguishable
  from the same-code noise floor). See
  [`../18-gpu-starvation-prerequisites-implementation.md`](../18-gpu-starvation-prerequisites-implementation.md) §2–3.

### ⑨ CPU back-stage regression: `detect_all_dots` +22.3% — **NEW, watch item**
| field | value |
|---|---|
| 現象 | Large/441: **239.41 s → 292.90 s (+53.5 s, +22.3%)** between the overlap round and round 3, for only **+1.8%** more cells (12,922 → 13,150). Per cell: 18.5 ms → **22.3 ms (+20.2%)**. Medium/121 agrees: 71.33 → 79.78 s (+11.8%) for +2.5% cells. Other BG-arm buckets did **not** regress (PNG encode flat 78.5 → 78.7 s), so this is specific to the dot-detection path, not to the arm or the thread. |
| 佔總時間 | 51.1% self-time, but **ceiling 1.013x** — it is on the slack arm |
| Amdahl 天花板 | **1.013x today.** Its importance is indirect: it consumes the slack that keeps ② and ③ hidden. |
| 所在 Phase | B3 |
| 量測規模 | 121 + 441 tiles |
| 是否為新現象 | **New regression**, appeared in round 3 |
| 分類方向 | **Class 7 (environment/config correctness)** primarily, possibly **Class 1** — `detect_all_dots` is LAB + H-morphology over skimage/numpy/opencv, and this round downgraded **scikit-image 0.25.2 → 0.24.0**, **numpy 2.2.6 → 1.26.4**, **opencv → 4.8.1.78**. That is the leading hypothesis; it is **not isolated** — cell-shape changes from the retrained checkpoints are a competing explanation and the two were changed together. |
| 信心 | **實測直接量到** (the regression), **推論待驗證** (the cause) |

- **Isolating measurement, if anyone wants it** (out of scope here): re-run `detect_all_dots`
  alone over a fixed, saved set of instance masks under both dependency sets. That holds cell
  geometry constant and separates "library downgrade" from "different cells".
- Do **not** action this as a wall-clock fix: at ceiling 1.013x it cannot move the anchor
  today. It matters only as the thing eating the BG arm's remaining margin.

---

## Memory-growth claim verification (plan §3.6 / §4.3)
- **VRAM**: flat **5.16 GB / 32 GB** at all scales — does **not** grow with tile count. ✔ bounded.
- **RSS**: 2.82 → 3.07 → 4.04 GB over 25 → 121 → 441 tiles (17.6× tiles, +43% RSS) — clearly **sub-linear in tiles**, but not perfectly flat: it tracks **total accumulated cell results** (`per_tile_owned` held in RAM until the final global merge), i.e. grows with cell count, not tile count. New-architecture "memory bounded, not linear in tiles" claim **holds**; the residual growth is cell-count-driven, not the deleted 400 GB canvas.
- **Round 3 (2026-07-22)**: invariant still holds, with materially more headroom.
  VRAM peak (dmon fb) **5159 → 2787 MB** at large / **2785 MB** at medium — flat across
  scales as before, but now **−46%**, consistent with Cellpose 4.2's `use_bfloat16=True`
  default. `cuda_reserved` peak 4.675 → **2.187 GB**. RSS peak 3.94 → **3.90 GB** (large),
  3.06 → **3.09 GB** (medium) — unchanged, still cell-count-driven.
- **Consequence for the dead `cellpose_batch_size` config (G-B, class 7)**: idle VRAM
  headroom has grown from ~27 GB to **~29.8 GB of 32 GB**. The field is still not wired
  into `Config`, so the `getattr(config, "cellpose_batch_size", 16)` fallback at
  `hybrid_pipeline.py:206,218` remains the only value ever used. Unchanged as a finding;
  strictly more headroom left on the table than when it was first recorded.
- **Round 4 (2026-07-22) — the config is now wired, but the headroom turns out to be
  unusable at this tile size.** A real `cellpose_batch_size` field was added and both
  `getattr` fallbacks replaced. Sweeping 16/32/64 is **flat** in wall, per-call forward time
  and VRAM alike: the `cpdino` backbone's `bsize=384` splits a 1024² tile into exactly 16
  patches, so batch size 16 already runs them in a single batch. The ~29.8 GB of idle VRAM
  cannot be spent this way — it would only become usable if `default_tile_size` grew to
  ≥1536. The G-B "dead config" defect is fixed; the performance opportunity it implied does
  not exist.

## Classification summary (plan §6)
| class | bottlenecks |
|---|---|
| 1 algorithm/model complexity | ② detect_all_dots; Cellpose/SAM postprocessing inside ① — `_extend_centers_gpu`/`get_masks_torch`/`steps_interp` (`cellpose/dynamics.py`) + `get_rel_pos` (`segment_anything/modeling/image_encoder.py`) are **already GPU-resident, kernel-launch-bound** (Python loop over iterations/seeds/blocks, not CPU placement — see `gil-contention-diag.md` "追加深挖" for per-function trace, 2026-07-08); `fill_holes_and_remove_small_masks` (`cellpose/utils.py`) is the one genuinely CPU-only function (no GPU path in `fill_voids`) |
| 3 parallel/concurrency | ① serial pipeline / GPU idle; D serial stitch; **⑧ CPU prep stranded on the MAIN arm (round 3)** |
| 4 memory lifecycle | ④ per-tile gc; RSS cell-result accumulation |
| 5 I/O & storage | ③ PNG encode; ⑤ precut & stitch |
| 6 architecture/framework | ① no cross-tile batching; ⑥ init; ⑦ API layer |
| 7 config/dead-code | `cellpose_batch_size` dead (G-B) — must fix before any batch-size sweep is meaningful; **⑨ `detect_all_dots` +22.3% coincident with the numpy/skimage/opencv downgrades (round 3, cause not isolated)** |

---

## Next stage (solution design, out of scope here)

Per plan §5.2/§6, this document classifies only. The solution-design follow-up for
**① (PRIMARY)** — the largest lever — lives in
[`../10-gpu-serial-pipeline-plan.md`](../10-gpu-serial-pipeline-plan.md). It also explains why
② and ③ are not independently actioned this round (their wall-clock cost overlaps almost exactly
with ①'s measured idle window and may be resolved for free if ①'s fix lands).

**Status update (2026-07-11, HEAD `0e27b20`, final control-vs-current comparison — see
[`current-status-comparison.md`](./current-status-comparison.md)):**
- **① actioned, confirmed at both scales** — idle 0.459→0.190 (large) / 0.477→0.163 (medium); wall −16.6%
  (large) / −14.5% (medium). GPU front is the standing critical path (81.8% wall @441) — and grew in
  *absolute* seconds too (386.0 s → 578.9 s), not just share; see item ①'s new-finding note above for the
  harness-relabeling + GIL-contention breakdown of that growth.
- **② resolved "for free" — confirmed.** `detect_all_dots` hidden behind the GPU front at all scales; cheap
  `disk()`-hoist landed bit-exact, heavier rewrites stopped-out ([`detect-all-dots-result.md`](./detect-all-dots-result.md)).
- **③ PNG encode** — also hidden behind the GPU front now; not independently actioned.
- **GIL-contention avenue (detect_all_dots process-backend, gc relocation, CUDA-graph-capture patch to
  cellpose internals) — already stop-lossed** by direct ablation in
  [`gil-contention-diag.md`](./gil-contention-diag.md): residual ceiling only ~1.18–1.23x, two of three
  candidate fixes measured negative, the third requires patching a pinned third-party package. **Do not
  reopen this round** without new evidence (e.g. an upstream cellpose fix, or the ceiling crossing a higher
  threshold after other changes land).

### Re-sorted priority for next stage

Previous ranking (see [`current-status-comparison.md`](./current-status-comparison.md) §6) put "shrink the
GPU front via CuPy/kernel work" at #1. That avenue is the one just stop-lossed above (small ceiling,
third-party-patch risk) — re-sorting by **cost × plausible ceiling**, not by raw current-% share:

1. **Wire `cellpose_batch_size` into `Config` (Class 7, config correctness).** `hybrid_pipeline.py:206,218`
   call `getattr(config, "cellpose_batch_size", 16)` — no such field exists on `Config` (the real field,
   `batch_size` at `config.py:184` = 4, feeds UNet++ only), so the fallback `16` is silently the only value
   that has ever run. VRAM sits at 5.16/32 GB — ~27 GB idle headroom unusable until this is wired. Nearly
   free (one field + wiring), and a **prerequisite** for #2 below — not itself a wall-clock fix.
2. **Batch-size sweep, once #1 lands.** Test whether raising Cellpose's internal batch size reduces
   per-call forward latency, using the idle VRAM headroom. This targets the *same* GPU front that's now
   the critical path, but via throughput (larger batches, fewer kernel-launch-bound Python iterations per
   unit of work) rather than the already-exhausted Python-overhead-shaving angle. **Measure, don't assume**
   — run the standard control-vs-current ablation (medium + large anchors, `--gpu-dmon`, bit-exact
   `report.csv` check) before trusting any speedup.
3. **Isolate the GIL-contention share of B1's absolute growth (optional, low-cost, closes an open
   question).** Nobody has directly compared B1's per-tile duration *with* vs *without* the background
   `detect_all_dots` thread running concurrently (only the much smaller `gc.collect` relocation was
   ablated). A short run with the background CPU stage's join point moved earlier (or the background call
   stubbed to a no-op, same tile inputs) would confirm how much of the +192.9 s (large, @441) is genuine
   contention vs. still-unexplained. Low priority given the ~1.2x ceiling already estimated — this closes
   the "why did GPU front get slower" question with direct evidence rather than inference, it doesn't
   promise new wall-clock.
4. **`gc.collect` frequency reduction (batch every N tiles).** Only untried lever from
   `gil-contention-diag.md`'s investigation; touches the memory-bounded invariant (RSS grows between
   sweeps), needs RSS re-validation at 441-tile scale before adoption. Backlog, not urgent (3.7–5.1% wall).
5. **Phase D stitch overlap.** The one remaining serial-but-structurally-overlappable stage; <1% of wall
   at these scales but becomes a genuine multi-minute serial block at full-WSI. Lowest priority of the
   actionable items — record and revisit if full-WSI timing work starts.

**Not next stage — already closed, don't re-litigate:** `detect_all_dots` process-backend swap, `gc.collect`
relocation to the background thread, CUDA-graph-capture/vectorization patches to `cellpose`/`segment_anything`
internals. All three have direct ablation or diagnostic evidence against them this round
([`gil-contention-diag.md`](./gil-contention-diag.md)).

### Re-sorted priority after round 3 (2026-07-22) — supersedes the list above

> The list above is **preserved as the 2026-07-11 record**. It is superseded because the
> Cellpose swap moved the numbers it was ranked on, and because the arm model (see the
> revised stop-loss section) changed how ranking must be computed. Measurement-only doc —
> **nothing below is a recommendation to implement, only a ranking of what measurement says
> is worth costing.**

Ranked by **critical-arm contribution × plausible reduction**, not by self-time %:

1. **The GPU forwards remain #1 (444.6 s, 77.5% wall, ceiling 1.382x).** Still the critical
   arm's dominant term even after −23.2%. **But the headroom is now bounded**: shrinking the
   forwards by more than **34%** buys nothing further until the BG arm also shrinks, because
   the background CPU arm becomes the critical path at that point. Any future GPU work should
   be sized against that 34% ceiling, not against the raw 77.5%.
2. **`gc.collect` frequency reduction (④) — DONE (2026-07-22).** Shipped as `gc.freeze()`
   (Option C), not the batching lever this ranking assumed — see item ④'s update and
   [doc 16](../16-gc-collect-frequency-result.md). Measured **1.069x attributable / 1.077x
   end-to-end** at the large anchor, matching the 1.083x ceiling predicted here almost exactly.
   The RSS re-validation caveat this entry called for was run and passed (peak RSS +1.1%,
   0.14% of the 32 GB machine; sawtooth shape intact). **Removed from the open backlog.**
3. **⑧ Move the stranded CPU prep off the MAIN arm — new, now the top open item.**
   `enlarge_cell_instances` + `build_all_positive_results` = 28.4 s / 4.96%, pure NumPy, no
   torch, and the BG arm has slack to absorb them. The **~1.14x combined (573.7 → ~501 s)**
   projection below assumed both #2 and #3 landed; #2 alone is done and confirmed the
   gc-side half of that projection (predicted −36.4 s, measured **−37.0 s** attributable —
   effectively exact), so the remaining gap to ~501 s is entirely this item, still unbuilt.
   **The MAIN/BG margin should be re-measured with `gc.freeze()` in place before sizing this
   or Priority 4's batch-size sweep** — the 34% figure below predates the gc fix and the
   arm balance has shifted now that gc's contribution is ~0 instead of 36.4 s.
4. **⑤ Overlap precut A and stitch D with the B loop.** 25.6 s / 4.5% combined, wholly
   serial and wholly outside the overlap, so 100% critical path. A scales with tile count →
   at full-WSI this is the largest of the sub-floor items in absolute terms.
5. **Wire `cellpose_batch_size` into `Config` (Class 7), then sweep.** Unchanged from the
   2026-07-11 list except that idle VRAM headroom grew from ~27 GB to **~29.8 GB / 32 GB**
   after the bfloat16 switch, so the argument for it is stronger, not weaker. Still a
   prerequisite for any batch-size measurement rather than a fix in itself.
6. **⑨ Isolate the `detect_all_dots` +22.3% regression.** No wall-clock payoff today
   (ceiling 1.013x) — worth doing only to protect the BG arm's shrinking margin, and cheap
   to settle (re-run dot detection over saved instance masks under both dependency sets).
7. **Record a `pip freeze` with every future round.** The overlap round has none, which is
   why ⑨'s cause cannot be attributed cleanly. Process fix, not a performance item.

### Re-sorted priority after round 4 (2026-07-22) — supersedes the round-3 list above

> The round-3 list is preserved as the record. It is superseded because items 3 (⑧) and 4 (⑤)
> in it are now **built and measured**, and the margin every remaining item must be sized
> against changed from 34.0% to **15.9%**. Full reasoning:
> [`../18-gpu-starvation-prerequisites-implementation.md`](../18-gpu-starvation-prerequisites-implementation.md) §6.

1. **`cellpose_batch_size` — wired (adopted), sweep measured NEGATIVE (closed).** The dead
   config is fixed: a real `cellpose_batch_size: int = 16` field now feeds both segmenters
   (**config hash `db2b7e6a` → `ad41c42f`**). The sweep it was a prerequisite for is a
   **negative result and should not be re-run**: wall/per-call forward time/VRAM are flat
   across 16/32/64 (per-call 517.6/519.0/518.6 ms, VRAM 2787 MB throughout). Cause, measured
   with `cellpose.transforms.make_tiles`: the `cpdino` backbone uses `bsize=384`, so a 1024²
   tile yields exactly **4×4 = 16 patches** — equal to the batch size already in use, i.e. one
   batch, nothing left to batch. The hardcoded fallback 16 was accidentally optimal for this
   tile size. **Only becomes live if `default_tile_size` grows to ≥1536** (25 patches). See
   [`../18-...-implementation.md`](../18-gpu-starvation-prerequisites-implementation.md) §6.1.
2. **Cross-tile multiprocessing — now the only remaining lever with a real ceiling.** Unlike every single-process
   lever it is not bounded by the `BG + outside = 389.3 s` floor (each process carries its own
   BG thread), so its ceiling is wider — roughly **1.23x–1.7x** — but it still requires solving
   fork-under-CUDA (per-process model reload: VRAM 2.79 GB × N, init 2.4 s × N) and carries the
   largest correctness risk of anything remaining. It is now the *only* lever that reaches the
   **intra-forward, launch-bound** idle (the larger half of all device idle): #1 cannot touch it
   at this tile size, and patching Cellpose internals was stop-lossed in `gil-contention-diag.md`.
3. **⑨ isolate the `detect_all_dots` regression (optional, unchanged priority).** Note it moved
   on its own in round 4 (279.9 → 257.6 s) purely from ⑧'s thread relocation, so any isolation
   attempt must control for thread placement.
4. **Closed this round, do not reopen without new evidence:** the CUDA-stream / pipeline-depth-2
   bubble redesign (sized at **≤1.065x** after ⑧ landed — see doc 18 §3), GPU-side tile/transform
   loading (`B2r_tile_read` = **1.22%** of wall, ceiling 1.012x, and no such pipeline exists to
   move), and stitch-D overlap (structurally ~0 s recoverable).

**Superseded by round 3, do not re-run as written:** item 3 of the 2026-07-11 list
("isolate the GIL-contention share of B1's +192.9 s growth"). Most of that growth reversed
on its own with the Cellpose upgrade (578.9 → 444.6 s), so the anomaly it was meant to
explain has largely dissolved. The 2026-07-11 stop-loss list ("Not next stage" above)
otherwise still stands.
