# Bottleneck list — hybrid pipeline (current state)

> **Compact, current-state-only.** This file lists every bottleneck this project has found —
> shipped, hidden, stop-lossed, or still open — with its latest measured result and a link to the
> document that has the full evidence. It does **not** narrate how each round got there; for the
> round-by-round history (control → overlap → Cellpose swap → ... → round 8), see
> [`bottleneck-list-history.md`](./bottleneck-list-history.md). For the exhaustive ledger of every
> candidate discovered but never shipped, see
> [`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md).
>
> Current HEAD: round 9 (config hash unchanged). Round 9 shipped doc 28 Option H (periodic gc re-freeze)
> and doc 30 Option L (tile-read prefetch), and closed doc 29's nvImageCodec route; all three are written up
> in [31](../31-gc-collect-round2-implementation.md) / [32](../32-phase-d-gpu-port-implementation.md) /
> [33](../33-tile-read-io-implementation.md). **No full-slide run was made this round**, so every full-slide
> figure below is still round 8's measurement and every round-9 full-slide number is a labelled projection.
> Round-8 baseline: config hash `3d1087f2` (unchanged since round 7). Machine: RTX 5090 /
> CUDA 13.0 / torch 2.11.0+cu130. Full round-8 record:
> [`../27-remaining-work-implementation.md`](../27-remaining-work-implementation.md); round-by-round
> narrative of how each number below was reached lives in
> [`bottleneck-list-history.md`](./bottleneck-list-history.md), not here.

## Current anchors

| anchor | tiles | background share | `workers=1` | `workers=4` |
|---|--:|--:|--:|--:|
| baseline (control, fully serial, no optimizations) | 441 (large/21×21) | 14.1% | 848.0 s | — |
| large/441 crop, current code | 441 | 14.1% (tissue-dense, **not** representative of a real slide) | 302.7 s | 128.8 s |
| match24 — composition-matched crop | 576 | 55.9% (real slide measured: 55.8%) | **188.8 s** | **88.3 s** |
| **full-WSI (27,565 tiles, real measured run)** | 27,565 | 55.8% (measured, not sampled) | **13,762 s = 3.82 h** | **6,211 s = 1.73 h** |

Cumulative single-process win on the tissue-dense crop: **848.0 s → 302.7 s (−64.3%)**.
**`workers=1` is the production default; `workers=4` is cleared to ship** — see "Cross-tile
multiprocessing" below for the one VRAM caveat that comes with it.

## Arm model (current, at the real slide's composition)

`wall ≈ max(MAIN, BG) + outside`. At `match24` (55.9% background, `workers=1`): **MAIN 187.4 s, BG
88.0 s, outside 7.5 s → BG/MAIN = 0.470 → MAIN must shed 53.0% before the BG arm (`detect_all_dots`,
PNG encode) becomes critical.** This is *wider* slack than every tissue-dense crop measured before
round 7 implied (as little as 15.9%–28%) — more tissue tiles load MAIN faster than they load BG, the
opposite of what a "mostly background" slide would predict. Full detail:
[`25-gpu-encode-decode-loop-acceleration-implementation.md`](../25-gpu-encode-decode-loop-acceleration-implementation.md) §2.3–§2.4.

## All bottlenecks — status and result

| item | status | measured result | doc |
|---|---|---|---|
| ① GPU forwards — serial pipeline / 3 sequential per-tile forwards (M1 UNet++ + 2× Cellpose) | **DONE (multi-stage)** — still PRIMARY | GPU/CPU overlap −16.6%; Cellpose 4.0.8→4.2.1.1 swap −18.9%; `dot_detect_n_jobs` fix made MAIN itself 43.4% faster; cumulative `workers=1` 848.0→302.7 s (−64.3%). Remaining internal ceiling ~1.118× (kernel-launch-bound Python loops, third-party) | [pipeline-overlap-result.md](./pipeline-overlap-result.md), [23-implementation §4/§7](../23-next-optimization-cycle-implementation.md) |
| ② `detect_all_dots` (M3 dot detection) | **RESOLVED "for free"** — hidden on the slack BG arm | ceiling 1.00×–1.013×; BG arm has 47–53% slack at real composition (was 15.9%–28% on tissue-dense crops) | [detect-all-dots-result.md](./detect-all-dots-result.md) |
| ③ PNG/TIFF per-tile encode+write | **HIDDEN**, same arm as ② | ceiling 1.00×–1.013× | [detect-all-dots-result.md](./detect-all-dots-result.md) |
| ④ Per-tile `gc.collect()` | ✅ **SHIPPED (round 9)** — periodic re-freeze; full-slide confirmation still outstanding | Doc 28 Option H shipped: `_frozen_gc_generation()` now yields a `_PeriodicFreezer` that calls `gc.freeze()` again every 5,000 accumulated cells, so results appended *after* the entry freeze stop being rescanned. **`gc.freeze()` measured to be O(1)** (0.00004–0.0006 ms/call; 27,565 calls = 1.2 ms total), which closed doc 28 §3's one open assumption and made Option I unnecessary. Synthetic harness at full-slide shape: collect **416.7 s → ~2.1 s**. On the match24 crop the per-call trend flips from **climbing (last/first 2.94)** to **flat (0.71)**, correctness and RSS unchanged. **Full-slide `workers=1` re-run (Exp 4) NOT yet run** — the ~1.19x end-to-end figure is a projection, not a measurement. `workers>1` is unaffected (workers accumulate nothing; the parent does not collect per tile) | [28-plan](../28-gc-collect-round2-plan.md) → **[31-implementation](../31-gc-collect-round2-implementation.md)** |
| ⑤a Precut A (tile cutting) | **DONE** — streamed into the analysis loop | 20.3 s → 0.004 s per batch; −3.1%/−3.8% wall at crop scale. See `B2r_tile_read` below for the full-slide read cost | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §4 |
| ⑤b Phase D slide stitch (`_stitch_overlay_slide`) | 🟡 **PLANNED** — GPU-codec route CLOSED, a faster CPU-container route found but **not correctness-cleared** | Cheap `tiffsave` knobs remain dead; `zstd` remains vetoed. **Round 9 (doc 32) killed the nvImageCodec route on output shape, not speed**: its TIFF encoder refuses tiling (`Tiling is not supported with TIFF encoder`) and its ~8 KB strip layout is internal, and independent LZW streams cannot be concatenated into a tile (measured). Separately, **`tifffile` accepts externally pre-compressed tile bytes** (verified lossless) and its container write is ~1% of the work — so doc 29 §1.2's "≈55% irreducible container assembly" is **a pyvips artefact, not a property of the problem**. Spike candidates beat the pyvips baseline; see doc 32 for the shape-matched table. **QuPath check run twice: after the fixes all three render identically, so candidate C's correctness veto is CLEARED** (what remains open for C is sizing, not correctness). That investigation also found a **pre-existing defect in the shipped stitch, now FIXED**: libvips tags Predictor=2 on level 0 only while differencing the reduced levels too, so every `overlay_slide.tiff` written before round 9 decodes its pyramid levels as ~90% black — confirmed in QuPath ("the old picture actually not normal"). `predictor="none"` fixes it for +9.4% file size and no time cost; 2 tests guard it. **Slides written before round 9 should be re-stitched.** Phase 2 not started | [29-plan](../29-phase-d-gpu-port-plan.md) → **[32-implementation](../32-phase-d-gpu-port-implementation.md)** |
| `B2r_tile_read` (GPU-side tile/transform loading) | ✅ **SHIPPED (round 9, `workers=1`)** — prefetched onto a background thread; full-slide magnitude still unconfirmed | Doc 30 Option L shipped: `prefetch_tile_reads` issues each tile's two reads **one tile ahead** on a dedicated `tile-read` thread, so they overlap the previous tile's GPU forward; `_process_precut_tile_gpu` gained an optional `preread`. **Split measured first (doc 30 §4 step 2): tissue tiles own 62.7% of the read cost and each has ~41× its own read time in GPU work to hide behind**; background tiles own 37.3% with only ~29.4 ms of UNet. Ablated over 3 interleaved repeats per arm on match24: **188.23 s → 184.93 s (−1.75%, 1.018×), non-overlapping ranges**, correctness identical, RSS unchanged. `B2r_tile_read` *rises* 5.61 → 7.11 s under the prefetch — the expected signature of work moved onto a parallel arm. **Full-slide 17.2% is unchanged as a measurement**; the projected ~81% hideable / ~1.16x ceiling is extrapolation. Option K (full-slide re-measure) and the multiprocess variant both still outstanding | [30-plan](../30-tile-read-io-plan.md) → **[33-implementation](../33-tile-read-io-implementation.md)** |
| ⑥ Model init (one-time) | Informational, negligible at scale | 0.37% of wall @441 tiles, amortizes further at full-WSI scale | [gil-contention-diag.md](./gil-contention-diag.md) |
| ⑦ API / job layer (Phase E) | Informational, negligible | ~10⁻⁷ of a multi-hour run | — |
| ⑧ CPU prep stranded on MAIN arm (`enlarge_cell_instances` + `build_all_positive_results`) | **DONE** — moved to BG arm | −8.0%/−5.0% wall; also removed GIL contention between the two arms (unmodified buckets got faster too) | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §2–3 |
| ⑨ `detect_all_dots` +22.3% regression (round-3 cause) | 🟢 **OPEN**, cause never isolated — but moot (no wall-clock payoff) | ceiling 1.013×; overtaken by the `dot_detect_n_jobs` fix below | [19-open-backlog.md](../19-open-backlog.md) #3 |
| `cellpose_batch_size` dead config | **DONE** — wired into `Config` | sweep (16/32/64) flat — no benefit at the current 1024px tile size; only becomes live if tile size ≥1536 | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §6.1 |
| `detect_all_dots` joblib fan-out (`n_jobs=-1` → `1`) | **DONE** | standalone: 2.77× faster serial than 20-thread fan-out; end-to-end: **1.60×** at `workers=1` (484.7 → 302.7 s); isolated the +192.9 s GIL-contention anomaly ① had left unproven | [23-next-optimization-cycle-implementation.md](../23-next-optimization-cycle-implementation.md) §4 |
| Cross-tile multiprocessing (`workers>1`) | ✅ **SHIPPED** — production gate satisfied | **2.216x measured** on the real 27,565-tile slide; correctness veto passed (356,255 vs 356,221 rows, −0.01%). **Ship `workers=4`** — but it peaks at 30,439 MB of 32,607 (93.3%, ~2.2 GB headroom): treat 32 GB VRAM as a hard floor, add no GPU library inside the worker pool without re-measuring, keep `workers≥5` off the table until the allocator balloon (next row) is root-caused | [21-...](../21-cross-tile-multiprocessing-implementation.md), [27-...](../27-remaining-work-implementation.md) §6.5–§6.6 |
| `workers≥6` allocator-fragmentation OOM | 🟢 **OPEN** — candidate fix tried, did not clear it | 2 failures in 6 `workers=6` runs, one worker balloons to exactly 24.76 GiB. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **does not reduce peak VRAM** (evidence against the fragmentation hypothesis) and costs +2.0% wall — default stays off. `workers=6` is also **not faster than `workers=4`** on this crop. Next step is root-causing the balloon directly | [19-open-backlog.md](../19-open-backlog.md) #7b, [27-...](../27-remaining-work-implementation.md) §5 |
| Full real-WSI-scale validation | ✅ **DONE — CLOSED.** First complete slide this project has ever run | `workers=1` 3.82 h, `workers=4` 1.73 h, 2.216x speedup; correctness veto passed. Peak RSS **61.13–61.67 GB** — new host requirement. Registration emits a different canvas per modality; `scripts/full_wsi_validate.py --conform` crops to the intersection (99.86% retained) before a run can start | [19-open-backlog.md](../19-open-backlog.md) #7, [27-...](../27-remaining-work-implementation.md) §6 |
| `_stitch_overlay_slide` — `RLIMIT_NOFILE` guard | ✅ **DONE** | Stitch opens all 27,565 overlay tiles as pyvips images at once; `_ensure_nofile_limit()` raises the soft limit itself when the hard limit permits, else fails loudly before opening anything. 7 tests | [27-...](../27-remaining-work-implementation.md) §1 |
| `run_batch` partial-resume / checkpointing | ✅ **DONE, opt-in** | `run_batch(checkpoint=True)` pickles each completed tile's `owned` result list (config-hash-guarded); resumed output byte-identical to a cold run. Fail-fast unchanged. Off by default; on via `--resume` / always in `full_wsi_validate.py`. 12 tests | [21-...](../21-cross-tile-multiprocessing-implementation.md) §10 follow-up #7, [27-...](../27-remaining-work-implementation.md) §2 |
| Per-bucket timing inside multiprocess workers | ✅ **DONE** | `perf_measure.py --worker-timings` reports 26 worker-side buckets (was parent-process-only, 4 buckets). Sums are aggregate CPU time across workers, not wall — stage-breakdown only, not for wall-clock comparison | [21-...](../21-cross-tile-multiprocessing-implementation.md) §10 follow-up #1, [27-...](../27-remaining-work-implementation.md) §4 |
| CUDA allocator config (`PYTORCH_CUDA_ALLOC_CONF`) knob | 🟡 **Built, swept, default off** | See "`workers≥6` allocator-fragmentation OOM" row above | [27-...](../27-remaining-work-implementation.md) §5 |
| Slide tissue/background composition premise | **DONE** — corrected via direct measurement | slide is **55.8% background**, not the previously assumed 39%; re-based every full-WSI projection and the BG-arm slack figure | [25-gpu-encode-decode-loop-acceleration-implementation.md](../25-gpu-encode-decode-loop-acceleration-implementation.md) §1 |
| Background-tile placeholder writes (Candidate F) | 🔴 **Measured, not built** — zero wall-clock payoff | 24 ms/tile, 7.5% of wall, entirely on the slack BG arm; `os.link` alternative is 272×–407× cheaper — a storage argument (~157 GB/slide of identical bytes), not a speed one | [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §4 |
| Redundant per-call `mkdir()` (Candidate G) | 🔴 **Built, measured, reverted** | 0.056% of wall; end-to-end ablation reads slightly negative. Patch preserved for one-edit revival if the storage backend ever moves to network filesystem | [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §3, §10 |
| Cross-tile Cellpose / UNet++ batching | 🔴 **Stop-lossed** | flat-to-worse at every group size, including G=16 (+5.9–6.6%, 15.8 GB VRAM — 48.6% of the card for one process) | [23-...](../23-next-optimization-cycle-implementation.md) §2–3, [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §7 |
| CUDA MPS (multi-context GPU sharing) | 🔴 **Stop-lossed** | +44% on a synthetic launch-bound microbenchmark, **0% end-to-end** — the real pipeline is no longer serialization-limited at its knee | [21-...](../21-cross-tile-multiprocessing-implementation.md) §5 |
| CPU-back-end-only process pool / deeper BG pipelining | 🔴 **Stop-lossed** | +2.8% slower single-process, flat under multiprocessing | [21-...](../21-cross-tile-multiprocessing-implementation.md) §6 |
| Fork-based model reuse across workers | 🔴 **Not built** — architecturally unsafe (CUDA contexts aren't fork-safe) | n/a | [20-cross-tile-multiprocessing-plan.md](../20-cross-tile-multiprocessing-plan.md) Candidate E |
| CUDA-stream / pipeline-depth-2 bubble redesign | 🔴 **Stop-lossed** | ceiling ≤1.065× after ⑧ landed | [18-...](../18-gpu-starvation-prerequisites-implementation.md) §3 |
| CUDA graph capture / vectorize Cellpose's internal kernel-launch loops | 🔴 **Stop-lossed** (re-confirmed round 6) | ceiling ~1.118×; requires patching pinned third-party `cellpose`/`segment_anything` internals | [23-...](../23-next-optimization-cycle-implementation.md) §7 |
| Multi-request / concurrent-job behavior (Phase E) | 🟢 **OPEN** — never measured | n/a | [19-open-backlog.md](../19-open-backlog.md) #8 |

## Memory (bounded — claim holds; host requirements below are the real full-slide numbers)

Crop-scale VRAM peak (single-process): **2787 MB**, flat regardless of tile count. Under
multiprocessing, VRAM per process grows **superlinearly** with worker count (2787 → 3117 → 4118 →
5167 MB at N=1–4 on crops) — this, not model-weight size, is what caps the safe worker count. Full
detail: [21-cross-tile-multiprocessing-implementation.md](../21-cross-tile-multiprocessing-implementation.md)
§4.4, [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §8.3.

**Real full-slide numbers** (driven by the stitch holding 27,565 lazy pyvips images, 12,027 open
fds mid-stitch): peak RSS **61.13 GB (`workers=1`) / 61.67 GB (`workers=4`)**; peak GPU **2,739 MB
(`workers=1`) / 30,439 MB (`workers=4`, 93.3% of the 32,607 MB card, ~2.2 GB headroom)**. Host
requirements this implies: **~64 GB RAM, ~32 GB VRAM for `workers=4`, ~350 GB disk per slide,
`RLIMIT_NOFILE` ≥ ~28,000**. Full detail: [27-...](../27-remaining-work-implementation.md) §6.4, §6.6.

## Classification summary

| class | bottlenecks |
|---|---|
| 1 algorithm/model complexity | ② `detect_all_dots`; Cellpose's kernel-launch-bound internals inside ① |
| 3 parallel/concurrency | ① serial→overlapped pipeline; ⑧ CPU-prep placement; cross-tile multiprocessing |
| 4 memory lifecycle | ④ per-tile gc; RSS cell-result accumulation |
| 5 I/O & storage | ③ PNG encode; ⑤a/⑤b precut & stitch; background-tile placeholder writes |
| 6 architecture/framework | ① no cross-tile GPU batching; ⑥ init; ⑦ API layer |
| 7 config/dead-code | `cellpose_batch_size` (fixed); `detect_all_dots` joblib fan-out (fixed); ⑨ regression cause (unresolved) |

## What's still open

See [`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md) for the ranked, complete
list of every candidate this project has found but not shipped, with disposition and source doc per
item.
