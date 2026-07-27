# Bottleneck list — hybrid pipeline (current state)

> **Compact, current-state-only.** This file lists every bottleneck this project has found —
> shipped, hidden, stop-lossed, or still open — with its latest measured result and a link to the
> document that has the full evidence. It does **not** narrate how each round got there; for the
> round-by-round history (control → overlap → Cellpose swap → ... → round 8), see
> [`bottleneck-list-history.md`](./bottleneck-list-history.md). For the exhaustive ledger of every
> candidate discovered but never shipped, see
> [`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md).
>
> Current HEAD: round 8, config hash `3d1087f2` (unchanged since round 7). Machine: RTX 5090 /
> CUDA 13.0 / torch 2.11.0+cu130.
>
> **Round 8 (2026-07-27) ran the first complete real WSI end-to-end** — see
> [`../27-remaining-work-implementation.md`](../27-remaining-work-implementation.md). It closed the
> `workers>1` production gate but overturned three numbers every prior round's crop-based
> extrapolation relied on: `gc.collect` (16.1% of wall, not ~0 — the `gc.freeze()` win from round 4
> does not survive to full-slide scale), tile read (17.2%, not the 1.22% doc 18 §6.3 stopped it out
> on), and Phase D stitch (8.6%/19.3% of wall at `workers=1`/`4`, not 3.5%/7.3%). See the "All
> bottlenecks" table below for each item's revised status.

## Current anchors

| anchor | tiles | background share | `workers=1` | `workers=4` |
|---|--:|--:|--:|--:|
| baseline (control, fully serial, no optimizations) | 441 (large/21×21) | 14.1% | 848.0 s | — |
| large/441 crop, current code | 441 | 14.1% (tissue-dense, **not** representative of a real slide) | 302.7 s | 128.8 s |
| match24 — composition-matched crop | 576 | 55.9% (real slide measured: 55.8%) | **188.8 s** | **88.3 s** |
| full-WSI projection (round 7, crop-rate extrapolation) | 27,565 | 55.8% (measured, not sampled) | ~2.6 h | ~1.25 h |
| **full-WSI, real run (round 8, measured, not projected)** | 27,565 | 55.8% (measured) | **13,762 s = 3.82 h (+47% vs projection)** | **6,211 s = 1.73 h (+38% vs projection)** |

Cumulative single-process win on the tissue-dense crop: **848.0 s → 302.7 s (−64.3%)**. The real
full-slide run overran its crop-based projection by 38–47% (§ below) — the composition prediction
was right to within one tile, so the miss is entirely in per-tile rates, not tissue/background mix.
**`workers=1` remains the production default; `workers=4` is now cleared to ship** (round 8 closed
the item-7 gate — see "Cross-tile multiprocessing" below — with one VRAM caveat, not a speed one).

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
| ④ Per-tile `gc.collect()` | 🟢 **REOPENED (round 8)** — `gc.freeze()` benefit does not survive to full-slide scale | Crop scale: 36.7 s → 0.52 s per batch (1.069×–1.077×). **Full slide: back to 80.5 ms/call, 2,218.4 s = 16.1% of wall** — `run_batch` accumulates 356,255 `CellAnalysisResult` dataclasses (created *after* `gc.freeze()`, fully tracked, rescanned every collection); invisible on a 441-tile crop (~6,000 objects), the 3rd-largest cost at full scale. Now the most attractive open target in the pipeline (cheap plausible fix: freeze again periodically, or keep accumulating results out of GC's reach) | [15-...](../15-gc-collect-frequency-implementation.md), [16-...](../16-gc-collect-frequency-result.md), [27-...](../27-remaining-work-implementation.md) §6.4 |
| ⑤a Precut A (tile cutting) | **DONE** — streamed into the analysis loop | 20.3 s → 0.004 s per batch; −3.1%/−3.8% wall. **But `B2r_tile_read` reopened at full scale** — see the tile-read row below | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §4 |
| ⑤b Phase D slide stitch (`_stitch_overlay_slide`) | 🔴 **Cheap knobs CLOSED, negative (round 8)** — GPU port reopened, ceiling revised up ~3x | Crop screening (4.055 GP, 13 configs): tile-size/pyramid-depth/predictor all dead; `zstd` won 1.2331x/13.8% smaller but **QuPath/BioFormats cannot open it — vetoed on correctness**. Real full-slide stitch: **1,185.4 s = 8.6% of wall at `workers=1`, 19.3% at `workers=4`** (3.7× the 322.7 s synthetic `stitch_probe.py` figure) → ceiling **1.094x/1.239x**, not 1.036x/1.078x. Now the largest single remaining lever in the pipeline | [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §5, [27-...](../27-remaining-work-implementation.md) §3, §6.4, §6.6 |
| `B2r_tile_read` (GPU-side tile/transform loading) | 🟢 **REOPENED (round 8)** — stop-loss was measured on a crop that doesn't hold at scale | Crop scale: 1.22% of wall, ceiling 1.012× (doc 18 §6.3, stopped out). **Full slide: 2,368.5 s = 17.2% of wall** — the ~49 GB precut scratch no longer fits page cache at full-slide scale, so reads that were free on a crop become real disk I/O. Sits on the arm with slack; ceiling needs re-deriving, not re-assuming | [18-...](../18-gpu-starvation-prerequisites-implementation.md) §6.3, [27-...](../27-remaining-work-implementation.md) §6.4 |
| ⑥ Model init (one-time) | Informational, negligible at scale | 0.37% of wall @441 tiles, amortizes further at full-WSI scale | [gil-contention-diag.md](./gil-contention-diag.md) |
| ⑦ API / job layer (Phase E) | Informational, negligible | ~10⁻⁷ of a multi-hour run | — |
| ⑧ CPU prep stranded on MAIN arm (`enlarge_cell_instances` + `build_all_positive_results`) | **DONE** — moved to BG arm | −8.0%/−5.0% wall; also removed GIL contention between the two arms (unmodified buckets got faster too) | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §2–3 |
| ⑨ `detect_all_dots` +22.3% regression (round-3 cause) | 🟢 **OPEN**, cause never isolated — but moot (no wall-clock payoff) | ceiling 1.013×; overtaken by the `dot_detect_n_jobs` fix below | [19-open-backlog.md](../19-open-backlog.md) #3 |
| `cellpose_batch_size` dead config | **DONE** — wired into `Config` | sweep (16/32/64) flat — no benefit at the current 1024px tile size; only becomes live if tile size ≥1536 | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §6.1 |
| `detect_all_dots` joblib fan-out (`n_jobs=-1` → `1`) | **DONE** | standalone: 2.77× faster serial than 20-thread fan-out; end-to-end: **1.60×** at `workers=1` (484.7 → 302.7 s); isolated the +192.9 s GIL-contention anomaly ① had left unproven | [23-next-optimization-cycle-implementation.md](../23-next-optimization-cycle-implementation.md) §4 |
| Cross-tile multiprocessing (`workers>1`) | ✅ **SHIPPED (round 8)** — item-7 gate satisfied on the real full slide | **2.216x measured** on the real 27,565-tile slide (in the 2.06x–2.17x predicted band); correctness veto passed (356,255 vs 356,221 rows, −0.01%). Recommendation: **ship `workers=4`**, but `workers=4` peaked at 30,439 MB of 32,607 (93.3%, ~2.2 GB headroom) — treat 32 GB VRAM as a hard floor, add no GPU library inside the worker pool without re-measuring, keep `workers≥5` off the table until the allocator balloon (next row) is root-caused | [21-...](../21-cross-tile-multiprocessing-implementation.md), [27-...](../27-remaining-work-implementation.md) §6.5–§6.6 |
| `workers≥6` allocator-fragmentation OOM | 🟢 **OPEN** — candidate fix tried (round 8), did not clear it | 2 failures in 6 `workers=6` runs, one worker balloons to exactly 24.76 GiB, reproduced again this round on a verified-clean GPU. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` swept (12 runs, interleaved): **does not reduce peak VRAM** (median 24,040 vs 22,968 MB control — evidence *against* the fragmentation hypothesis), 0-in-6 vs 1-in-6 OOM is not a statistically distinguishable result (p≈1.0), and costs +2.0% wall. Default stays off; next step is root-causing the 24.76 GiB balloon directly, not sweeping more allocator flags. Also found: `workers=6` is **not faster than `workers=4`** on this crop (65.46 s vs 65.55 s) | [19-open-backlog.md](../19-open-backlog.md) #7b, [27-...](../27-remaining-work-implementation.md) §5 |
| Full real-WSI-scale validation | ✅ **DONE (round 8) — CLOSED.** First complete slide this project has ever run | `workers=1` 13,762 s / 3.82 h (**+47%** vs 2.6 h projected), `workers=4` 6,211 s / 1.73 h (**+38%** vs 1.25 h projected); composition prediction right to within one tile (15,385 vs predicted 15,386 background tiles), so the overrun is entirely per-tile-rate, not composition. Peak RSS **61.13–61.67 GB** (no prior round exceeded ~4 GB) — new host requirement. Preflight surfaced a blocker no crop-based round could hit: per-modality canvas sizes differ (HER2 141818×114366 vs DISH 141658×114415), `PrecutStream` fail-fasts on it; `scripts/full_wsi_validate.py --conform` crops to the intersection (99.86% retained) | [19-open-backlog.md](../19-open-backlog.md) #7, [27-...](../27-remaining-work-implementation.md) §6 |
| `_stitch_overlay_slide` — no `RLIMIT_NOFILE` guard | ✅ **DONE (round 8)** | Stitch opens all 27,565 overlay tiles as pyvips images at once (12,027 open fds observed mid-stitch on the real run); on a host with the common 1,024 soft-limit default this would fail *after* the whole multi-hour analysis completed. `_ensure_nofile_limit()` raises the soft limit itself when the hard limit permits, else fails loudly before opening anything; exercised for real on the full-slide run, passed silently. 7 tests | [27-...](../27-remaining-work-implementation.md) §1 |
| `run_batch` partial-resume / checkpointing | ✅ **DONE (round 8), opt-in** | `run_batch(checkpoint=True)` pickles each completed tile's `owned` result list (tmp+rename, config-hash-guarded); resumed output byte-identical to a cold run (`report.csv`/`summary.txt`). Fail-fast is unchanged — resume makes a retry cheap, not a failure survivable mid-run. Off by default (API path would only pay I/O); on via `--resume` / always in `full_wsi_validate.py`. 12 tests | [21-...](../21-cross-tile-multiprocessing-implementation.md) §10 follow-up #7, [27-...](../27-remaining-work-implementation.md) §2 |
| Per-bucket timing inside multiprocess workers | ✅ **DONE (round 8)** | `perf_measure.py` previously instrumented the **parent** process only, so `workers>1` runs had 4 parent-only buckets and zero visibility into which worker-side stage moved. Env-gated probe hook (`HYBRID_MP_WORKER_PROBE`) now reports 26 worker-side buckets via `perf_measure.py --worker-timings`, with no pipeline dependency on the measurement harness. Sums are aggregate CPU time across workers, not wall — stage-breakdown only, not for wall-clock comparison | [21-...](../21-cross-tile-multiprocessing-implementation.md) §10 follow-up #1, [27-...](../27-remaining-work-implementation.md) §4 |
| CUDA allocator config (`PYTORCH_CUDA_ALLOC_CONF`) knob | 🟡 **Built, swept, default stays off** | See "`workers≥6` allocator-fragmentation OOM" row above | [27-...](../27-remaining-work-implementation.md) §5 |
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

## Memory (bounded — claim holds, but crop-scale numbers understated the real host requirements)

VRAM peak (single-process, crop scale): **2787 MB**, flat regardless of tile count (was 5159 MB
before the Cellpose bfloat16 swap). RSS tracks accumulated cell count, not tile count, and stays
sub-linear at every crop scale tested. Under multiprocessing, VRAM per process grows
**superlinearly** with worker count (2787 → 3117 → 4118 → 5167 MB at N=1–4 on crops) — this, not
model-weight size, is what caps the safe worker count. Full detail:
[21-cross-tile-multiprocessing-implementation.md](../21-cross-tile-multiprocessing-implementation.md)
§4.4, [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §8.3.

**Round 8 — real full-slide numbers, first time measured (not extrapolated).** Peak RSS
**61.13 GB (`workers=1`) / 61.67 GB (`workers=4`)** — driven by the stitch holding 27,565 lazy
pyvips images (12,027 open fds observed mid-stitch); no prior round exceeded ~4 GB. Peak GPU
**2,739 MB (`workers=1`) / 30,439 MB (`workers=4`, 93.3% of the 32,607 MB card, ~2.2 GB
headroom)** — worse than the ~6 GB transient headroom doc 25 §8.3 inferred from crops. New,
previously unstated host requirements: **~64 GB RAM, ~32 GB VRAM for `workers=4`, ~350 GB disk per
slide, `RLIMIT_NOFILE` ≥ ~28,000**. Full detail: [27-...](../27-remaining-work-implementation.md) §6.4, §6.6.

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
