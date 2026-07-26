# Bottleneck list — hybrid pipeline (current state)

> **Compact, current-state-only.** This file lists every bottleneck this project has found —
> shipped, hidden, stop-lossed, or still open — with its latest measured result and a link to the
> document that has the full evidence. It does **not** narrate how each round got there; for the
> round-by-round history (control → overlap → Cellpose swap → ... → round 7), see
> [`bottleneck-list-history.md`](./bottleneck-list-history.md). For the exhaustive ledger of every
> candidate discovered but never shipped, see
> [`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md).
>
> Current HEAD: git `025f9a5`, config hash `3d1087f2`. Machine: RTX 5090 / CUDA 13.0 /
> torch 2.11.0+cu130.

## Current anchors

| anchor | tiles | background share | `workers=1` | `workers=4` |
|---|--:|--:|--:|--:|
| baseline (control, fully serial, no optimizations) | 441 (large/21×21) | 14.1% | 848.0 s | — |
| large/441 crop, current code | 441 | 14.1% (tissue-dense, **not** representative of a real slide) | 302.7 s | 128.8 s |
| **match24 — composition-matched to the real slide** | 576 | 55.9% (real slide measured: 55.8%) | **188.8 s** | **88.3 s** |
| full-WSI projection (27,565 tiles, measured composition + measured Phase D) | 27,565 | 55.8% (measured, not sampled) | **~2.6 h** | **~1.25 h** |

Cumulative single-process win: **848.0 s → 302.7 s (−64.3%)** on the same tissue-dense crop every
round has used; **`workers=1` remains the production default** — `workers>1` is built and measured
but not yet shipped (see "Cross-tile multiprocessing" below).

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
| ④ Per-tile `gc.collect()` | **DONE** — `gc.freeze()` adopted | 36.7 s → 0.52 s per batch; 1.069×–1.077× measured, matched its 1.083× predicted ceiling | [15-gc-collect-frequency-implementation.md](../15-gc-collect-frequency-implementation.md), [16-gc-collect-frequency-result.md](../16-gc-collect-frequency-result.md) |
| ⑤a Precut A (tile cutting) | **DONE** — streamed into the analysis loop | 20.3 s → 0.004 s per batch; −3.1%/−3.8% wall | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §4 |
| ⑤b Phase D slide stitch (`_stitch_overlay_slide`) | 🟢 **OPEN** — measured at real scale, not built | 322.7 s at 16.2 gigapixels (1.8× the crop-based extrapolation, superlinear); ceiling 1.036× (`workers=1`) – 1.078× (`workers=4`, share *grows* as everything else shrinks). GPU path exists (nvImageCodec, 19.2× faster encode) but needs pyramid/container engineering; cheaper `tiffsave` knobs untried | [25-gpu-encode-decode-loop-acceleration-implementation.md](../25-gpu-encode-decode-loop-acceleration-implementation.md) §5 |
| ⑥ Model init (one-time) | Informational, negligible at scale | 0.37% of wall @441 tiles, amortizes further at full-WSI scale | [gil-contention-diag.md](./gil-contention-diag.md) |
| ⑦ API / job layer (Phase E) | Informational, negligible | ~10⁻⁷ of a multi-hour run | — |
| ⑧ CPU prep stranded on MAIN arm (`enlarge_cell_instances` + `build_all_positive_results`) | **DONE** — moved to BG arm | −8.0%/−5.0% wall; also removed GIL contention between the two arms (unmodified buckets got faster too) | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §2–3 |
| ⑨ `detect_all_dots` +22.3% regression (round-3 cause) | 🟢 **OPEN**, cause never isolated — but moot (no wall-clock payoff) | ceiling 1.013×; overtaken by the `dot_detect_n_jobs` fix below | [19-open-backlog.md](../19-open-backlog.md) #3 |
| `cellpose_batch_size` dead config | **DONE** — wired into `Config` | sweep (16/32/64) flat — no benefit at the current 1024px tile size; only becomes live if tile size ≥1536 | [18-gpu-starvation-prerequisites-implementation.md](../18-gpu-starvation-prerequisites-implementation.md) §6.1 |
| `detect_all_dots` joblib fan-out (`n_jobs=-1` → `1`) | **DONE** | standalone: 2.77× faster serial than 20-thread fan-out; end-to-end: **1.60×** at `workers=1` (484.7 → 302.7 s); isolated the +192.9 s GIL-contention anomaly ① had left unproven | [23-next-optimization-cycle-implementation.md](../23-next-optimization-cycle-implementation.md) §4 |
| Cross-tile multiprocessing (`workers>1`) | 🟡 **BUILT & MEASURED, not shipped** — gated on full-WSI validation | 2.06×–3.51× depending on composition/worker count; `workers=4` recommended for unattended jobs, `workers=5` when a restart is cheap | [21-cross-tile-multiprocessing-implementation.md](../21-cross-tile-multiprocessing-implementation.md) |
| `workers≥6` allocator-fragmentation OOM | 🟢 **OPEN** — reliability defect, root cause not isolated | 2 failures in 6 `workers=6` runs, one worker balloons to exactly 24.76 GiB; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` untried | [19-open-backlog.md](../19-open-backlog.md) #7b |
| Full real-WSI-scale validation | 🟢 **OPEN** — never done, any round | binding gate on shipping `workers>1` to production | [19-open-backlog.md](../19-open-backlog.md) #7 |
| Slide tissue/background composition premise | **DONE** — corrected via direct measurement | slide is **55.8% background**, not the previously assumed 39%; re-based every full-WSI projection and the BG-arm slack figure | [25-gpu-encode-decode-loop-acceleration-implementation.md](../25-gpu-encode-decode-loop-acceleration-implementation.md) §1 |
| Background-tile placeholder writes (Candidate F) | 🔴 **Measured, not built** — zero wall-clock payoff | 24 ms/tile, 7.5% of wall, entirely on the slack BG arm; `os.link` alternative is 272×–407× cheaper — a storage argument (~157 GB/slide of identical bytes), not a speed one | [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §4 |
| Redundant per-call `mkdir()` (Candidate G) | 🔴 **Built, measured, reverted** | 0.056% of wall; end-to-end ablation reads slightly negative. Patch preserved for one-edit revival if the storage backend ever moves to network filesystem | [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §3, §10 |
| Cross-tile Cellpose / UNet++ batching | 🔴 **Stop-lossed** | flat-to-worse at every group size, including G=16 (+5.9–6.6%, 15.8 GB VRAM — 48.6% of the card for one process) | [23-...](../23-next-optimization-cycle-implementation.md) §2–3, [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §7 |
| CUDA MPS (multi-context GPU sharing) | 🔴 **Stop-lossed** | +44% on a synthetic launch-bound microbenchmark, **0% end-to-end** — the real pipeline is no longer serialization-limited at its knee | [21-...](../21-cross-tile-multiprocessing-implementation.md) §5 |
| CPU-back-end-only process pool / deeper BG pipelining | 🔴 **Stop-lossed** | +2.8% slower single-process, flat under multiprocessing | [21-...](../21-cross-tile-multiprocessing-implementation.md) §6 |
| Fork-based model reuse across workers | 🔴 **Not built** — architecturally unsafe (CUDA contexts aren't fork-safe) | n/a | [20-cross-tile-multiprocessing-plan.md](../20-cross-tile-multiprocessing-plan.md) Candidate E |
| CUDA-stream / pipeline-depth-2 bubble redesign | 🔴 **Stop-lossed** | ceiling ≤1.065× after ⑧ landed | [18-...](../18-gpu-starvation-prerequisites-implementation.md) §3 |
| CUDA graph capture / vectorize Cellpose's internal kernel-launch loops | 🔴 **Stop-lossed** (re-confirmed round 6) | ceiling ~1.118×; requires patching pinned third-party `cellpose`/`segment_anything` internals | [23-...](../23-next-optimization-cycle-implementation.md) §7 |
| GPU-side tile/transform loading | 🔴 **Stopped out** | ceiling 1.012×; no existing CPU→GPU transform pipeline to move | [18-...](../18-gpu-starvation-prerequisites-implementation.md) §6.3 |
| Multi-request / concurrent-job behavior (Phase E) | 🟢 **OPEN** — never measured | n/a | [19-open-backlog.md](../19-open-backlog.md) #8 |

## Memory (bounded — claim holds)

VRAM peak (single-process): **2787 MB**, flat regardless of tile count (was 5159 MB before the
Cellpose bfloat16 swap). RSS tracks accumulated cell count, not tile count, and stays sub-linear at
every scale tested. Under multiprocessing, VRAM per process grows **superlinearly** with worker
count (2787 → 3117 → 4118 → 5167 MB at N=1–4) — this, not model-weight size, is what caps the safe
worker count. Full detail: [21-cross-tile-multiprocessing-implementation.md](../21-cross-tile-multiprocessing-implementation.md)
§4.4, [25-...](../25-gpu-encode-decode-loop-acceleration-implementation.md) §8.3.

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
