# Current status vs. baseline — hybrid pipeline

> **Compact, baseline-vs-current-only.** This file compares the original serial "dumb-version"
> control against the current HEAD. It does **not** narrate the eight rounds in between; for that
> chain, see [`current-status-comparison-history.md`](./current-status-comparison-history.md). For
> the current per-bottleneck ledger, see [`bottleneck-list.md`](./bottleneck-list.md); for
> discovered-but-unshipped candidates, see
> [`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md).
>
> **Baseline:** git `96a28ba`, fully serial `run_batch` (one tile at a time, no optimizations),
> `_metrics/`. **Current:** round 8, config hash `3d1087f2` (unchanged since round 7). Same machine
> (RTX 5090 / CUDA 13.0, torch 2.11.0+cu130), same `scripts/perf_measure.py` harness. Full round-8
> record: [`../27-remaining-work-implementation.md`](../27-remaining-work-implementation.md);
> round-by-round narrative lives in
> [`current-status-comparison-history.md`](./current-status-comparison-history.md), not here.

## 1. Headline — end-to-end anchors

| scale | tiles | **baseline wall** | **current wall** (`workers=1`) | **current wall** (`workers=4`) | Δ (`workers=1`) |
|---|--:|--:|--:|--:|--:|
| small | 25 | 54.3 s | — | — | — |
| medium | 121 | 243.3 s | — | — | — |
| large | 441 | 848.0 s | **302.7 s** | **128.8 s** | **−64.3%** |
| **match24 (55.9% background — matches the real slide's measured 55.8%)** | 576 | — | **188.8 s** | **88.3 s** | — |

`workers=1` remains the production default; **`workers=4` is cleared for production** (see
[`bottleneck-list.md`](./bottleneck-list.md) "Cross-tile multiprocessing" row for the one VRAM
caveat that comes with it). No negative optimization at any scale measured.

### GPU utilization — baseline vs current

| metric | baseline | current |
|---|--:|--:|
| GPU idle_frac (large/441, sm==0) | 0.459 | 0.06–0.19 depending on `workers` (multiprocessing fills most of the remaining idle) |
| mean SM % | 28.3 | 16.6–78.1 depending on `workers` (lower single-process SM% is a *side effect* of Cellpose's own speedup, not a regression — see [`bottleneck-list.md`](./bottleneck-list.md) ①) |

## 2. Per-bottleneck status

See [`bottleneck-list.md`](./bottleneck-list.md)'s "All bottlenecks — status and result" table for
the current disposition and measured result of every item (①–⑨ plus every later candidate). Not
duplicated here to avoid the two files drifting apart.

## 3. Memory (bounded — claim holds; real full-slide numbers are much larger than crop scale)

| | baseline | current (crop scale) | current (real full slide) |
|---|--:|--:|--:|
| VRAM peak | 5159 MB (large/441) | **2787 MB** (single-process; grows superlinearly with `workers`, see `bottleneck-list.md` "Memory") | **2,739 MB (`workers=1`) / 30,439 MB (`workers=4`, 93.3% of the 32,607 MB card)** |
| peak RSS | 4.04 GB (large/441) | ~3.9 GB (tracks accumulated cell count, not tile count) | **61.13 GB (`workers=1`) / 61.67 GB (`workers=4`)** |

Host requirements implied by the real full-slide numbers: ~64 GB RAM, ~32 GB VRAM for `workers=4`,
~350 GB disk/slide, `RLIMIT_NOFILE` ≥ ~28,000. Full detail:
[`27-...`](../27-remaining-work-implementation.md) §6.4/§6.6.

## 4. Full-WSI — real measured run (27,565 tiles, 16.2 gigapixels)

| | baseline (extrapolated) | current (`workers=1`) | current (`workers=4`) |
|---|--:|--:|--:|
| end-to-end wall | ~18.9 h | **13,762 s = 3.82 h** | **6,211 s = 1.73 h** (2.216x speedup) |
| success / skipped tiles | — | 10,801 / 16,764 | 10,800 / 16,765 |
| `report.csv` rows | — | 356,255 | 356,221 (**−0.01%**, within the correctness-veto noise floor) |
| Phase D stitch | — | 1,185.4 s (8.6% of wall) | 1,200.8 s (19.3% of wall) |

Baseline is a 3-tile crop extrapolation, upper bound, never run at full scale; current is the real
measured run — see [`19-open-backlog.md`](../19-open-backlog.md) item 7 (closed) and
[`27-...`](../27-remaining-work-implementation.md) §6 for the full record. Registration emits a
different canvas per modality; `scripts/full_wsi_validate.py --conform` crops the pair to their
intersection (99.86% retained) before a run can start.

## 5. What is still worth optimizing

See [`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md) for the ranked, complete
list. Top items, current state:

1. **`gc.collect` at full-slide scale — 16.1% of wall.** `run_batch`'s accumulating
   `per_tile_owned` results are created after `gc.freeze()` and fully tracked, so every one of
   27,565 collections rescans 356,255 objects by the end. Most attractive open target in the
   pipeline. [`27-...`](../27-remaining-work-implementation.md) §6.4.
2. **Phase D GPU port — ceiling ~1.24x.** Cheap `tiffsave` knobs are closed negative (the one
   winner, `zstd`, is unreadable by QuPath/BioFormats and vetoed); the real stitch is 19.3% of wall
   at `workers=4`. Largest single remaining lever. [`27-...`](../27-remaining-work-implementation.md) §3, §6.6.
3. **Tile read (`B2r_tile_read`) — 17.2% of wall at full scale.** The ~49 GB precut scratch no
   longer fits page cache at full-slide scale. [`27-...`](../27-remaining-work-implementation.md) §6.4.
4. **`workers≥6` allocator-fragmentation OOM** — candidate fix (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
   does not reduce peak VRAM; root-causing the 24.76 GiB balloon directly is the next step, not
   more allocator flags. [`19-open-backlog.md`](../19-open-backlog.md) #7b.

## 6. Reproduce

```bash
cd /data/taro_Projects/tsgh
ROI="$PWD/backend/algorithms/hybrid/test_picture/_roi_crops"
M=docs/hybrid-pipeline/measurement/_metrics_r7
SLIDE=/data/nvmessd/storge_tsgh/<case>/output

# composition-matched crop (match24), against the pipeline's own core-mask map
.venv/bin/python scripts/core_mask_map.py --ihc $SLIDE/HER2_processed.tiff --out $M/core_mask_map.npz
.venv/bin/python scripts/composition_crop.py --ihc $SLIDE/HER2_processed.tiff \
    --dish $SLIDE/DISH_processed.tiff --grid 24 --map $M/core_mask_map.npz \
    --out-ihc "$ROI/match24_ihc.tiff" --out-dish "$ROI/match24_dish.tiff" --report $M/match24_crop.json

# baseline vs current, workers=1 and workers=4
.venv/bin/python scripts/perf_measure.py --ihc "$ROI/match24_ihc.tiff" --dish "$ROI/match24_dish.tiff" \
    --output docs/hybrid-pipeline/measurement/runs_r7/match24_w1 --label match_w1 \
    --workers 8 --gpu-dmon --stream-precut --mp-workers 1 --metrics-dir "$M"
.venv/bin/python scripts/perf_measure.py --ihc "$ROI/match24_ihc.tiff" --dish "$ROI/match24_dish.tiff" \
    --output docs/hybrid-pipeline/measurement/runs_r7/match24_w4 --label match_w4 \
    --workers 8 --gpu-dmon --stream-precut --mp-workers 4 --metrics-dir "$M"

# Phase D at real scale + full-WSI projection
.venv/bin/python scripts/stitch_probe.py --overlay-src <run>/overlay_annotated \
    --slide-w 141818 --slide-h 114366 --out $M/stitch_probe_full.json
.venv/bin/python scripts/wsi_projection.py --timings $M/match_w1_timings.json \
    --background-share 0.5582 --stitch-s 322.7 --out $M/wsi_projection.json
```

Raw artifacts (preserved): baseline in `_metrics/`, current in `_metrics_r7/` (incl.
`env_stamp_r7.txt`, `pip_freeze_r7.txt`, `core_mask_map.npz`). Full command set:
[`25-gpu-encode-decode-loop-acceleration-implementation.md`](../25-gpu-encode-decode-loop-acceleration-implementation.md) §12.
