# Current status vs. baseline — hybrid pipeline

> **Compact, baseline-vs-current-only.** This file compares the original serial "dumb-version"
> control against the current HEAD. It does **not** narrate the seven rounds in between; for that
> chain, see [`current-status-comparison-history.md`](./current-status-comparison-history.md). For
> the current per-bottleneck ledger, see [`bottleneck-list.md`](./bottleneck-list.md); for
> discovered-but-unshipped candidates, see
> [`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md).
>
> **Baseline:** git `96a28ba`, fully serial `run_batch` (one tile at a time, no optimizations),
> `_metrics/`. **Current:** git `025f9a5`, config hash `3d1087f2`, `_metrics_r7/`. Same machine
> (RTX 5090 / CUDA 13.0, torch 2.11.0+cu130), same `scripts/perf_measure.py` harness.

## 1. Headline — end-to-end anchors

| scale | tiles | **baseline wall** | **current wall** (`workers=1`) | **current wall** (`workers=4`) | Δ (`workers=1`) |
|---|--:|--:|--:|--:|--:|
| small | 25 | 54.3 s | — | — | — |
| medium | 121 | 243.3 s | — | — | — |
| large | 441 | 848.0 s | **302.7 s** | **128.8 s** | **−64.3%** |
| **match24 (55.9% background — matches the real slide's measured 55.8%)** | 576 | — | **188.8 s** | **88.3 s** | — |

`workers=1` is the production default; `workers>1` is built and measured but **not yet shipped**
(gated on full real-WSI validation — see [`bottleneck-list.md`](./bottleneck-list.md) "Cross-tile
multiprocessing" row). No negative optimization at any scale measured.

### GPU utilization — baseline vs current

| metric | baseline | current |
|---|--:|--:|
| GPU idle_frac (large/441, sm==0) | 0.459 | 0.06–0.19 depending on `workers` (multiprocessing fills most of the remaining idle) |
| mean SM % | 28.3 | 16.6–78.1 depending on `workers` (lower single-process SM% is a *side effect* of Cellpose's own speedup, not a regression — see [`bottleneck-list.md`](./bottleneck-list.md) ①) |

## 2. Per-bottleneck status

See [`bottleneck-list.md`](./bottleneck-list.md)'s "All bottlenecks — status and result" table for
the current disposition and measured result of every item (①–⑨ plus every later candidate). Not
duplicated here to avoid the two files drifting apart.

## 3. Memory (bounded — claim holds)

| | baseline | current |
|---|--:|--:|
| VRAM peak (large/441, `dmon fb`) | 5159 MB | **2787 MB** (single-process; grows superlinearly with `workers`, see `bottleneck-list.md` "Memory") |
| peak RSS (large/441) | 4.04 GB | ~3.9 GB (tracks accumulated cell count, not tile count, at every scale and every `workers` value tested) |

## 4. Full-WSI reprojection

| | figure | basis |
|---|--:|---|
| baseline | ~18.9 h | 3-tile crop extrapolation, upper bound, assumed tissue-dense |
| **current** | **~2.6 h (`workers=1`) / ~1.25 h (`workers=4`)** | composition-matched crop rates × the slide's **measured** 55.8%/44.2% tissue/background split (27,565 tiles), plus a directly-measured Phase D stitch cost (322.7 s) |

Still a rate-based extrapolation from crops, not the real full-WSI run — see
[`19-open-backlog.md`](../19-open-backlog.md) item 7, still open, the single highest-leverage
remaining item in this whole document set.

## 5. What is still worth optimizing

See [`../DISCOVERED-NOT-IMPLEMENTED.md`](../DISCOVERED-NOT-IMPLEMENTED.md) for the ranked, complete
list. Top items, current state:

1. **Full real-WSI-scale validation** — closes the gate on shipping `workers>1` and replaces every
   projection above with a real number. [`19-open-backlog.md`](../19-open-backlog.md) #7.
2. **Phase D slide stitch — cheap non-GPU knobs** (`tiffsave` tile-size/pyramid-depth params, skip
   re-encoding constant background) before any GPU port.
   [`25-...`](../25-gpu-encode-decode-loop-acceleration-implementation.md) §5, §11.
3. **`workers≥6` allocator-fragmentation OOM** — reliability defect, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   untried. [`19-open-backlog.md`](../19-open-backlog.md) #7b.

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
