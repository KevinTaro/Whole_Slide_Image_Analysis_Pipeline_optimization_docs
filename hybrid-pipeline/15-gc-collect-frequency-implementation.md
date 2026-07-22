# 15 — `gc.collect` optimisation: implementation record

> Code record for [`14-gc-collect-frequency-plan.md`](./14-gc-collect-frequency-plan.md).
> Measurements and the Choose decision that selected this shape:
> [`16-gc-collect-frequency-result.md`](./16-gc-collect-frequency-result.md).
>
> **What shipped: Option C (`gc.freeze()` after model init), unconditional. Nothing else.**
> Option A (fixed-N `gc.collect()` batching) was built, measured, found to be a regression at the
> large anchor, and deleted — see doc 16 §4.2. This document describes the final state and the two
> plan assumptions that did not survive contact with the code.

## 1. The change

One wrapper around the tile loop in `run_batch` (`backend/algorithms/hybrid/hybrid_pipeline.py`):

```python
@contextmanager
def _frozen_gc_generation():
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()
```

applied by extending the loop's existing `with`, so nothing needed re-indenting:

```python
with _frozen_gc_generation(), \
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") as pool:
```

That is the entire optimisation. `gc.collect()` and `torch.cuda.empty_cache()` still run once per
tile, unchanged. **No config field, no CLI flag, no branch in the hot loop.**

The three GPU models (UNet++ and two Cellpose SAM-ViT) are alive and reachable for the whole batch,
so every full collection was re-walking them. `gc.freeze()` moves everything currently tracked into a
permanent generation the collector skips, which cuts the cost of *each* collection — measured
83.2 ms → 1.2 ms per call, 36.71 s → 0.52 s per 441-tile batch (doc 16 §1).

### Why it is unconditional

Plan §3 recommended this if adopted ("there's no scenario where you'd want it off"), and the ablation
found none. A permanently-on config knob is dead surface (`karpathy_rule` §2). The control group is
git history: revert the wrapper to get the baseline back.

## 2. Two plan assumptions that were wrong

### 2.1 `gc.freeze()` alone would have leaked in the API server

Plan §3 describes Option C as "a single unconditional call after init". Implemented literally, that
is a **memory leak**, and the plan did not account for it.

`run_batch` is not only a CLI entry point — `backend/api/hybrid.py:39` calls it from a background job
inside a long-lived FastAPI server, so it runs many times per process. `gc.freeze()` moves
**everything currently tracked** into a generation that is never scanned again. One unpaired call per
request would permanently pin whatever garbage happened to be tracked at that instant: RSS growth
unbounded in *request count* — the precise failure mode this repo's memory-bounded invariant exists
to prevent, and completely invisible to a single-batch benchmark, which calls `run_batch` once.

Hence the context manager: `finally: gc.unfreeze()`, so the process is left exactly as it was found,
including when the batch fail-fasts mid-loop. `gc.unfreeze()` unfreezes everything with no scoped
variant, which is safe here only because nothing else in the codebase calls `gc.freeze()` (verified
by search). If that changes, this needs revisiting.

**Known caveat:** FastAPI can run background tasks concurrently, and two overlapping `run_batch`
calls would have the first to finish unfreeze globally. The effect is loss of the optimisation for
the other batch, never incorrectness. Two concurrent `run_batch` calls are already unsafe for an
unrelated and older reason (three GPU models, one CUDA context), so this adds no new constraint.

### 2.2 Freezing at the `with` is better than "right after the three `_init_*` calls"

Plan §1 C specifies the call site as immediately after `dish_cellpose = _init_dish_cellpose_segmenter()`.
It is placed ~20 lines later, at the `with`. The objects created in between (`stats`,
`per_tile_owned`, the `_collect` closure) live for the whole batch too, so freezing them is equally
correct — and `per_tile_owned` is the batch's largest growing container, so keeping the collector
from walking it every sweep is more of the effect Option C is after, not less.

Objects *appended* to a frozen list are not themselves frozen and are collected normally; they stay
alive only because the frozen list genuinely references them. No reachability semantics change.

## 3. Files

| File | State |
|---|---|
| `backend/algorithms/hybrid/hybrid_pipeline.py` | `contextmanager` import + `_frozen_gc_generation()` + one-line `with` extension (+32 / −1) |
| `backend/algorithms/hybrid/config.py`, `config_example.py` | **unchanged** — the Option A fields were added, measured, and removed; config hash is back to `db2b7e6a` |
| `scripts/perf_measure.py` | **unchanged** — the `--gc-every-n` / `--gc-freeze` ablation flags were removed with Option A |
| `scripts/verify_gc_freeze.py` | **new** — invariant guard (§4) |
| `scripts/gc_ablation_report.py` | **new** — ablation analysis (wall / RSS / correctness) used to produce doc 16 |

## 4. Invariant guard — `scripts/verify_gc_freeze.py`

Two properties must never regress, and **neither shows up in a wall-clock number**, so they get a
test rather than a comment:

1. **Cadence is unchanged.** The optimisation reduces per-call scan cost, not call frequency;
   `gc.collect()` must still run once per tile or the memory-bounded invariant is silently in play.
   Checked against the pre-change `hybrid_pipeline.py` read out of git as a control group.
2. **`freeze` is always paired with `unfreeze`**, including on the fail-fast path (§2.1).

It drives the real `run_batch` loop with only the heavy stages stubbed (model init, GPU forward, CPU
back-end, export, stitch), so it needs no GPU, model weights, or slide data, and runs in about a
minute. It is not a pipeline correctness test — that bar is `report.csv` against the baseline
(doc 16 §4.3).

```
$ .venv/bin/python scripts/verify_gc_freeze.py [--control-ref <pre-change SHA>]
  PASS  control collects once per tile: got 441, want 441
  PASS  current collects once per tile (freeze changes cost, not cadence): got 441, want 441
  PASS  freeze called exactly once / unfreeze called exactly once
  PASS  nothing left frozen after run_batch: got 0, want 0
  PASS  fail-fast still raises / nothing left frozen after fail-fast
```

`--control-ref` defaults to `HEAD`, correct only while the change is uncommitted; pass the pre-change
SHA afterwards. The tool hard-errors if the ref it is handed already contains the change, rather than
silently comparing the new code against itself.
