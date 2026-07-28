# 33 — `B2r_tile_read` / precut-scratch I/O: implementation and results

> Implements [`30-tile-read-io-plan.md`](./30-tile-read-io-plan.md) — **Option L, one-tile-ahead
> read prefetch**, plus the plan's §4 step 2 measurement that had to come first. Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md).
> Round 9. Companions: [`31-gc-collect-round2-implementation.md`](./31-gc-collect-round2-implementation.md),
> [`32-phase-d-gpu-port-implementation.md`](./32-phase-d-gpu-port-implementation.md).
>
> **Round 9's most important change is in neither of this document's own plans**: chasing a
> QuPath crash found that every `overlay_slide.tiff` written before this round decodes its
> pyramid levels as noise. Fixed and tested — see
> [`32-phase-d-gpu-port-implementation.md`](./32-phase-d-gpu-port-implementation.md) §5.1(b).

## 0. Outcome in one paragraph

Doc 30 §4 sequences this work as **K → step 2 → L → (M only if L disappoints)**. Option K
(re-measure on a full-slide run after doc 28 ships) **could not be done** — it needs the same
3.82 h run doc 31 §7 also could not afford — so this round did step 2 first and used its result
to decide. Step 2 says build: the read cost is **concentrated on tissue tiles (62.7%), and each of
those tiles has ~41× more GPU work of its own to hide its read behind**. Option L is built, tested, and ablated. Option M and all
of Option N remain unbuilt, exactly as the plan sequenced them.

## 1. Step 2 — the tissue/background split, measured

Doc 30 §3 flagged this as *"the first thing to measure before sizing Option L's real ceiling, not
something to assume from the aggregate average"*. It could not be read off any existing bucket:
a tile's population is decided by the UNet forward that runs **after** both its reads, so
`perf_measure.py` was extended to stash each read under its file path and attribute it
retroactively when `_process_precut_tile_gpu` returns (`chunk is None` ⇒ background).

Measured on the `match24` composition-matched anchor (576 tiles, 55.9% background — the same
composition as the real slide's 55.8%), on the pre-Option-L code:

| population | tiles | reads | read cost | share of read cost | per tile | GPU work per tile |
|---|--:|--:|--:|--:|--:|--:|
| tissue | 254 | 508 | 3.561 s | **62.7%** | 14.02 ms | **~574 ms** (UNet 29.4 ms + 2× Cellpose 544.5 ms) |
| background | 322 | 644 | 2.195 s | 37.3% | 6.82 ms | **~29.4 ms** (UNet only) |
| all | 576 | 1,152 | 5.756 s | 100% | 9.99 ms | — |

**This is the favourable outcome for Option L, and it was not the obvious one.** Doc 30 §3
worried the aggregate might be masking the opposite shape — most of the cost on background tiles,
which have almost nothing to hide a read behind. Instead the cost is concentrated on exactly the
population with the largest hiding place: a tissue tile's read is **2.4% of its own GPU work**,
so a one-deep prefetch can absorb essentially all of it. Background tiles are the harder case
(read is 23% of their GPU work) but own only 37.3% of the cost.

The mechanism behind the split is worth recording because it also predicts how it scales:
background tiles are near-uniform white, so their `deflate`-compressed scratch files are far
smaller and decode faster. The split is a property of slide *content*, not of cache state, so it
should transfer to full-slide scale even though the magnitude does not (§4).

## 2. What shipped — Option L

**`m0_tile_runner.py`** — the two reads move out of `_process_precut_tile_gpu`'s critical
position into a generator that runs them one tile ahead:

```python
def prefetch_tile_reads(tiles, pool, depth: int = 1):
    q = deque()
    for item in tiles:
        q.append((item, pool.submit(_read_tile_pair, item[0], item[1])))
        if len(q) > depth:
            yield q.popleft()
    while q:
        yield q.popleft()
```

`_process_precut_tile_gpu` gains an optional `preread` parameter; when given it skips the read
entirely. The parameter is **optional on purpose** — the multiprocess worker and every existing
caller keep today's inline-read behaviour with no change (its 5 call sites were checked, not
assumed).

**`hybrid_pipeline.py`** — `run_batch`'s single-process loop drives it on a **second, dedicated**
single-thread pool:

```python
with _frozen_gc_generation() as refreezer, \
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") as pool, \
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-read") as rpool:
    for idx, ((ihc_path, dish_path, (ax, ay)), read_fut) in enumerate(
            prefetch_tile_reads(tiles, rpool), start=1):
        try:
            preread = read_fut.result()
        except Exception as exc:
            logger.error("Tile tile_x%d_y%d 讀取失敗: %s", ax, ay, exc)
            preread = None
        tg = None if preread is None else _process_precut_tile_gpu(..., preread=preread)
        if tg is None:
            ...  # unchanged fail-fast
```

Doc 30 §3 offered a choice between reusing the existing `tile-cpu` pool and a dedicated one. A
**dedicated pool** is right and the plan's parenthetical was correct: the `tile-cpu` pool is the
background arm running `detect_all_dots`, and sharing it would queue each read *behind* the
previous tile's CPU tail — serialising precisely what this change exists to overlap.

### 2.1 Fail-fast, preserved exactly

Doc 30 §3 calls this out as *"a real correctness requirement for the design, not just an
implementation detail"*. Before Option L a read error raised inside `_process_precut_tile_gpu`,
which logged and returned `None`, and `run_batch` turned `None` into a batch abort. Now the
exception is raised on another thread, one tile early. It is re-raised out of `read_fut.result()`
**at the moment that specific tile is consumed**, converted to the same `preread = None` →
`tg = None` path, and aborts the batch identically. Two failure modes are ruled out by test:
swallowing the exception (batch continues past a hole), and surfacing it against the wrong tile.

`tests/test_tile_read_prefetch.py`, 6 tests: every tile yielded exactly once in order; empty
stream; single-tile stream (the read-ahead window never fills); reads issued ahead of consumption
(asserted on *submission* order, since a real thread pool makes execution order
non-deterministic); a failing read re-raising for its own tile and no other; and `preread`
genuinely bypassing the inline read — without which the prefetch would *double* every read
instead of moving it.

Memory stays bounded: `depth=1` means at most two tiles' pixel data in flight (~12 MB), matching
the existing "at most two tiles in flight" property of the two-stage pipeline. Processing order
is untouched — only *when bytes are decoded* changes — so `_finish_batch`'s global sort and the
coordinate-keyed stitch are unaffected (doc 18 §4.2 already established processing order is free).

## 3. Ablation — Option L on its own

Doc 30 §4 requires K → L → M to be *"ablated individually"*. `perf_measure.py --no-prefetch`
restores the pre-round-9 inline read through the same generator contract, so the two arms differ
in exactly one thing: when the bytes are decoded. Three repeats per arm, **interleaved**
(A,B,A,B,A,B) so any thermal or page-cache drift loads onto both arms equally.

| rep | control (`--no-prefetch`) | Option L prefetch |
|---|--:|--:|
| 1 | 188.06 s | 184.90 s |
| 2 | 188.08 s | 186.25 s |
| 3 | 188.55 s | 183.63 s |
| **mean** | **188.23 s** (sd 0.22) | **184.93 s** (sd 1.07) |

**−3.30 s, −1.75%, 1.018× — and the two arms' ranges do not overlap** (worst prefetch run,
186.25 s, is still faster than the best control run, 188.06 s). Correctness veto passed: `stats`
identical (224 success / 352 skipped) across all six runs. Peak RSS 4.26–4.33 GB (control) vs
4.31–4.39 GB (prefetch) — no meaningful change, as expected for ~12 MB of extra in-flight data.

Sanity-checking the size of the win against its own mechanism, which is the check that catches a
result that is really something else: the read cost removed from the critical path is 5.605 s =
**2.98% of wall**, and the measured saving is **1.75%** — i.e. the prefetch recovers ~59% of the
theoretically available amount. That is the right order and the right side of the bound. The
shortfall is expected from §1: background tiles cannot fully hide a read behind 29.4 ms of UNet,
and the prefetch itself costs some contention (§3.1).

Bucket detail (means across the 3 reps):

| bucket | control | prefetch |
|---|--:|--:|
| `B2r_tile_read` | 5.605 s | 7.111 s |
| ⤷ tissue | 3.503 s | 4.454 s |
| ⤷ background | 2.102 s | 2.657 s |
| `B4_gc_collect` | 0.347 s | 0.346 s |

### 3.1 The counter-intuitive bucket movement, and why it is the expected signature

`B2r_tile_read` **rises** under the prefetch (5.756 s → 7.121 s in the first pair) while wall
falls. This is not a regression and not a measurement error: the reads now execute concurrently
with the main thread's GPU forwards, so they contend for the GIL and CPU and each one takes
*longer in wall-clock terms* — while no longer being on the critical path. A bucket total getting
bigger as total time gets smaller is the signature of work successfully moved onto a parallel
arm, and it is exactly why doc 30 §4's success criterion is stated in terms of **share of wall on
a real run, never a microbenchmark of `_read_rgb` in isolation**.

## 4. Full-slide projection — and why it is a projection

The crop cannot show this bottleneck's magnitude, by construction: doc 27 §6.4 traced the
regression to the ~49 GB precut scratch falling out of page cache once process RSS peaks at
61.13 GB, and at 576 tiles the scratch fits cache comfortably (`B2r_tile_read` is 3.0% of wall
here vs the 17.2% doc 27 measured at full slide). What the crop *does* give is the **split
ratio**, which is a content property and should transfer.

Applying the measured per-read ratio (tissue : background = 2.06) to doc 27's full-slide
aggregate of 2,368.5 s over 12,184 tissue / 15,381 background tiles:

| | per tile | population total | hideable behind that tile's own GPU work |
|---|--:|--:|--:|
| tissue | 120.5 ms | 1,468 s | **all of it** (120.5 ms vs ~574 ms) |
| background | 58.6 ms | 901 s | ~29.4 ms of 58.6 → **452 s** |
| total | — | 2,369 s | **~1,920 s = 81.1%** |

That is 13.9% of the 13,762 s `workers=1` wall → an end-to-end ceiling of about **1.16×**.

**Stated as a projection, with its assumptions visible:** it assumes the tissue/background cost
ratio survives the cache-miss regime, and that per-tile GPU times at full slide match the crop's.
Neither is verified. The number that would settle it is the same full-slide `workers=1` run doc
31 §7 needs, which is why §6 recommends running them together.

**One limitation the projection makes visible.** Background tiles at full slide would read for
58.6 ms with only ~29.4 ms of UNet to hide behind, and background tiles cluster spatially — so a
*run* of them becomes read-bound and a one-deep prefetch cannot keep up. `depth` is already a
parameter of `prefetch_tile_reads`; raising it to 2–3 is the obvious follow-up, but it costs
proportional in-flight memory and there is no measurement yet that justifies it. Recorded, not
built.

## 5. Corrections to the record

- **`measurement/bottleneck-list.md`'s "on the arm with slack"** — doc 30 §1 already corrected
  this from the code (the read sits inline on the MAIN/critical arm, not the slack BG arm). As of
  this round the phrase is finally true *as a plan*: the read now executes on a background thread
  concurrently with the MAIN arm. The row is updated.
- **`scripts/stitch_probe.py` was broken** and nobody had noticed: it still did
  `from m0_reader import ...` / `from m0_stitch import ...`, which the M0 split invalidated
  (the same breakage class the previous commit fixed for `perf_measure.py`). It imported
  `_join_overlay_tiles` and `_stitch_overlay_slide` off `hybrid_pipeline`, which no longer
  re-exports the former. Fixed to import from `m0_module.m0_stitch` directly; the now-unused
  `hybrid_pipeline` import was dropped. Doc 29/32's Phase D work depends on this script.

## 6. Disposition

| doc 30 item | status |
|---|---|
| Option K — re-measure after doc 28, free, do first | **NOT DONE** — needs a 3.82 h full-slide run. The step-2 measurement was done instead and was sufficient to decide |
| §4 step 2 — tissue/background split | **DONE** — tissue owns 62.7% of read cost, and each tissue tile has ~41× its own read time in GPU work to hide behind |
| Option L — one-tile-ahead prefetch | **SHIPPED**, single-process path, `depth=1`, dedicated `tile-read` pool, 6 tests |
| Option M — bypass the scratch round trip | **NOT BUILT** — gated on L disappointing; it did not |
| Option N1/N2/N3 — scratch codec, pyvips read, remove scratch | **NOT BUILT**, per the plan's own "record, do not schedule" |
| Correctness veto | **PASSED** — identical `stats` across all arms; fail-fast semantics pinned by test |

**Follow-ups this created:**

1. Run Option K properly on the next full-slide `workers=1` run, together with doc 31's Exp 4 and
   doc 32's Phase 2 decision — one 3.8 h run settles all three.
2. Consider `depth=2–3` for background-tile runs (§4), but only with a measurement behind it.
3. The multiprocess worker still reads inline. Doc 30 §3 argues a loop-level prefetch benefits
   every worker; applying `prefetch_tile_reads` inside `_mp_tile_worker` is a small, separate
   change, and `workers=4` is the shipped production recommendation — so this is the highest-value
   remaining piece of Option L, not an afterthought.
