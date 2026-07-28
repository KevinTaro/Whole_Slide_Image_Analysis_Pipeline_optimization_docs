# 35 — Round 10: closing the round-9 backlog — implementation and results

> Implements [`34-round-10-backlog-plan.md`](./34-round-10-backlog-plan.md), which sequenced the
> open items left by [`31-gc-collect-round2-implementation.md`](./31-gc-collect-round2-implementation.md),
> [`32-phase-d-gpu-port-implementation.md`](./32-phase-d-gpu-port-implementation.md) and
> [`33-tile-read-io-implementation.md`](./33-tile-read-io-implementation.md). Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose. Round 10.

## 0. Outcome in one paragraph

Doc 34 scheduled five steps and all five are resolved. **Step 1 (correctness):** both shipped
full-slide overlays were confirmed broken by direct measurement — not inferred from their date —
and both have been regenerated and verified clean. **Step 2:** `_mp_tile_worker` now prefetches,
with its correctness properties pinned by test, but six interleaved `workers=4` repeats put the
win at −0.83%, **not separable from noise at 95%** even though the effect size lands exactly where
the mechanism predicts. **Step 3 reverses doc 32 §4:** with the read/join half included at
4.055 GP, Phase D candidates B and C are **slower than the shipped baseline** (0.788× and 0.884×),
not the 1.400× and 2.147× the in-memory spike measured — Phase 2 is dead as specified. **Step 4,
the 2.8 h full-slide run three documents were waiting on, lands at 1.347× (13,762.5 → 10,217.7 s)
and settles doc 31 Exp 4 exactly as projected** (`gc.collect` 2,218.4 → 58.8 s, 70 freezes, peak
RSS *down* 1.09 GB); doc 33's Option K is only partly settled, because an unrelated commit
confounds the read bucket's absolute size. **Step 5's gate is met but its prize is 1.04% of wall,
so it is measured and deliberately not built.**

Three defects were found along the way that nobody was looking for (§6), one of them in a script
this document depended on, plus one unexplained regression the full-slide run exposed (§4.4).

## 1. Step 1 — the broken overlays, measured and repaired

Doc 32 §5.2 left this as *"any `overlay_slide.tiff` produced before this fix has broken pyramid
levels and should be regenerated if it is still in use,"* and doc 34 §2.1 correctly called it an
inventory-and-repair procedure rather than a code change. Two full-slide overlays exist on this
machine, both written 2026-07-27, both predating the `34d909b` fix.

**The defect was confirmed on the real files, not assumed from their date.** `scripts/overlay_pyramid_audit.py`
(new) applies `tests/test_stitch_pyramid_levels.py`'s two checks to a deployed artifact:

| | `fullwsi_w1` before | `fullwsi_w4` before | `fullwsi_w1` after re-stitch |
|---|---|---|---|
| Predictor tags, L0 → L11 | `2, 1, 1, …, 1` | `2, 1, 1, …, 1` | `1, 1, 1, …, 1` |
| L3 decoded mean / zero px | 25.03 / **81.1%** | 25.03 / 81.1% | **240.71 / 0.0%** |
| L5 | 35.34 / 73.3% | 35.34 / 73.3% | 240.76 / 0.0% |
| L8 | 81.82 / 36.8% | 81.81 / 36.8% | 240.88 / 0.0% |
| L11 | 69.94 / 47.5% | 69.98 / 47.5% | 241.14 / 0.0% |
| verdict | **FAIL** | **FAIL** | **PASS** |

This is exactly the shape doc 32 §5.1(b) described — Predictor=2 declared on the full-resolution
IFD only, while the reduced levels' data is differenced anyway — reproduced here on 16.2-gigapixel
clinical output rather than the 18,688² file doc 32 measured. A pathologist zooming out on either
of these saw noise.

**Repair did not recompute the batch.** `_stitch_overlay_slide` reads `_stitch_scratch/`, which the
shipped path deletes on success — but both runs kept their per-tile annotated overlays (27,565
tiles, 46 GB) in a sibling directory. The audit script hard-links those into a fresh scratch dir, so
the (deliberate) `rmtree` at the end of a successful stitch removes only the links and the 46 GB
source survives — which matters, because step 3 needs those same tiles as its input.

`fullwsi_w1` re-stitched in **1,352.9 s** and now passes both checks. Cost: **5.86 → 7.50 GB,
+28.0%**. Doc 32 measured +9.4% for `predictor="none"` on an 18,688² crop; the penalty is
**three times larger at full-slide scale**, which is worth recording because doc 32's number is the
one anyone would extrapolate from. It does not change the trade — doc 32 §5.1(b)'s reasoning ("an
overlay a pathologist cannot read is worth zero") holds at 28% just as at 9.4% — but the number
should be the measured one.

The old file is kept alongside as `overlay_slide.tiff.predictor-broken` rather than deleted, so a
stitch that died halfway could not leave the run with no overlay at all.

`fullwsi_w4` was re-stitched by the same procedure (**1,315.0 s, also 5.86 → 7.50 GB**) after step
4's run was off the machine — a 22-minute pyvips stitch peaks around 45 GB RSS, and running it
alongside would have contaminated exactly the peak-RSS and page-cache numbers doc 31 §4 wanted from
that run. Both regenerated slides now decode identically at every level (means 240.71 → 241.14,
zero-fraction 0.0%), which is itself a small correctness check: they are two runs of the same
slide, so their pyramids *should* agree.

**Step 4's own output is the third data point, and the one that matters most**: it was written
fresh by the fixed code rather than repaired, and it audits clean at every level (means 240.71 →
241.14, zero-fraction 0.0%, Predictor consistent across all 12 IFDs). So the fix is confirmed at
full-slide scale on both paths — repairing an old file and writing a new one.

## 2. Step 2 — prefetch on the multiprocess path

### 2.1 What shipped

Doc 33 follow-up 3 called this *"the highest-value remaining piece of Option L, not an
afterthought"*, because `workers=4` is the shipped production recommendation and round 9 wired
Option L only into `run_batch`'s single-process loop — so the entire measured −1.75% belonged to a
code path production does not run.

`_mp_tile_worker` now drives `prefetch_tile_reads` on its own dedicated `tile-read` pool, one per
worker process:

```python
read_tiles = (_inline_tile_reads
              if os.environ.get(_MP_NO_PREFETCH_ENV) == "1"
              else prefetch_tile_reads)
with _frozen_gc_generation(), \
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") as pool, \
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-read") as rpool:
    for (ihc_path, dish_path, (ax, ay)), read_fut in read_tiles(
            _drain_task_queue(task_q), rpool):
        try:
            preread = read_fut.result()
        except Exception as exc:
            logger.error("Tile tile_x%d_y%d 讀取失敗: %s", ax, ay, exc)
            preread = None
        tg = None if preread is None else _process_precut_tile_gpu(..., preread=preread)
```

Two pieces of glue were needed and nothing else:

- **`_drain_task_queue`** turns the cross-process work queue plus its poison sentinel into the
  iterable `prefetch_tile_reads` consumes. Queue semantics are unchanged: each tile is still
  `get`-ted once by one worker, and the poison still ends that worker.
- **A dedicated pool per worker**, for the same reason doc 33 §2 gave for the single-process path:
  the `tile-cpu` pool is the background arm running `detect_all_dots`, and sharing it would queue
  each read *behind* the previous tile's CPU tail, serialising precisely what the change exists to
  overlap. Four workers means four independent one-thread pools, sharing nothing.

The one behavioural change worth naming: a prefetching worker holds **one extra task** off the
shared queue. Under the dynamic work queue that is just "take work one tile earlier"; every tile is
still processed exactly once, and four workers holding four extra tiles is irrelevant against
27,565.

**Doc 34 §2.2 asked for the fail-fast path to be re-traced rather than assumed to match
`run_batch`'s, and it does differ** — the worker converts a `None` chunk into an `("error", …)`
message on the result queue, where `run_batch` raises. The prefetched read's exception is therefore
re-raised out of `read_fut.result()` at the moment its own tile is consumed, logged against that
tile, and converted into the same `preread = None` → `tg = None` → worker-level abort. Two failure
modes are ruled out by test: swallowing the exception (batch ships a slide with an undocumented
hole) and surfacing it against the wrong tile (the message blames an innocent tile).

`tests/test_mp_tile_read_prefetch.py`, 6 tests: the queue adapter's shape and poison handling; the
control arm matching the prefetch generator's contract, including where its exceptions surface; the
worker processing every tile once, in order, with prefetched pixels; a failing read aborting the
batch **for its own tile** and no other; and the ablation env var actually reaching the worker. Full
suite: **60 → 66 tests, all passing.**

### 2.2 The ablation switch had to be an environment variable

`perf_measure.py --no-prefetch` replaces `prefetch_tile_reads` in the **parent's** namespace, and
`spawn`ed workers re-import the module and get the clean original — the same trap doc 25 §2.2
recorded for `_save_tile_array`. A control arm that silently fails to take effect makes both arms
identical and the measurement meaningless (playbook anti-pattern #3), so the worker reads
`HYBRID_MP_NO_PREFETCH` instead, following the `HYBRID_MP_WORKER_PROBE` precedent already in that
module. `--no-prefetch` now sets both. A test pins it.

Two measurement gaps were closed at the same time, both of which would have made this ablation
unreadable:

- **`wrap_tile_read_split` was patching only `hybrid_pipeline`'s binding** of
  `_process_precut_tile_gpu`, so the tissue/background split buckets came back **empty at
  `workers>1`** — the configuration production runs. It is now installed on `m0_multiprocess`'s
  binding too.
- **The Option H freeze count was unreadable from outside `run_batch`.** Doc 31 §7 follow-up 2 asks
  to watch it on the full-slide run specifically (~71 freezes vs the crop's one); `perf_measure.py`
  now reports `gc_refreeze`.

### 2.3 Result — the mechanism is confirmed, the wall-clock claim is not

Six interleaved repeats per arm (A,B,A,B,…) on the `match24` composition-matched anchor at
`workers=4`. **Correctness veto passed: `stats` identical (224 success / 352 skipped) in all twelve
runs**, matching round 9's single-process runs exactly. Peak RSS 11.70–11.81 GB across every run —
no meaningful difference, as expected for ~12 MB of extra in-flight data per worker.

| | control (`--no-prefetch`) | prefetch | Δ | ranges overlap? |
|---|--:|--:|--:|---|
| **all 6 pairs** | **105.30 s** (sd 1.16) | **104.43 s** (sd 0.84) | **−0.83%**, 1.0084× | **yes** |
| pairs 1–3 | 105.43 s (sd 1.74) | 104.99 s (sd 0.89) | −0.42% | yes |
| pairs 4–6 | 105.17 s (sd 0.52) | 103.86 s (sd 0.08) | −1.24% | no |

**The headline is the all-six number, and it does not clear the noise floor**: |Δ|/SE = 1.50, where
about 2 is needed to call it. The last three pairs are much tighter — both arms' variance collapses
(sd 1.74 → 0.52 and 0.89 → 0.08) and the ranges separate cleanly — which is the page-cache warming
effect doc 31 §6 already caught and warned about. But **selecting that subset after seeing it is
weaker evidence than pre-registering it would have been**, and it is reported here as a
consistency check, not as the result.

**What is unambiguous is the mechanism.** `B2r_tile_read` rises **5.769 s → 6.626 s (+14.9%)** — in
every single one of the six pairings, with no exceptions — while wall does not rise. That is doc 33
§3.1's signature of work successfully moved onto a parallel arm, reproduced across four processes
instead of one thread. Reads genuinely execute concurrently with the GPU forwards; they contend and
each takes longer in wall-clock terms while no longer sitting on the critical path.

**And the effect size is exactly where the arithmetic says it should be**, which is the check that
distinguishes "small real effect" from "measured the wrong thing":

| | hideable read (share of wall) | measured saving | recovered |
|---|--:|--:|--:|
| `workers=1` (doc 33 §3) | 3.06% | 1.75% | 57% |
| `workers=4` (here) | 1.37% | 0.83% | 61% |

At `workers=4` each worker's own reads total ~1.44 s against a ~105 s wall, so the prefetch's
Amdahl ceiling on this anchor is ~1.37% — about half what it was at `workers=1`, because the wall
shrank 1.8× while the read work did not. Recovering ~61% of a 1.37% ceiling is a ~0.8% win, which
is precisely what was measured and precisely too small for a 576-tile anchor to resolve in six
runs.

**Disposition: keep it, and do not claim a win from this anchor.** The playbook's step 4 says
zero-contribution layers get cut — but it also says the crop is the wrong instrument here, and doc
33 §4 already established *why*: at 576 tiles the precut scratch fits page cache and `B2r_tile_read`
is 3% of wall, where at full slide it was 17.2%. Cutting a change because a deliberately
cache-friendly anchor cannot see it would be measuring at the wrong scale. The change costs nothing
measurable (RSS flat, `stats` identical), its mechanism is confirmed, and step 4 is the instrument
that can size it.

**One thing doc 34 §1 hoped for did not happen.** Its argument for sequencing step 2 before step 4
was that the expensive run would then *"sit next to evidence that the production path benefits
too."* That evidence came back inconclusive. The sequencing was still right — step 2 is shipped and
tested, so no second 3.8 h run is needed to re-validate it — but the claim it was meant to support
is not available.

## 3. Step 3 — Phase D at scale, and why it reverses doc 32 §4

### 3.1 What doc 32 could not measure

Doc 32 §6 is explicit: the Phase 1 spike ran on a contiguous in-memory slab, while *"real Phase D
never holds its input in memory — it joins 27,565 tile files lazily through pyvips … The spike
measures encode + pyramid + container given data in memory; it does not measure the read/join half."*
Doc 32 §4's 1.055× / 1.109× ceilings, and its "this justifies a scale-accurate Phase 1.5"
recommendation, were both gated on that half.

`stitch_probe.py` gained a `--candidates` mode that closes it. Candidates B and C get the lazy
source they were missing: a top-to-bottom band walk that joins **one row of source tiles at a time**
(~163 MB at 4 GP), cuts its rows into the container's 128 px tile rows, and pushes the same rows
once through the pyramid reduction. Nothing larger than one band of any level is ever resident —
the same boundedness property `_join_overlay_tiles` has, and the reason Phase D can run on a 16 GP
slide at all. A **read-only arm** drains that band source and does nothing else, which is the first
direct measurement of the cost every candidate shares.

Input is 6,900 real annotated overlay tiles at the 4.055 GP / 6,900-tile scale doc 29 §2.3 asked
for, at the slide's measured 44.74% annotated share.

### 3.2 The measurement

| | wall | read | pyramid | encode | container | output | vs A |
|---|--:|--:|--:|--:|--:|--:|--:|
| read-only (shared by all) | **30.44 s** | | | | | | |
| **A** `pyvips_tiffsave` (shipped) | **62.55 s** | (fused) | (fused) | (fused) | (fused) | 2.44 GB | **1.000×** |
| **B** `tifffile_cpu` | 79.33 s | 36.87 | 18.39 | 18.22 | 0.55 | 1.89 GB | **0.788×** |
| **C** `tifffile_gpu_pyramid` | 70.71 s | 36.04 | **6.59** | 22.23 | 0.54 | 1.89 GB | **0.884×** |

All three are shape-matched to the baseline (**11 pages, 128×128 tiles, identical level-0 shape**)
and all three pass the per-level decode audit (every level bright, zero-fraction 0.0%, Predictor
consistent across IFDs). That shape check earned its keep immediately — see §6.3.

**Both candidates are slower than the code they were meant to replace.** Against doc 32 §3's
in-memory 1.400× and 2.147×, this is a full reversal.

### 3.3 Why, and what it means for Phase 2

The read is **48.7% of candidate A's entire wall** (30.44 s of 62.55 s), which means A's actual
encode + pyramid + container work is only ~32 s. Candidate C's equivalent work is 29.4 s — barely
faster — and B's is 37.2 s, i.e. slower outright. Doc 32 §4's claim that the ≈55% doc 29 attributed
to container assembly *"is not irreducible; it is pyvips-specific"* survives as a statement about
container writing (0.55 s, under 1% of either candidate) but **not** as a route to a faster Phase D:
what pyvips spends that time on is overlapping the read with the encode, and the streamed
candidates pay it serially instead.

That is also the honest caveat, and it is quantifiable rather than hand-waved. If B's and C's reads
were fully hidden behind their own encode work — a threaded implementation nobody has written —
they would land at 42.5 s (**1.47×**) and 34.7 s (**1.80×**). So the remaining case for Phase 2
rests entirely on **pipelining the read**, and that, not container assembly and not GPU encoding, is
now the measured gate. Translating the best hypothetical onto doc 27 §6.5's `workers=4` numbers
(Phase D = 1,200.8 s of a 6,211 s wall) gives ~1.094× end-to-end — below doc 32 §4's projected
1.109×, and requiring work that does not exist.

**Decision, per doc 34 §2.3: Phase 1.5 is done and it closed the question negatively. Phase 2 is
not justified.** Doc 32 §7 step 4 favoured candidate B on simplicity grounds *"if it holds up at
scale"*; it does not hold up at scale — it is the worst of the three. Doc 29 §3's "prefer the
simplest solution that clears the bar" resolves cleanly: nothing clears the bar, so the shipped
pyvips path stays.

This is the playbook's red flag #1 ("your optimised version is slower than the dumbest baseline")
firing on a candidate that looked like a 2.15× win three weeks ago, and the reason it fired is red
flag #5: the spike was a micro-benchmark of half the problem.

### 3.4 What is still outstanding here

The programmatic correctness checks are complete (shape match, per-level decode, Predictor
consistency). **The QuPath open-and-render-identically pass doc 34 §2.3 also asks for has not been
run** — it needs a human at the application. The three outputs are kept at `/home/taro/r10_phase_d/`
for exactly that. Given §3.2's result it is no longer decision-relevant — no candidate is being
adopted — so it is recorded as outstanding rather than blocking.

## 4. Step 4 — the full-slide `workers=1` run

The run doc 31 §7, doc 33 §6 and doc 34 §2.4 all asked for: one `workers=1` pass over the whole
16.2 GP slide, no `--resume` (a resumed run does less work and its wall is not comparable),
collecting everything three documents were waiting on. **2 h 50 m of analysis + 19 m 40 s of
stitch.**

| | doc 27 baseline | round 10 | |
|---|--:|--:|--:|
| end-to-end wall | 13,762.5 s | **10,217.7 s** | **1.347×**, −25.8% |
| `stats` | 10,801 / 16,764 | 10,800 / 16,765 | one tile changed population |
| peak RSS | 61.13 GB | **60.04 GB** | −1.09 GB |

The single tile that flipped between success and skipped is GPU non-determinism at the
`core_mask.sum() == 0` threshold, which doc 20 already established is judged against a noise floor
rather than exact equality. Everything else about the output is unchanged.

### 4.1 Attribution — and the confound that had to be ruled out first

**The 1.347× is not attributable to Options H and L on its own, and checking that was not
optional.** Between doc 27's baseline run and this one, `d6592c3` removed every per-tile
intermediate write: the baseline run left 275 GB of `instance_mask`, `dish_nucleus_mask`,
`cell_crops`, `masked_ihc` and `core_mask` on disk, and this run writes none of them. Crediting
Options H and L with a speedup that a third change caused is playbook anti-pattern #7 applied
across runs, which is exactly the mistake doc 34 §2.4 sequenced the round to avoid.

Decomposing by arm settles it, because `wall ≈ max(MAIN, BG) + outside`:

| arm | baseline | round 10 | Δ |
|---|--:|--:|--:|
| **MAIN** (critical) | 13,084.7 s | **9,493.9 s** | **−3,590.8** |
| BG (slack) | 5,292.9 s | 2,826.9 s | −2,466.1 |
| `tile-read` (new arm) | — | 1,581.2 s | — |
| outside (Phase D) | 1,185.4 s | 1,182.4 s | −3.0 |
| **wall** | 13,762.5 s | 10,217.7 s | **−3,544.8** |

**The wall change (−3,544.8 s) is the MAIN-arm change (−3,590.8 s).** The 2,466 s of write work
that `d6592c3` deleted was *all on the BG arm*, which had ~7,800 s of slack against MAIN — so it
contributed approximately **nothing** to wall. That is counter-intuitive enough to be worth
stating plainly: deleting 275 GB of disk writes from this pipeline did not make it faster, because
those writes were never on the critical path (doc 18 §6.2's `BG + outside` bound, still not
binding). The confound is real but it lands where it cannot move the number.

What did move MAIN:

| | Δ on MAIN |
|---|--:|
| `B4_gc_collect` — **Option H** | **−2,159.6 s** |
| `B2r_tile_read` leaving MAIN entirely — **Option L** | **−2,368.5 s** |
| `B1_m3b_cellpose` | **+751.0 s** |
| `B1_unet_coremask`, `BM1_*`, misc | +144 s |

### 4.2 Doc 31 Exp 4 — settled, and its projection was accurate

| | doc 27 baseline | round 10 |
|---|--:|--:|
| `B4_gc_collect` total | 2,218.4 s | **58.8 s** (−97.3%, 37.7×) |
| per-call decile means (ms) | 1.2 → … → 80.5 (**climbing**) | 16.19 → 0.94 → 0.41 → 0.42 → 0.40 → 0.39 → 0.41 → 0.39 → 0.75 → **1.04** |
| last ÷ first decile | ~67 (climbing) | **0.06** (flat, then falls) |
| `gc.freeze()` count | 1 | **70** |
| peak RSS | 61.13 GB | **60.04 GB** |

Three things this pins down that no crop could:

- **The regression is gone, not reduced.** Doc 27's per-call cost climbed monotonically with
  accumulation; here it is flat at ~0.4 ms from the second decile onward. The elevated *first*
  decile (16.19 ms) is model-init garbage being collected before the first freeze lands, which is
  a one-off, not a trend.
- **Doc 31 §7's projection was right.** It predicted "~2,185 s off a 13,762 s wall = 15.9%, a
  1.19× ceiling" from a synthetic harness plus one crop run. Measured: **2,159.6 s = 15.7% of the
  baseline wall, 1.186×.** The remaining floor it predicted (~33 s from 1.2 ms × 27,565 calls)
  came in at 58.8 s — same order, ~2.1 ms per call.
- **Doc 31 §4's RSS-pinning concern is answered: no.** 70 freezes — against doc 31 §7 follow-up
  2's "~71 freezes is the real exposure count" and the crop's single one — and peak RSS went
  *down* 1.09 GB. The crop's weak evidence held at 70× the exposure.

### 4.3 Doc 33 Option K — the critical-path claim holds; the isolated number does not

`B2r_tile_read` **left the critical arm entirely**: 2,368.5 s of inline read on MAIN became
1,581.2 s running concurrently on the `tile-read` thread, which has ~7,900 s of slack against
MAIN. That −2,368.5 s off MAIN is 17.2% of the baseline wall, a **1.208×** contribution — slightly
better than doc 33 §4's projected 1.16× ceiling.

**But the bucket's absolute size is confounded and the honest reading has to say so.** Doc 27 §6.4
traced the read cost specifically to page-cache eviction under RSS pressure — and removing 275 GB
of concurrent writes is precisely the kind of change that relieves page-cache pressure. So the
2,368.5 → 1,581.2 s drop cannot be credited to Option L; some unknown share of it is `d6592c3`.
What is *not* confounded is the critical-path question, because the reads now execute on a thread
with thousands of seconds of slack no matter what they cost.

**Doc 33's Option K therefore remains formally unsettled.** The clean experiment is a second
full-slide run with `--no-prefetch` on today's code — ~2.9 h — and it is the only thing that would
turn 1.208× from an upper bound into a measurement. It is not run here.

Splitting the read by population, at full scale, for the first time:

| | reads | total | per read | per tile |
|---|--:|--:|--:|--:|
| tissue | 24,360 | 1,037.8 s | 42.6 ms | 85.2 ms |
| background | 30,770 | 543.4 s | 17.7 ms | 35.3 ms |

The tissue : background per-read ratio is **2.41**, against the 2.06 doc 33 §1 measured on the
crop and projected would transfer. It transferred, within 17%.

### 4.4 One regression this run exposes

**`B1_m3b_cellpose` is 751.0 s slower than the baseline (6,358.9 → 7,109.9 s, +11.8%)**, on the
critical arm, with the same tile count. Nothing in this round touched M2/M3b. The likely candidate
is `e806938` ("更改判讀依據"), which changed judgement criteria between the two runs, but that is a
hypothesis, not a finding — it has not been measured. It is worth chasing precisely because it is
on MAIN: it gave back 21% of what Option H won.

## 5. Step 5 — `depth=2/3`: gate met, prize too small, not built

Doc 33 §4 recorded the concern and doc 34 §2.5 made building it strictly conditional on step 4
showing that background-tile runs actually exhaust the `depth=1` window. **They do** — and this is
now measured rather than argued:

| | measured |
|---|--:|
| background tile's read (2 files) | **35.32 ms** |
| background tile's GPU work (UNet only, `core_mask.sum() == 0` fast path) | **28.39 ms** |
| deficit per background tile | **+6.93 ms** |
| × 15,385 background tiles | **106.6 s** |
| share of wall | **1.04%** |

So doc 33 §4's arithmetic was directionally right: at full scale a background tile's read does
outrun the only GPU work available to hide it behind, by about 24%. Doc 34 §2.5's condition is
technically satisfied.

**It is still not worth building, and the reason is Amdahl, not doubt about the mechanism.** The
entire prize is 1.04% of wall — below the playbook's own "don't optimise anything under ~10%"
line by an order of magnitude, and *below what any anchor this project owns can measure*: §2.3
just demonstrated that six interleaved repeats of the composition-matched crop could not resolve a
0.83% effect. Validating a ≤1.04% change would require another ~2.9 h full-slide run, and it would
be competing for that run against the `--no-prefetch` arm §4.3 actually needs.

**Not built. Recorded with its measured ceiling**, so a future round decides with 106.6 s and
1.04% in hand rather than with doc 33 §4's projection.

## 6. Three defects found on the way

### 6.1 `stitch_probe.py` could never have found its own input

Doc 33 §5 repaired this script's imports after the M0 split and doc 32 §6 called it "unblocked".
It was not: `build_inputs` wrote its synthetic tile grid to `overlay_annotated/` while
`_join_overlay_tiles` reads `_STITCH_SCRATCH` (`_stitch_scratch/`), so every run since that rename
would have died with `FileNotFoundError`. It now takes the directory name from the pipeline
constant instead of repeating the string. Round 9 fixed the imports it could see from a stack trace
and stopped; this one needed the script actually to be run.

### 6.2 A `workers=4` run OOM'd the GPU and then hung for 49 minutes

Mid-ablation, one control-arm run died with `torch.OutOfMemoryError` — *"Process 2264989 has
24.76 GiB memory in use"* out of a 31.36 GiB card, starving its three siblings. That is the
allocator balloon recorded as doc 19 #7b / DISCOVERED #2 for `workers≥6`, **observed here at
`workers=4`, the shipped production recommendation.** It is intermittent: the same command
succeeded on eleven other runs.

Fail-fast worked correctly — the failing tile aborted the batch rather than shipping a holed slide.
**What did not work is the exit.** After raising, the parent process sat alive for 49 minutes doing
nothing, and had to be killed. The signature (a terminated worker plus a `multiprocessing.Queue`
whose feeder thread can no longer flush to a closed pipe, joined at interpreter exit) points at the
`_kill_all()` path in `_run_tiles_multiprocess`, which is untouched by this round's changes.

Both belong to the multiprocess path as it already shipped, not to the prefetch wiring — the OOM
hit the `--no-prefetch` control arm, which is the pre-round-9 code shape. Neither is fixed here;
both are recorded because a full-slide production run at `workers=4` can hit them, and the second
turns a clean fail-fast into an unattended job that never returns.

### 6.3 The candidates were writing a shallower pyramid than the baseline

The first `--candidates` run had B and C writing **9 pages against A's 11**, because the level-count
rule used `min(h, w)` where pyvips uses `max(h, w)` — it stops two levels early on a landscape
slide. The candidates were doing less work than the baseline, exactly the non-comparability doc 32
§3 caught the first time and the reason it added a shape check at all. The numbers in §3.2 are from
the corrected run. (The conclusion was unaffected in direction — B and C were already losing while
doing *less* work — but the comparison was not quotable until it was fixed.)

## 7. Disposition

| doc 34 item | status |
|---|---|
| Step 1 — regenerate broken overlays | **DONE** — defect confirmed by measurement on both deployed slides; w1 re-stitched (1,352.9 s, +28.0% size) and w4 likewise, both passing. `scripts/overlay_pyramid_audit.py` added. Step 4's fresh full-slide output also audits clean, confirming the fix on the write path as well as the repair path |
| Step 2 — `_mp_tile_worker` prefetch | **SHIPPED**, 6 tests, env-var ablation switch. **Wall claim withheld**: −0.83% over six `workers=4` pairs is inside the noise floor; mechanism confirmed (+14.9% `B2r_tile_read`, every pair) and effect size matches the 1.37% Amdahl ceiling |
| Step 3 — Phase D at 4.055 GP from disk | **DONE — result reverses doc 32 §4.** B 0.788×, C 0.884× against the shipped baseline once the read/join half is included. **Phase 2 not justified** |
| Step 4 — full-slide `workers=1` run | **DONE** — 10,217.7 s, **1.347×** over doc 27's baseline. **Doc 31 Exp 4 settled** (`gc.collect` −97.3%, decile climb eliminated, 70 freezes, peak RSS −1.09 GB — §4.2). **Doc 33 Option K only partly settled**: the read left the critical arm (−2,368.5 s, 1.208×) but its absolute size is confounded by `d6592c3` (§4.3) |
| Step 5 — `depth=2/3` tuning | **MEASURED, NOT BUILT** — the gate is met (background read 35.32 ms vs 28.39 ms of UNet) but the ceiling is **106.6 s = 1.04% of wall**, below both the playbook's threshold and any anchor's resolution (§5) |
| Option J (spill to disk) | **No action**, as doc 34 §4 left it |
| Option M / N1–N3 | **No action** — correctly closed in round 9 |
| Correctness veto | **PASSED** — `stats` identical across all twelve `workers=4` runs, and 10,800/16,765 vs the baseline's 10,801/16,764 at full slide (one tile, GPU non-determinism at the empty-mask threshold); all Phase D candidates shape-matched and per-level clean |

## 8. Follow-ups this round creates

Ordered by what a next round should actually do first.

1. **`B1_m3b_cellpose` is 751 s slower than the baseline, on the critical arm (§4.4).** It gave
   back 21% of what Option H won and nothing this round touched it. `e806938` is the suspect but
   that is unverified. This is the largest single unexplained number in the round and it sits on
   MAIN — measure it before optimising anything else.
2. **The post-fail-fast hang (§6.2) is the most dangerous defect found.** It converts a correct
   abort into a job that never returns, on exactly the unattended full-slide runs `--resume`
   exists for — and it was seen **twice**: once after a `workers=4` fail-fast (49 min) and once
   after a `workers=1` run had already printed its final JSON (2 h). The second has no
   multiprocessing in it at all, so the cause is not only the task queue. Worth a bounded join on
   the queue *and* an audit of what non-daemon threads outlive `main()`.
3. **The `workers=4` CUDA allocator balloon (§6.2)** — one worker took 24.76 GiB of a 31.36 GiB
   card and starved its siblings. `workers=4` is the shipped production recommendation.
   `cuda_alloc_conf=expandable_segments:True` is the existing knob, has never been measured at
   `workers=4`, and sweeping it is cheap.
4. **A full-slide `--no-prefetch` arm would settle Option L properly (§4.3)** — ~2.9 h, and it is
   the only way to turn the 1.208× upper bound into a measurement. Worth more than item 5.
5. **If Phase D is ever revisited, the gate is now read pipelining**, not codecs or containers
   (§3.3). Measured hypothetical ceiling ~1.47×/1.80× on Phase D alone, ~1.094× end-to-end at
   `workers=4` — below what doc 32 projected, for work that does not yet exist. Note Phase D's
   share of wall *grew* this round (8.6% → 11.6%) purely because everything around it got faster.
6. **`depth=2/3` (§5)** — 1.04% ceiling, measured. Do not build without a full-slide instrument.
7. **A QuPath pass on `/home/taro/r10_phase_d/cand_{A,B,C}.tiff`** remains unrun (§3.4), now
   informational rather than decision-relevant.
