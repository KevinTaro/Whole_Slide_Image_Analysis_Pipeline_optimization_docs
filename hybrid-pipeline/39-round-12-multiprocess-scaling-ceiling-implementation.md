# 39 — Round 12: the multiprocess scaling ceiling — implementation and results

> Implements [`38-round-12-multiprocess-scaling-ceiling-plan.md`](./38-round-12-multiprocess-scaling-ceiling-plan.md),
> which sequenced the answer to the user's question — *"`workers=1` vs `workers=4` only gets about
> 2x — can load balancing, pipelined stitching, or opening more workers get closer to using the
> hardware fully?"* Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose. Round 12.

## 0. Outcome in one paragraph

Doc 38 scheduled five items and all five are resolved. **Item 0** was already applied to the two
files doc 38 named; the same stale sentence survived in three more, now fixed (§1). **Item 1's
full-slide `workers=4` re-run is the headline, and it fires doc 38 §4's first stop-loss**: on
today's code `workers=4` takes **5,854.9 s**, and against the current `workers=1` baseline that is
**1.745x, not 2.216x** — the ratio the whole of doc 38 §2 was built on has fallen by 21%. The
cause is not the tile-parallel arm, which doc 38 §2 predicted correctly and which this run confirms
from a third independent direction (**2.279x**, against the 2.1x–2.5x band §2 derived). It is
**Phase D, which grew from 1,239.2 s to 1,889.8 s and is now 32.3% of the `workers=4` wall** — an
Amdahl ceiling of **1.477x** for the one phase, against the 1.239x doc 27 §6.6 recorded.
**Item 3 closes negative and decisively**: dispatching all 27,565 tiles with zero work costs
**0.356 s**, i.e. **0.006% of wall** — batch-claiming the queue has an end-to-end ceiling of
1.00006x, so nothing was built (§3). **Item 2 reverses doc 35 §3.3's disposition**: the unbuilt
lever now exists, and with the band read pipelined the streamed candidates go from *slower* than
the shipped path to **1.365x and 1.581x faster** than it, controls reproducing doc 35 §3.2 almost
exactly (§4). The obvious way that spike could have been lying — a cache-resident source — was
measured instead of argued: on the real 46 GB source read cold, the read is **51.9% of Phase D at
full scale against doc 35's 48.7% at crop scale**, so the ratio the win depends on does transfer
(§4.4). Projected end-to-end value is **1.135x**, with a perfect-overlap floor at 1.184x.
**Item 4 was not done, deliberately** (§5).

The correctness veto passed on every axis checked (§2.2). One measurement gap this round
introduced is recorded rather than hidden (§6.2).

## 1. Item 0 — the stale `expandable_segments` line, and the three files doc 38 missed

Doc 38 §3 item 0 scoped this to `measurement/bottleneck-list.md` and `19-open-backlog.md`, and
both were **already correct** when this round opened — doc 38's author fixed them alongside
writing the plan, exactly as its item-0 row said ("Done alongside this document").

Re-reading for the same sentence elsewhere found it alive in three more places, all still telling
a reader that the knob is *not* the shipped default when HEAD (`b3fa47d`) has shipped it since
2026-07-28:

| file | what it said | now |
|---|---|---|
| `README.md:265` | "recommended for `workers>1` but not yet the shipped default in `config_example.py`" | states the flip, and that production headroom is therefore the tighter **92.2%-of-card** figure, not round 8's 93.3% / ~2.2 GB |
| `docs/BACKLOG.md` (header, item 1, item 7b) | "recommended but not yet the shipped default" / status 🟡 "default not yet flipped" | flip recorded; item 7b promoted 🟡 → ✅ |
| `DISCOVERED-NOT-IMPLEMENTED.md:50` | "not yet flipped as the shipped default (deployment decision)" | states the decision was made and by which commit |

This matters beyond tidiness for one reason doc 38 §0 already gave: the knob **removes a failure
mode, it does not create headroom**. Every one of those three files was pointing a reader at the
looser round-8 VRAM figure while production runs on the tighter one — which is the number any
"can we run `workers=5`" conversation has to start from.

**Still open, and still out of scope** (doc 38 §0 recorded it and this round does not change it):
`run_batch(..., workers: int = 4, ...)` at `hybrid_pipeline.py:128` still defaults to `4` while
every document says `1`. Neither production call site is affected (`backend/api/hybrid.py:89`
passes `workers=1`; the CLI's argparse default is `1`), so it remains a latent footgun for a
future caller that omits the argument, not a live bug.

## 2. Item 1 — the clean full-slide `workers=4` re-run

Doc 38 §1.1 flagged the problem precisely: *"This number has never been re-run at `workers=4`
since round 8 (2026-07-27)"*, while rounds 9–11's wins were only ever confirmed at `workers=1`, so
2.216x's numerator and denominator were **not measured on the same code**. This run closes that.

One full 27,565-tile pass, `--mp-workers 4 --stream-precut`, no `--resume`, on HEAD.
**`--worker-timings` was deliberately not passed** — its own help says the in-worker shims make a
run unusable for wall-clock comparison, and a clean wall is the entire point of this item. The
cost of that choice is §6.2.

### 2.1 Result

| | round 8 `w=1` | round 10 `w=1` | round 11 `w=1` (no-prefetch) | **round 12 `w=4`** |
|---|--:|--:|--:|--:|
| **end-to-end wall** | 13,762.5 s | **10,217.7 s** | 10,666.1 s | **5,854.9 s** |
| **Phase D stitch** | 1,185.4 s | 1,182.4 s | 1,239.2 s | **1,889.8 s** |
| Phase D share of wall | 8.6% | 11.6% | 11.6% | **32.3%** |
| peak RSS | 61.13 GB | 60.04 GB | 60.17 GB | **45.56 GB** |
| analysis output | 347 GB | — | 55.24 GB | **55.25 GB** |
| `overlay_slide.tiff` | 5.86 GB (broken) | 7.50 GB | 7.50 GB | **7.50 GB** |
| `config_hash` | `3d1087f2` | — | `d2ccc46b` | **`d2ccc46b`** |
| `cuda_alloc_conf` | `""` | `""` | `""` | **`expandable_segments:True`** |

Against the two candidate `workers=1` baselines:

```
vs round 10 (prefetch ON  = the shipped default)   10,217.7 / 5,854.9  =  1.745x
vs round 11 (prefetch OFF = the ablation control)  10,666.1 / 5,854.9  =  1.822x
```

**1.745x is the honest number**, because round 10's arm is the one that shares today's shipped
configuration. Round 8's `workers=4` measured 6,211 s, so in *absolute* terms this run is 1.061x
faster — rounds 9–11's wins did transfer to `workers=4`, just far less than they transferred to
`workers=1`.

### 2.2 Correctness veto — PASSED

Compared against round 11's `workers=1` run, the only full-slide run on the same `config_hash`:

| | r11 `w=1` | **r12 `w=4`** | delta |
|---|--:|--:|--:|
| `report.csv` rows | 356,226 | 356,220 | **−0.002%** |
| valid cells | 45,736 | 45,732 | −0.009% |
| HER2 / CEP17 dots | 148,405 / 108,464 | 148,385 / 108,458 | −0.013% / −0.006% |
| HER2:CEP17 ratio, mean HER2/cell | 1.37, 3.24 | **1.37, 3.24** | identical |
| verdict | Not amplified | **Not amplified** | identical |
| `stats` success / skipped | 10,800 / 16,765 | 10,801 / 16,764 | one tile |

This is **tighter than round 8's −0.01%** and well inside the same-code noise floor doc 21 §4
established. `scripts/overlay_pyramid_audit.py` on the produced 7.50 GB overlay: **PASS at every
level** — Predictor=1 across all 12 IFDs, per-level means 240.71 → 241.14, zero-fraction 0.0%,
i.e. byte-for-byte the same decode profile as round 10's and round 11's slides.

### 2.3 Doc 38 §2's Amdahl account, redone — right about the mechanism, wrong about the headline

Doc 38 §2 did its arithmetic on the round-8 pair and flagged that it might already be stale. It
was. Redoing it on measured current-code numbers:

```
parallel portion, w=1 (round 10)  = 10,217.7 − 1,182.4 = 9,035.3 s
parallel portion, w=4 (this run)  =  5,854.9 − 1,889.8 = 3,965.1 s
tile-parallel speedup             =  9,035.3 / 3,965.1 = 2.279x
```

**Doc 38 §2's central claim survives, and this is now its third independent confirmation.** Three
different routes to "pure tile-parallel speedup with Phase D excluded" now agree:

| source | method | value |
|---|---|--:|
| match24 crop (composition-matched, 55.9% background) | measured directly | **2.138x** |
| round 8 full slide | backed out of the wall | 2.510x |
| **round 12 full slide (this run)** | **backed out of the wall** | **2.279x** |

So doc 38 §2's answer to the user stands on its main point: **the real slide's 55.8% background
composition caps tile parallelism itself at ~2.1x–2.5x**, and that is not a tuning failure — a
background tile short-circuits on an empty core mask and has almost no GIL contention left to
recover, which is precisely why the tissue-dense crop measured 3.51x and the real slide never can.

**What doc 38 got wrong is the headline, and the reason is entirely Phase D.** Its §2 conclusion —
"the achievable ceiling for this pipeline is roughly ~2.5x–2.8x" — assumed Phase D stays near
1,185–1,199 s. It did not:

```
Phase D share of the w=4 wall     = 1,889.8 / 5,854.9  = 32.3%
Amdahl ceiling for that phase     = 1 / (1 − 0.323)    = 1.477x
```

Doc 38 §1.3 predicted the direction — *"as long as anything else gets optimized (or worker count
goes up), its share on `workers=4` will keep rising — eventually it becomes the dominant term"* —
and quoted `measurement/bottleneck-list.md` ⑤b saying the same. That has now happened, harder
than the trend implied, and it moves item 2 from "the only lever with a known small ceiling" to
**the dominant remaining term in the whole pipeline**.

### 2.4 Phase D grew 52%, and this document does not know why

The comparison that matters is against runs producing the **same artifact**. Round 8's 1,200.8 s
produced the 5.86 GB file doc 35 §1 later proved had broken pyramid levels; the honest
same-artifact comparisons are round 10's re-stitches (1,352.9 s and 1,315.0 s, both 7.50 GB) and
round 11's fresh full run (1,239.2 s, 7.50 GB). Against those, this run's **1,889.8 s for a
byte-size-identical 7.50 GB output is +40% to +52%**.

What is known:
- The output is not different: 7,500,478,515 B here vs 7,500,468,791 B (round 11) and
  7,500,511,297 B (round 8's re-stitch), and it audits identically at every level (§2.2).
- Peak RSS for the whole run was **45.56 GB**, *lower* than round 11's 60.17 GB. The stitch window
  itself (t=3,965 s → 5,855 s) ran at 35.0 GB mean / 45.6 GB peak.
- Phase D is unchanged code and runs single-threaded in the parent either way.

What is **not** known, and this document deliberately does not pick between them:
1. **Writeback/page-cache contention.** At `workers=4` the analysis wrote its 55 GB of output and
   27,565 `_stitch_scratch` tiles in 1.2 h instead of 2.5 h — roughly double the dirty-page rate,
   so the stitch starts against a much deeper writeback backlog.
2. **Ordinary full-slide run-to-run variance.** Doc 37 §5 already flagged that this project has
   **never established an error bar on a full-slide number** — every full-slide comparison in
   every round is n=1 vs n=1, and doc 37 left an unexplained regression on exactly that basis.

**This is n=1 and must be read as n=1.** It does not change §2.3's tile-parallel conclusion (which
three sources corroborate), but the specific figures 32.3% and 1.477x rest on a single
measurement of a quantity now known to move by ±50%. Settling it needs repeated full-slide runs,
which is the same ~3 h-each cost doc 37 §5 declined to pay.

## 3. Item 3 — batch-claiming the dynamic queue: closed negative

Doc 38 §3 item 3 asked for exactly one cheap measurement and pre-committed to the outcome: *"If it
comes back at the noise floor (likely …), close it out without building anything."*

`scripts/mp_queue_claim_probe.py` (new) reproduces `_run_tiles_multiprocess`'s IPC shape — spawn
context, feeder thread, `ctx.Queue()`, poison sentinels, one `("ok", …)` result message per tile
carrying a real `CellAnalysisResult` list — over the real 185×149 grid at the real 55.81%
background composition, with tissue clustered in a centred ellipse (scattering it would average
every batch out and hide the one failure mode worth looking for). Only the **claim** side varies
between arms; results stay one-message-per-tile in all of them.

**The `null` arm is decisive on its own.** With zero work per tile, its wall *is* the queue
round-trip cost, and it bounds what any claim-strategy change could ever save:

| claim strategy | wall (median of 5) | claim-side cost | vs `per_tile` |
|---|--:|--:|--:|
| `per_tile` (shipped) | **0.356 s** | 18.84 µs/tile | 1.000x |
| `batch:8` | 0.291 s | 2.08 µs/tile | 1.221x |
| `batch:32` | 0.288 s | 0.93 µs/tile | 1.233x |

```
entire claim cost for all 27,565 tiles  =   0.356 s
as a share of the measured w=4 wall     =   0.356 / 5,854.9  =  0.0061%
end-to-end Amdahl ceiling               =   1 / (1 − 0.000061) = 1.00006x
```

**That is four orders of magnitude below anything worth building.** The playbook's anti-pattern #2
("optimizing something under ~10% of total time") does not need a second opinion at 0.006%.

The modelled-work arm confirms it from the other side and adds the reason to *not* build it even
if it were free. Per-tile durations are the measured ones, compressed 50x so the probe stays
cheap — which is **conservative for this question**, because compressing the work while leaving
IPC at full cost inflates the claim overhead's share by exactly 50x:

| claim | wall (median of 2) | vs `per_tile` | tile spread (busiest − idlest worker) |
|---|--:|--:|--:|
| `per_tile` | 48.857 s | 1.0000x | 160, 493 |
| `batch:8` | 48.530 s | 1.0068x | 304, 512 |
| `batch:32` | 48.459 s | 1.0082x | **1,344, 1,011** |

Even with IPC's share inflated 50x, batch-claiming buys **0.7–0.8%** — about **0.015%** at real
per-tile durations, consistent with the null arm. Meanwhile the **tile spread grows monotonically
with K**: the load imbalance that doc 20 §2 Candidate B chose a dynamic queue to avoid is real,
visible, and gets worse exactly as the batch gets bigger. So the trade is a rounding error of
saving against a growing tail risk.

**Nothing was built. Doc 38 §1.4's read of this line was right**: no round's shortfall analysis
ever pointed at load balancing, and now there is a measurement saying why it never could.

## 4. Item 2 — pipelining Phase D's read, which reverses doc 35 §3.3

Doc 35 §3.3 closed Phase 2 negative but named one lever it had not built: *"If B's and C's reads
were fully hidden behind their own encode work — a threaded implementation nobody has written —
they would land at 42.5 s (**1.47x**) and 34.7 s (**1.80x**)."* Doc 38 §3 item 2 scheduled that
implementation, gated on running the cheap spike first. This is that spike.

`_prefetch_bands()` (new, in `scripts/stitch_probe.py`) fetches band k+1 on a background thread
while band k is encoded — the same shape as the pipeline's own `prefetch_tile_reads`, depth 1, so
at most two bands are ever resident. `--pipelined` runs each pipelined arm **interleaved with its
own serial control**, and nothing else changes between them (playbook anti-pattern #7).

### 4.1 The controls reproduce doc 35 §3.2 almost exactly

Run at doc 35's own 4.055 GP / 6,900-tile scale so the numbers are directly comparable:

| arm | doc 35 §3.2 | **this round** |
|---|--:|--:|
| read-only | 30.44 s | **30.53 s** |
| A `pyvips_tiffsave` (shipped) | 62.55 s | **62.40 s** |
| B `tifffile_cpu` | 79.33 s / 0.788x | **79.52 s / 0.785x** |
| C `tifffile_gpu_pyramid` | 70.71 s / 0.884x | **72.90 s / 0.856x** |

Four independent reproductions inside 3% is a working control group, so the new arms can be read
as the change and not as drift.

### 4.2 The new arms — the lever works

| arm | wall | read | pyramid | encode | container | vs A | audit |
|---|--:|--:|--:|--:|--:|--:|:--:|
| A `pyvips_tiffsave` (shipped) | 62.40 s | (fused) | (fused) | (fused) | (fused) | 1.000x | PASS |
| B `tifffile_cpu` | 79.52 s | 36.07 | 18.67 | 18.80 | 0.54 | 0.785x | PASS |
| **B + pipelined read** | **45.72 s** | **1.56** | 19.11 | 19.09 | 0.57 | **1.365x** | PASS |
| C `tifffile_gpu_pyramid` | 72.90 s | 36.56 | 6.98 | 23.24 | 0.55 | 0.856x | PASS |
| **C + pipelined read** | **39.47 s** | **4.23** | 6.23 | 22.87 | 0.56 | **1.581x** | PASS |

The read is very nearly gone: **36.07 → 1.56 s (95.7% hidden)** for B, **36.56 → 4.23 s (88.4%
hidden)** for C. Both land within 7–12% of doc 35's idealised ceilings (1.47x / 1.80x), which is
what "hidden, but not perfectly" looks like.

Every arm is shape-matched to the baseline (**11 pages, 128×128 tiles, identical level-0 shape**)
and passes the per-level decode audit. **Doc 35's disposition is reversed on its own terms**: the
two candidates that were 0.785x and 0.856x are now 1.365x and 1.581x, and the reason is exactly
the one doc 35 identified — pyvips was winning by overlapping the read with the encode, and the
streamed candidates were paying it serially. They no longer are.

### 4.3 What it is worth end-to-end — and the measurement that gates it

Translating the measured 1.581x onto §2.1's `workers=4` numbers:

```
Phase D today                      1,889.8 s of a 5,854.9 s wall
Phase D at the measured 1.581x     1,195.3 s        (saves 694.5 s)
new wall                           5,160.4 s   ->   1.135x end-to-end
workers=4 speedup would go         1.745x      ->   1.980x
```

That is above doc 35 §3.3's ~1.094x estimate and above doc 38 §3's expectation for this item —
because Phase D's share grew (§2.3), not because the lever got better.

**That projection had one obvious way to be wrong, so it was measured rather than argued.**
`stitch_probe.py` builds its input by hard-linking a pool of ~24 distinct tiles into 6,900
positions, so its source is **page-cache resident**. Real Phase D walks 27,565 *distinct* tiles,
46 GB, off disk — and pipelining can only ever save `min(read, other work)`, so if the real
uncached read dwarfs the rest, hiding it saves much less than the spike suggests. The absolute
per-pixel costs say that risk is real:

```
probe A:        4.055 GP in    62.4 s   =   15.4 s/GP
real Phase D:  16.201 GP in 1,889.8 s   =  116.6 s/GP     (7.6x)
```

### 4.4 The uncached read, measured on the real 27,565-tile source

`scripts/phase_d_real_read_probe.py` (new) hard-links the real 46 GB `overlay_annotated`
directory into a scratch dir and drains `stitch_probe.py`'s own band walk over it at the real
141,658×114,366 geometry — the read half of Phase D, alone, at full scale, from cold cache
(`/proc/<pid>/io` confirms **46.5 GB of `read_bytes`**, i.e. it really came off the disk):

```
REAL read-only:  981.00 s over 149 bands, 16.20 GP  (60.55 s/GP)
```

**The read share transfers almost exactly, and that is the result that matters:**

| | read | total | read share |
|---|--:|--:|--:|
| doc 35 §3.2, 4.055 GP, cache-resident source | 30.44 s | 62.55 s | **48.7%** |
| **this round, 16.20 GP, real source, cold** | **981.0 s** | 1,889.8 s | **51.9%** |

So the spike's *absolute* numbers do not extrapolate (7.6x per-pixel error — doc 27 §6.4 already
caught Phase D extrapolation being wrong by 1.8x, and this is worse), but **the one ratio that
drives the pipelining win does**. The read is about half of Phase D at both scales, on both a
cached synthetic source and a cold real one, which is exactly the regime where hiding it behind
the other half pays. The failure mode §4.3 was worried about — a read so dominant that
pipelining has nothing to hide it behind — **does not occur**.

Bounding it from the other side, the residual non-read work is `1,889.8 − 981.0 = 908.8 s`, so a
*perfectly* overlapped Phase D could not go below **981.0 s (1.926x)**, which puts the measured
spike's 1.581x inside a plausible band rather than beyond one:

| assumption | Phase D | `w=4` wall | end-to-end |
|---|--:|--:|--:|
| today | 1,889.8 s | 5,854.9 s | 1.000x |
| measured spike ratio (1.581x) | 1,195.3 s | 5,160.4 s | **1.135x** |
| perfect overlap (floor) | 981.0 s | 4,946.1 s | 1.184x |

**Doc 38 §4's gate — "if the cheap spike doesn't come back close to the ~1.09x signal, close it
out" — is passed comfortably, and §4.4 removes the main reason to distrust it.** Two caveats keep
this a *recommendation to build and re-measure*, not a result:

- The 981.0 s read is **cold**, whereas real Phase D reads tiles its own analysis phase wrote
  minutes earlier, so the in-situ read is somewhere at or below that figure. The 51.9% is
  therefore an upper bound on the read share, not a point estimate.
- Everything downstream of the spike is still a spike. Only a full-scale candidate run settles the
  end-to-end number, and this project's own history (doc 32 → doc 35) is one reversal at exactly
  that step.

One further condition, which doc 35 §3.4 recorded and then set aside: candidates B/C write
**1.89 GB against A's 2.44 GB** because they apply Predictor 2 where the shipped path applies
none. Doc 35 parked the QuPath open-and-render-identically check as *"no longer decision-relevant
— no candidate is being adopted."* With a candidate now viable, **it is decision-relevant again
and has still never been run.** No adoption without it.

## 5. Item 4 — not done, deliberately

Doc 38 §3 marks `workers≥5` as *"the one item in this document explicitly marked 'do not do
this'"*, and nothing this round measured argues for reopening it. If anything §2.3 strengthens the
case against: the marginal return on more workers is bounded by the same composition ceiling that
caps the tile-parallel arm at ~2.28x, while the term that actually dominates the wall (Phase D,
32.3%) does not scale with worker count at all. Adding a fifth worker cannot touch a third of the
wall.

This round adds **no new VRAM evidence** either way — see §6.2.

## 6. Doc 38 §4's decision gates, evaluated

### 6.1 The gates

| gate | outcome |
|---|---|
| *"If step 1's re-run comes back materially outside a reasonable noise band around 2.216x: stop and find out why before proceeding."* | **Fired.** 1.745x vs 2.216x is a 21% miss. §2.3/§2.4 do the "find out why": the tile-parallel arm is fine (2.279x, corroborated three ways); Phase D grew 40–52% and now dominates. §2.4 states plainly what is and is not known about that growth. |
| *"If step 2's cheap spike doesn't come back close to the ~1.09x signal: close it out."* | **Passed** — 1.581x on Phase D, projecting to **1.135x** end-to-end. The spike's main credibility risk (cache-resident source) was then closed by direct measurement: the read share is 51.9% at full scale vs 48.7% at crop scale (§4.4). Recommended to build and re-measure at full scale; not adopted on a spike. |
| *"If item 3 comes back at the noise floor, close it out without building anything."* | **Closed. Nothing built.** 0.006% of wall. |
| *"Any correctness veto: no per-cell correctness vote, no ship."* | **Passed everywhere.** §2.2 (−0.002% rows, identical verdict, pyramid audit clean at all 12 levels); every Phase D candidate arm shape-matched and level-audited (§4.2). |

### 6.2 The measurement gap this round introduced, stated rather than hidden

Passing `--worker-timings` would have voided the wall-clock (§2), so it was omitted — and that has
two consequences worth recording before anyone reads the metrics JSON literally:

- **No per-stage breakdown exists for this run.** Under `workers>1` the parent's monkeypatched
  timers only see the parent, so every bucket except `D_stitch_overlay` and the two `C_export_*`
  calls is empty. `gc_refreeze` reads `n_freezes: 0` for the same reason — the freezes happen
  inside the workers. That is an artifact of the measurement, **not** a regression.
- **No peak-VRAM figure for this run.** `--gpu-dmon` was not passed either, and
  `peak_cuda_reserved_gb` reads `0.0` because the parent never allocates on the device under
  multiprocessing. Two spot `nvidia-smi` samples during the run read 9,181 MiB and 5,017 MiB, but
  those are single samples and **not** peaks — round 8's 30,439 MB figure came from `dmon`
  sampling and this round does nothing to refresh or contradict it. Any `workers≥5` discussion
  still rests on round 8's and round 11's VRAM numbers, unchanged.

## 7. What is outstanding

1. **Build the pipelined read into `_stitch_overlay_slide` and re-measure at full slide scale.**
   §4 sized it (1.581x on Phase D, 1.135x end-to-end) and §4.4 removed the reason to distrust the
   spike, but every number downstream of a spike in this project's history has needed a full-scale
   confirmation, and once it reversed outright (doc 32 → doc 35).
2. **An error bar on a full-slide number.** §2.4's +40–52% Phase D growth is n=1 vs n=1, like every
   other full-slide comparison this project has ever made (doc 37 §5 flagged the same gap and
   declined the ~3 h-per-run cost). Until it exists, 32.3% and 1.477x are single measurements of a
   quantity now demonstrated to move a lot.
3. **The QuPath open-and-render pass on a Predictor-2 candidate** (doc 35 §3.4) — parked as
   irrelevant when no candidate was viable, decision-relevant again now (§4.3).
4. **`run_batch`'s `workers: int = 4` default** (§1) — latent footgun, no live call site affected,
   still unfixed and still out of scope.

## 8. Reproduce

```bash
# §2 — the full-slide workers=4 run (1 h 38 m on the reference host)
.venv/bin/python scripts/perf_measure.py \
  --ihc  /home/taro/full_wsi_validation/_conformed/ihc_conformed_141658x114366.tiff \
  --dish /home/taro/full_wsi_validation/_conformed/dish_conformed_141658x114366.tiff \
  --output /home/taro/r12_fullwsi_w4 --label r12_fullwsi_w4 \
  --mp-workers 4 --stream-precut --metrics-dir /home/taro/r12_fullwsi_w4/_metrics

# §2.2 — correctness veto on the produced overlay
.venv/bin/python scripts/overlay_pyramid_audit.py /home/taro/r12_fullwsi_w4/overlay_slide.tiff

# §3 — queue claim cost (the null arm is the decisive one)
.venv/bin/python scripts/mp_queue_claim_probe.py --work null --workers 4 --repeats 5 \
    --claims per_tile,batch:8,batch:32 --out .../queue_claim_null.json
.venv/bin/python scripts/mp_queue_claim_probe.py --work sleep --time-scale 0.02 \
    --workers 4 --repeats 2 --claims per_tile,batch:8,batch:32 --out .../queue_claim_sleep.json

# §4 — Phase D pipelined read, at doc 35 §3.2's directly comparable 4.055 GP scale
.venv/bin/python scripts/stitch_probe.py \
    --overlay-src /home/taro/full_wsi_validation/fullwsi_w1/overlay_annotated \
    --candidates --pipelined --slide-w 70909 --slide-h 57183 \
    --root /home/taro/r12_phase_d --out .../phase_d_pipelined_4gp.json

# §4.4 — the uncached read on the real 27,565-tile source (16 min, reads 46 GB cold;
#         drop caches or use a source not read recently, or it measures the page cache)
.venv/bin/python scripts/phase_d_real_read_probe.py
```

Metrics for this round are archived under
[`measurement/_metrics_r12/`](./measurement/_metrics_r12/).
