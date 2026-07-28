# 31 — `gc.collect` round 2 (periodic re-freeze): implementation and results

> Implements [`28-gc-collect-round2-plan.md`](./28-gc-collect-round2-plan.md) — **Option H,
> periodic re-freeze**. Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md).
> Round 9. Companion documents this round:
> [`32-phase-d-gpu-port-implementation.md`](./32-phase-d-gpu-port-implementation.md) (doc 29),
> [`33-tile-read-io-implementation.md`](./33-tile-read-io-implementation.md) (doc 30).
>
> **Round 9's most important change is in neither of this document's own plans**: chasing a
> QuPath crash found that every `overlay_slide.tiff` written before this round decodes its
> pyramid levels as noise. Fixed and tested — see
> [`32-phase-d-gpu-port-implementation.md`](./32-phase-d-gpu-port-implementation.md) §5.1(b).

## 0. Outcome in one paragraph

Option H shipped, in the shape doc 28 §3 recommended and for the reason it gave. The plan's one
self-declared unverified assumption — *"`gc.freeze()`'s own cost is not necessarily free at these
object counts… **this is the one unverified assumption in this whole plan and must be the first
thing measured**"* — was measured first, and resolves **decisively in Option H's favour**:
`gc.freeze()` is an **O(1) linked-list splice**, costing 0.00004–0.0006 ms per call regardless of
how much is already frozen or newly tracked. 27,565 freeze calls cost **1.2 milliseconds in
total**. Option I (the pickle-to-`bytes` fallback) is therefore **not needed and not built**, per
the plan's own stopping rule. What the measurement *also* found is a risk doc 28 §4 did not
identify, and it is the reason the shipped cadence is not "every tile" — see §4.

**Not claimed:** the full-slide win. Doc 28 §5 Exp 4 (a 3.82 h `workers=1` run) was not run, and
this document does not report a number as if it had been. §7 gives a projection, labelled as one.

## 1. What shipped

Three files, ~60 lines net.

**`backend/algorithms/hybrid/m0_module/m0_tile_runner.py`** — new `_PeriodicFreezer`, and
`_frozen_gc_generation()` now yields one:

```python
class _PeriodicFreezer:
    def __init__(self, every_cells: int) -> None:
        self.every_cells = every_cells
        self.pending = 0
        self.n_freezes = 0

    def note(self, n_cells: int) -> None:
        self.pending += n_cells
        if self.pending >= self.every_cells:
            gc.freeze()
            self.pending = 0
            self.n_freezes += 1


_REFREEZE_EVERY_CELLS = 5_000
```

`_frozen_gc_generation()` is otherwise untouched: one `gc.freeze()` on entry, one
`gc.unfreeze()` in `finally`. Freezing is **additive** — each call merges whatever is currently
tracked into the permanent generation on top of what is already there — so N freezes still need
exactly one unfreeze, on both the success and the fail-fast path.

**`backend/algorithms/hybrid/hybrid_pipeline.py`** — `run_batch`'s single-process loop binds the
freezer and feeds it from `_collect`:

```python
def _collect(entry: Tuple[int, int, Future]) -> None:
    e_ax, e_ay, fut = entry
    owned = fut.result()
    _record(e_ax, e_ay, owned)
    refreezer.note(len(owned))

with _frozen_gc_generation() as refreezer, \
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tile-cpu") as pool, \
        ...
```

**`scripts/verify_gc_freeze.py`** — invariant guard extended, not replaced (doc 28 §6's
requirement). See §5.

### 1.1 Why the hook is in `_collect` and not `_record` — a scoping correction to the plan

Doc 28 §1 calls `_record` *"the single parent-side convergence point for both the single-process
and multiprocess paths"*, and that is exactly why the re-freeze must **not** go there.

Traced against the current code (post-M0-split): at `workers>1`, `_record` is invoked by
`_run_tiles_multiprocess(on_tile=_record)` in the **parent**, which is *outside* any
`_frozen_gc_generation()` scope. A `gc.freeze()` there would be a freeze with no matching
unfreeze — precisely the unbounded-RSS-growth failure mode in a long-lived API server that doc 15
built the context manager to prevent. `_collect` is the single-process path's own convergence
point and runs only inside the `with` block, so the hook is safe there by construction.

**This also narrows the scope of the fix, and the plan does not say so explicitly:** the
regression is a `workers=1` phenomenon. The multiprocess *workers* run their own
`_frozen_gc_generation()` and their own per-tile `gc.collect()`
(`m0_multiprocess.py:135,160`), but they accumulate nothing — each tile's result goes straight
out over `result_q` and is dropped. The parent accumulates `per_tile_owned` but never collects
per tile. So at `workers>1` there is no growing scan scope to bound, and nothing to fix. Doc 28
§5 Exp 4 asking for a **`workers=1`** full-slide re-run is consistent with this; the reasoning
just was not written down.

## 2. Exp 0 — the climb, reproduced synthetically

`scripts/gc_refreeze_probe.py` (new) drives `run_batch`'s accumulate-plus-collect structure with
the **real** `CellAnalysisResult` dataclass, at the real slide's shape (27,565 tiles, 55.8%
background, 29.24 cells per tissue tile → 356,255 cells), with no GPU, model, or slide data.

| arm | collect total | per-call, first → last 5% |
|---|--:|--:|
| frozen once (today's shipped state) | **369.4 s** | 0.001 → **31.2 ms** |
| never frozen (control, capped at 5,000 tiles) | 81.2 s | 12.3 → 21.2 ms |

The shape doc 27 §6.4 measured is reproduced: with a single entry freeze, per-call cost climbs
without bound as results accumulate. The **absolute** figure is ~2.6× below the pipeline's
measured 80.5 ms/call, because the harness deliberately omits the transient per-tile numpy/torch
garbage the real loop creates. That offset is additive and constant per tile, so it cannot change
the *shape* the cadence is designed against — but it does mean no number in §2–§4 may be quoted
as a pipeline figure.

## 3. Exp 1 — `gc.freeze()`'s own repeated cost: the plan's open question, closed

Cadence 1 freezes ~29 new objects per call against a permanent generation growing to hundreds of
thousands; cadence 1,000 freezes ~29,000 at a time. If freeze were O(cumulative), per-call cost
would rise with run length; if O(new), with batch size.

| re-freeze cadence | freeze calls | mean per call | **total freeze cost, whole slide** |
|---|--:|--:|--:|
| every 1 tile | 27,565 | 0.000043 ms | **0.0012 s** |
| every 10 tiles | 2,757 | 0.000055 ms | 0.0002 s |
| every 100 tiles | 276 | 0.000088 ms | 0.00002 s |
| every 1,000 tiles | 28 | 0.000596 ms | 0.00002 s |

Per-call cost is flat to within sub-microsecond noise across a 1,000× range of batch sizes. This
is the expected behaviour of CPython's `gc_freeze_impl`, which splices the generation lists into
the permanent generation — a constant-time pointer operation, not a walk. **Freeze is free.**

Consequences, stated plainly because they change the plan's conclusions:

- Doc 28 §3's ceiling for Option H stands as written, with no recomputation needed.
- Doc 28 §3's Option I (pickle each tile's results to `bytes` so they are never GC-tracked) was
  gated on this measurement coming back unfavourable. It did not. **Option I is not built.**
- Doc 28 §3's *"calling it too often could reproduce the exact problem it's solving"* is
  disproven. Too-frequent freezing is not a performance risk. It is a different kind of risk
  (§4).

## 4. Exp 2 — cadence sweep, and the finding that actually set the cadence

Total collect + freeze cost across the full 27,565-tile batch:

| keyed by tile count | combined | | keyed by accumulated cells | combined | freezes |
|---|--:|---|---|--:|--:|
| every 1 | 0.004 s | | every 100 | 0.031 s | 3,046 |
| every 5 | 0.020 s | | every 500 | 0.164 s | 676 |
| every 25 | 0.105 s | | every 2,000 | 0.757 s | 176 |
| every 100 | 0.490 s | | every 10,000 | 4.292 s | 35 |
| every 500 | 2.942 s | | every 50,000 | 27.833 s | 7 |
| every 2,000 | 12.072 s | | | | |
| **none (today)** | **416.7 s** | | | | |

**Doc 28 §5 Exp 2's stated pass bar was "a clear minimum-cost region should exist (too-frequent
freezing pays its own O(n) cost too often; too-infrequent reverts toward today's climb)". There
is no minimum.** Cost is monotonically decreasing in freeze frequency, because the first half of
that sentence is false — freeze never pays an O(n) cost to trade against. The plan predicted a
U-shape and the data is a slope. Recording this as the plan's prediction failing, per the
playbook's own discipline, rather than quietly picking a number off the curve.

So the cadence is **not** set by the collect-cost curve, where every option from 100 to 10,000
cells is negligible against a 13,762 s wall. It is set by a risk doc 28 §4 did not identify:

> **`gc.freeze()` freezes everything currently tracked, not only the results we want pinned.**
> Doc 28 §4 argues there is no RSS re-validation burden because `CellAnalysisResult` and its
> siblings contain no reference cycles. That is true of the results, and it is not the whole
> object graph the call touches. Any *transient* object that happens to be alive at the freeze
> instant — the in-flight `ChunkResult` on the background arm, torch/cellpose internals — is also
> moved to the permanent generation, and if it later becomes part of an unreachable cycle it is
> never collected until `run_batch` returns. Each freeze is one exposure to that. Freezing every
> tile is 27,565 exposures; freezing every 5,000 accumulated cells is ~71.

`_REFREEZE_EVERY_CELLS = 5_000` sits where the collect cost is still trivial (~2.1 s across the
slide, interpolated between the 2,000- and 10,000-cell rows) and the exposure count is small.
Keying on **cells rather than tiles** is doc 28 §3's own recommendation and is load-adaptive for
free: the 55.8% of tiles that are background contribute zero cells, so quiet regions of the slide
trigger no freezes at all — verified as an explicit test (§5, check 2b).

## 5. Invariant guard — `scripts/verify_gc_freeze.py`

Extended, per doc 28 §4's requirement that the "freeze called exactly once" assertion be
*updated, not dropped*:

```
=== 1. cadence unchanged (control: none) ===
  PASS  current collects once per tile (freeze changes scope, not cadence): got 441, want 441
=== 2. freeze/unfreeze cardinality (API-server leak guard) ===
  PASS  freeze called once at entry plus once per cadence crossing: got 3, want 3
  PASS  periodic re-freeze actually fired: got True, want True
  PASS  unfreeze called exactly once: got 1, want 1
  PASS  nothing left frozen after run_batch: got 0, want 0
=== 2b. no cells accumulated -> no re-freeze (cadence is cell-keyed) ===
  PASS  all-background batch freezes only at entry: got 1, want 1
=== 3. unfreeze survives a mid-loop fail-fast ===
  PASS  re-freezes happened before the failure: got True, want True
  PASS  unfreeze called exactly once on the fail path: got 1, want 1
  PASS  nothing left frozen after fail-fast: got 0, want 0
```

Changes worth noting:

- The expected freeze count is **read from `TR._REFREEZE_EVERY_CELLS`**, not duplicated, so
  retuning the cadence cannot silently invalidate the assertion.
- The harness previously stubbed `_process_precut_tile_cpu` to return `[]`; with a cell-keyed
  cadence that would never fire, so it now fabricates 29 real `CellAnalysisResult`s per tile.
- Section 3 previously failed on tile 1; it now fails at the midpoint, so the fail-fast path is
  exercised with re-freezes already behind it.
- **Two pre-existing breakages fixed.** `run_batch`'s default is `workers=4`, so the script was
  silently exercising the multiprocess path (where its monkeypatched stubs do not apply) and
  aborting; it now passes `workers=1` explicitly. And `--control-ref` is now optional: **no ref
  predating the freeze change survives in this repository's history** (it was squashed away), so
  the control arm is unobtainable. The invariant it protected — one collect per tile — is now
  asserted directly against the tile count instead, which is strictly stronger than comparing to
  another implementation.

## 6. Exp 3 — real data, real objects, at a scale that shows the mechanism

Doc 28 §5 is explicit that the 441-tile crop is *"the wrong scale to validate this fix on"*. It
is, for the *magnitude*. It is the right scale to show the *mechanism* works on real
`CellAnalysisResult` objects produced by real inference, which is what Exp 3 asks for. Two runs
on the `match24` composition-matched anchor (576 tiles, 55.9% background), differing **only** in
whether Option H is active (`scripts/perf_measure.py --no-refreeze` neuters the cadence and
leaves the doc-15 entry freeze in place, so the arms differ in exactly one thing):

| | control (Option H ablated) | Option H |
|---|--:|--:|
| per-call `gc.collect()`, decile means (ms) | 0.47 → 0.50 → 0.53 → 0.57 → 0.66 → 0.73 → 0.80 → 0.93 → 1.11 → **1.37** | 0.47 → 0.50 → 0.53 → 0.57 → 0.69 → 0.72 → 0.83 → 0.97 → 0.71 → **0.33** |
| last decile ÷ first decile | **2.94 (climbing)** | **0.71 (flat, then falls)** |
| `B4_gc_collect` total | 0.447 s | 0.363 s (−18.8%) |
| peak RSS | 4.338 GB | 4.309 GB |
| `stats` | 224 success / 352 skipped | 224 success / 352 skipped |
| end-to-end wall | 193.6 s | 200.2 s |

The decile series is the result: the control reproduces doc 27's regression in miniature (cost
per call rising monotonically with accumulation), and Option H flattens it and then visibly drops
it when the single cadence crossing at ~5,000 cells fires. This required adding a per-call
`gc.collect()` series to `perf_measure.py` — the bucket **total** cannot distinguish "flat and
cheap" from "climbing but short", which is exactly how this regression survived rounds 1–7.

Two things reported straight rather than spun:

- **The absolute saving here is 0.084 s, ~0.04% of wall.** Expected and stated in advance by the
  plan; a crop accumulates ~7,400 cells where the slide accumulates 356,255.
- **The Option H arm's end-to-end wall was 3.4% *higher* (200.2 s vs 193.6 s), and that number
  is noise — demonstrably so.** The mechanism under test accounts for 0.084 s of that 6.6 s gap,
  which already made it implausible as a regression signal. It was then settled directly: doc 33
  §3 later ran six more runs of this same anchor and measured the *same* configuration at
  **188.2 s ± 0.22 s**, i.e. both of these two runs sit above a much tighter later distribution
  and differ from it by more than they differ from each other. They were the first two runs
  against a cold page cache. **No wall claim is made at this scale**, in either direction; the
  crop's job here is the mechanism plus the correctness and RSS invariants, and a single pair of
  cold runs cannot resolve a 6 s difference. (Lesson worth carrying: this pair was very nearly
  written up as "Option H costs 3.4%". Three interleaved repeats per arm cost 20 minutes and
  turned a misleading number into a tight one — doc 33 §3 did exactly that for its own change.)

**RSS did not regress** (4.338 → 4.309 GB), which is the first evidence against §4's pinning
concern — but at ~71× fewer freezes than a full slide would see, it is weak evidence. It remains
the thing to watch in Exp 4.

## 7. Exp 4 — not run, and what is projected instead

Doc 28 §5 Exp 4 is a full-slide `workers=1` run: **3.82 hours**. It was not run, so the win at
the scale that motivated this plan is **projected, not measured**:

- The harness's no-re-freeze arm costs 416.7 s where the pipeline measured 2,218.4 s — a 5.3×
  factor, being the transient garbage the harness omits.
- At the shipped 5,000-cell cadence the harness's collect cost falls from 416.7 s to ~2.1 s, i.e.
  the accumulation-driven component is essentially eliminated rather than reduced.
- What remains is the per-call floor doc 16 already measured on real data: ~1.2 ms × 27,565 calls
  ≈ **33 s**, against today's 2,218.4 s.
- That is ~2,185 s off a 13,762 s wall = **15.9%**, an end-to-end ceiling of about **1.19×**.

Treat 1.19× as a projection with a measured mechanism behind it, not a result. The one number
that would settle it is Exp 4, and it is the natural companion to doc 33's Option K, which needs
the same run.

## 8. Disposition

| doc 28 item | status |
|---|---|
| Option H — periodic re-freeze | **SHIPPED**, cell-keyed, 5,000 cells |
| Option I — pickle results to `bytes` | **NOT BUILT** — gated on Exp 1 coming back unfavourable; it came back favourable |
| Option J — spill `per_tile_owned` to disk under `checkpoint=True` | **Still flagged, not scheduled**, exactly as the plan left it |
| Exp 0 / 1 / 2 | **DONE** — `scripts/gc_refreeze_probe.py`, raw JSON in `measurement/_metrics_r9/gc_refreeze_probe_full.json` |
| Exp 3 | **DONE** — mechanism, correctness and RSS confirmed on real data; no wall claim |
| Exp 4 (full slide) | **NOT RUN** — 3.82 h; §7 projects, does not measure |
| Correctness veto | **PASSED** — identical `stats` across arms |
| `verify_gc_freeze.py` cardinality guard | **UPDATED** to freeze N≥1 / unfreeze exactly once |

**Follow-ups this created, for the next round:**

1. Exp 4 (full-slide `workers=1`) is the only outstanding item, and should be run together with
   doc 33's Option K and Option L ablation so one 3.8 h run settles all three.
2. Watch peak RSS on that run specifically for §4's pinning concern; ~71 freezes is the real
   exposure count, and the crop only exercised one.
3. `_REFREEZE_EVERY_CELLS` is a module constant, not config. That is deliberate (it is not a
   user-facing knob), but if Exp 4 shows RSS sensitivity it is the dial to turn.
