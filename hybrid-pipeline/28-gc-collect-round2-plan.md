# 28 — `gc.collect` round 2: fixing the full-slide accumulation regression (design plan)

> **Design-only document — no pipeline code changed here.** Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose. Picks up where
> [`14-gc-collect-frequency-plan.md`](./14-gc-collect-frequency-plan.md) /
> [`15-gc-collect-frequency-implementation.md`](./15-gc-collect-frequency-implementation.md) /
> [`16-gc-collect-frequency-result.md`](./16-gc-collect-frequency-result.md) left off. Those three
> documents shipped `gc.freeze()` (Option C) and closed the item on the 441-tile crop. This
> document exists because [`27-remaining-work-implementation.md`](./27-remaining-work-implementation.md)
> §6.4 found, on the first real full-slide run, that the win **does not survive to production
> scale** — see doc 16's own round-8 addendum at the top of that file, plus
> [`19-open-backlog.md`](./19-open-backlog.md) item 6b and
> [`DISCOVERED-NOT-IMPLEMENTED.md`](./DISCOVERED-NOT-IMPLEMENTED.md) #43. This plan does not
> re-litigate doc 14's already-closed Options A/D/E/F/G (fixed-N batching, gen-limited collection,
> adaptive triggers, `gc.disable()`, streaming the merge) — those addressed *call frequency* and
> were shown in doc 16 §4.1 to be aimed at the wrong variable. This is a **second** problem with
> the same call site, and the fix is in the same family as the one that already shipped (scan
> *scope*, not call frequency), continued one step further.

## 0. Recap — the regression, in one paragraph

`gc.freeze()` (doc 15) wraps `run_batch`'s whole tile loop and moves everything tracked **at that
moment** — the three GPU models, their weights, `stats`, the empty `per_tile_owned` list, the
`_collect` closure — into a permanent generation the collector never scans again. Doc 16 measured
this cutting `gc.collect()` from 83.2 ms to 1.2 ms per call on a 441-tile crop and shipped it
unconditionally. It works exactly as designed — **for objects that exist at freeze time.** But
`run_batch`'s tile loop keeps running for hours after that one `gc.freeze()` call, and every tile
appends a new, never-frozen object to a list that itself *was* frozen. At full-slide scale
(27,565 tiles) that accumulation is **356,255 `CellAnalysisResult` dataclasses**, and every one of
27,565 subsequent `gc.collect()` calls has to walk all of them that exist so far. Doc 27 measured
this landing back at **80.5 ms/call, 2,218.4 s = 16.1% of wall** — indistinguishable from the
pre-`gc.freeze()` cost doc 16 eliminated. On the 441-tile crop that validated the original fix,
the accumulated list tops out around ~6,000 objects, which is why the regression was invisible
then and is the third-largest cost in the pipeline now.

## 1. Root cause, verified against the current code (not assumed)

Read directly via `codegraph_explore`, `backend/algorithms/hybrid/hybrid_pipeline.py`:

```python
# _record — the single parent-side convergence point for both the
# single-process and multiprocess paths (hybrid_pipeline.py:1114-1118)
def _record(abs_x: int, abs_y: int, owned: List[CellAnalysisResult]) -> None:
    per_tile_owned.append((abs_x, abs_y, owned))
    stats["skipped" if len(owned) == 0 else "success"] += 1
    if checkpoint:
        _checkpoint_save(ckpt_dir, abs_x, abs_y, owned)
```

`per_tile_owned` is created before the loop (so the empty list object *is* frozen), but each
`(abs_x, abs_y, owned)` tuple appended by `_record` — and every `CellAnalysisResult` inside
`owned` — is a **new** object created after `gc.freeze()` ran. Per doc 15 §2.2's own observation,
already written down and correct: *"Objects appended to a frozen list are not themselves frozen
and are collected normally; they stay alive only because the frozen list genuinely references
them."* That is precisely the mechanism this plan addresses — those objects don't leak, they just
never stop being *scanned*.

`CellAnalysisResult` itself (`hybrid_data_types.py:29-50`) is a flat `@dataclass` of scalar fields
only — `int`, `float`, `bool` — with no back-references, no nested containers, no cycles. Verified
directly, not assumed: this matters because it means there is **no correctness risk** in pinning
these objects permanently for the rest of the batch (§4 below) — they contain nothing that could
form a reference cycle needing active collection before the batch ends anyway.

## 2. Why this wasn't caught by doc 14/16's own re-validation requirement

Doc 14 §2 required an RSS re-validation for every option **except Option C**, reasoning that
`gc.freeze()` "does not change collection frequency... no RSS re-validation burden beyond the
standard check" (doc 16 §4.4). That reasoning is correct as far as it goes — freeze doesn't change
*how often* `gc.collect()` runs, and RSS itself was never the problem here. What neither document
anticipated is that freeze's *benefit* (cheap scans) has its own hidden scope limit: it only
covers the object graph that exists at the moment it's called, and `run_batch`'s dominant
long-lived container is built entirely **after** that moment, by design (the whole point of
`per_tile_owned` is to accumulate across the batch for the final global merge in `_finish_batch`).
This is a scope problem, not a frequency or memory problem, and neither of doc 14's two
re-validation questions (absolute RSS bound, sawtooth growth shape) is the right instrument for
detecting it — it only shows up in the **per-call cost trendline over the length of one long
batch**, which no round before 8 ever ran.

## 3. Design space

### Option H — periodic re-freeze (primary, recommended)

Call `gc.freeze()` again, periodically, from inside the existing `_frozen_gc_generation()` scope
(the same context manager doc 15 built), so tiles' worth of accumulated `per_tile_owned` entries
get folded into the permanent generation and stop being rescanned. Concretely: a counter inside
the tile loop, and every `N` tiles (or every time accumulated cell count crosses a threshold —
see the cadence question below), one extra `gc.freeze()` call. No `gc.unfreeze()` in between —
freezing is additive; each call moves whatever is *currently* tracked into the permanent
generation on top of what's already there. The single `finally: gc.unfreeze()` at the end of the
`with` block (unchanged from doc 15) still correctly restores the process to its original state on
both the success and fail-fast paths, exactly as it does today.

**Why this is the cheapest lever, not just the first one tried.** It doesn't touch call
frequency at all — `gc.collect()` still runs once per tile, same call site, same semantics as
today. It's a strict continuation of the exact mechanism doc 16 already measured, ablated, and
shipped: same risk profile, same correctness argument (§1's "no cycles" finding applies to every
future append, not just the ones existing today), same "no RSS re-validation burden" reasoning
(§4 below re-derives this explicitly rather than assuming it transfers). The code change is a
counter and one extra line inside a loop that already exists.

**Ceiling.** If re-freeze cadence is tuned so no more than a small, roughly-constant number of
objects accumulate between freezes, per-call `gc.collect()` cost should stay close to doc 16's
crop-scale figure (≈1.2 ms) for the whole batch instead of climbing to 80.5 ms by the end. That
would take the 2,218.4 s / 16.1%-of-wall cost down toward the low single digits of seconds —
comparable in relative terms to what Option C already delivered once (doc 16: 36.71 s → 0.52 s at
441 tiles), just re-applied at a scale where it currently fails to hold. This is very plausibly
the single largest remaining lever in the pipeline measured in raw seconds — bigger than Phase D
(doc 29) at `workers=1`, and cheaper to build than either Phase D or the tile-read item (doc 30).

**What must actually be measured, not assumed:**

1. **`gc.freeze()`'s own cost is not necessarily free at these object counts.** Doc 15/16 never
   had to measure it because it was called exactly once, on a small object graph (the three model
   graphs), and its one-time cost was invisible against a multi-minute run. Called repeatedly
   against a growing `per_tile_owned`, freeze itself must at minimum touch every object it moves —
   plausibly the same order of cost as the collection it's meant to avoid. If freeze's own cost
   scales with the *cumulative* tracked-object count rather than just the *newly tracked since last
   freeze* count, calling it too often could reproduce the exact problem it's solving. **This is
   the one unverified assumption in this whole plan and must be the first thing measured** (§5 Exp
   0/1), not the cadence sweep.
2. **The right cadence variable is cell count, not tile count** — the same lesson doc 14 §1 Option
   E already flagged for a different reason (`bottleneck-list.md`'s "growth tracks accumulated
   cell count, not tile count"). A fixed tile-count `N` tuned against one crop's cell density does
   not generalize across the slide the way a threshold on `sum(len(o) for _, _, o in
   per_tile_owned)` (already available in `_record`'s own scope, effectively free to read) would.
   Recommend keying cadence off accumulated object count from the start, not tile count, so this
   plan doesn't repeat doc 14 Option A's crop-tuned-`N` fragility.

### Option I — keep accumulated results structurally out of GC's reach (fallback)

Instead of periodically re-freezing tracked objects, stop tracking them in the first place: have
`_record` immediately pickle each tile's `owned` list to a `bytes` blob and store that in
`per_tile_owned` instead of the live `CellAnalysisResult` objects. `bytes` objects are immutable
and never GC-tracked, so this removes the scan cost entirely rather than bounding it, at zero
`gc.freeze()`-cadence risk. `_finish_batch`'s global merge would need to unpickle before its
existing sort/renumber pass.

This is explicitly a **fallback**, not a co-equal option, per the same "cheapest first" discipline
every other document in this folder uses:

- It touches more of the hot path (`_record` and `_finish_batch` both change, vs. Option H's
  isolated counter) and adds a pickle/unpickle round trip for every tile's results even when
  `checkpoint=False` — today that serialization only happens when checkpointing is explicitly
  requested (`_checkpoint_save`, round 8).
- It's a larger diff for what should be the same ceiling as Option H, if Option H's cadence
  question (above) resolves favorably.
- **Only reach for this if Option H's own measurement (§5 Exp 0/1) shows `gc.freeze()`'s repeated
  cost doesn't amortize well** — i.e., if the thing Option H assumes is cheap turns out not to be.

### Option J — spill `per_tile_owned` to disk when checkpointing is already on (opportunistic, not scheduled)

When `run_batch(checkpoint=True)` (round 8's feature, always on in `full_wsi_validate.py`), every
tile's `owned` is *already* durably pickled to `output_dir/_resume/tile_x{ax}_y{ay}.pkl` by
`_checkpoint_save` before `_record` appends the same data to `per_tile_owned` in RAM. In that
configuration specifically, the in-memory copy is redundant with the on-disk copy the moment it's
written, and `_finish_batch`'s global merge could stream tiles back from `_resume/*.pkl` instead
of holding them live for the whole batch — roughly halving the peak tracked-object count by
construction, independent of any `gc.freeze()` cadence question.

This is flagged, not scheduled: it only benefits the `checkpoint=True` path (the API's default
`checkpoint=False` single-tile-request path gets nothing from it), and it's a bigger structural
change (turns `run_batch`'s accumulate-then-merge pattern into a spill-and-reload pattern) than
either H or I for a benefit Option H should already capture generally. Recorded here so it isn't
rediscovered from scratch if H's cadence tuning ever turns out to interact badly with the
checkpoint path specifically.

## 4. Correctness and the memory-bounded invariant

**Correctness veto.** Freezing (repeated or not) only changes what the collector *scans*, never
what's *reachable* — this is the same argument doc 16 §4.3 already made and verified end-to-end
(13,148 vs. 13,149 cells, nearest-centroid matched, max|Δ|=0 on every field for matched cells).
Nothing about calling `gc.freeze()` more than once changes that argument; it applies per-call, not
once globally. The re-validation this plan actually owes is narrower: confirm the **existing**
correctness check still passes with periodic re-freeze in place (cheap — it's the same
`report.csv` comparison every prior round already runs).

**RSS / memory-bounded invariant.** Per §1, `CellAnalysisResult` and its siblings
(`CellDotResult`, `DetectedDot` — all checked directly in `hybrid_data_types.py`) contain no
cycles, so nothing about freezing them early changes whether or when they'd otherwise have been
collected: they live until `run_batch` returns regardless, exactly as today. This is the same
"doesn't touch cadence, so no RSS re-validation burden" argument doc 16 §4.4 used for the original
Option C, and it transfers cleanly to periodic re-freeze because periodic re-freeze **also**
doesn't touch cadence (`gc.collect()` still runs once per tile) — only scan scope changes, twice
now instead of once. The one genuinely new thing to check, because it's new to *this* round of the
change: `gc.freeze()` called repeatedly must still be exactly paired with a **single**
`gc.unfreeze()` in `finally`, on both the success and fail-fast paths — `scripts/verify_gc_freeze.py`
(doc 15 §4) already asserts "freeze called exactly once"; that assertion needs updating to "freeze
called N≥1 times, unfreeze called exactly once" once this ships, not dropped.

## 5. Recommended experiment matrix

Cheapest-first, per the playbook. The 441-tile crop that validated the *original* fix is
explicitly the wrong scale to validate *this* fix on — the whole point is that the bug is
invisible there. A synthetic stress harness is required before spending any multi-hour real run.

| # | What | Purpose | Pass bar |
|---|---|---|---|
| 0 | **Synthetic accumulation harness.** Extend `scripts/verify_gc_freeze.py`'s existing pattern (heavy stages stubbed, no GPU/model/slide data needed) to also fabricate `CellAnalysisResult`-shaped garbage at a chosen cells-per-tile rate and drive the real `_frozen_gc_generation()` / `gc.collect()` loop structure for a chosen tile count (e.g. 500 / 5,000 / 27,565 tiles, all in well under a minute since nothing GPU-bound runs) | Reproduce doc 27's 1.2 ms → 80.5 ms/call climb cheaply, without a multi-hour run, and get a cost-vs-accumulated-object-count curve to design cadence against | Curve reproduces the qualitative shape (cost grows with accumulated count) — doesn't need to hit 80.5 ms exactly, since the harness's fabricated objects and real ones may differ slightly in size |
| 1 | **Measure `gc.freeze()`'s own repeated cost** on the same harness — call it every 1 / 10 / 100 / 1,000 tiles and time the freeze calls themselves, not just the collections | Resolves §3 Option H's one open assumption before designing around it | If freeze cost per call scales with *cumulative* tracked count rather than *since-last-freeze* count, Option H's ceiling must be recomputed before proceeding — don't assume, this is the step that finds out |
| 2 | **Cadence sweep** on the harness: total (fabricated) `gc.collect()` + `gc.freeze()` wall cost as a function of re-freeze interval `N` (tile-count-keyed and cell-count-keyed variants both), at the full-slide-equivalent object count | Pick 1-2 candidate cadences | A clear minimum-cost region should exist (too-frequent freezing pays its own O(n) cost too often; too-infrequent reverts toward today's climb) |
| 3 | **Confirm on real (non-synthetic) data at a cheap-but-non-trivial scale** — a crop concatenated several times back-to-back (same technique doc 14 §4 Exp 6 proposed for a different reason) to reach a few thousand real tiles without needing the full 27,565-tile run | Real `CellAnalysisResult` objects may differ in GC-relevant size/shape from the harness's synthetic ones; this step catches that before committing to the expensive step | `B4_gc_collect` bucket share stays flat (not climbing) across the concatenated run, correctness veto passes against the existing report.csv discipline |
| 4 | **Full-slide `workers=1` re-run**, held until 0–3 have already picked a specific, confident cadence | The only step needing a multi-hour run; confirms the real win at the scale that motivated this plan | `B4_gc_collect`'s share of wall drops substantially below today's 16.1% (target: single digits of seconds, matching doc 16's own crop-scale result in relative terms); total wall drops correspondingly; correctness veto passes on the real 356k-row `report.csv` |

## 6. Success criteria and stopping rule

- Judged by the `B4_gc_collect` bucket's absolute cost and share of wall on the **full-slide** run
  (Exp 4), never on the synthetic harness alone — the harness is for cheap iteration on cadence,
  not for claiming the win.
- Correctness veto at every step: cell counts and per-cell fields must stay within the same-code
  noise floor already established (doc 16 §4.3, doc 18 §2's per-cell veto method).
- `scripts/verify_gc_freeze.py`'s invariant guard is extended (not replaced) to assert the new
  freeze/unfreeze cardinality once the cadence is decided.
- **Stop at Option H if Exp 0-4 clear the bar.** Escalate to Option I only if Exp 1 shows
  `gc.freeze()`'s repeated cost doesn't amortize the way §3 hopes. Option J stays flagged, not
  scheduled, regardless of how H performs — it solves a narrower problem (checkpoint-path RAM,
  not GC scan cost) that H does not address and does not need to.
- A flat or negative result at any step is a valid, recordable outcome, exactly as doc 16 treated
  Option A's zero contribution — write it up the same way rather than pushing forward on
  optimism once the data says otherwise.

## 7. Relationship to doc 30 (tile read)

Worth stating explicitly because it changes sequencing, not because it changes this plan's
content: [`30-tile-read-io-plan.md`](./30-tile-read-io-plan.md) traces `B2r_tile_read`'s round-8
regression to full-slide memory pressure (peak RSS 61.13 GB) squeezing the OS page cache that the
~49 GB precut scratch depends on. `per_tile_owned`'s 356,255 live objects are part of what's
occupying that RAM. **This plan's fix reduces the tracked-object count, not the RSS footprint
directly** (the objects are still allocated, just not scanned) — so it should not be assumed to
free meaningful RAM back to the page cache on its own. Doc 30 §1 recommends re-measuring
`B2r_tile_read` after this plan ships anyway, cheaply, before investing in tile-read-specific
engineering — but don't oversell the connection: this document's win is CPU time inside
`gc.collect()`, not memory headroom.
