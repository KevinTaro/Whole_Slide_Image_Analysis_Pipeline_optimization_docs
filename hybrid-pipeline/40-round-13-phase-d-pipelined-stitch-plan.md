# 40 — Round 13: shipping Phase D's pipelined stitch — plan

> Sequences the four items
> [`39-round-12-multiprocess-scaling-ceiling-implementation.md`](./39-round-12-multiprocess-scaling-ceiling-implementation.md)
> §7 left outstanding, per
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)'s
> Discover → Analyze → Plan → Choose, cheapest-first. **Planning document only — no pipeline code
> changed here.** §1 cites doc 39's own numbers; §2 is the new work this document does — deciding
> what "build it into `_stitch_overlay_slide`" concretely means (it is not what it sounds like —
> see §1.2) and in what order the four items should run so a correctness gate doc 35 already
> flagged isn't skipped a second time.

## 0. What doc 39 left outstanding (§7, recapped)

1. Build the pipelined read into `_stitch_overlay_slide` and re-measure at full-slide scale.
2. An error bar on a full-slide number — every full-slide comparison in this project's history is
   still n=1 vs n=1 (doc 37 §5, doc 39 §2.4).
3. The QuPath open-and-render pass on a Predictor-2 candidate (doc 35 §3.4) — parked as
   irrelevant when no candidate was viable, decision-relevant again now that one is. **Done
   2026-07-29: the user opened the candidate output in QuPath and confirmed it's fine** — see §1.4,
   this gate is now cleared rather than outstanding.
4. `run_batch(..., workers: int = 4, ...)`'s documented-vs-shipped mismatch (`hybrid_pipeline.py:128`)
   — the function signature already defaulted to `4`; every doc in this folder and the CLI's own
   `--workers` argparse default still said `1`. **Decided 2026-07-29: `workers=4` is the intended
   default**, not a bug to revert — see below, this is now resolved rather than merely re-flagged.

`docs/BACKLOG.md` item 1b (currently uncommitted on this branch, written alongside doc 39) already
states the same next step in one sentence: *"build it into `_stitch_overlay_slide` and confirm at
full scale; blocked on the never-run QuPath render check for the Predictor-2 output."* This
document is that sentence, expanded into an ordered plan.

## 1. Discover — what round 13 inherits, and one thing it must not assume

### 1.1 The lever's payoff is already measured from two independent directions

Doc 39 §4 built `_prefetch_bands` (new, `scripts/stitch_probe.py`) and measured it at doc 35's
4.055 GP scale, controls reproducing doc 35 §3.2 inside 3%:

| arm | wall | vs shipped pyvips | audit |
|---|--:|--:|:--:|
| B `tifffile_cpu` + pipelined read | 45.72 s | **1.365x** | PASS |
| C `tifffile_gpu_pyramid` + pipelined read | 39.47 s | **1.581x** | PASS |

Doc 39 §4.4 then closed the one obvious way that could be lying (a cache-resident spike source):
on the real 46 GB `overlay_annotated` source read cold, the read is **51.9% of Phase D at full
scale** against doc 35's 48.7% at crop scale — the ratio the win depends on transfers. Projected
end-to-end value on the `workers=4` `w=4` wall: **1.135x** (perfect-overlap floor 1.184x), pushing
the round-12 `workers=1→4` speedup from 1.745x to **~1.980x**. None of that arithmetic is redone
here — it is cited, not re-derived, exactly as doc 38 §1 cited rounds 8–11 rather than re-deriving
them.

### 1.2 "Build it into `_stitch_overlay_slide`" is not "add a prefetch thread to the shipped path"

This is the one place a reader could misread doc 39 §7 item 1, so it needs stating plainly before
any code is scoped. The shipped path (`m0_stitch.py:438` `_stitch_overlay_slide` → `:396`
`_join_overlay_tiles` → one `pyvips` `tiffsave` call) already hides its own read behind its own
encode, for free, by construction — that is precisely *why* it beat both tifffile candidates
0.788x/0.884x in doc 35. `_prefetch_bands` cannot be bolted onto pyvips because pyvips has no read
step to pipeline against; the win lives entirely on the **tifffile-streamed candidates' side**,
which pay their read serially today and don't need to.

So "build it" means: **replace** the pyvips join+encode call with a ported version of
`scripts/stitch_probe.py`'s `cand_tifffile_streamed()` (band-at-a-time from `_band_source`,
`_prefetch_bands` wrapping it, `_encode_tile_row`'s Predictor-2 LZW path, CPU or GPU `_shrink2_*`
for the pyramid) — not an incremental patch to the existing function. That is a bigger, riskier
change than doc 39 §7's one-line phrasing suggests, and §3 sequences it accordingly.

### 1.3 Two candidates, different cost and a real architectural difference — B first, C as a stretch

| | B `tifffile_cpu` | C `tifffile_gpu_pyramid` |
|---|---|---|
| measured speedup (§1.1) | 1.365x | **1.581x** |
| pyramid shrink | CPU (`_shrink2_cpu`, box filter) | GPU (`_shrink2_gpu`, `torch`/CUDA) |
| touches CUDA in the `workers>1` parent? | no | **yes — for the first time** |

Reading `hybrid_pipeline.py:250` and `m0_multiprocess.py:250`'s own comments: today, whenever
`workers>1`, the **parent process never touches CUDA** — models load only inside the spawned
workers, by design (`_run_tiles_multiprocess`'s docstring: 父行程完全不碰 CUDA "or it would pay
for a context + weights for nothing"). `_finish_batch` → `_stitch_overlay_slide` runs in that same
parent, after every worker has exited. Candidate C's `_shrink2_gpu` would be the **first CUDA
allocation the parent ever makes on the `workers=4` path** — not unsafe (the workers have already
exited and freed their contexts by the time Phase D starts), but a real architectural first that
doc 39 never flagged and this document should, since it changes what "the parent never touches
CUDA" means as an invariant other code may currently rely on.

**Recommendation: ship B first.** 1.365x vs 1.581x is a real gap, but B is a strictly smaller
change (no new CUDA surface in a process that has never had one on this path), and the playbook's
"prefer the simplest solution that clears the bar" applies directly — C is worth a follow-up
measurement once B is shipped and stable, not a reason to hold B back.

### 1.4 The QuPath render check — done, passed, gate cleared

Doc 35 §3.4 recorded, and doc 39 §4.3 repeated: candidates B/C write **1.89 GB against the shipped
path's 2.44 GB** because they apply TIFF Predictor 2 where the shipped `pyvips` call applies none
(same `predictor="none"` correctness fix `m0_stitch.py:475`'s comment documents — pyvips's own
*horizontal* predictor default was the round-9 defect; the tifffile candidates use Predictor 2
correctly by construction, per `_encode_tile_row`'s docstring in `stitch_probe.py:220`). Doc 35
parked "open one of these in QuPath and confirm it renders identically to the shipped path" as
*"no longer decision-relevant — no candidate is being adopted."* A candidate was proposed for
adoption in this document (§1.1, §1.3), and this check had still never been run.

**It has now been run: the user opened the candidate output (doc 39 §4.2's audit-passed
`B + pipelined read` / `C + pipelined read` TIFF) directly in QuPath on 2026-07-29 and confirmed it
renders correctly** — no visible difference from the shipped pyvips output. This closes doc 35
§3.4's precondition for adoption. It corroborates, rather than replaces, `stitch_probe.py`'s
programmatic `--verify` audit (every pyramid level already confirmed decoding correctly via
`imagecodecs`/`tifffile`) with the human-render check that audit cannot stand in for — the same
distinction round 9 (doc 32 §5.1) got burned by trusting a programmatic check alone. **This gate is
cleared; item 2 (the build) is no longer blocked on it.**

### 1.5 The error-bar item is real but does not gate the build

Doc 39 §2.4 was explicit that Phase D's 32.3%-of-wall / 1.477x figures are a **single**
full-slide measurement, and doc 37 §5 already flagged that no full-slide number in this project's
history has ever had a repeat run. That is a legitimate gap, but it does not block round 13's
build decision: §1.1's payoff (1.365x/1.581x) is sourced from the **spike**, corroborated by an
**independent cold-read measurement** (§1.1's 51.9%-vs-48.7% transfer check) — neither number
depends on the 32.3% figure being exactly right. Building and shipping the lever, then
re-measuring the *whole* `workers=4` wall once more, produces one more full-slide data point almost
for free; a dedicated repeated-runs campaign (~3 h/run × several reps) is a separate, expensive
item this document deliberately does not fold into the build's critical path (§3 item 4).

### 1.6 The `run_batch` default mismatch — decided, and now fixed as documentation/CLI alignment

Doc 38 §0 and doc 39 §1 both flagged that `hybrid_pipeline.py`'s top-level `run_batch(..., workers:
int = 4, ...)` signature (changed by commit `a64b92e`, 2026-07-25) contradicted every document in
this folder stating `workers=1` was the default, and called it a "latent footgun" to revert back to
`1`. **That framing is superseded: the user decided 2026-07-29 that `workers=4` is the intended
default** — consistent with round 8/round 12's own recommendation to *ship* `workers=4` for
production. Reverting the signature to `1` would have gone the wrong direction.

Resolved directly, alongside this document, rather than left as a round-13 action item:

- `run_batch`'s own docstring (`hybrid_pipeline.py`, the `workers:` `Args` entry) updated from
  "`1`（預設）" to "`4`（預設）", and its rationale reworded: the reason `workers` must stay low for
  a single-tile request is no longer "the default protects you," it's that
  **`backend/api/hybrid.py:89` explicitly passes `workers=1`** to override the new default — that
  explicit override is now *more* load-bearing than before, not less, and this document does not
  touch it (a single API request paying 4 workers' model-init cost for one tile would be a real
  regression, unrelated to what the *batch/CLI* default should be).
- The CLI's `--workers` argparse default (`hybrid_pipeline.py`, `build_arg_parser`) flipped from `1`
  to `4`, and its help text's stale "`>1` 尚未通過整片驗證" (not yet cleared for full-slide
  validation) removed — round 8 cleared it and round 12 re-confirmed it.
- `backend/algorithms/hybrid/CLAUDE.md`'s "Running" section and M0 architecture section, and
  `README.md`'s cross-tile-multiprocessing paragraph, both corrected from "defaults to `workers=1`
  ... not yet cleared for production" to state the `workers=4` default and cite round 8/round 12.

No pipeline *behavior* changed for the one production call site that matters today
(`backend/api/hybrid.py:89` is unaffected, since it passes `workers=1` explicitly either way); what
changed is the CLI's own default for unattended invocations that omit `--workers`, which now matches
this project's own stated production recommendation instead of contradicting it.

## 2. Analyze — why this ordering and not another

The four items don't compete for the same resource (engineering time is the only shared
constraint), so the ordering question is really "what must happen before what, and what's cheapest
to clear first." Two hard dependencies exist:

- **QuPath (item 1 in §3's table) had to precede the build (item 2)**, not run alongside it or
  after. Doc 35's own discipline ("no adoption without it") exists precisely so a candidate doesn't
  get wired into production on the strength of a programmatic audit alone when the whole reason
  Predictor 2 matters is *human-visible* rendering in the tool pathologists actually use. **That
  check is now done and passed (§1.4)**, so the build is unblocked.
- **The error-bar campaign (item 2) does not need to precede or follow the build** — it's an
  independent, expensive measurement whose absence doesn't change whether B's 1.365x is real
  (§1.5). It is sequenced last purely because it's the highest-cost item with the least urgency.

The `run_batch` default question (item 4) had no dependency on anything and was free, so — matching
doc 38 §3's own "item 0" pattern — it was resolved directly while writing this document (§1.6)
rather than left as a round-13 action item.

## 3. Plan — next steps, cheapest-first

| # | item | cost | gates / gated by | why this ordering |
|---|---|---|---|---|
| 0 | ✅ **Done alongside this document.** Decided `workers=4` is the intended default (§1.6); aligned `run_batch`'s docstring, the CLI `--workers` argparse default, `backend/algorithms/hybrid/CLAUDE.md`, and `README.md` to it. `backend/api/hybrid.py:89`'s explicit `workers=1` override is unchanged and still required. | near zero (doc/comment + one argparse default) | none | Flagged three rounds running (doc 38 §0, doc 39 §1) as a bug to revert; resolved this round in the other direction per the user's decision, not deferred a fourth time. |
| 1 | ✅ **Done, passed (2026-07-29).** QuPath open-and-render check on the existing Predictor-2 B/C spike output (doc 39 §4.2's audit-passed files) — user opened it directly in QuPath, confirmed it renders correctly, no visible difference from shipped pyvips output. | near zero — the files already existed from doc 39's run | **was gating item 2 — gate cleared** | Doc 35 §3.4's parked precondition for adoption, never run before now, and the cheapest possible check to clear before spending real engineering time on item 2. |
| 2 | **Port candidate B (`tifffile_cpu` + `_prefetch_bands`) into `_stitch_overlay_slide`**, behind a config switch defaulting to the current pyvips path (mirrors round 11's `expandable_segments` rollout pattern: measure behind a flag, flip the default as a separate deployment decision once confirmed), keeping `backend/tests/test_stitch_scratch_cleanup.py` and `tests/test_stitch_pyramid_levels.py` green (extend them for the new path rather than replacing their pyvips-path coverage) | medium — a real port from `scripts/stitch_probe.py`'s spike code into `m0_stitch.py`, not a one-line change (§1.2) | **unblocked — item 1 passed**; gates item 3 | The only item with a real, already-measured, already-corroborated ceiling (§1.1), now clear to start; everything else in doc 39 §7 is either free (item 4, here as item 0) or expensive-and-non-gating (item 2 above, here as item 4 below). |
| 3 | **Full-slide re-measurement of the ported path at `workers=4`**, same protocol as doc 39 §2 (`perf_measure.py --mp-workers 4 --stream-precut`, correctness veto against the current `d2ccc46b` baseline via `report.csv` diff + `overlay_pyramid_audit.py`) | high (~1.6–1.7 h run, same order as doc 39's) | gated by item 2 | This project's own history (doc 32 → doc 35's reversal) is the reason a spike is never adopted on projection alone; item 2's port is not "done" until this number exists. If it lands materially below the projected ~1.135x end-to-end, apply doc 38 §4's stop-loss: stop and find out why before flipping the default. |
| 4 | **Full-slide error-bar campaign** — repeat the `workers=1` and `workers=4` full-slide runs (2–3 reps each) to put a variance band on the 32.3%/1.477x Phase D-share figures doc 39 §2.4 flagged as n=1 | high (~3 h × 4–6 runs, the same cost doc 37 §5 and doc 39 §2.4 both declined to pay) | none — independent of items 1–3 (§1.5) | Real gap, not urgent: it doesn't change whether to build (§1.5) and doesn't block shipping item 2/3. Lowest priority purely on cost, not on validity of the concern. |
| 5 | *(stretch, optional)* Swap candidate B for candidate C (`tifffile_gpu_pyramid`) once B is shipped and stable, for the incremental 1.365x → 1.581x | medium, plus the architectural change flagged in §1.3 (first CUDA use in the `workers>1` parent) | gated by item 2/3 shipping cleanly | Real upside, but not worth taking on the CUDA-surface change simultaneously with the encoder swap itself (playbook anti-pattern #7: change one thing at a time). Only reopen if item 3's confirmed number leaves headroom worth the extra risk. |

## 4. Decision gates / stop-loss

- **Item 1 (QuPath) — PASSED 2026-07-29.** No visible difference from the shipped pyvips output was
  found. Had it shown one, at any zoom level, the rule was: stop, do not proceed to item 2 — exactly
  the failure mode round 9 (doc 32 §5.1) already shipped once by trusting a programmatic check
  alone. That rule doesn't apply now, but it stands for any future candidate (e.g. §3 item 5's
  candidate C) that hasn't had the same check run against it yet.
- **If item 3's full-scale re-measurement comes back materially below the ~1.135x end-to-end
  projection** (or, worse, reverses the way doc 32→35 once did): stop, do not flip the default, and
  write up why — the in-situ read (tiles written minutes earlier, likely still partly page-cache
  warm per doc 39 §4.4's caveat) is the most probable source of a gap, but it must be measured, not
  assumed.
- **Any correctness veto**: unchanged from every prior round — `report.csv` parity, verdict/ratio
  identity, `overlay_pyramid_audit.py` PASS at every level, and now item 1's QuPath check besides.
  No per-cell correctness vote, no ship.
- **`workers≥5` stays closed** (doc 39 §5) — nothing in this document reopens it; if anything,
  §1.3's note that Phase D "does not scale with worker count at all" is one more reason a fifth
  worker wouldn't touch the term that now dominates the wall.

## 5. What this document is not

- **Not a record of completed work** — no pipeline code was changed; §3 is a sequenced list of next
  steps.
- **Not a redo of doc 39's Amdahl account** — §1 cites doc 39's numbers, it does not re-derive them.
- **Not a commitment to candidate C** — B is this document's recommendation; C is an explicitly
  optional stretch item gated on B shipping cleanly first (§1.3, §3 item 5).
- **Not a promise that 1.135x/1.980x survives at full scale** — every spike-derived number in this
  project's history has needed full-scale confirmation before being trusted, and one already
  reversed outright (doc 32 → doc 35). Item 3 is that confirmation step, not a formality.
- **Not reopening `workers≥5`** — still closed, per doc 39 §5 and this document's §4.
