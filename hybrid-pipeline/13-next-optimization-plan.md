# 13 — Next optimization plan (post-round-3 Cellpose swap, re-sorted 2026-07-22)

> Solution-design follow-up to [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md)
> and [`measurement/current-status-comparison.md`](./measurement/current-status-comparison.md), both
> updated through **round 3** (git `f95a573`, Cellpose 4.2.1.1 `cpdino` swap, 2026-07-22). Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose, cheapest lever first, ablation-proof every layer, correctness is
> a veto. **Planning document — no pipeline code changed here.**
>
> **This supersedes the previous version of this doc wholesale.** The old #1/#2 priorities (wire
> `cellpose_batch_size`, then sweep it) are now #4, demoted by measurement: round 3 changed which
> ranking method is even valid (see §2) and delivered a bigger win than anything in the old plan,
> for free, from outside this project's own work.

## 0. Operational note — shared server

This machine is a multi-user server; other tenants' jobs can hold the GPU at 100% / most of VRAM at
any time. **Before running any measurement in this plan**, run `nvidia-smi` (+ `who` / `ps aux
--sort=-%cpu` if usage looks nonzero) and confirm the GPU is idle — round 3's own run log recorded
"89 MiB, 0%, no other compute processes" before launch, and that check-first discipline is what makes
every number in `bottleneck-list.md` trustworthy. If the GPU is contended, wait or coordinate; don't
launch a run that will OOM on a contended slice or produce numbers contaminated by a concurrent
tenant that look like a regression/improvement but are just noise. Also drop a `uv pip freeze` (or
`pip freeze`) next to every new round's metrics directory — round 2 (`_metrics_current/`) has none,
which is why ⑨ below can't be attributed cleanly to a single dependency. Don't repeat that gap.

## 1. What's already closed — do not re-litigate

| Item | Status | Evidence |
|---|---|---|
| ① serial pipeline / GPU idle | **Fixed** (two-stage overlap). idle 0.459→0.190 (large) / 0.477→0.163 (medium); wall −16.6%/−14.5% at that step | [`pipeline-overlap-result.md`](./measurement/pipeline-overlap-result.md) |
| ② `detect_all_dots` | **Resolved for free** — fully hidden behind the GPU front (BG/slack arm) at all scales measured | [`detect-all-dots-result.md`](./measurement/detect-all-dots-result.md) |
| ③ PNG encode | **Hidden** behind the GPU front (BG/slack arm), not independently actioned | `current-status-comparison.md` §3 |
| `detect_all_dots` → process backend | **Tested, negative** | `gil-contention-diag.md` §"結論" |
| `gc.collect` → background thread (relocation) | **Tested, negative** (ablation: idle 0.183→0.221, worse) — only *relocation* is closed; *frequency reduction* is not (§3 P1) | `gil-contention-diag.md` §"方案 (d) ablation" |
| CUDA-graph-capture / vectorize Cellpose loops | **Stop-lossed** — ceiling only ~1.18–1.23x, requires patching pinned third-party `cellpose`/`segment_anything` internals | `gil-contention-diag.md` §"追加深挖" |
| **Cellpose 4.0.8→4.2.1.1 / `cpdino` swap (round 3)** | **Landed, largest win to date.** Wall 707.4→573.7 s (−18.9%, large); cumulative −32.3% vs the original control. Delivered by a coworker's model swap, not this project's own optimization work | `bottleneck-list.md` "Round 3" note under ①; `current-status-comparison.md` §8 |
| Old §2 open question ("why did B1 grow +192.9 s absolute") | **Superseded, mostly moot.** Most of the growth reversed on its own with the Cellpose upgrade (578.9→444.6 s); isolating the residual GIL-contention share is no longer worth doing as originally scoped | `bottleneck-list.md` "Superseded by round 3" note |
| **`gc.collect` frequency reduction (Priority 1, this doc)** | **Landed as `gc.freeze()` (Option C), not the batching lever this plan specified.** Measured 1.069x attributable / 1.077x e2e at the large anchor — matches this doc's 1.083x ceiling almost exactly. Batching (Option A) was built, measured, found to add nothing on top of freeze, and deleted | [`15-gc-collect-frequency-implementation.md`](./15-gc-collect-frequency-implementation.md) / [`16-gc-collect-frequency-result.md`](./16-gc-collect-frequency-result.md) |

Reopening any of these needs new evidence (upstream fix, or a ceiling crossing a materially higher
threshold after something else lands) — not just "let's try it again."

## 2. Why the ranking method itself changed

Before round 3, items were ranked by **self-time ÷ wall**. That stopped being valid once the ①
overlap made the pipeline two concurrent arms (confirmed by reading `hybrid_pipeline.py:766-810`,
not inferred from timings):

- **MAIN arm** (main thread): 3 GPU forwards, `_read_rgb`, M1 overlay ops, `clear_slide_edge_cells`,
  `build_all_positive_results`, `enlarge_cell_instances`, `gc.collect` + `empty_cache`.
- **BG arm** (one background thread): `detect_all_dots`, merge, PNG/TIFF encode, `render_overlay_image`,
  per-cell crops, `filter_and_absolutize`.

`wall ≈ max(MAIN, BG) + outside` (outside = precut A + stitch D + model init), validated at both
scales to within 1.3%. An item's self-time-% is meaningless if it's on the **slack** arm (BG) — its
ceiling is ≈1.00 no matter how large the percentage looks. Ranking must instead use each item's
**ceiling if reduced to zero, on its own arm**:

| large/441, round-3 anchor 573.7 s | self-time | % wall | arm | **ceiling if →0** |
|---|--:|--:|---|--:|
| GPU forwards | 444.6 s | 77.5% | MAIN (critical) | **1.382x** |
| `gc.collect` | 36.4 s | 6.3% | MAIN (critical) | **1.083x** — **DONE**, measured 1.069–1.077x (§4 Priority 1) |
| CPU prep (`enlarge_cell_instances`+`build_all_positive_results`) | 28.4 s | 5.0% | MAIN (critical) | ~1.05x |
| precut A + stitch D | 25.6 s | 4.5% | outside | ~1.05x |
| `detect_all_dots` | 292.9 s | 51.1% | BG (slack) | **1.013x** |
| PNG encode | 78.7 s | 13.7% | BG (slack) | **1.013x** |

Read that table before trusting any "% of wall" figure elsewhere in the docs: `detect_all_dots` is
51.1% of wall and worth 1.3%; `gc.collect` is 6.3% and worth 8.3% — six times more, for looking eight
times smaller. The flat "<10% ⇒ drop it" floor from the original playbook is **suspended** (not
abandoned) for items that are (a) on a critical arm and (b) plausibly reducible to near zero — at a
~12.6 h full-WSI run, a 6% MAIN-arm item is ~44 minutes.

**The one number that governs everything below:** BG arm (387.3 s) is **72% of** MAIN arm (538.3 s).
The MAIN arm must shed **34.0%** (large) / **36.8%** (medium) of the GPU forwards before the BG arm
(`detect_all_dots` + PNG encode, i.e. ②③) becomes the new critical path and gets re-exposed. Every
item below should be sized against that margin, not against its raw self-time%.

## 3. Correctness caveat — read before treating round 3 as a free baseline

Round 3's wall-clock win did **not** come with bit-exact output. Retrained Cellpose checkpoints
changed segmentation: cells 12,922→13,150 (+1.8%, large), 3,558→3,647 (+2.5%, medium), one tile
flipped success→skipped (379→378, large). Rounds 1–2 agreed to within ±3 cells (pure GPU-nondeterminism
noise floor); round 3 does not. This is a **model-quality change requiring clinical/pathologist
validation on its own terms** — it is out of scope for this performance track, but every ablation
below must use **round 3's own cell counts as its new correctness reference point**, not round 1/2's,
or a real regression from the plan items below will be invisible against the noise the Cellpose swap
already introduced.

## 4. Re-sorted priority for next stage

Ranked by **critical-arm contribution × plausible reduction toward zero** (§2's method), not by raw
self-time%. All are measured against the round-3 anchors (166.6 s medium / 573.7 s large).

### Priority 1 — `gc.collect` frequency reduction (Class 4, MAIN/critical arm) — **DONE (2026-07-22)**

> Full record: [`15-gc-collect-frequency-implementation.md`](./15-gc-collect-frequency-implementation.md) /
> [`16-gc-collect-frequency-result.md`](./16-gc-collect-frequency-result.md). Kept below (with the
> original framing struck through in spirit, not text) so the plan-vs-actual gap is visible.

**What this section originally called for**: batch `gc.collect()`/`empty_cache()` every N tiles
instead of every tile, sweeping N ∈ {4, 8, 16}, with RSS re-validation at 441-tile scale because
skipping sweeps lets garbage accumulate between them.

**What measurement found instead**: the cost driver was never call *count* — it was **per-call scan
volume**. The three resident GPU models (UNet++ + two Cellpose SAM-ViT) were being re-walked by every
full collection regardless of how often it ran (ms/call flat at 81.9–83.0 across N = 1/4/8/16 with
freeze off). Batching alone (Option A) therefore recovered almost nothing, and adding it on top of
`gc.freeze()` (Option C) — freezing the resident models into a generation the collector never
scans — added nothing measurable either (indistinguishable from freeze-alone; peak RSS actually
*highest* of any config tested, 3.991 GB). **Shipped: `gc.freeze()` alone**, unconditional, wrapped
in a context manager (`finally: gc.unfreeze()`) so a long-lived API server process isn't left with a
permanently-frozen generation across requests — a leak the original plan's "single unconditional
call" framing didn't account for.

**Result**: gc cost 36.71 s → 0.52 s per 441-tile batch (83.2 ms → 1.2 ms/call, cadence unchanged at
441 calls). Large anchor: **1.069x attributable / 1.077x end-to-end** — matches this doc's predicted
1.083x ceiling almost exactly. The memory-bounded invariant this section required re-validating is
**untouched by construction** (collection cadence never changed) and was measured anyway: peak RSS
+1.1% (3.881→3.925 GB), 0.14% of the 32 GB machine, sawtooth reclamation shape intact. Correctness
veto passed (max|Δ| = 0 for reddot/blackdot/score among matched cells). No config knob shipped — the
`--gc-every-n`/`--gc-freeze` harness flags and Option A's config fields were built, measured, and
removed; `config.py`/`config_example.py` are byte-identical to pre-change (`db2b7e6a`).

**Consequence for Priority 2 below**: the "~1.14x combined" projection assumed both this item and
Priority 2 landed. This item alone confirms the gc half almost exactly (predicted −36.4 s, measured
−37.0 s attributable). The MAIN/BG arm margin (34.0%/36.8%, §2 table) predates this fix and should be
**re-measured**, not reused, before sizing Priority 2 or Priority 4 — MAIN is now smaller by ~36 s.

### Priority 2 — Move stranded CPU prep off the MAIN arm (Class 3, new in round 3) — **now the top open item**

> Priority 1 above is done. This is the next unbuilt lever, and the only one left that the ~1.14x
> combined projection below still depends on.

`enlarge_cell_instances` (19.60 s) and `build_all_positive_results` (8.81 s) run inside
`_process_one_chunk_gpu` (`hybrid_pipeline.py:569-573`) — i.e. on the **main/GPU thread**, between the
M2 and M3b Cellpose forwards. Both are pure NumPy/skimage; neither touches torch or CUDA. This was
invisible before round 3: the control had no arms, and the overlap round's MAIN arm had so much
headroom (BG/MAIN 0.496) that placement didn't matter. Now BG/MAIN is 0.719 and MAIN is critical, so
28.4 s of pure-CPU work sitting on the wrong arm is a real, free-looking win: BG arm has 151.0 s of
measured slack to absorb it.

**Plan**: move the two function calls from the MAIN-thread `_process_one_chunk_gpu` path into the
BG-thread `_finish_chunk_cpu` path (same functions, different thread — a placement change, not an
algorithm change). The arm model projected `max(538.3−36.4−28.4, 387.3+28.4) + 28.0 ≈ 501.5 s` —
i.e. **wall 573.7 → ~501 s (~1.14x)** combining this item with Priority 1. Priority 1's half of that
arithmetic is now confirmed (predicted −36.4 s, measured −37.0 s, effectively exact), so the
remaining gap to ~501 s depends entirely on this item, still unbuilt. **Verify, don't assume**: this
is a threading/ordering change, so confirm no data race is introduced (the two functions must not be
read by anything still running on MAIN after the move), **re-measure the actual MAIN/BG split with
`gc.freeze()` already in place** before trusting the ~501 s projection (§4 Priority 1's closing note
flags the old 34.0%/36.8% margin as stale), and re-run the standard medium+large ablation.

**Important second-order effect — re-measure the margin after P1+P2 land.** Moving 28.4 s of work
from MAIN to BG doesn't just shrink MAIN, it also *grows* BG. Projected new arms: MAIN ≈ 473.5 s, BG ≈
415.7 s → margin drops from 34.0% to **~12.2%**. In other words, landing P1+P2 eats most of the slack
that currently keeps ②③ hidden. **Priority 4's batch-size sweep (below) must be evaluated against
this post-P1/P2 margin, not the pre-P1/P2 34%** — the GPU-forward headroom that's actually left to
spend may be much smaller than it looks today.

### Priority 3 — Overlap precut A and stitch D with the B loop (Class 3/5, backlog)

The only two stages that are 100% serial **and** entirely outside the overlap (`outside` in the arm
model): precut A (~20.5 s / 3.6%) and stitch D (~5.1 s / 0.9%), combined 25.6 s / 4.5%, ceiling ~1.05x.
Both flat across all three rounds in absolute seconds, including through pyvips 3.1.1→2.2.3 — the
pyvips downgrade cost nothing measurable here. A scales linearly with tile count, so at full-WSI scale
(35,700 tiles) it becomes the largest of the sub-floor items in absolute terms, even though today it's
below the 10% floor.

**Plan**: no design work has been done on *how* to overlap precut A with the analysis loop or stitch D
with the tail of the B loop — that's the next step if this priority is picked up, not something to
improvise inline. Lowest priority of the three "actionable now" items (P1–P3) because its ceiling is
smallest at current scale; revisit sooner if full-WSI timing work starts, since its full-WSI share is
proportionally larger than its 121/441-tile share suggests.

### Priority 4 — Wire `cellpose_batch_size` into `Config`, then sweep (Class 6/7)

**Confirmed still broken** (read directly): `backend/algorithms/hybrid/hybrid_pipeline.py:206` and
`:218` call `getattr(config, "cellpose_batch_size", 16)`. No such field exists on `Config` — the only
`batch_size` field (`config.py:184`, default `4`) feeds `_init_unet_inferencer` (UNet++), not Cellpose.
Cellpose's batch size has **always** silently been the hardcoded fallback `16`. VRAM headroom for this
experiment is now **larger than when this item was first written**: round 3's bfloat16 default cut
peak VRAM from 5159 MB to 2787 MB (large) / 2785 MB (medium) — idle headroom is ~29.8 GB of 32 GB,
up from ~27 GB.

**Plan** (unchanged in mechanics from the previous version of this doc, re-ordered by priority only):
1. Add a `cellpose_batch_size: int` field to `Config` (mirror the `batch_size` pattern at
   `config.py:184`), thread it through the two `getattr(...)` call sites. Default must match the
   current fallback (`16`) — this step alone should be a **no-op ablation** (bit-exact `report.csv`
   against round 3's own cell counts per §3, flat wall-clock). Verify before moving on.
2. Sweep batch size (16 → 32 → 64) at the medium anchor first — cheap, catches regressions before a
   large-scale run. Ablation-required checks: end-to-end wall-clock (`--gpu-dmon`), peak VRAM (must
   stay bounded — same discipline that caught the 22.2 GB `cuda_alloc_peak` sampling artifact; read
   VRAM from `dmon fb`, not the torch-allocated counter), and correctness (cell-count diff against
   round 3's own noise floor, not round 1/2's — see §3).
3. **Size the expectation against the post-P1/P2 margin (§ Priority 2's note), not the pre-P1/P2 34%.**
   **P1 has now landed** (Priority 1 above, `gc.freeze()`) but P2 has not — the MAIN/BG margin has
   shifted from its pre-P1 34.0%/36.8% value by roughly P1's measured −37.0 s off MAIN, but has not
   been re-measured end-to-end. Re-measure the actual margin before this sweep runs rather than
   reusing either the pre-P1 34%/36.8% or the pre-measurement post-P1/P2 ~12.2% estimate — a
   batch-size win larger than the real current margin will just shift the bottleneck to the BG arm
   (②③) rather than move wall further.
4. If flat or negative: stop, write up the negative result (`gil-contention-diag.md`-style), don't
   proceed to cross-tile batching.
5. If positive: re-confirm at the large anchor before calling it done. Only then does cross-tile
   batching (batching multiple tiles' forwards together, not just one tile's sliding windows) become
   worth scoping — a bigger architectural change with its own accuracy tradeoffs, needing its own plan
   document.

### Priority 5 — Isolate the `detect_all_dots` +22.3% regression (optional, Class 7/1)

Large/441: `detect_all_dots` went 239.4 s → **292.9 s (+22.3%)** between the overlap round and round 3,
for only +1.8% more cells (per-cell 18.5→22.3 ms, +20.2%). Medium agrees (+11.8% for +2.5% cells).
Leading hypothesis: the same round bundled scikit-image 0.25.2→0.24.0, numpy 2.2.6→1.26.4, and
opencv→4.8.1.78 downgrades, and `detect_all_dots` is LAB + H-morphology over exactly those libraries —
but retrained-checkpoint cell-geometry change is a competing explanation, and the two changed together
so **the cause is not isolated**. Ceiling today is **1.013x** (it's on the BG/slack arm) — this cannot
move wall now. It matters only because it's eating the margin computed in §2/Priority 2's note: it's
76% of the BG arm, so it's the thing that will re-expose ②③ first once MAIN sheds enough.

**Plan, if picked up**: re-run `detect_all_dots` alone over a fixed, saved set of instance masks under
both dependency sets (old vs new numpy/scikit-image/opencv). That holds cell geometry constant and
separates "library downgrade" from "different cells." Low priority — no wall-clock payoff, pure
documentation-completeness to protect the margin's shrinking runway.

### Priority 6 — Record `pip freeze` with every future round (process fix, not perf)

Round 2 (`_metrics_current/`) has no dependency snapshot, which is exactly why Priority 5 above can't
be settled cleanly. Not a performance item — just discipline (see §0).

## 5. Success criteria (playbook §4 "Choose")

- Every priority above is judged by **end-to-end wall-clock** on the medium/large anchors, never a
  micro-benchmark of the changed function alone.
- Every ablation's correctness check is against **round 3's own cell counts** (§3), not round 1/2's —
  round 3 is not bit-exact with earlier rounds, so comparing against the wrong baseline will either
  mask a real regression or flag a phantom one.
- Priority 1 is **done**: validated with the standard `--gpu-dmon` medium+large run (n=3 at large
  per the noise floor found during this measurement) and its RSS check at 441-tile scale passed
  (+1.1%, invariant intact). Priority 2 still needs the same medium+large validation once built.
- Priority 2's projected `~501 s` combined result (with P1) must be **re-measured, not assumed** —
  it's an arm-model projection, not a measured number, until P2 lands and the combination is
  ablated together (P1's half is already confirmed at −37.0 s, almost exactly the −36.4 s assumed).
- Priority 4 step 1 (`Config` wiring) must be a bit-exact no-op before step 2 (the sweep) is attempted.
  Priority 4 step 2's expected ceiling must be checked against the **current** MAIN/BG margin at the
  time it's run (34%/36.8% pre-P1/P2, ~12.2% projected post-P1/P2) — not a fixed number.
- A flat or negative result at any priority is itself a valid, recordable outcome (write it up like
  `gil-contention-diag.md`'s negative results) — do not chase a positive result past what the data
  supports.
- All measurement runs follow §0: confirm the GPU is idle before launching, and drop a `pip freeze`
  next to the metrics.
