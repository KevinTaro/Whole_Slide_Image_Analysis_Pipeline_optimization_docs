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
| `gc.collect` | 36.4 s | 6.3% | MAIN (critical) | **1.083x** |
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

### Priority 1 — `gc.collect` frequency reduction (Class 4, MAIN/critical arm)

**Confirmed still per-tile** (read directly, `hybrid_pipeline.py:798`): a full `gc.collect()` +
`torch.cuda.empty_cache()` runs on the MAIN thread once per tile, between the GPU front and the
`_collect(pending)` join. It costs 36.3–36.4 s in every round measured so far (r1/r2/r3: 36.31 / 36.33
/ 36.39 s) — a fixed per-tile overhead, independent of everything else that's changed. Being on the
critical arm, its ceiling is a real **1.083x** (~44 min at the ~12.6 h full-WSI projection).
Relocating it to the background thread was **already tried and reverted** (idle_frac got worse,
0.183→0.221) — that lever is closed. The one untried lever is **frequency reduction**: batch the
sweep every N tiles instead of every tile.

**Plan**: change the `run_batch` loop to call `gc.collect()` / `empty_cache()` every N tiles (start
with N=4 or 8) instead of unconditionally every tile. This directly touches the memory-bounded
invariant this repo has protected throughout (peak RSS 2.82→3.07→4.04 GB across 25/121/441 tiles) —
**RSS must be re-measured at the 441-tile scale for each N** before adopting it, since skipping sweeps
lets garbage accumulate between them. Sweep N ∈ {4, 8, 16}; pick the largest N whose peak RSS still
sits comfortably under the existing bound (headroom is large — machine has 32 GB VRAM and RSS peaks
have never exceeded ~4 GB — but the invariant, not the headroom, is the actual bar).

### Priority 2 — Move stranded CPU prep off the MAIN arm (Class 3, new in round 3)

`enlarge_cell_instances` (19.60 s) and `build_all_positive_results` (8.81 s) run inside
`_process_one_chunk_gpu` (`hybrid_pipeline.py:569-573`) — i.e. on the **main/GPU thread**, between the
M2 and M3b Cellpose forwards. Both are pure NumPy/skimage; neither touches torch or CUDA. This was
invisible before round 3: the control had no arms, and the overlap round's MAIN arm had so much
headroom (BG/MAIN 0.496) that placement didn't matter. Now BG/MAIN is 0.719 and MAIN is critical, so
28.4 s of pure-CPU work sitting on the wrong arm is a real, free-looking win: BG arm has 151.0 s of
measured slack to absorb it.

**Plan**: move the two function calls from the MAIN-thread `_process_one_chunk_gpu` path into the
BG-thread `_finish_chunk_cpu` path (same functions, different thread — a placement change, not an
algorithm change). Combined with Priority 1, the arm model projects `max(538.3−36.4−28.4,
387.3+28.4) + 28.0 ≈ 501.5 s` — i.e. **wall 573.7 → ~501 s (~1.14x)** without touching the GPU
forwards at all. **Verify, don't assume**: this is a threading/ordering change, so confirm no
data race is introduced (the two functions must not be read by anything still running on MAIN after
the move) and re-run the standard medium+large ablation before trusting the projected number.

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
   If P1/P2 haven't landed yet when this sweep runs, treat 34%/36.8% as the ceiling; if they have,
   use the smaller re-measured margin — a batch-size win larger than the current margin will just
   shift the bottleneck to the BG arm (②③) rather than move wall further.
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
- Priority 1 and 2 must each be validated with the standard `--gpu-dmon` medium+large run, and
  Priority 1 additionally needs an RSS check at 441-tile scale (memory-bounded invariant).
- Priority 2's projected `~501 s` combined result (with P1) must be **re-measured, not assumed** —
  it's an arm-model projection, not a measured number, until both land and are ablated together.
- Priority 4 step 1 (`Config` wiring) must be a bit-exact no-op before step 2 (the sweep) is attempted.
  Priority 4 step 2's expected ceiling must be checked against the **current** MAIN/BG margin at the
  time it's run (34%/36.8% pre-P1/P2, ~12.2% projected post-P1/P2) — not a fixed number.
- A flat or negative result at any priority is itself a valid, recordable outcome (write it up like
  `gil-contention-diag.md`'s negative results) — do not chase a positive result past what the data
  supports.
- All measurement runs follow §0: confirm the GPU is idle before launching, and drop a `pip freeze`
  next to the metrics.
