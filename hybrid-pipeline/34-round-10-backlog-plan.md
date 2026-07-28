# 34 — Round 10: closing the round-9 backlog — design plan

> **Design-only document — no pipeline code changed here.** Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose. Synthesizes the open items left by round 9's three
> implementation documents — [`31-gc-collect-round2-implementation.md`](./31-gc-collect-round2-implementation.md),
> [`32-phase-d-gpu-port-implementation.md`](./32-phase-d-gpu-port-implementation.md),
> [`33-tile-read-io-implementation.md`](./33-tile-read-io-implementation.md) — into one sequenced
> plan. No code has landed since round 9: `git log` on `backend/algorithms/hybrid/` and
> `docs/hybrid-pipeline/` shows nothing past `b028810` (round-9 tooling) / `d7e6cf1` (Option H +
> Option L) / `34d909b` (predictor fix), so every disposition table cited below is still current,
> verified against the code directly (`m0_multiprocess.py:143-146`, `m0_tile_runner.py:207,611`)
> rather than trusted from the prior documents' prose.

## 0. Discover — everything the three documents left open

| # | source | item | status left in round 9 |
|---|---|---|---|
| 1 | doc 32 §5.2, §7 step 1 | Any `overlay_slide.tiff` predating the `predictor="none"` fix decodes its pyramid levels as noise | **Unregenerated.** Correctness defect, clinical-facing (breaks on zoom-out), not a performance item |
| 2 | doc 33 disposition, follow-up 3 | `_mp_tile_worker` still reads tiles inline; `prefetch_tile_reads` is wired only into the single-process path | **Not built.** doc 33 calls this "the highest-value remaining piece of Option L, not an afterthought" — because `workers=4` is the shipped production recommendation, and it currently gets none of Option L's benefit |
| 3 | doc 32 §6, §7 step 2 | `stitch_probe.py` (now repaired) has not been re-run at the 4.055 GP scale doc 29 §2.3 asked for, sourcing real tiles from disk rather than an in-memory slab | **Not run.** This is what §6 calls "the unmeasured read/join half," and doc 32 §4's 1.109×/1.055× ceilings are gated on it |
| 4 | doc 31 §7, doc 33 disposition follow-up 1 | Full-slide `workers=1` run (3.82 h) — settles doc 31 Exp 4 (real RSS-pinning exposure, real Option H win) and doc 33 Option K (real Option L win, replacing the projected 1.16× ceiling) simultaneously | **Not run** — both documents explicitly ask for it to be one run, not two |
| 5 | doc 33 §4, disposition follow-up 2 | Background tiles cluster spatially and, at full scale, a read (58.6 ms) can exceed the UNet-only hide budget (29.4 ms); raising `prefetch_tile_reads`'s `depth` to 2–3 is flagged | **Recorded, not built** — explicitly gated on a measurement that does not exist yet |
| 6 | doc 31 disposition | Option J — spill `per_tile_owned` to disk under `checkpoint=True` | **Still flagged, not scheduled** — no new information changes this |
| 7 | doc 33 disposition | Option M (bypass scratch round trip), Option N1–N3 (scratch codec / pyvips read / remove scratch) | **Closed / record-only** — Option M was gated on Option L disappointing, and it didn't; Option N is explicitly "record, do not schedule" in the plan that proposed it |

Items 6 and 7 require no new action — they are correctly closed already and are listed here only so
this document is a complete accounting of the three source documents' disposition tables, not a
partial one.

## 1. Analyze — what actually depends on what

**Item 1 (broken overlays) is not a performance question and does not belong in the Amdahl
sequencing below.** It is a shipped correctness defect a pathologist can hit today by zooming out
on any slide processed before the `34d909b` fix. It is cheap to reason about — no measurement is
needed to know a noise-decoding pyramid level is wrong — so it is not blocked on anything in this
document and should not wait for it.

**Items 2 and 4 look independent but interact through what "production" means.** Doc 31 Exp 4 must
run at `workers=1` — doc 31 §1.1 traced the RSS-pinning risk as a `workers=1`-only phenomenon,
since multiprocess workers accumulate nothing across tiles. But `workers=4` is the actual shipped
default (`run_batch`'s default, per doc 31 §1.1), and today the entire measured Option L win
(doc 33 §3's −1.75%, doc 33 §4's projected 1.16× ceiling) is **single-process-path evidence for a
code path production does not run.** Shipping item 2 first, before spending 3.82 h on item 4,
changes what item 4's run can honestly claim: without item 2, a full-slide `workers=1` run
validates a configuration nobody deploys; with item 2 shipped and its own crop-scale ablation done
first (cheap, ~20 minutes per doc 31 §6's own lesson about interleaved repeats), the expensive run
at least sits next to evidence that the production path benefits too, even though the 3.82 h run
itself still only covers `workers=1`.

**Item 3 is cheaper and independent of item 4.** Doc 29 §2.3 screened at 4.055 GP / 6,900 tiles —
two orders of magnitude below the 27,565-tile full-slide run item 4 needs — because it exercises
only the stitch stage (`_stitch_overlay_slide` / `_join_overlay_tiles`), not the whole pipeline. It
has no dependency on items 1, 2, 4, or 5 and can run in parallel with any of them.

**Item 5 is downstream of item 4 (and, for the production path, item 2).** doc 33 §4 is explicit
that raising `depth` is "recorded, not built" for lack of a measurement, and the measurement that
would justify it — real background-tile run-length statistics at full scale — is exactly what item
4's run produces. Building it first would repeat the mistake the playbook's anti-pattern list
warns against (#1, optimizing before measuring): there is no evidence yet that `depth=1` actually
falls behind at full scale, only the arithmetic argument in doc 33 §4.

## 2. Plan — sequencing

```
1. Regenerate broken overlays        →  verify: pyramid levels decode correctly in QuPath
                                          (independent, do first — correctness, not perf)
2. Wire prefetch into _mp_tile_worker →  verify: tests/test_tile_read_prefetch.py-style
                                          coverage for the mp path + crop-scale interleaved
                                          ablation at workers=4 (doc 33 §3's method)
3. Re-run stitch_probe.py at 4.055 GP →  verify: candidates A/B/C shape-matched and
   (Phase D read/join half)              QuPath-clean, as doc 32 §3/§5 already established
                                          at the small scale; decide Phase 1.5 vs. Phase 2
4. Full-slide workers=1 run (3.82 h)  →  verify: doc 31 Exp 4's RSS-pinning check +
   (doc 31 Exp 4 + doc 33 Option K,       doc 33 Option K's real (not projected) B2r_tile_read
   one run settles both)                 share of wall
5. Depth=2/3 tuning, if item 4's data →  verify: crop-scale ablation shows a measured
   shows background runs starving         win before touching the shipped depth=1 default
   the depth=1 window
```

Step ordering follows the playbook's own rule (§3, "cheapest-first," and §4, "decide by
end-to-end wall-clock, not micro-benchmark, and let every layer justify itself"): items 1–3 are
each independently cheap and either urgent (1) or unlock a real decision on their own (2, 3);
item 4 is the one expensive, non-repeatable resource in this whole backlog, so it is sequenced
last among the "do soon" items — after item 2 so it says something about production, not only
about `workers=1`. Item 5 is explicitly barred from starting before item 4 supplies the
measurement doc 33 §4 says is missing.

### 2.1 Step 1 detail — regenerating broken overlays

This document cannot enumerate which deployed `overlay_slide.tiff` files predate the fix — that is
an inventory question about what is in active clinical/review use, not something derivable from
this repository. What the repo does fix is the *mechanism*: per doc 32 §5.2, where
`_stitch_scratch/` still exists for a slide, `run_batch(checkpoint=True)` short-circuits straight
to `_finish_batch` (re-stitch only, cheap); where it has been cleaned up, the full batch must be
re-run. The action here is procedural, not code: identify affected slides, check scratch survival
per slide, re-stitch or re-run accordingly, and re-verify each with
`tests/test_stitch_pyramid_levels.py`'s method (every level bright and non-degenerate, Predictor
tag consistent across IFDs) plus a QuPath open-and-zoom, matching how doc 32 §5.1 caught the
original defect.

### 2.2 Step 2 detail — multiprocess prefetch

The shape to replicate is `hybrid_pipeline.py`'s existing wiring (doc 33 §2): a dedicated
single-thread pool per worker process feeding `prefetch_tile_reads`, with `_process_precut_tile_gpu`
called with `preread=` instead of letting it read inline. `_mp_tile_worker`
(`m0_multiprocess.py:143-146`) currently calls `_process_precut_tile_gpu` with no `preread`
argument, so it falls into the `else` branch (`m0_tile_runner.py:273-281`) and reads synchronously
on each worker's own main thread — structurally the same pre-Option-L shape doc 30 described for
the single-process path, just replicated across `workers=4` processes instead of fixed once.
Fail-fast semantics need the same treatment doc 33 §2.1 gave the single-process path: a read
exception raised on the prefetch thread must surface for its own tile, at consumption time, inside
whichever per-worker error path today converts a `None` chunk into a worker-level abort — that
path should be re-traced with `codegraph_explore` before editing (project rule: verify callers
before modifying), not assumed to match `hybrid_pipeline.py`'s shape exactly, since the
multiprocess worker's error handling is a different function than `run_batch`'s loop.

Verification mirrors doc 33 §3 exactly, at `workers=4`: `perf_measure.py --no-prefetch` (or its
multiprocess-path equivalent, extended if it does not already cover this) vs. prefetch-on,
interleaved repeats, identical `stats`, and a bucket-level check that `B2r_tile_read`'s total rises
while wall falls (§3.1's "signature of work successfully moved onto a parallel arm" — the same
counter-intuitive-but-expected pattern, now across 4 processes instead of 1 thread).

### 2.3 Step 3 detail — Phase D at scale

Doc 32 §7 step 2 is explicit about what changes: source real overlay tiles from
`_stitch_scratch/` in the container writer's tile order (a lazy source, not the in-memory slab
`phase_d_container_spike.py` used), at 4.055 GP / 6,900 tiles. Re-run all three candidates (A
baseline, B CPU-only, C GPU-pyramid) with the same shape-matching and every-pyramid-level
correctness check §5.1 added (not just level 0 — that gap is exactly what let the predictor bug
through to QuPath last round), plus the QuPath open-and-render-identically check doc 32 §5 already
passed at small scale. Doc 32 §7 step 4 already flags the decision this produces: if candidate B
holds up at scale and through QuPath, it needs no GPU at all and is lower-risk than C, and doc 29
§3's "prefer the simplest solution that clears the bar" favors it unless C's margin (already thin
at small scale — 1.6 percentage points over the ceiling doc 29 §1.3 set) survives at scale.

### 2.4 Step 4 detail — the full-slide run

Budget this as the single 3.82 h `workers=1` run doc 31 §7 and doc 33 §6 both ask for, not two.
Collect, in one run: per-tile `gc.collect()` deciles (doc 31 §2's method) for the real climb/flatten
shape at scale, peak RSS (doc 31 §4's pinning concern — ~71 freezes is the real full-slide
exposure count, against the crop's single freeze), and `B2r_tile_read`'s tissue/background bucket
split (doc 33 §1's method, already extended into `perf_measure.py`) for Option K's real number
against doc 33 §4's projected 1.16×. Because this run is expensive and non-repeatable at will, do
not start it until step 2 has shipped and passed its own cheap ablation — otherwise a second
3.82 h run is needed later just to re-validate Option L once the production path changes, which is
the "changing several things at once" anti-pattern (playbook checklist #7) applied across runs
instead of within one.

### 2.5 Step 5 — explicitly conditional

Only scope this if step 4's data shows background-tile runs actually exhausting the `depth=1`
window (doc 33 §4's concern). If it does not, doc 33 §6 follow-up 2's own caveat applies unchanged:
raising `depth` costs proportional in-flight memory (currently ~12 MB at `depth=1`,
`m0_tile_runner.py:611`) for no demonstrated win, and the playbook's Amdahl discipline says don't.

## 3. Choose — what this document commits to and what it doesn't

This document sequences the backlog; it does not pre-decide step 3's Phase D candidate (A/B/C) or
step 5's depth value — both are explicitly measurement-gated by the documents that raised them, and
picking a number here would violate the same "optimizing before measuring" rule this whole series
has been enforcing since round 1. What it does commit to: step 1 is correctness and goes first
regardless of the performance sequencing; step 2 goes before step 4 because production runs
`workers=4` and the expensive run should not spend 3.82 h validating a path production doesn't use;
step 3 and step 5 are correctly independent/downstream as analyzed in §1, not by convention.

## 4. Disposition

| item | this document's action |
|---|---|
| Broken overlay regeneration | **Scheduled, step 1** — procedural, no code change, correctness-driven |
| `_mp_tile_worker` prefetch | **Scheduled, step 2** — code change, own crop-scale ablation at `workers=4` before the full-slide run |
| Phase D at scale (`stitch_probe.py`) | **Scheduled, step 3** — independent, can run in parallel with steps 1–2 |
| Full-slide `workers=1` run | **Scheduled, step 4** — after step 2 ships, one run settles doc 31 Exp 4 and doc 33 Option K |
| `depth=2/3` tuning | **Conditional, step 5** — only if step 4's data supports it |
| Option J (spill to disk) | **No action** — still correctly deprioritized |
| Option M / N1–N3 | **No action** — correctly closed in round 9 |
