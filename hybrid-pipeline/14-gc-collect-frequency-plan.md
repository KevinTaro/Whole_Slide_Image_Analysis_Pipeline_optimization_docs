# 14 — `gc.collect` frequency-reduction design plan (Priority 1 detail)

> Detail design for [`13-next-optimization-plan.md`](./13-next-optimization-plan.md) **Priority 1**
> (`gc.collect` frequency reduction, Class 4, MAIN/critical arm). Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose, cheapest lever first, ablation-proof every layer, correctness
> is a veto. **Design-only document — no pipeline code changed here.** Every option below ends in a
> concrete, independently ablatable experiment; §5 gives the recommended run order.

## 0. Recap — why this is even worth doing (don't re-derive, just cite)

Already established, not re-litigated here:

- `gc.collect()` runs once per tile on the **MAIN thread**, between the GPU front and
  `_collect(pending)` (`backend/algorithms/hybrid/hybrid_pipeline.py:798`, inside `run_batch`'s tile
  loop at `hybrid_pipeline.py:766-806`). `torch.cuda.empty_cache()` runs immediately before it
  (`:792-797`), same call site, but is a **separate, much cheaper** cost (<0.3% wall,
  `bottleneck-list.md` ④) — the two must be reasoned about independently, not as one blob (see §1
  Option B).
- Cost: 36.31 / 36.33 / 36.39 s (r1/r2/r3, large/441) — flat, fixed per-tile overhead, **6.3% of
  round-3 wall**, MAIN/critical arm ⇒ ceiling **1.083x** if reduced to zero
  (`13-next-optimization-plan.md` §2/§4 P1).
- `gil-contention-diag.md`: `gc.collect()` is the **single largest GIL holder** (33.6% of GIL-holding
  samples, 3.7% of wall) — a main-thread sequential stall where GPU is idle **and** the background
  thread is blocked on the GIL too.
- **Relocation already tried and reverted** (`gil-contention-diag.md` "方案 (d)"): moving the same
  call to the background thread made `idle_frac` *worse* (0.183→0.221) because the pipeline's overlap
  is background-CPU-bound (`_collect(pending)` blocks MAIN waiting on the BG thread) — moving gc onto
  BG lengthens the arm MAIN is already waiting on. **Do not re-propose relocation.** The only
  open lever is **reducing how often the call happens**, not where it runs.
- **Hard constraint — the memory-bounded invariant**: peak RSS 2.82 → 3.07 → 4.04 GB across
  25/121/441 tiles (`bottleneck-list.md`), and per `bottleneck-list.md` line 350 this growth **tracks
  accumulated cell count** (`per_tile_owned` held in RAM until the final global merge), **not tile
  count**. This matters for every option below: a cadence tuned safe on the 441-tile test crop (dense
  tissue, high cells/tile) is not automatically safe at full-WSI scale (35,700 tiles) if a real slide's
  cell density per swept window differs. Treat "safe at N=8 on this crop" as scoped to this crop until
  re-validated, not as a general result.

## 1. Design space — every lever that touches call frequency or per-call cost

The playbook's "cheapest first" ordering below is by **implementation risk × blast radius**, not by
expected win size — several of these are near-zero-cost to try and should be ablated before the
riskier ones even if their theoretical ceiling is smaller.

### Option A — Fixed-N batching (the lever doc 13 already named)

Call `gc.collect()` + `empty_cache()` every N tiles instead of every tile, via a simple loop counter.
Doc 13's own starting point: sweep N ∈ {4, 8, 16}.

- **Ceiling**: at N=8, ~7/8 of the 36.4 s disappears → ≈32 s saved (≈1.075x on top of ceiling budget).
  Diminishing returns as N grows (N=16 saves only ~2 s more than N=8) while RSS risk keeps climbing —
  so the useful sweep range is small; don't bother with N>16 without a specific reason.
- **Risk**: RSS grows monotonically with N (garbage from N-1 tiles accumulates before each sweep).
  This is the one the memory-bounded invariant is about — see §2 for the exact re-validation
  requirement.
- **Effort**: trivial (one counter, one `if idx % N == 0` guard around the existing two lines).

### Option B — Decouple `gc.collect()` cadence from `empty_cache()` cadence

Doc 13's plan text (and the current code) treats the two calls as one unit. They shouldn't be swept
together:

| call | cost (large/441) | % wall |
|---|--:|--:|
| `gc.collect()` | 36.4 s | 6.3% |
| `torch.cuda.empty_cache()` | <1.7 s (bottleneck-list.md: "<0.3%") | <0.3% |

`empty_cache()` is ~20x cheaper and serves a different invariant (VRAM fragmentation, not RSS) —
VRAM headroom is enormous relative to what's ever been used (peak VRAM 4.68 GB of 32 GB across every
round measured). There is no evidence-based reason to also stretch its cadence, and doing so adds a
second RSS-unrelated risk (VRAM fragmentation growth) to the same experiment for a payoff that's
already <0.3% of wall — the kind of "several things at once" the playbook's anti-pattern #7 warns
against. **Recommendation: keep `empty_cache()` every tile, unconditionally; only `gc.collect()`'s
cadence is a variable in this plan.** This isn't a separate experiment — it's a design constraint on
every other option below (all of them keep `empty_cache()` per-tile).

### Option C — `gc.freeze()` after model init (orthogonal to A, near-zero risk)

The three GPU models (`unet`, `cellpose`, `dish_cellpose`) are loaded once before the tile loop
(`run_batch`, `hybrid_pipeline.py:741-743`) and stay alive and reachable for the entire batch. A
generational GC re-scans *every* live object it can reach on each full collection, including those
permanently-alive model objects, every single time. `gc.freeze()` (stdlib, Python ≥3.7) moves
everything currently tracked into a **permanent generation the collector skips** on subsequent
collections — a well-known technique in long-lived-process servers (gunicorn/uwsgi preload use
exactly this). Calling it once, right after the three `_init_*` calls, doesn't change *how often*
`gc.collect()` runs — it changes **how much each call has to scan**, so it stacks additively with
Option A rather than competing with it.

- **Ceiling**: unknown until measured — depends on how large the frozen (model) object graph is
  relative to per-tile garbage. Could be small if per-tile garbage dominates scan cost, or could be a
  meaningful fraction of the 36.4 s if model graphs are large (Cellpose/SAM ViT backbones are not
  small). **Must be isolated (measured alone, gc.collect() still every tile) before combining with
  A** — otherwise a combined win can't be attributed (playbook anti-pattern #6/#7).
  - **Note on scope**: this only moves the *originally-loaded model objects* into the frozen generation. It has no interaction with the RSS-growth mechanism (`per_tile_owned` accumulation), so it does **not** touch the memory-bounded invariant — no RSS re-validation burden beyond the standard check.
- **Risk**: essentially none — it's a scan-scope optimization, not a correctness-affecting change; the collector still runs identically often and still collects the same non-frozen garbage. Still must be measured, not assumed (karpathy_rule: no unverified "should work" claims).
- **Effort**: one line, one call site (`run_batch`, right after `dish_cellpose = _init_dish_cellpose_segmenter()`).

### Option D — Split full collection from generation-limited collection

Instead of a full `gc.collect()` (implicitly generation 2 — scans everything) every tile, call the
cheaper `gc.collect(0)` (or `gc.collect(1)`) every tile, and reserve the expensive full
`gc.collect(2)` (equivalent to bare `gc.collect()`) for every Nth tile. This is a hedge between Option
A (which does *zero* collection on off-sweep tiles) and doing nothing: off-sweep tiles still get
cheap young-generation collection, which catches small short-lived cycles (e.g. per-tile intermediate
objects) without paying the full-graph scan cost every time.

- **Ceiling**: smaller guaranteed win than A at the same N (gen-0/1 collection isn't free, just
  cheaper), but **lower RSS risk at the same N** — this is the fallback if Option A's RSS check fails
  at the N the team wants and a smaller N isn't acceptable for the wall-clock win needed.
- **Risk**: low; RSS growth between full sweeps should be smaller than plain A at equal N since
  young-gen garbage is still being reclaimed every tile — but must still be measured, not assumed
  (the invariant is empirical per `bottleneck-list.md`, not something to reason about from CPython
  internals alone).
- **Effort**: trivial — swap the argument, add the same counter as Option A.
- **When to reach for this**: only if Option A's RSS check (§2) fails at every N in the useful range
  (§1 Option A) — don't pre-emptively build this; it's a fallback, not a parallel track (karpathy_rule
  §2: no speculative flexibility ahead of a demonstrated need).

### Option E — Adaptive, content-triggered sweep (RSS- or cell-count-gated, not tile-count-gated)

Given the memory-bounded invariant is driven by **accumulated cell count**, not tile count
(`bottleneck-list.md` line 350), a fixed tile-count N is a proxy that only holds for slides whose
cell density resembles the 441-tile test crop. A more robust trigger: sweep when
`len(per_tile_owned)`'s cumulative cell count (already computed per tile — `stats["success"]` /
`owned` length) crosses a threshold, or when RSS itself (already sampled by `perf_measure.py`'s
`ResourceSampler`, cheap `psutil` call) crosses a bound.

- **Ceiling**: same as Option A at equivalent effective cadence — this is a *safety/generalization*
  improvement, not a speed improvement. Its payoff is avoiding a full-WSI RSS blowup that a
  crop-tuned fixed-N wouldn't catch, not a bigger wall-clock number on the 441-tile anchor.
  Reading a cell-count counter that's already in scope is free; reading RSS via `psutil` costs a
  syscall per check but is O(µs), negligible next to a 36 ms/tile budget.
- **Risk**: adds a second, less-tested code path (branch logic) — must not become the default without
  first showing fixed-N (Option A) is provably unsafe at full-WSI scale. Per karpathy_rule §2, do not
  build this speculatively.
- **When to reach for this**: **not part of the initial experiment matrix (§4).** Only design/build if
  (a) Option A or D is adopted, and (b) before full-WSI rollout, someone needs to argue a fixed N
  chosen on the 441-tile crop generalizes to real-slide cell density variance. Flagged here so it's
  not rediscovered from scratch later — this doc's own §2 constraint is exactly the reason it might be
  needed.

### Option F — Disable cyclic GC entirely during the tile loop, full collect only at batch end/N

The most aggressive lever: `gc.disable()` after model init, rely purely on CPython refcounting for the
entire tile loop (numpy arrays / torch tensors are non-cyclic in the common case, so most per-tile
garbage is freed immediately on refcount-zero regardless of the cyclic collector), and run a bounded
number of full `gc.collect()` calls (e.g. every N tiles as a backstop, or only once at loop end).

- **Ceiling**: largest of all options — approaches the full 1.083x if the loop truly creates no
  reference cycles worth collecting.
- **Risk**: **highest of all options, direct hit on the memory-bounded invariant.** If *any* per-tile
  code path creates reference cycles (common culprits: exception objects with tracebacks, some
  matplotlib/PIL/third-party patterns, closures capturing `self`), those cycles never get reclaimed
  between sweeps, and RSS growth could become **unbounded in tile count** instead of the current
  "sub-linear, cell-count-driven" pattern — exactly the invariant this repo has protected since doc 10.
  This is not hypothetical: Cellpose/SAM (third-party, not audited here) are exactly the kind of
  complex object graphs where cycles are plausible.
- **When to reach for this**: **last, and only if** Options A–D plateau below the wall-clock target
  *and* a long-run RSS stress test (§4 Exp 6) shows no unbounded growth. Never adopt without the
  stress test — a 441-tile ablation alone is too short to distinguish "no cycles" from "cycles that
  accumulate too slowly to see in 441 tiles but matter at 35,700."

### Option G (out of scope, noted only) — Reduce what's held in RAM instead of collection frequency

The deeper fix for the RSS-growth *mechanism* itself (not `gc.collect`'s frequency) would be
incremental/streaming write of `per_tile_owned` instead of holding all tiles' results until the final
global merge. That's an architecture change to the merge step, not a "frequency reduction" — out of
scope for Priority 1, noted so it isn't silently lost; revisit only if Options A–F together still
leave RSS as the binding constraint on how far N can go.

## 2. The RSS re-validation requirement (applies to every option except C)

Every option above except C (Option C doesn't change collection frequency, only scan scope) changes
how much garbage is allowed to accumulate before it's swept. Per doc 13 §"Plan" and the existing
invariant history, **peak RSS must be measured at the 441-tile scale for every configuration before
it's considered adoptable** — headroom is large (32 GB machine RAM, peaks have never exceeded ~4 GB)
but the invariant itself, not the headroom, is the bar (doc 13 is explicit about this).

Two checks, not one:

1. **Absolute bound**: peak RSS at 441 tiles stays comfortably under existing peaks (4.04 GB
   round-1-era, re-confirm round-3's own peak as the live reference — round 3 hasn't re-stated this
   number independently of round 1/2 in the docs read for this plan, so measure it fresh as part of
   Exp 0 in §4).
2. **Growth-shape check** (new, motivated by §0's cell-count-not-tile-count finding): plot/inspect RSS
   over the run (the `ResourceSampler` CSV `perf_measure.py` already emits at 0.5 s resolution) and
   confirm growth between sweeps is **bounded and resets on each sweep** — i.e. sawtooth, not a
   monotonic ramp that never comes back down. A monotonic ramp is the early signature of exactly the
   "unbounded in tile count" failure mode Option F risks (§1) — catching it at 441 tiles, where it'd
   still be small in absolute GB, is the cheap place to catch it, not at full-WSI scale where it'd be
   catastrophic.

## 3. Instrumentation gap — what `scripts/perf_measure.py` needs before these experiments can run

No code changes in this session, but naming the gap now saves a false start later:

- `perf_measure.py` already buckets `B4_gc_collect` / `B4_empty_cache` timing (via monkeypatching
  `gc.collect`/`torch.cuda.empty_cache` at the `gc`/`torch.cuda` module level — `install_wrappers()`,
  `scripts/perf_measure.py:207-225`) and already samples `rss_gb` every 0.5 s
  (`ResourceSampler`). **Both are cadence-agnostic** — they'll correctly report "N calls, T seconds
  total" and the RSS timeseries regardless of which option is implemented, so no harness change is
  needed to *measure* any option in §1.
- What's missing: a way to select the cadence/mode without editing `hybrid_pipeline.py` per
  experiment. Recommended (for whoever implements, not this session): a `Config` field
  (`gc_collect_every_n_tiles: int = 1`, mirroring the `batch_size: int = 4` pattern at
  `config.py:184`) plus, if Option D or F end up warranted by the ablation results, a
  `gc_collect_mode: str = "full"` field (`"full" | "gen0" | "gen1" | "disabled"`). Default must
  reproduce current behavior exactly (N=1, mode="full") so wiring the field is itself a no-op ablation
  before any sweep touches it — same discipline doc 13 Priority 4 step 1 requires for
  `cellpose_batch_size`.
- Option C (`gc.freeze()`) needs no new config surface to *test* (it's a single unconditional call
  after init, on or off via code presence) but if adopted permanently should probably be unconditional
  rather than flagged — there's no scenario where you'd want it off.

## 4. Experiment matrix (Choose phase — ablation-proof each layer)

All runs: `scripts/perf_measure.py --gpu-dmon`, medium (121-tile) anchor first for cheap iteration,
large (441-tile) anchor to confirm before adopting anything — same harness discipline as every prior
round (§0 of doc 13). Correctness check: cell counts against **round 3's own** `report.csv` reference
(doc 13 §3 — round 3 is not bit-exact with rounds 1/2, so that's the only valid comparison point).
`gc.collect` is memory-only and participates in no computation, so any cell-count delta beyond the
existing GPU-nondeterminism noise floor (±1 cell class, per `gil-contention-diag.md`'s own ablation)
is a red flag, not expected variance.

| # | Option(s) | Config | Scale | Purpose | Pass bar |
|---|---|---|---|---|---|
| 0 | — (baseline) | current code, unmodified | medium + large | Fresh reference point for this specific ablation session (env/driver may have drifted since round 3's own numbers) | n/a — this *is* the reference |
| 1 | C alone | `gc.freeze()` after init, collect still every tile | medium | Isolate freeze's own contribution before combining with anything | Wall/GC-time delta vs Exp 0, correctness bit-exact-equivalent (gc.freeze changes no computation) |
| 2a/b/c | A alone (+ B constraint: empty_cache stays per-tile) | N=4 / 8 / 16, full collect | medium | Cheap sweep to find the useful N range | RSS growth-shape check (§2) passes at each N; pick candidate N(s) for large-scale confirm |
| 3 | A (winning N from 2) | best N from Exp 2 | large | Confirm medium-anchor N choice holds at large scale | Wall-clock win vs Exp 0 large; RSS absolute + growth-shape check (§2) both pass |
| 4 | A + C combined | best N + freeze | medium then large | Confirm the two stack additively (not redundant, not interfering) — required by playbook anti-pattern #6 before adopting both | Combined win ≈ (Exp1 win) + (Exp3 win) within noise; if not additive, investigate before adopting both |
| 5 | D (fallback) | gen0/1 every tile, full collect every N | medium | **Only run if Exp 2/3's RSS check fails at every N in the useful range** | Same bars as Exp 2/3, on the fallback path |
| 6 | F (stretch) | `gc.disable()`, full collect every N (or loop-end only) | large, **extended stress run** (repeat the 441-tile crop back-to-back, e.g. 3-4x concatenated, to approximate a much longer tile sequence within a feasible wall-clock budget) | Rule out unbounded cross-tile RSS growth (§1 Option F risk) before this is ever considered adoptable | RSS growth-shape check (§2) passes over the **entire extended run**, not just 441 tiles — a monotonic ramp anywhere fails this option outright |

Run order follows §1's risk ordering: **0 → 1 → 2 → 3 → 4**, stop and adopt if 4 clears its bar and
meets the wall-clock target. Only fall through to **5** if 2/3's RSS check fails, and only reach **6**
if 4's ceiling is measured and judged insufficient against the ~44 min/full-WSI-run target *and*
someone is willing to pay for the extended stress run's engineering time — per the playbook's Choose
step, a flat or negative result at any stage is a valid, written-up outcome, not a reason to
automatically escalate to the next riskier option.

## 5. Recommended order and stopping rule

1. **Exp 0 → 1 (C alone)**: near-zero cost to try, isolates a number that's needed anyway before Exp
   4 can attribute anything. If it shows a real win with zero RSS/correctness risk, it's close to a
   free adopt regardless of what happens with A.
2. **Exp 2 (A sweep, medium)**: this is the lever doc 13 named as "the one untried lever" — run it
   first among the frequency-changing options since it's the simplest to implement and reason about.
3. **Exp 3 (A confirm, large) → Exp 4 (A+C combined)**: only proceed to large-scale and combination
   once medium-scale numbers look promising; don't burn large-anchor runs (which cost more wall-clock
   to execute, per doc 13 §0's server-sharing note) on options that already look weak at medium scale.
4. **Stop here if Exp 4 clears the target** (recall: the whole Priority-1 ceiling is 1.083x / ~44 min
   at full-WSI scale — Exp 4 doesn't need to hit that exactly, but should capture the large majority of
   it while keeping RSS bounded). Write up whichever N and freeze combination wins, exactly like
   `gil-contention-diag.md`'s ablation table, including the negative results from any N that failed
   the RSS check.
5. **Only escalate to Exp 5/6** if 2-4 underperform relative to the 1.083x ceiling by enough to matter
   *and* someone explicitly decides the extra risk (Exp 6) or extra complexity (Exp 5) is worth it —
   this is a judgment call for whoever runs the experiments, not a default continuation.

Do not skip straight to Option F (Exp 6) because it has the largest theoretical ceiling — that inverts
the playbook's cheapest-first ordering and stacks the single highest-risk change (direct hit on the
memory-bounded invariant, §1) before cheaper options have even been measured.

## 6. Success criteria (mirrors doc 13 §5, scoped to this priority)

- Judged by **end-to-end wall-clock** on medium/large anchors via `--gpu-dmon`, never a micro-benchmark
  of `gc.collect()` alone.
- Every option's correctness check is against **round 3's own cell counts**, not round 1/2's (doc 13
  §3).
- Every option except C requires the two-part RSS check in §2, at the **441-tile scale**, before
  being considered adoptable — this is non-negotiable per the existing memory-bounded invariant
  history (doc 10 §5.4/§7.2, doc 11's own ablation write-up).
- Options must be ablated **individually before being combined** (Exp 1 before Exp 4) — a combined
  win that can't be decomposed into "how much did each layer contribute" doesn't clear the playbook's
  anti-pattern #6/#7 bar.
- A flat or negative result at any experiment is a valid, recordable outcome — write it up
  `gil-contention-diag.md`-style (that file's own "方案 (d) ablation" section is the template) rather
  than silently dropping it or chasing a positive result past what the data supports.
- All measurement runs follow doc 13 §0: confirm the GPU is idle (`nvidia-smi`) before launching, and
  drop a `pip freeze` next to the metrics for every round.
