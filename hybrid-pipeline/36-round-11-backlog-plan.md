# 36 — Round 11: closing the round-10 backlog — design plan

> **Design-only document — no pipeline code changed here.** Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose. Synthesizes the seven follow-ups
> [`35-round-10-backlog-implementation.md`](./35-round-10-backlog-implementation.md) §8 left open into one
> sequenced plan. `git log` on `backend/algorithms/hybrid/` shows nothing past `7726bc5` ("perf: pipeline
> tile read prefetching into multiprocess workers (#34)", doc 35's Step 2), so doc 35's disposition table is
> still current. The working tree does carry uncommitted changes to `scripts/perf_measure.py`,
> `scripts/stitch_probe.py`, and a new `tests/test_mp_tile_read_prefetch.py` — these match doc 35's own
> `gc_refreeze` reporting, `--candidates` mode, and Step 2 test additions, i.e. round 10's own measurement
> tooling, not new work. This document doesn't touch them.

## 0. Discover — everything doc 35 §8 left open

| # | source | item | status left in round 10 |
|---|---|---|---|
| 1 | doc 35 §4.4, §8 item 1 | `B1_m3b_cellpose` is 751.0 s slower than the doc 27 baseline (6,358.9 → 7,109.9 s, **+11.8%**), on the critical arm, same tile count. Nothing round 10 touched M2/M3b | **Unattributed.** `e806938` was named a suspect but not verified. Largest single unexplained number in round 10, and it sits on MAIN — it gave back 21% of what Option H won |
| 2 | doc 35 §6.2, §8 item 2 | After a correct fail-fast abort, the parent process doesn't exit — sat alive doing nothing for **49 minutes** (`workers=4` OOM) and separately for **2 hours** (a `workers=1` run that had already printed its final JSON, no multiprocessing involved) | **Not fixed.** Doc 35 calls it "the most dangerous defect found" — it converts a correct abort/completion into a job that never returns, on exactly the unattended full-slide runs `--resume` exists for |
| 3 | doc 35 §6.2, §8 item 3 | `workers=4` (the shipped production recommendation) hit the `workers≥6` CUDA allocator balloon (doc 19 #7b / DISCOVERED #2) — one worker took 24.76 GiB of a 31.36 GiB card and starved its three siblings. Intermittent (11 other runs succeeded). `cuda_alloc_conf="expandable_segments:True"` is an existing, wired knob that has **never been measured at `workers=4`** | **Not swept.** `scripts/alloc_conf_probe.py` already exists for this; it hasn't been pointed at `workers=4` |
| 4 | doc 35 §4.3, §8 item 4 | Doc 33's Option K (Option L's real MAIN-arm contribution) is only an **upper bound**: the measured 2,368.5 → 1,581.2 s `B2r_tile_read` drop is confounded by `d6592c3` removing 275 GB of concurrent per-tile writes, which independently relieves the page-cache pressure doc 27 §6.4 traced the read cost to | **Formally unsettled.** Doc 35: *"The clean experiment is a second full-slide run with `--no-prefetch` on today's code — ~2.9h — and it is the only thing that would turn 1.208× from an upper bound into a measurement."* |
| 5 | doc 35 §3.3, §8 item 5 | Phase D Phase 2 (read pipelining) — the only remaining lever after Phase 1.5 killed candidates B/C on codec/container grounds | **Measured hypothetical ceiling only** (~1.47×/1.80× on Phase D alone, ~1.094× end-to-end at `workers=4`) — below what doc 32 projected, for a threaded reader that does not exist |
| 6 | doc 35 §5, §8 item 6 | `depth=2/3` tuning for `prefetch_tile_reads` | **Measured, not built** — gate is met (background read 35.32 ms > 28.39 ms UNet-only hide budget) but ceiling is 106.6 s = **1.04% of wall** |
| 7 | doc 35 §3.4, §8 item 7 | QuPath open-and-render-identically pass on `/home/taro/r10_phase_d/cand_{A,B,C}.tiff` | **Unrun**, and doc 35 already notes it is "no longer decision-relevant" since no Phase D candidate is being adopted |

## 1. Analyze — what actually depends on what, and what's already been ruled in/out

**Item 1 is cheaper than it looks, and the obvious suspect is very likely wrong.** `e806938`'s full diff
touches exactly two code files: `backend/algorithms/hybrid/hybrid_pipeline.py` (one added docstring comment
line) and `backend/algorithms/hybrid/m4_module/csv.py` (`DotStatsSummary.from_results`'s valid-cell filter and
`write_summary_csv`'s labels — dropping the `her2_dot_count >= 1` requirement and relabeling output columns).
**Nothing in that diff touches M2 (`m2_segmentation.py`), M3b, or any Cellpose call** — the change is entirely
downstream, in CSV export filtering that runs after all GPU work for a tile is done. It cannot plausibly move a
GPU-forward timing bucket. `d6592c3` ("M0 split", landed in the same window between doc 27's baseline and doc
35's run) is the more plausible structural candidate: it moved 1,065 lines out of `hybrid_pipeline.py` and
added 547 new lines to `m0_tile_runner.py`, which is exactly where the M2/M3b GPU-call orchestration
(`_process_precut_tile_gpu` per `backend/algorithms/hybrid/CLAUDE.md`) now lives. A pure code-motion refactor
changing a hot-path timing bucket by +11.8% would itself be worth knowing about, independent of any
"judgement criteria" story. The remaining candidate is plain **n=1 run-to-run variance** — doc 27's baseline
and doc 35's Step 4 are each a single ~3h run with no repeat, so a 751 s / 6,358.9 s (11.8%) swing has no error
bar to compare against yet, the same caveat doc 35 §2.3 raised about six repeats not resolving a 0.83% crop
effect.

**Items 2 and 3 are both reliability defects already shipped in `_run_tiles_multiprocess`
(`m0_multiprocess.py:250-398`) and its `_kill_all()` helper, but they are not obviously the same bug.** The
`workers=4` OOM hang fits the mechanism doc 35 §6.2 already names — a terminated worker plus a
`multiprocessing.Queue` whose feeder thread can't flush to a closed pipe, joined at process exit
(`_kill_all`'s `p.join(timeout=10)` at line 306, or the final `p.join(timeout=60)` / `_kill_all()` at
lines 396-397). **But the second hang has no multiprocessing in it at all** — it was a `workers=1` run, which
never calls `_run_tiles_multiprocess`, after its final JSON had already printed. Whatever kept that process
alive is not a queue-join problem; it's a non-daemon thread (or an unclosed resource with an `atexit`/finalizer
hook) surviving past `main()` on the single-process path too — candidates include the `tile-cpu`/`tile-read`
`ThreadPoolExecutor`s Option L added in round 9, though those use `with` blocks that should join on their own.
Item 2 therefore needs its own diagnosis before assuming item 2 and item 3 share a root cause; they are listed
together only because both live in the multiprocess path as shipped and both surfaced in the same round-10
ablation session.

**Item 3 is independent of item 2 and cheap**: `config.cuda_alloc_conf` (`config_example.py:181-186`) and
`scripts/alloc_conf_probe.py` already exist for exactly this sweep; it has just never been pointed at
`workers=4` specifically. It does not block or get blocked by anything else in this document.

**Item 4 is the expensive, non-repeatable resource, and its dependency on items 1-3 is about protecting
operator time, not measurement validity.** The 2.9h run is at `workers=1` (matching doc 35 Step 4, so its
`B2r_tile_read` number is directly comparable), and item 3 (`workers=4`-specific) genuinely doesn't touch it.
Item 1's regression, whatever its cause, is present in both arms of item 4's `--no-prefetch` on/off comparison
equally (same code, only the flag differs) — it doesn't confound item 4's read-only question, so item 4 does
not need to wait on item 1 being resolved. **Item 2 does matter operationally**: one of the two hangs observed
was on exactly this run shape (`workers=1`, no multiprocessing), so running item 4 before item 2 lands risks
repeating the "job never returns, has to be noticed and killed by hand" failure this round already hit twice —
the same reasoning doc 34 §1 used to sequence Option L's prefetch before doc 34's own expensive run.

**Items 5 and 6 require no new action.** Doc 35 already closed both against the playbook's own Amdahl
threshold (§3.3's "requiring work that does not exist" for item 5; §5's 1.04% ceiling for item 6) — they are
listed here only so this document's accounting is complete, the same convention doc 34 §0 used for its own
already-closed items 6-7.

**Item 7 needs no scheduling either.** Doc 35 §3.4 already reclassified it as informational: no Phase D
candidate is being adopted, so nothing currently depends on its outcome. It is dropped from active tracking
below rather than carried forward as a step.

## 2. Plan — sequencing

```
1. Attribute B1_m3b_cellpose's +751 s regression  →  verify: crop-scale composition-matched A/B across
   (cheapest; largest unexplained MAIN-arm number)    d6592c3 (or targeted profiling of today's M3b/
                                                        Cellpose call) produces a measured cause, not a
                                                        named suspect
2. Diagnose + fix the post-fail-fast hang         →  verify: a deliberately-triggered fail-fast (existing
   (most dangerous defect: turns a correct abort     verify_mp_failfast.py-style trigger) and a normal
   or completion into an unattended multi-hour        workers=1 completion both exit within seconds of
   stall)                                             their last log line, no manual kill needed
3. Sweep cuda_alloc_conf=expandable_segments:True →  verify: enough workers=4 repeats to have a shot at
   at workers=4 (existing knob, never measured here)  the intermittent balloon (doc 35 saw it in 1/12);
                                                        no OOM across the sweep, wall/RSS not regressed
4. Full-slide workers=1 --no-prefetch run (~2.9h) →  verify: B2r_tile_read's real, unconfounded inline
   (settles Option K's real number, not an upper       total on today's code vs doc 35's 1,581.2 s
   bound)                                              prefetch-on number — turns 1.208x into a
                                                        measurement
5. No action: Phase D read pipelining, depth=2/3, →  recorded only — already closed by doc 35 against
   QuPath pass on Phase D candidates                   the Amdahl threshold or no longer decision-relevant
```

Sequencing follows the same rule doc 34 used: cheap and independent items go first (1-3, in order of
information value — largest unexplained number, then most dangerous defect, then a cheap existing-knob sweep);
the one expensive, non-repeatable run (4) goes last, after the defect that could waste its own wall-clock time
is addressed. Item 5's contents are not re-opened — they were closed on Amdahl grounds doc 35 already
established, and re-litigating a closed item without new information is exactly the "optimizing before
measuring" anti-pattern this series has been enforcing since round 1.

### 2.1 Step 1 detail — attributing the cellpose regression

Do not start from `e806938`. §1 above already rules it out by direct diff inspection — the diff has zero
overlap with M2/M3b/Cellpose code, so no amount of further reasoning about "judgement criteria" changes will
explain a GPU-forward timing shift. The concrete, cheap check: run doc 35's own composition-matched `match24`
anchor (576 tiles, the same anchor Step 2's ablation used) once on the code immediately before `d6592c3` and
once on current `HEAD`, both at `workers=1`, comparing `B1_m3b_cellpose`'s bucket total. This isolates whether
the M0-split refactor itself — not any GPU model or algorithm change — shifted the cost, and it's minutes, not
hours, unlike re-running the full slide. If the crop anchor shows no shift, the two remaining candidates are
(a) some other landed commit not yet considered, checked the same way, or (b) plain run-to-run variance on an
n=1 full-slide comparison — in which case this is recorded as *"not reproduced at crop scale, unresolved
without a second full-slide run"* rather than chased further, since a second 2.9-3.8h run to chase an 11.8%
regression that might just be noise is exactly the kind of spend item 4 already has a stronger claim on.

### 2.2 Step 2 detail — the post-fail-fast / post-completion hang

Two different run shapes hung (`workers=4` after fail-fast; `workers=1` after a clean finish with no
multiprocessing), so treat this as **two candidate mechanisms until proven otherwise**, not one bug with two
symptoms:

- **The `workers=4` case** matches doc 35 §6.2's own diagnosis: a terminated sibling worker plus a
  `multiprocessing.Queue` whose feeder thread can't flush to a closed pipe, joined at interpreter exit. Fix
  candidates: a bounded/timed join on `task_q`/`result_q` before process exit, or draining the queue before
  calling `p.terminate()` in `_kill_all()` (`m0_multiprocess.py:301-306`) rather than after.
- **The `workers=1` case** cannot be this mechanism — `_run_tiles_multiprocess` is never called on that path.
  It needs its own audit of what non-daemon threads or open resources outlive `main()` on the single-process
  route (`hybrid_pipeline.py`'s `run_batch`), per doc 35's own framing: *"worth a bounded join on the queue
  and an audit of what non-daemon threads outlive `main()`."* `codegraph_explore` on `main()`'s call graph
  before editing (project rule: verify callers before modifying), not assumed to be the same fix as the
  `workers=4` case.

Verification: force both shapes deliberately (an injected OOM or bad-tile fail-fast for the `workers=4` case,
a normal small run for the `workers=1` case) and confirm the process exits within seconds of its last log line
in both, rather than requiring a human to notice and kill it.

### 2.3 Step 3 detail — `expandable_segments` at `workers=4`

`config.cuda_alloc_conf` (`config_example.py:186`) already wires `PYTORCH_CUDA_ALLOC_CONF` into the spawned
workers' environment before their first CUDA allocation (`m0_multiprocess.py:280-284`), and
`scripts/alloc_conf_probe.py` already exists to sweep it — this step is pointing an existing tool at an
untested configuration, not building anything. Because the balloon is intermittent (doc 35: 1 failure in 12
runs), the sweep needs enough repeats at `workers=4` with `expandable_segments:True` to plausibly catch it if
it still occurs, not a single pass. Report both outcomes: whether the OOM signature reproduces at all with the
knob on, and whether the knob costs anything in wall/RSS on the runs that don't OOM (expandable-segments
allocation has its own overhead in some PyTorch/CUDA combinations).

### 2.4 Step 4 detail — the clean Option K experiment

One `workers=1` full-slide run (~2.9h) with `--no-prefetch` (which `perf_measure.py` already sets on both the
parent's binding and `HYBRID_MP_NO_PREFETCH`, per doc 35 §2.2, so the control genuinely takes effect) on
today's code — i.e., after `d6592c3` and everything since, so no write-removal confound is possible; the only
difference from doc 35's Step 4 run is the prefetch flag. Collect `B2r_tile_read`'s total the same way Step 4
did (doc 33 §1's tissue/background split, already in `perf_measure.py`) and compare directly against Step 4's
1,581.2 s prefetch-on number — both numbers now come from the same code, so the difference is Option L's real
contribution, not an upper bound. This is the only way left to answer that question; a second crop-scale
ablation was already tried at `workers=4` in round 10 and landed inside the noise floor (doc 35 §2.3), and this
project's own crop anchor cannot resolve an effect this small — only full slide scale can.

### 2.5 Step 5 — explicitly no action, listed for completeness

Nothing here is scheduled. Phase D Phase 2 stays closed per doc 35 §3.3 (ceiling below what was projected, for
a threaded reader that doesn't exist); `depth=2/3` stays closed per doc 35 §5 (1.04% ceiling, below the
playbook's own ~10% line by an order of magnitude); the QuPath pass on the Phase D candidates is dropped from
active tracking since no candidate is being adopted and nothing depends on its result. Re-opening any of these
without new information would repeat anti-pattern #1 (optimizing before measuring) on questions doc 35 already
measured and closed.

## 3. Choose — what this document commits to and what it doesn't

This document sequences the backlog; it does not pre-decide **why** `B1_m3b_cellpose` regressed (only that
`e806938` is ruled out by direct diff inspection and `d6592c3` is the more plausible remaining candidate,
pending the crop-scale check in §2.1) or **which mechanism** explains each hang shape (§2.2 explicitly treats
them as two candidates, not one) — both are measurement-gated by design, the same discipline doc 34 §3 applied
to its own step 3/step 5 unknowns. What it does commit to: step 1 goes first because it's the cheapest way to
close the largest unattributed number on the critical arm, and because a structural refactor quietly eating
back optimization wins is worth catching regardless of its raw Amdahl share (11.8% relative to its own bucket,
7.35% of full-slide wall — not huge, but it's a regression against prior work, not a proposed new
optimization, so the playbook's ~10%-floor framing for *pursuing* a win doesn't cleanly apply to *recovering*
one). Steps 2 and 3 are both cheap and independent of step 1, ordered by how dangerous each defect is. Step 4
is sequenced last because it is this backlog's one expensive, non-repeatable resource, and because item 2's
hang risk makes it worth landing (or at least understanding) before spending another unattended 2.9h run —
mirroring exactly the reasoning doc 34 §1 used to put Option L's prefetch before its own full-slide run. Step
4 is scheduled despite being expensive not because Amdahl favors it (Option L is already shipped either way)
but because doc 35 leaves a shipped change's headline number as an unconfounded upper bound, and turning that
into ground truth protects round 10's own disposition table from being cited more strongly than it can support.

## 4. Disposition

| item | this document's action |
|---|---|
| `B1_m3b_cellpose` +751 s regression | **Scheduled, step 1** — `e806938` ruled out by diff inspection (touches only CSV export, not M2/M3b); crop-scale A/B across `d6592c3` is the cheap next check |
| Post-fail-fast / post-completion hang | **Scheduled, step 2** — treated as two candidate mechanisms (multiprocess-queue-join vs. single-process thread/resource leak) until the audit says otherwise |
| `workers=4` CUDA allocator balloon | **Scheduled, step 3** — sweep the existing `cuda_alloc_conf` knob at `workers=4` with `scripts/alloc_conf_probe.py`, enough repeats to catch an intermittent failure |
| Full-slide `workers=1` `--no-prefetch` run | **Scheduled, step 4** — after steps 1-3 ship, ~2.9h, turns Option K's 1.208× upper bound into a measurement |
| Phase D Phase 2 (read pipelining) | **No action** — correctly closed in round 10, ceiling below what was projected |
| `depth=2/3` tuning | **No action** — correctly closed in round 10, 1.04% ceiling |
| QuPath pass on Phase D candidates | **No action, dropped from active tracking** — informational only, nothing depends on it |
