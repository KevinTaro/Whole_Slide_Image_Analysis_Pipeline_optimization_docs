# 21 — Cross-tile multiprocessing: implementation & measurement record (round 5)

> Executes [`20-cross-tile-multiprocessing-plan.md`](./20-cross-tile-multiprocessing-plan.md) §3,
> steps 1–6, in order. Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose; cheapest signal first; every layer ablation-proved;
> **correctness is a veto**. **This document changes pipeline code** (doc 20 was planning-only) —
> see §8 for the complete diff surface.
>
> Round-5 anchors, git `9f02be1` + the changes in §8, RTX 5090 / driver 580.159.03 / Compute Mode
> `Default`, cellpose 4.2.1.1, torch 2.11.0+cu130, config hash `ad41c42f` (unchanged — no config
> field was added this round), same `test_picture/_roi_crops/{med,large}` crops as rounds 1–4 plus a
> new 25-tile `small` crop (§3). Raw artifacts: `measurement/_metrics_r5/` (incl. `env_stamp_r5.txt`,
> `pip_freeze.txt`), per-run `report.csv`/`summary.txt` in `measurement/runs_r5/`.

## 0. Protocol

Same protocol as doc 18 §0, and the same checkpoint hashes, so this round is correctness-comparable
to rounds 3–4:

- GPU confirmed idle before **every** launch — 89 MiB / 0%, no compute processes. The sweep loops
  block on `memory.used < 200 MiB` between runs rather than assuming it.
- `--gpu-dmon --workers 8` on every run; `pip freeze` + env stamp (incl. checkpoint SHA-256) beside
  the metrics (`19-open-backlog.md` §1 item 9).
- **n=2 at every anchor/configuration, n=3 for the recommended config at large.** Measured spreads
  this round were unusually tight — medium 0.4–2.2%, see §4 — so n=2 is interpretable here; doc 16
  follow-up #1's warning was about ±3% B1 variance, which the multiprocess configurations do not
  exhibit.
- Correctness reference: doc 18 §1's recorded round-4 figures — large **13152–13153** cells / 378
  success / 63 skipped; medium **3647–3649** / 103 / 18.

**One measurement capability is genuinely lost under multiprocessing, and it is not a small
caveat.** `perf_measure.py` measures by monkeypatching the `hybrid_pipeline` namespace **in the
parent process**. Under `--mp-workers N` every tile is processed in a spawned child, so all
worker-side buckets (`B1_*`, `B2_*`, `B3_*`, …) come back **empty**, and `arm_report.py`'s
`wall ≈ max(MAIN, BG) + outside` decomposition does not merely lack data — it no longer describes
the run, which now has N MAIN arms and N BG arms. Doc 20 §3 step 4 anticipated this and asked for
"an N-arm model (or per-process wall/idle breakdown)". What this round actually reports is
end-to-end wall plus the device-side `dmon` view (§4), which is what the playbook's step 4 says
decides anyway ("decide by end-to-end wall-clock only, never by micro-benchmark"). Per-bucket
attribution inside workers is recorded as follow-up #1 (§10), not claimed.

## 1. Step 1 — the gate: do independent CUDA contexts actually overlap on this card?

Doc 20 §3 step 1 made this the gate for everything: *"If they fully serialize, the ceiling for
B/C/D collapses toward ~1.0x and this whole line of work should stop right here."* It is also the
document's own "biggest open unknown" (§2 Candidate B), because without MPS the GPU's scheduler
time-slices between contexts rather than co-executing them.

### 1.1 Method

`scripts/mp_concurrency_probe.py` (new, §8) measures **throughput scaling, not a micro-benchmark**.
Every process performs the same fixed work *W*; a `multiprocessing.Barrier` makes the timed sections
genuinely coincide, so what is compared is one process doing *W* against *N* processes each doing
*W*:

```
speedup(N) = N · T(1) / T(N)      speedup ≈ N → full overlap;  ≈ 1.0 → full serialization
```

All processes are `spawn` (never `fork` — doc 20 Candidate E). Warmup runs before the barrier so JIT
/ autotune / allocator effects are outside the timed window.

### 1.2 Two synthetic controls that bracket the answer

The answer depends on *why* the device is idle, so the real workload is bracketed by two controls
that must come back with known answers — if they don't, the harness is wrong, not the GPU.

| workload | N=1 | N=2 | N=3 | N=4 | N=6 |
|---|--:|--:|--:|--:|--:|
| `sm` — large matmuls (SM-saturated) | 1.00 | **0.94** | — | **0.94** | — |
| `launch` — tiny-kernel Python loop (launch-bound) | 1.00 | **1.45** | **1.45** | **1.45** | **1.45** |

*(speedup; `sm` at 400 units, `launch` at 1000 units, so each timed window is 5–25 s.)*

Both controls behave exactly as theory demands, which validates the harness:

1. **SM-saturated work does not overlap — it degrades slightly (0.94x).** A device already at 71%
   mean SM in one process has nothing left to give a second; the 6% loss is context-switch overhead.
   This is the "adding processes buys nothing" signature, measured directly rather than assumed.
2. **Launch-bound work overlaps, but hits a hard ceiling of 1.45x at N=2 and never moves again.**
   Throughput is **236.65 units/s at N=2, 3, 4 and 6 — identical to four significant figures**,
   against 163.22 at N=1. That flatness is the important part: it is not gradual saturation but a
   sharply-defined driver/context-scheduler limit. Mean SM was still only 55–64% at those points, so
   the device was *not* out of SM capacity — this is serialization, not exhaustion.

Finding (2) is the one that matters for this pipeline, because doc 18 §3 established that the
*larger half* of all device idle is intra-forward, kernel-launch-bound Cellpose/SAM Python loops
that no single-process restructuring can reach. **1.45x is the upper bound on what multiprocessing
can recover from that component on this card without MPS.**

### 1.3 The decisive real-model probe

Controls are not the pipeline. The `models` workload loads **all three real models per process**
(its own CUDA context, exactly as Candidate B/D would) and runs the **unmodified**
`_process_precut_tile_gpu` over real precut tiles:

| N | wall (16 tiles/proc) | throughput | **speedup** | efficiency | mean SM% |
|--:|--:|--:|--:|--:|--:|
| 1 | 10.198 s | 1.57 | **1.00** | 100% | 35.2 |
| 2 | 12.904 s | 2.48 | **1.58** | 79% | 45.7 |
| 3 | 15.944 s | 3.01 | **1.92** | 64% | 73.6 |
| 4 | 18.632 s | 3.43 | **2.19** | 55% | 74.0 |

**Step 1 is decisively positive.** Kernels from independent processes overlap; the B/C/D line is not
stopped out. The real per-tile GPU front scales *better* than the launch-bound control's 1.45x
because it is a mixture: alongside the three forwards it carries genuinely parallel CPU work
(`_read_rgb`, the M1 overlay glue, `clear_slide_edge_cells`) that spreads across the 20 available
cores, on top of the launch-gap interleaving.

**Read this as the GPU-arm number only, not as a pipeline prediction.** A real Candidate D worker
*also* runs the BG CPU arm (`detect_all_dots`, PNG encode, per-cell crops) concurrently, which
competes for the same cores. Whether the end-to-end number lands above or below 2.19x is a question
only §4 can answer — and it landed materially *above*, for a reason worth recording (§4.2).

### 1.4 Per-process cost — doc 20 §5's unresolved prerequisite

Doc 20 §5 lists per-process CUDA context memory overhead as unmeasured, and notes it — not the
2.79 GB weight figure — determines the real *N* that fits in 32 GB. Measured at N=1 (see the caveat
below):

| quantity | measured |
|---|--:|
| CUDA context overhead (before any weights) | **88.1 MB** |
| the three models' weights | **828.4 MB** |
| per-process total after init | **916.5 MB** |
| model init, per worker | **3.14 s** |

**Caveat on this table, found while doing it:** `torch.cuda.mem_get_info()` reports **device-wide**
free memory, so with N>1 concurrent children each child's before/after delta silently includes its
siblings' allocations. **Only the N=1 row is trustworthy** and the probe's N>1 per-process VRAM
columns should be ignored. The trustworthy per-N figure is the device-level `dmon fb` peak, reported
in §4 — which is also what doc 18 §7.2 already established as the reliable VRAM counter.

The 916.5 MB steady-state figure is far below the 2.79 GB the backlog quotes; that 2.79 GB is the
*peak during forwards* (activations included), which `dmon fb` confirms at ~2787 MB per process
(§4). Both numbers are right, they measure different things. `2.79 GB × N` remains the correct
sizing basis.

## 2. Step 2 — the concurrency knee

Doc 20 §3 step 2 asks where adding a process stops buying wall-clock. The synthetic and real answers
differ, and the difference is informative:

- **Launch-bound synthetic: the knee is exactly N=2.** N=3, 4 and 6 return byte-identical
  throughput. Nothing beyond two processes reaches that idle component at all.
- **Real GPU front (§1.3): still rising at N=4** (marginal speedup +0.58, +0.34, +0.27), because the
  parallel CPU portion keeps scaling after the GPU portion has stopped.
- **Full pipeline (§4): the knee is N=3**, where efficiency is still 93% and marginal gain collapses
  from +0.70 to +0.27 at N=4.

The knee is therefore a property of the whole workload, not of the GPU alone, which is why doc 20
was right to require the prototype (step 3) rather than sizing *N* from the probe.

## 3. Step 3 — Candidate D at the smallest scale, correctness first

Doc 20 §3 step 3 requires the prototype validated at the small anchor — per-cell correctness **and**
fail-fast — **before** ever running at medium/large. Both were done first, in that order.

A 25-tile `small` anchor did not exist (`_roi_crops/` had only med/large), so one was cut: a
4096×4096 crop of `med_*`, giving exactly the 5×5 = 25-tile grid the earlier rounds' "small" anchor
used.

### 3.1 What was built

Candidate D as doc 20 §2 specifies it — the smallest new-code surface, reusing proven code:

- `run_batch(..., workers: int = 1)`. **`workers=1` is the default and is today's code path
  unchanged** — the diff shows exactly three deleted lines in the single-process path (the model
  `_init_*` calls, moved below the new branch). This is doc 20 §1 item 7 satisfied by construction:
  `backend/api/hybrid.py`'s `/api/hybrid/tile` does not pass `workers`, so single-tile API requests
  never pay N× model init.
- `_mp_tile_worker` — a `spawn`ed worker that inits the three models once, then drains a shared work
  queue. Its loop is a **line-for-line transcription of `run_batch`'s single-process loop**,
  including the depth-1 GPU/CPU overlap (background thread runs the previous tile's CPU back end
  while the main thread runs the next tile's GPU front). Keeping that was not optional: it is worth
  the −8.0% doc 18 §2 measured, and dropping it would have made each worker a pre-round-3 serial
  version.
- `_run_tiles_multiprocess` — **dynamic work queue**, not static round-robin, exactly as doc 20 §2
  recommends: per-tile cost varies enormously (background tiles short-circuit; tissue-dense tiles
  run all three forwards) and is unknown upfront. A feeder thread pumps the queue so a streaming
  `PrecutStream` still overlaps cutting with analysis.
- `_finish_batch` — the global merge/renumber/export/stitch tail, **extracted verbatim** and shared
  by both paths. This is how doc 20 §1 item 1 is discharged structurally rather than by promise:
  there is exactly one implementation of the global renumbering, it runs once, in the parent, and
  workers never renumber.

### 3.2 Correctness veto — passed

Small anchor, `w1` (single-process) as reference:

| config | wall | cells | matched | reddot max\|Δ\| | blackdot max\|Δ\| | score max\|Δ\| | X-flips |
|---|--:|--:|--:|--:|--:|--:|--:|
| `w1_small` (reference) | 37.0 s | 1015 | 1015 | 0 | 0 | 0 | 0 |
| `w2_small` | 19.8 s | **1015** | 1014 | **0** | **0** | **0** | **0** |
| `w3_small` | 16.5 s | **1015** | 1015 | **0** | **0** | **0** | **0** |

Identical cell counts, identical tile outcomes (24 success / 1 skipped everywhere), and **zero**
delta on every per-cell field. This is a stronger result than doc 18's adopted changes achieved,
which is expected: unlike moving work between threads, nothing about the per-tile computation
changed here at all.

### 3.3 Fail-fast injection — passed, after fixing the test

Doc 20 §3 step 3(b) requires deliberately injecting a bad tile and confirming the **whole** run
aborts and **every sibling process actually terminates**. `scripts/verify_mp_failfast.py` (new, §8)
corrupts one tile **in the middle of the grid** (aborting on tile 1 would pass even with a pool that
never stops siblings) and checks three things on the real `run_batch` path:

| check | `workers=3` | `workers=1` (control) |
|---|---|---|
| `run_batch` raises rather than returning stats | ✅ `RuntimeError` | ✅ `RuntimeError` |
| aborts early (fewer tiles written than the grid) | ✅ 9/25 | ✅ 12/25 |
| no worker process survives the raise | ✅ | ✅ |

**The first version of this test reported a false failure, and the reason is worth recording.** It
flagged a surviving child process — but flagged it on the `workers=1` path too, which spawns no
workers at all. The survivor was `multiprocessing.resource_tracker` / joblib's loky semaphore
tracker: legitimate, shared, long-lived children that outlive any batch by design. The check now
matches `spawn_main` children and excludes trackers. A test that fails on the control was testing
the wrong thing.

### 3.4 Two real defects found by writing the fail-fast test

Both were in code I had just written, both are invisible to a green benchmark, and both are fixed:

1. **The feeder thread kept cutting the slide after the batch aborted.** It is a daemon thread
   iterating `PrecutStream`; on fail-fast the workers were killed but the feeder went on pulling
   tiles, i.e. it would cut an entire WSI to disk for a batch that had already been abandoned —
   and `run_batch` runs inside a long-lived API server. Fixed with a `stop_feeding` event checked
   per tile and set before `_kill_all()`.
2. **All workers exiting cleanly with results missing would have hung, then looked successful.**
   The liveness check only raised on a *non-zero* exit code. Now, if every worker has exited and
   fewer than `total` tiles were collected, the batch fail-fasts rather than proceeding to stitch a
   slide with an undocumented hole — the exact failure mode doc 20 §1 item 2 exists to prevent.

## 4. Step 4 — medium and large anchors

Full protocol per §0. `scripts/mp_scaling_report.py` (new, §8) produces these tables.

### 4.1 Results

**Medium / 121 tiles (n=2 per configuration):**

| workers | wall | spread | **speedup** | efficiency | mean SM% | near-idle (SM≤3) | FB peak | peak RSS | cells |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | **138.1 s** | 136.6–139.6 | 1.00 | 100% | 22.8 | 0.43 | 2787 MB | 3.15 GB | 3646–3648 |
| 2 | **65.8 s** | 65.2–66.4 | **2.10** | 105% | 49.1 | 0.12 | 6233 MB | 6.42 GB | 3647–3648 |
| 3 | **49.3 s** | 49.1–49.4 | **2.80** | 93% | 58.5 | 0.15 | 11340 MB | 9.13 GB | 3646–3648 |
| 4 | **45.0 s** | 44.9–45.1 | **3.07** | 77% | 69.7 | 0.16 | 21159 MB | 11.83 GB | 3649 |

**Large / 441 tiles (n=2, n=3 at the recommended `workers=3`):**

| workers | wall | spread | **speedup** | efficiency | mean SM% | near-idle (SM≤3) | FB peak | peak RSS | cells |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | **482.8 s** | 482.4–483.2 | 1.00 | 100% | 20.5 | 0.46 | 2787 MB | 4.04 GB | 13147–13149 |
| 2 | **208.9 s** | 207.8–210.0 | **2.31** | 116% | 48.4 | 0.09 | 6233 MB | 6.52 GB | 13147 |
| 3 | **156.1 s** | 155.8–156.3 | **3.09** | 103% | 70.6 | 0.06 | 12354 MB | 9.26 GB | 13146–13149 |
| 4 | **137.4 s** | 137.4–137.4 | **3.51** | 88% | 78.1 | 0.08 | 20667 MB | 11.97 GB | 13146–13148 |

**Tile outcomes are identical at every worker count** — large 378 success / 63 skipped, medium
103 / 18 — matching doc 18 §1's recorded round-4 reference exactly.

Two sanity checks that the baseline is the right baseline, not a strawman: the `workers=1` control
measures **482.8 s** at large against doc 18's recorded `p3` anchor of **480.3 s** (+0.5%) and
**138.1 s** at medium against its **140.8 s** (−1.9%). The single-process path reproduces the
previous round, so the speedups below are against a real anchor. Run-to-run spread this round was
extremely tight — **0.16% at large/`w1`, 0.34% at large/`w3`** — tighter than any configuration in
doc 18.

### 4.2 The result is far above what doc 20 predicted, and the reason matters

Doc 20 §0/§2 sized this lever at **1.23x–1.7x**. Measured: **3.09x** at `workers=3`, **3.51x** at
`workers=4`. A result that beats its own prediction by 2x is exactly the kind of thing the playbook
says to distrust, so: the cells, tile counts and per-cell veto (§4.3) all confirm the same work is
being done, and the baseline reproduces the prior round. The number is real. The estimate was wrong,
and it was wrong for an identifiable reason.

Doc 20's ceiling was derived from **device-busy time** — "its real bound is the GPU's serialized
device-busy time... mean SM is only ~20%". That accounting treats the single-process wall as if it
were GPU-limited with idle gaps to fill. But a large share of the single-process wall was never GPU
time at all: it was **GIL-serialized Python across the two arms**. Doc 18 §2 had already
demonstrated this mechanism in miniature and flagged that the arm model cannot capture it — moving
⑧ off the MAIN arm made **two unmodified buckets** faster (B1 forwards −12.8 s, `detect_all_dots`
−22.3 s) purely by removing GIL contention between the arms.

Separate processes do not share a GIL at all. So multiprocessing collects three things at once,
where doc 20 only counted the first:

1. GPU device idle filled by interleaving forwards (the launch-bound component, §1.2's 1.45x bound),
2. **the entire GIL-contention cost between the MAIN and BG arms**, which no thread-based design can
   recover — this is the term the 1.23x–1.7x estimate omitted,
3. genuinely parallel CPU work across 20 cores (`detect_all_dots`, PNG encode, per-cell crops,
   `_read_rgb`, M1 glue) that the single BG thread previously serialized.

The signature of (2) is visible in the superlinear efficiency at `workers=2` (**116%** at large,
105% at medium): two processes do more than twice the work of one, which is impossible if the only
thing gained were device overlap. Near-idle (SM≤3) collapsing **0.46 → 0.06** confirms (1) and (3)
jointly filled the device.

### 4.3 Correctness veto — passed at both anchors

Judged against the **same-code noise floor**, per doc 20 §1 item 6 and doc 18 §2's method — not
exact equality, since GPU forward nondeterminism is an accepted, characterized noise source.

**Large anchor**, all pairings vs `w1_large_r1`:

| comparison | cells | reddot max\|Δ\| | blackdot max\|Δ\| | score max\|Δ\| | differing cells | X-flips |
|---|--:|--:|--:|--:|--:|--:|
| `w1` vs `w1` (**same code**) | 13147 | 0 | 0 | 0 | 0 | **21** |
| `w2` vs `w1` | 13147 | 2 | 8 | 4 | 1 | 3–15 |
| `w3` vs `w1` | 13146–13149 | 2 | 8 | 4 | 1–2 | 12–15 |
| `w4` vs `w1` | 13146–13148 | 1 | 1 | 2 | 0–1 | 15–18 |

**The decisive points**, both identical in form to doc 18 §2's adopted change:

- The `reddot 2 / blackdot 8 / score 4` signature is **precisely the fingerprint doc 18 §2 recorded
  between two runs of identical `p2` code**. It is the known GPU-nondeterminism signature, not a
  computation change.
- **X-flips never exceed the same-code floor**: 3–18 across every multiprocess configuration against
  **21** between two runs of identical single-process code.
- Only **1–2 of ~13,140 matched cells** differ in any pairing — the same figure doc 18 reported.

At the medium anchor the same holds (same-code floor: 12 X-flips, ±2 cells; multiprocess: 6–21
X-flips, reddot ≤1, blackdot ≤1, score ≤2), and at the small anchor every delta is **exactly zero**
(§3.2). Cell counts stay inside doc 18's recorded reference ranges at both anchors.

This is a stronger correctness result than any previous round's adopted change, and structurally it
should be: unlike moving work between threads, **nothing about the per-tile computation changed** —
the same functions run in the same order on the same inputs, only in a different process.

### 4.4 Memory invariants at N-process scale (doc 20 §1 item 5)

- **VRAM.** `dmon fb` peak per process is ~2787 MB at `workers=1`, and the device peak grows
  2787 → 6233 → 12354 → 20667 MB. Note this is **superlinear** (per-process 2787 / 3117 / 4118 /
  5167 MB) — allocator behaviour under contention, not weight growth. It is the main scaling risk
  and the reason `workers=4` is not the default recommendation: 20.7 GB of 32 GB leaves thin margin.
- **RSS.** Peak 4.04 → 6.52 → 9.26 → 11.97 GB at large, i.e. roughly linear in N, as expected when
  each worker holds its own model set and in-flight arrays. **The bounded-memory shape is preserved
  at every N**: the sawtooth is intact (85–131 drawdowns >50 MB per run, max drawdown 2.6–7.7 GB,
  ramp fraction 0.48–0.57 against the `w1` control's 0.54). No monotonic ramp — the signature doc 14
  §2 says to reject — appears at any worker count.

### 4.5 Full-WSI extrapolation

Linear refit on the two anchors per worker count, 35,700 tiles @ 1024px:

| workers | s/tile | intercept | **full WSI** | vs round 4 |
|--:|--:|--:|--:|--:|
| 1 | 1.0772 | 7.76 s | **10.68 h** | (round 4 recorded 10.52 h ✔) |
| 2 | 0.4472 | 11.69 s | **4.44 h** | 2.41x |
| 3 | 0.3337 | 8.92 s | **3.31 h** | **3.23x** |
| 4 | 0.2888 | 10.06 s | **2.87 h** | 3.73x |

The `workers=1` refit reproducing round 4's recorded 10.52 h is the check that this extrapolation is
built the same way as every previous round's. **The same tissue-density caveat applies as in every
round 1–4**: these are crops at ~85% tissue density, so all hour figures are upper bounds, and
`19-open-backlog.md` §1 item 7 (never-done real-WSI validation) is **not** closed by this round.

### 4.6 Recommended configuration

**Superseded by §4.7.** This subsection originally recommended `workers=3` from the N≤4 data alone.
§4.7 extends the sweep to find this machine's actual ceiling and settles on **`workers=6`** as the
value that minimizes wall-clock while staying inside the failure-free range this session could
reproduce. Kept here as the record of what was recommended before the ceiling was found.

### 4.7 Finding the actual limit — round 5b, 2026-07-23

Doc 20 never sized *N* beyond "fits in 32 GB alongside model weights" (§1 item 5, §5). §4.1–4.6 only
measured up to `workers=4`, which was itself picked from a knee in a four-point curve — not from
knowing where the machine actually breaks. This section pushes `workers` until it breaks, on both
anchors, with enough repeats to tell a real failure rate from a lucky or unlucky single run.

**Method.** Same protocol as §0 (GPU confirmed idle between runs, same checkpoints, same config
hash). The medium anchor (121 tiles, ~40 s/run) was used to explore quickly; the two candidates that
looked best were then confirmed with 3 repeats at the large anchor (441 tiles) before being trusted.

**Full sweep, medium anchor (121 tiles):**

| workers | wall (s), all runs | success rate | note |
|--:|---|--:|---|
| 1–4 | see §4.1 | 100% (n=2 each) | — |
| 5 | 42.77, 43.05, 42.43 | **3/3** | |
| 6 | 42.23, 41.84, 42.62 | **3/3** | |
| 7 | 42.02, 41.89, **OOM** | **2/3** | first observed failure |
| 8 | 41.61 | 1/1 | not repeated |
| 9 | **OOM** | 0/1 | |
| 10 | 42.30, **OOM**, 42.40 | 2/3 | flaky, not a hard cutoff |
| 11 | **OOM** ×3 | **0/3** | reliably broken |
| 12, 14, 16, 20 | **OOM** ×1 each | 0/1 each | fails during model **load**, before any tile |

**Full sweep, large anchor (441 tiles):**

| workers | wall (s), all runs | success rate |
|--:|---|--:|
| 1–4 | see §4.1 | 100% (n=2–3 each) |
| 5 | 127.39, 126.79, 129.60 | **3/3** |
| 6 | 123.23, 123.40, 123.29 | **3/3** |
| 7 | 121.22 | 1/1 (not repeated at this scale) |
| 8 | **OOM** | 0/1 |

**The ceiling is not a clean cutoff — it is a probability that rises with N, and independently with
job size.** Three findings, in order of how much they should change how this number gets used:

1. **A "success" at N=7/8/9/10 is not proof those values are safe — repeating them found failures
   that single runs missed.** `workers=7` looked clean on its first outing (large anchor, 121.2 s)
   and then failed on its very next run (medium anchor) with three individual worker processes
   holding **7.7 / 7.7 / 9.4 GB** each — 3–4x their ~2.8 GB steady-state footprint. `workers=10`
   alternated success/fail/success across three consecutive medium-anchor runs. This is allocator
   behavior, not a deterministic function of N: PyTorch's own OOM message names the mechanism
   (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation), consistent with
   follow-up #3's already-flagged superlinear per-process VRAM growth (§4.4).
2. **Risk scales with total tile count, not just N.** `workers=8` succeeded once on the 121-tile
   medium crop but failed on the 441-tile large crop at the same worker count — more tiles means more
   forward passes means more chances for an unlucky simultaneous multi-worker memory spike to land at
   once. A full WSI is **~80x more tiles than the large crop**, so a value that "worked" once or twice
   here carries meaningfully more exposure over a real slide than these numbers suggest.
3. **The hard, deterministic wall is `workers=12`.** At and above 12, every trial fails **during
   model loading**, before a single tile is processed: `2.79 GB × 12 ≈ 33.5 GB` already exceeds the
   32.6 GB card before any per-tile activation spike is even possible. This is the one number in this
   section that is not probabilistic.

**Correctness veto — passed at `workers=5` and `workers=6`**, both anchors, same method as §4.3: all
deltas inside the same-code noise floor (reddot ≤2, blackdot ≤8, score ≤4), cell counts within the
established reference range, no signature distinguishable from ordinary GPU nondeterminism.

**The answer to "what should I set for the shortest time":**

| | wall (large, mean of 3) | vs `workers=4` | observed failures |
|---|--:|--:|--:|
| `workers=4` | 137.4 s | baseline | **0 / 5** (this + prior rounds) |
| **`workers=6`** | **123.3 s** | **−10.3%** | **0 / 6** (3 medium + 3 large) |
| `workers=7` | 121.2 s (1 run) | −11.8% | **1 / 4** (medium anchor, ~25%) |

**Set `workers=6`.** It is the highest worker count this session could push to zero observed failures
on *both* crop sizes (6 trials total), it is measurably faster than `workers=4` (−10.3% at large,
−6.2% at medium), and `workers=7` is exactly where cracks start showing — a ~25% failure rate on the
smaller crop, which fail-fast turns into a *full batch restart*, not a retry of one tile. Going past 6
trades a single-digit percent further speedup for a real and rising chance of losing the entire run.

**One caveat that should govern production, not just this benchmark: "0/6" is not "impossible to
fail," and the risk compounds over a longer job.** Six trials is enough to distinguish `workers=6`
from `workers=7`'s already-visible ~25% failure rate, not enough to bound `workers=6`'s true failure
probability tightly, and finding (2) above means that probability is higher again over a full WSI's
~35,700 tiles than over these 441-tile crops. Combined with `run_batch`'s fail-fast design (§1 item 2
— any failure aborts and discards the *whole* batch, no partial resume), the practical guidance is:

- **Batch jobs sized like these crops or smaller, where a restart is cheap**: `workers=6`.
- **A full, unattended WSI run, where an OOM at tile 30,000 of 35,700 means starting over**:
  `workers=4` is the more defensible choice until either (a) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is verified to remove the fragmentation spikes finding (1) identified, or (b) `run_batch` gains
  partial-resume/checkpointing so an OOM late in a long run does not discard completed tiles (neither
  exists today — recorded as follow-up #7, §10).

## 5. Step 5 — Candidate C (CUDA MPS): **measured, works on this card, and stopped out**

Doc 20 §3 step 5 says to size MPS only if the knee looked *serialization-limited* rather than
*SM-limited*. §1.2's launch-bound control is unambiguous on that: throughput pinned at exactly
236.65 units/s from N=2 through N=6 while mean SM was still only 55–64%. That is serialization with
SM capacity to spare — the condition that justifies testing MPS.

**MPS does start and does work on this RTX 5090**, which doc 20 §2 Candidate C explicitly flagged as
unverified ("this RTX 5090 is a consumer GeForce card, not on NVIDIA's published MPS support
matrix"). The daemon started, a server was spawned (PID confirmed via `get_server_list`), and
clients connected. **Recording this as a positive finding on an open question**, independent of the
disposition below.

**And it substantially raises the synthetic ceiling:**

| launch-bound control | N=1 | N=2 | N=4 |
|---|--:|--:|--:|
| no MPS | 1.00 | 1.45 | 1.45 |
| **with MPS** | 1.00 | **1.94** (97% eff) | **2.09** |

Throughput 236.65 → **316.77** units/s at N=2. The mechanism doc 20 describes is real and measurable.

**But on the real pipeline it is flat**, which is what decides it:

| medium anchor | no MPS | with MPS | Δ |
|---|--:|--:|--:|
| `workers=2` | 65.8 s | 64.9 s | −1.3% |
| `workers=3` | 49.3 s | 49.4 s | +0.2% |
| `workers=4` | 45.0 s | 44.4 s | −1.4% |

Every difference is inside the ~2% medium run-to-run band. **Disposition: stopped out, not backlog.**
The reason is not that MPS fails — it demonstrably works — but that *by the time the real pipeline
reaches its knee it is no longer serialization-limited*. Mean SM is already 58–78% and near-idle is
0.06–0.16 (§4.1); the pipeline has enough concurrent CPU and GPU work that context-switch overhead
has stopped being what costs time. MPS fixes a bottleneck the real workload no longer has.

This is the playbook's anti-pattern #5 — *"treating a micro-benchmark win as an end-to-end win"* —
caught in the act, on a 44% synthetic improvement that delivers 0% end-to-end. Adopting it would also
have added an operational dependency (a daemon that must be running, unsupported by NVIDIA on
GeForce) for nothing. Reopen only if a future change makes the pipeline serialization-limited again.

## 6. Step 6 — Candidate A: **premise tested directly and refuted, not built**

Doc 20 §3 step 6 offers Candidate A (a `ProcessPoolExecutor` for the CPU back end, raising pipeline
depth from 1 to M) as independently prototypable. Doc 20 §2 already flagged its expected payoff as
"likely small today" because BG/MAIN was 0.841 — BG is not the bottleneck at depth 1.

Rather than build the IPC path first (which requires moving ~7 MB of arrays per tile across a
process boundary, the cost doc 20 §2 says "must be measured, not assumed away"), its **premise** was
tested directly and far more cheaply: does *more BG concurrency* help at all? A temporary
`HYBRID_BG_DEPTH` knob raised the per-worker CPU-back-end thread pool from depth 1 to depth 2. This
is the strictly cheaper version of the same idea — no IPC, no serialization — so if it cannot win,
the process-based version carrying extra cost cannot either.

| medium anchor | BG depth 1 | BG depth 2 | Δ |
|---|--:|--:|--:|
| `workers=1` (most BG slack) | 138.1 s | **142.0 s** | **+2.8% slower** |
| `workers=3` (at the knee) | 49.3 s | 48.7 s | −1.2% (inside the noise band) |

**Single-process is measurably *worse*** — the extra BG thread adds GIL contention with the MAIN arm
rather than useful concurrency, the same mechanism doc 18 §2 documented in the opposite direction.
Under Candidate D it is flat, and the reason is structural: at `workers=3` the device near-idle is
already 0.06, i.e. the GPU is busy 94% of the time, so the critical path is no longer the background
CPU arm and adding depth to it cannot move wall-clock.

**Disposition: stopped out with direct evidence, not carried as backlog.** The knob was built,
measured, and **deleted** — the same discipline doc 16 §4.2 applied to fixed-N `gc.collect` batching.
The final diff contains no trace of it. Per the playbook, "zero-contribution layers get cut no matter
how clever".

Note this does **not** re-litigate the already-closed `detect_all_dots` joblib process-backend item
(`19-open-backlog.md` §1 "already closed"); doc 20 §2 was right that they are different experiments.
This one is closed on its own new evidence.

## 7. How each doc 20 §1 invariant is discharged

| # | invariant | how it is satisfied | evidence |
|--:|---|---|---|
| 1 | Global renumbering stays a single deterministic post-pass | `_finish_batch` holds the **only** implementation of the sort+renumber and is shared verbatim by both paths; workers return the same `(abs_x, abs_y, owned)` tuples the single-process loop builds internally and **never renumber** | §3.1; the merge/renumber block is textually unchanged in the diff |
| 2 | Fail-fast is whole-batch, not per-worker | First error → parent stops the feeder, `terminate()`s every sibling, then re-raises | §3.3 injection test, incl. the two defects it caught (§3.4) |
| 3 | Exactly one worker touches each tile | Dynamic work queue: each tile is `put` once and `get` by exactly one worker | §3.1; identical 378/63 and 103/18 tile outcomes at every N (§4.1) |
| 4 | `gc.freeze()` semantics must not leak across a persistent pool | Workers are **spawned fresh per `run_batch` call and torn down**, so the per-call freeze/unfreeze contract holds automatically. Doc 20 §5's persistent-pool design surface is **avoided, not solved** — see §10 follow-up #2 | `_frozen_gc_generation()` wraps each worker's whole tile loop |
| 5 | VRAM/RSS bounded at N-process scale | Verified at both anchors, all N | §4.4 — sawtooth preserved, ramp fraction 0.48–0.57 vs control 0.54 |
| 6 | Per-cell correctness veto vs round-4 reference, read against the same-code noise floor | Passed at all three anchors | §3.2, §4.3 |
| 7 | Small/API-server requests must not regress | `workers=1` is the default; `backend/api/hybrid.py` does not pass it, so the API path is unchanged. The single-process diff is **three deleted lines** (the `_init_*` calls, relocated) | §3.1; `git diff` verified |

## 8. Code changed

| file | change |
|---|---|
| `backend/algorithms/hybrid/hybrid_pipeline.py` | `run_batch(..., workers: int = 1)`; new `_mp_tile_worker` (spawned worker, own models, mirrors the single-process depth-1 loop), `_run_tiles_multiprocess` (dynamic queue, feeder thread, fail-fast + sibling termination), `_finish_batch` (global merge/renumber/export/stitch tail, **extracted verbatim** and shared by both paths) |
| `scripts/perf_measure.py` | `--mp-workers N`; records `mp_workers` in the timings JSON |
| `scripts/mp_concurrency_probe.py` | **new** — step-1/2 concurrency probe (`sm` / `launch` / `models` workloads, barrier-synced throughput scaling, per-process VRAM + init) |
| `scripts/mp_scaling_report.py` | **new** — N-worker scaling table (wall, speedup, efficiency, `dmon` SM/near-idle/FB, RSS, cells); documents why `arm_report.py` does not apply |
| `scripts/verify_mp_failfast.py` | **new** — corrupt-tile injection guard: raises, aborts early, no surviving worker |
| `backend/algorithms/hybrid/test_picture/_roi_crops/small_{ihc,dish}.tiff` | **new** 4096² crop = the 25-tile small anchor (§3). **Gitignored** (`*.tiff`), like every other crop — regenerate with the one-liner in §9 |

**Config hash is unchanged at `ad41c42f`** — no config field was added this round, so every round-4
and round-5 CSV is directly comparable. `workers` is a `run_batch` call argument, deliberately not a
`Config` field: it is a deployment/host-capacity choice, not a analysis parameter, and putting it in
`Config` would change the hash and make every prior CSV incomparable for no benefit.

## 9. Reproducing

```bash
# the 25-tile small anchor (gitignored, like every crop -- cut it once)
.venv/bin/python -c "
import pyvips
for n in ('ihc','dish'):
    p='backend/algorithms/hybrid/test_picture/_roi_crops/'
    pyvips.Image.new_from_file(p+f'med_{n}.tiff', access='random').crop(0,0,4096,4096) \
        .write_to_file(p+f'small_{n}.tiff', compression='deflate')"

# step 1/2 -- concurrency probe (controls first, then the real models)
.venv/bin/python scripts/mp_concurrency_probe.py --workload sm     --procs 1,2,4     --units 400
.venv/bin/python scripts/mp_concurrency_probe.py --workload launch --procs 1,2,3,4,6  --units 1000
.venv/bin/python scripts/mp_concurrency_probe.py --workload models --procs 1,2,3,4 --units 16 \
    --tiles-dir <precut scratch> --tile-limit 24

# step 3 -- correctness + fail-fast, BEFORE any scale run
.venv/bin/python scripts/perf_measure.py --ihc .../small_ihc.tiff --dish .../small_dish.tiff \
    --output <out> --label w3_small_r1 --workers 8 --gpu-dmon --stream-precut --mp-workers 3
.venv/bin/python scripts/verify_mp_failfast.py --tiles-dir <precut scratch> --workers 3

# step 4 -- anchors (GPU confirmed idle before each; loop on memory.used < 200 MiB)
.venv/bin/python scripts/perf_measure.py --ihc .../large_ihc.tiff --dish .../large_dish.tiff \
    --output <out> --label w3_large_r1 --workers 8 --gpu-dmon --stream-precut --mp-workers 3 \
    --metrics-dir <m>

# scaling table + correctness veto
.venv/bin/python scripts/mp_scaling_report.py --metrics-dir <m> --runs-dir <runs> --pattern large
.venv/bin/python scripts/gc_ablation_report.py --metrics-dir <m> --runs-dir <runs> \
    --reference <runs>/w1_large_r1/report.csv --baseline w1_large_r1

# step 5 -- MPS (Candidate C)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
nvidia-cuda-mps-control -d       # ... run as above ...      echo quit | nvidia-cuda-mps-control

# §4.7 -- pushing N to find the ceiling (GPU idle-check loop as above, repeat 3x per N)
for W in 5 6 7 8 9 10 11 12; do
  .venv/bin/python scripts/perf_measure.py --ihc .../med_ihc.tiff --dish .../med_dish.tiff \
      --output <out>/w${W} --label w${W}_med_r1 --workers 8 --gpu-dmon --stream-precut \
      --mp-workers $W --metrics-dir <m>
done
```

## 10. Follow-ups this round raised

1. **Per-bucket timing inside workers does not exist.** `perf_measure.py`'s monkeypatches are
   parent-only, so under `--mp-workers N` every worker-side bucket is empty and `arm_report.py` does
   not apply (§0). Nothing in this round needed it — end-to-end wall decided everything — but any
   future *intra-worker* optimization is currently unmeasurable. The fix is to have each worker emit
   its own `TIMINGS` dict back over the result queue and merge them in the parent; this pairs
   naturally with doc 18 follow-up #1 (record the executing thread per bucket).
2. **The persistent-pool design surface is avoided, not solved.** Workers are spawned per
   `run_batch` call, which is what makes doc 20 §1 item 4's `gc.freeze()` contract automatic — but it
   means the API server path would pay `3.14 s × N` model init **per request** if it ever enabled
   `workers>1`. That is why the API deliberately still uses `workers=1`. Making multiprocessing
   useful for the API needs a persistent pool, which reopens the freeze/unfreeze-per-call design doc
   20 §5 describes, and is adjacent to `19-open-backlog.md` §1 item 8 (multi-request behaviour, never
   measured).
3. **VRAM per process grows superlinearly with N** (2787 → 3117 → 4118 → 5167 MB, §4.4). Unexplained;
   most likely caching-allocator behaviour under contention. It is what caps *N*, so if anyone wants
   `workers>4` this needs isolating first (`PYTORCH_CUDA_ALLOC_CONF` is the obvious first knob).
4. **`19-open-backlog.md` §1 item 7 (real-WSI validation) is now the binding constraint on
   shipping this.** Doc 20 §4's stop-loss is explicit: "Do not ship any multiprocessing change to
   production ahead of that validation." Everything here is crop-scale, like every round before it.
   The gap matters more now than it did: worker count interacts with tissue density (a real slide is
   mostly cheap background tiles, so the dynamic queue's balancing behaviour at 35,700 tiles is
   untested), and peak VRAM/RSS scale with N.
5. **The probe's per-process VRAM columns are only valid at N=1** (§1.4) — `mem_get_info` is
   device-wide. Either subtract a measured sibling baseline or read per-PID from
   `nvidia-smi --query-compute-apps` if per-process VRAM at N>1 is ever needed.
6. **`workers` has no CLI exposure on `hybrid_pipeline.py` itself.** Only `run_batch`'s argument and
   `perf_measure.py --mp-workers` reach it. Deliberate for now — the default must stay 1 until
   follow-up #4 clears — but a `--workers` CLI flag is the obvious next step once it does.
7. **No partial-resume/checkpointing.** §4.7 found the practical ceiling is a *probability of OOM
   that rises with N and with total tile count*, not a hard cutoff — and `run_batch`'s fail-fast
   design (correctly, for correctness) discards the whole batch on any failure. For a full WSI this
   means an OOM at tile 30,000 of 35,700 costs the entire run, not just the failed tile. This is the
   real reason §4.7 recommends the more conservative `workers=4` for unattended full-slide runs rather
   than the faster `workers=6` — the fix is either a persisted per-tile completion log `run_batch` can
   resume from, or accepting the current all-or-nothing contract and choosing *N* conservatively.
8. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was never tried.** §4.7 traced the `workers≥7`
   instability to individual worker processes' VRAM spiking 3–4x above their steady-state footprint —
   allocator fragmentation, per PyTorch's own OOM message — not a clean function of aggregate demand.
   This flag is PyTorch's documented mitigation for exactly that failure mode and was not tested this
   round; if it removes the fragmentation spikes, the safe ceiling above `workers=6` could move.
