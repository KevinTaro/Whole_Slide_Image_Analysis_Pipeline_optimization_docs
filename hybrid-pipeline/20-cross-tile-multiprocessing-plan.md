# 20 — Cross-tile multiprocessing: design space and experiment plan

> Solution-design follow-up to [`19-open-backlog.md`](./19-open-backlog.md) §1 item 1 (the only
> remaining lever with a real ceiling), [`17-gpu-starvation-prerequisites-plan.md`](./17-gpu-starvation-prerequisites-plan.md)
> §4.4, and [`18-gpu-starvation-prerequisites-implementation.md`](./18-gpu-starvation-prerequisites-implementation.md)
> §6.2. Follows [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose, cheapest-and-lowest-risk experiment first, every layer
> ablation-proved, **correctness is a veto, not a tradeoff**.
>
> **Planning document — no pipeline code changed here.** Grounded by reading current HEAD
> (`backend/algorithms/hybrid/hybrid_pipeline.py`, line numbers below, plus this machine's actual
> `nvidia-smi`/`torch` state — checked 2026-07-23, not assumed) and the module's own
> `CLAUDE.md`. This is a **menu of candidates to try, ranked and sequenced** — it does not pick a
> winner in advance, and nothing in it should be read as "build this." Every candidate below still
> has to clear the same correctness veto as every prior optimization round before it can be adopted.

## 0. Why this document exists, and what it inherits

Per `19-open-backlog.md` §1 item 1, cross-tile multiprocessing is **the only remaining lever with a
measured ceiling** (~1.23x–1.7x) after four rounds of single-process work closed everything cheaper:
`gc.freeze()` (done), moving stranded CPU prep off the MAIN arm (done), precut streaming (done),
`cellpose_batch_size` wiring (done, sweep negative), the CUDA-stream/depth-2 bubble redesign
(sized and stopped out at ≤1.065x). It is also flagged as **the highest correctness risk of
anything left**. This document exists to turn "scope it" into a concrete, sequenced set of
experiments instead of one all-or-nothing rewrite.

**The constraint, read from source, not assumed:** `backend/algorithms/hybrid/CLAUDE.md` states
directly — *"`run_batch()` is intentionally sequential (not parallelized) across tiles — the 3 GPU
models are loaded once in the main process and share one CUDA context, so cross-tile process
parallelism is unsafe (fork-under-CUDA)."* Confirmed in the current implementation: `run_batch`
(`hybrid_pipeline.py:710`) calls `_init_unet_inferencer` / `_init_cellpose_segmenter` /
`_init_dish_cellpose_segmenter` once (lines 786–788), then loops tiles sequentially on the main
thread, with a single `ThreadPoolExecutor(max_workers=1, ...)` (line 812) running the previous
tile's CPU back end (`_process_precut_tile_cpu` → `_finish_chunk_cpu`, line 617) while the main
thread runs the next tile's GPU front (`_process_precut_tile_gpu` → `_process_one_chunk_gpu`, line
546). Pipeline depth is 1; nothing forks; the background thread never touches `torch`/CUDA.

**Why the ceiling is wider here than any single-process lever.** Doc 18 §6.2: every single-process
optimization is bounded by `BG + outside = 389.3 s` (1.23x from the current 480.3 s at the large
anchor) — the floor where the background CPU arm becomes the new critical path. Multiprocessing is
not bounded by that floor, because **each additional process carries its own BG thread too** — both
arms parallelize, not just the GPU front. Mean GPU SM utilization is only ~20% even today (doc 17
§0), so there is genuine headroom for multiple processes' forwards to interleave on the device — if
the hardware actually lets them (see §2 Candidate B and §3 step 1, the biggest open unknown).

**This machine's actual state (checked 2026-07-23, not assumed from the docs):**

| fact | value | relevance |
|---|---|---|
| GPU | RTX 5090, 32607 MiB | VRAM budget for N × 2.79 GB model sets |
| Compute Mode | `Default` (not `EXCLUSIVE_PROCESS`) | multiple processes can already open contexts on this GPU without extra config |
| CUDA MPS | not running | would need to be started explicitly (Candidate C) |
| torch | 2.11.0+cu130 | matches doc 18's recorded round-4 environment |
| driver | 580.159.03 (per doc 18) | — |

## 1. Correctness invariants every candidate must preserve

These are not negotiable per-candidate design choices — every candidate in §2 must satisfy all of
them before it can be adopted, the same way doc 18's ⑧/precut-streaming changes were only adopted
after passing an explicit correctness veto.

1. **Global cell-ID renumbering stays a single deterministic post-pass.** Today, `run_batch`
   accumulates `(abs_x, abs_y, owned_results)` from every tile into `per_tile_owned` (regardless of
   completion order — precut streaming already proved processing order doesn't matter, doc 18 §4.2),
   then sorts by `(abs_y, abs_x, cell_id)` and renumbers 1..N exactly once (lines 861–869). Any
   multiprocessing design must still gather results from all workers into one list and perform this
   *same* sort+renumber step centrally in the parent — never renumber inside a worker, never merge
   partial orderings.
2. **Fail-fast is whole-batch, not per-worker.** `run_batch` raises immediately and aborts the whole
   run if any tile hits a real error (lines 821–831) — "any tile fails → the whole slide is
   untrustworthy," not "skip and continue." Under multiprocessing, a worker's exception must
   propagate to the parent, and the parent must terminate every sibling worker before returning —
   letting siblings run to completion after one worker has already failed reproduces exactly the
   "slide with an undocumented hole" failure mode the current fail-fast design exists to prevent.
3. **Exactly one worker touches each tile.** Per-tile output files are already keyed by
   `tile_x{abs_x}_y{abs_y}` and written independently (no shared mutable state across tiles except
   the final global merge/stitch) — this makes file writes inherently race-free *as long as no tile
   is double-dispatched*. Any work-queue design (§2 Candidate B/D) must guarantee each tile is
   claimed by exactly one worker.
4. **`gc.freeze()` semantics must not leak across a persistent pool.** `_frozen_gc_generation`
   (line 223) explicitly warns that its `unfreeze` on exit is required specifically because
   `run_batch` "will be called repeatedly inside a long-lived API server process" — freezing without
   unfreezing per call would permanently freeze whatever's tracked at that moment, defeating GC
   across requests. If any candidate below reuses a persistent worker pool across multiple
   `run_batch` calls (rather than spawning fresh workers per call), each worker needs the *same*
   freeze/unfreeze contract applied per call, not once at worker startup — this is new design surface
   that doesn't exist in the current one-shot-per-process model.
5. **VRAM/RSS bounded invariant must be re-verified at N-process scale**, the same way doc 18 §2
   re-verified it after moving ⑧ off the MAIN arm. `2.79 GB × N` must fit in 32 GB *alongside* the
   per-process CUDA context overhead itself (unmeasured — see §5), not just the model weights.
6. **Output must clear the same per-cell correctness veto as every prior round** — reuse
   `scripts/gc_ablation_report.py`'s nearest-centroid-matched comparison (reddot/blackdot/score
   max|Δ|, X-flips, differing-cell count) against the recorded round-4 reference (large:
   13152–13153 cells / 378 success / 63 skipped; medium: 3647–3649 / 103 / 18), read against the
   **same-code noise floor** (doc 18 §2's table), not exact equality — GPU forward nondeterminism is
   already an accepted, characterized noise source.
7. **Small/API-server requests must not regress.** `backend/api/hybrid.py`'s `/api/hybrid/tile`
   endpoint calls this same pipeline for potentially single-tile requests. A design that
   unconditionally spins up N worker processes (each paying model-init cost, ~2.4 s × N if
   sequential, or a large one-time cost if parallel) for every request would add latency the current
   single-process path doesn't pay today for small jobs. `19-open-backlog.md` §1 item 8
   ("multi-request/concurrent-job behavior... never measured") is adjacent to this but explicitly
   out of scope here — flagged, not solved, so it isn't silently made worse.

## 2. Design space — candidate architectures

Ranked roughly cheapest/lowest-risk → most invasive. None of these are mutually exclusive; §3
sequences them.

### Candidate A — CPU-back-end-only process pool (GPU stays single-process, untouched)

Keep model init and all 3 GPU forwards exactly as today — one CUDA context, main thread, unmodified.
Replace the single `ThreadPoolExecutor(max_workers=1)` (line 812) with a `ProcessPoolExecutor` of
*M* worker processes that run only `_finish_chunk_cpu`'s pure-CPU code (`detect_all_dots`,
`build_all_positive_results`, `enlarge_cell_instances`, PNG/TIFF encode, per-cell crop export,
merge). Pipeline depth goes from 1 to *M* — up to *M* tiles' CPU back ends in flight concurrently
instead of 1.

- **Sidesteps fork-under-CUDA entirely** — these workers never import `torch`/CUDA; they receive
  plain numpy arrays (`core_mask`, `masked_ihc`, `dish_mask_overlay`, `instance_mask`,
  `dish_nucleus_mask`) plus `config`, and return an `owned` results list. Lowest correctness risk of
  anything in this document.
- **Not the same experiment as the already-closed `detect_all_dots` process-backend attempt**
  (`19-open-backlog.md` §1 "already closed" list). That prior attempt used `joblib` to
  process-parallelize *inside* `detect_all_dots` per tile (fine-grained, many small process
  launches, found negative). This candidate parallelizes *across tiles* coarsely, with a small
  number of long-lived worker processes reused for the whole batch — a different granularity that
  has not been tried. **Flag this distinction explicitly** so nobody stop-losses Candidate A by
  citing the earlier negative result; they are not the same design.
- **Cost to watch**: cross-process array transfer has no shared-memory shortcut like threads get —
  pickling a ~3 MB RGB tile + ~4 MB int32 mask per tile has real serialization overhead that must be
  measured, not assumed away. Mitigations: `multiprocessing.shared_memory`, or reuse the pattern
  already proven elsewhere in this pipeline (precut streaming passes file *paths*, not arrays) by
  having the GPU front write intermediate arrays to disk and handing the worker a path instead.
- **Expected payoff on its own is likely small today**: BG/MAIN is currently 0.841 (doc 18 §5) — BG
  is *not yet* the bottleneck at depth 1, so more BG concurrency alone may buy little until the GPU
  front also gets faster or more concurrent (Candidate B/D). Record as a prerequisite/pairs-with
  item, not a standalone win — but it is cheap enough to measure on its own regardless.

### Candidate B — N-way GPU worker process pool with per-process model reload (the canonical "doc 18 §6.2" design)

Spawn *N* worker processes using `multiprocessing.get_context("spawn")` (never `fork` — see
Candidate E for why). Each worker independently calls `_init_unet_inferencer` +
`_init_cellpose_segmenter` + `_init_dish_cellpose_segmenter` once (its own CUDA context, its own
2.79 GB VRAM footprint), then pulls tiles from a shared work source and runs the **full** existing
per-tile pipeline end to end (`process_precut_tile`, unmodified) — including its own internal
depth-1 GPU/CPU thread overlap, exactly as the current single process does today.

- **Static vs. dynamic partitioning.**
  - *(i) Static round-robin* (tile *i* → worker `i % N`): simplest, but risks load imbalance — tile
    cost varies a lot (background/empty tiles short-circuit cheaply; tissue-dense tiles cost the
    full three-forward pipeline), and per-tile cost isn't known upfront.
  - *(ii) Dynamic work queue* (`multiprocessing.Queue`, or `ProcessPoolExecutor.map`/`submit` fed
    incrementally): each worker claims the next tile when free — self-balancing. **Recommended**,
    given `19-open-backlog.md` §1 item 7 notes a real WSI is mostly background (~85% tissue-dense in
    the measured crops, but a full slide skews far more toward cheap background tiles) — static
    partitioning would leave idle workers waiting on a sibling stuck with a dense cluster.
- **VRAM/init cost**: `2.79 GB × N` model weights + **per-process CUDA context overhead
  (unmeasured — must be profiled, see §5)** must fit in 32 GB. Init cost is `2.4 s` per model set,
  but paid **once per worker at startup, in parallel** if all workers spawn concurrently — not
  `N × 2.4 s` serial.
- **GPU concurrency reality check — the single biggest unknown in this whole document.** This
  machine's `Compute Mode` is `Default` (confirmed §0), meaning multiple processes *can* open
  independent CUDA contexts without extra configuration — but without CUDA MPS, the GPU's hardware
  scheduler time-slices between contexts by default rather than truly co-executing their kernels.
  Historically this adds context-switch overhead and can leave concurrent-process throughput far
  below the naive "N processes = N× the work" expectation. **Do not trust the 1.7x upper bound
  without measuring this directly first** (§3 step 1) — it is exactly the kind of assumption the
  playbook's "real speedup far below what theory predicted" red flag warns about.
- **Fail-fast**: submit each tile as a future; on the first raised exception, call
  `pool.shutdown(cancel_futures=True)` (or explicitly `terminate()` every worker process), then
  re-raise — mirroring today's raise-on-`None`-return contract in `run_batch` (§1 item 2).
- **Global merge**: workers return the same `(abs_x, abs_y, owned_results)` tuples the current
  single process already produces internally; the parent gathers from N sources instead of 1 and
  performs the *identical* sort+renumber step (§1 item 1) — no change to that logic.
- **Ceiling**: 1.23x–1.7x per doc 18 §6.2 — the widest ceiling of anything left in this pipeline,
  and the largest correctness risk (per doc 19's own ranking).

### Candidate C — CUDA MPS-mediated multi-context (an add-on to B, not a replacement)

Run NVIDIA CUDA MPS (`nvidia-cuda-mps-control`) in front of Candidate B's worker processes. MPS lets
multiple client processes share GPU execution resources through one server-side context, reducing
the per-context switch overhead that plain multi-process CUDA (no MPS) pays, and can let independent
processes' kernels interleave more efficiently than default time-slicing.

- **Caveat found on this exact machine, not assumed**: MPS is officially validated by NVIDIA
  primarily on data-center GPUs (Tesla/A100/H100-class); this RTX 5090 is a consumer GeForce card,
  not on NVIDIA's published MPS support matrix, though unofficial reports of it working exist. This
  must be verified empirically here (§3 step 1's toy probe, extended to test with and without an MPS
  daemon running) before betting any redesign on it. If MPS doesn't work reliably on this card,
  Candidate C simply collapses to Candidate B without the efficiency gain — not a blocker to B, just
  an unproven upside.
- **Not the same as the already-closed CUDA-stream/depth-2 bubble redesign** (`19-open-backlog.md`
  §1 item 4, closed at ≤1.065x, doc 18 §3). That item was single-process, single CUDA context,
  multiple *streams* — a much smaller ceiling because it only ever reaches *inter-forward* bubbles
  within one process, never the fork-under-CUDA-blocked *intra-forward* launch-bound idle (the
  larger half of all device idle, per doc 18 §3). MPS/multi-context is a different mechanism
  (separate processes, separate contexts, mediated by MPS) that specifically targets that other,
  larger idle component. **This distinction should be stated explicitly whenever this document is
  read**, so nobody re-opens or conflates it with the already-stopped-out item.

### Candidate D — the concretely recommended shape of B ("B done right")

Small *N* (2–3, pending §3 step 2's concurrency-knee measurement) GPU worker processes, each running
an **unmodified copy** of today's existing per-tile logic — `_process_precut_tile_gpu` /
`_process_precut_tile_cpu` / `_frozen_gc_generation`, verbatim, reused as-is inside each worker —
fed a subset of tiles by an outer dynamic dispatcher (Candidate B's queue design). This is not a new
architecture; it is Candidate B scoped down to the smallest new-code surface: the only genuinely new
code is the outer process pool, the work-stealing queue, result-gathering, and fail-fast
propagation. Recommended as the concrete first build target once §3's measurements support it,
specifically *because* it reuses proven code rather than rewriting the per-tile pipeline.

### Candidate E — fork-based reuse of already-loaded models (not recommended; recorded to close the door on it)

Use `os.fork()` *after* model init in the main process, relying on copy-on-write memory pages so
children inherit the already-loaded models without paying N× VRAM or N× init cost. **This is exactly
the pattern `backend/algorithms/hybrid/CLAUDE.md` already flags as unsafe** — CUDA contexts are not
fork-safe; a forked child typically inherits a corrupted/unusable CUDA context and hangs or crashes
on its first CUDA call. This is *why* Candidates B/C/D all specify the `spawn` start method, not
`fork` — spawn re-imports and re-initializes each child cleanly, sidestepping this failure mode
entirely (at the cost of paying init/VRAM per worker). Do not attempt Candidate E without first
testing, in complete isolation from this pipeline (a 5-line repro script, not real model code),
whether this specific torch/CUDA/driver combination (2.11.0+cu130 / 580.159.03) has any documented
safe fork-after-CUDA-init workaround. Recorded only so "just fork, it's cheaper" isn't quietly
re-attempted later without this context.

### Summary table

| candidate | touches CUDA in workers? | new code surface | ceiling | correctness risk | prerequisite for |
|---|---|---|---|---|---|
| A — CPU-back-end pool | no | moderate (IPC transfer) | small alone, unproven | low | pairs with B/D |
| B — N-way GPU worker pool | yes, own context/worker | large | 1.23x–1.7x | high | canonical design |
| C — MPS-mediated multi-context | yes, MPS-shared | small on top of B | upside over B, unproven here | high (inherits B's) | only if B is serialization-limited |
| D — small-N "B done right" | yes, own context/worker | smallest of B/C/D | same as B, smaller N | high (inherits B's) | **recommended first build** |
| E — fork-based reuse | yes, unsafe | n/a | n/a | architecturally blocked | do not build |

## 3. Recommended experiment order (cheapest signal first)

1. **Toy CUDA multi-context concurrency probe — no pipeline code at all.** Before writing any
   pipeline multiprocessing code: two independent `spawn`-context Python processes, each running a
   real model forward (or a representative dummy CUDA workload), instrumented with
   `nvidia-smi dmon` and/or `torch.cuda.Event` (the same Event-based technique doc 18 §3 already
   validated for intra-process bubble measurement) to check whether their kernels actually overlap
   in wall time or effectively serialize. **This one experiment answers the biggest open question
   before any of B/C/D are worth building.** If they fully serialize, the ceiling for B/C/D collapses
   toward ~1.0x and this whole line of work should stop right here — the cheapest possible way to
   invalidate the avenue, per the playbook's "measure before optimizing" rule.
2. **If step 1 is positive, find the concurrency knee.** Repeat with 2, 3, 4 (and more, budget
   permitting) concurrent processes to find where adding another process stops buying wall-clock —
   likely well before VRAM or raw process-count limits, since GPU SM/memory-bandwidth is the actual
   shared resource, not process count. This number is *N* for Candidate D.
3. **Candidate D prototype at the winning N, smallest scale first** (the existing small/25-tile
   crop). Verify: (a) the per-cell correctness veto vs. the recorded round-4 baseline (§1 item 6),
   (b) fail-fast propagation — deliberately inject a bad tile and confirm the *whole* run aborts and
   every sibling process actually terminates, not just the one that errored. Do this **before**
   ever running at medium/large scale.
4. **Scale to medium (121) and large (441) anchors** under the full existing measurement protocol
   (idle-GPU check, `--gpu-dmon --workers 8`, `pip freeze` + env stamp, n≥2 repeats per doc 18 §0) —
   same rigor as every prior round, extending `scripts/arm_report.py`'s two-arm model to an N-arm
   model (or per-process wall/idle breakdown) since the current model assumes exactly one MAIN + one
   BG arm.
5. **Only after D is measured, decide on Candidate C (MPS).** Size it only if step 2's concurrency
   knee looked *serialization-limited* rather than *SM-limited* — MPS specifically targets reduced
   context-switch/serialization overhead; if that's not what's costing time in step 2, MPS will not
   help and should not be built.
6. **Candidate A can be prototyped independently, in parallel with 1–5.** It never touches CUDA, so
   it carries none of B/C/D's risk and doesn't need to wait on the concurrency probe. Worth an early,
   cheap, low-risk ablation on its own regardless of how B/C/D turn out — it is a strict subset of
   the total engineering work either way.

## 4. Decision gates / stop-loss

- **Step 1 negative (kernels effectively serialize across processes on this GPU):** stop the B/C/D
  line entirely, record it stopped-out with the measured evidence — same discipline as the
  already-closed CUDA-stream/depth-2 item (doc 18 §3). Fall back to Candidate A only, which remains
  bounded by the existing single-process 1.23x floor but carries none of the fork-under-CUDA risk.
- **Per-process CUDA context overhead (step 1/§5) too large to fit N≥2–3 in 32 GB** alongside real
  workloads: cap *N* accordingly and re-derive the ceiling downward from the 1.7x upper bound before
  committing further engineering time to it.
- **Any correctness veto failure** — a per-cell delta outside the established same-code noise floor
  (§1 item 6), a fail-fast test that doesn't actually terminate every sibling worker, or any cell
  miscount outside the recorded reference range — is an immediate, non-negotiable stop. Per the
  playbook: "correctness is a veto," not a tunable tradeoff against speed.
- **Real-WSI-scale validation is still outstanding** (`19-open-backlog.md` §1 item 7 — every round
  to date, including this design, only has crop-scale evidence). Do not ship any multiprocessing
  change to production ahead of that validation — every ceiling estimate in this document inherits
  the crop-scale tissue-density assumption every prior round has flagged as an upper bound, not a
  measured full-slide number.

## 5. Open questions / prerequisites to resolve during step 1

- **`spawn` vs. `forkserver`**, and whether `CellposeSegmenter`/`UNetPPInference` are cleanly
  picklable/importable in a spawned child. Likely yes — `run_batch` already does clean top-level
  module imports (lines 50–123) with no interpreter-global state beyond `config` — but this has
  never been exercised in a multiprocessing context and must be tested directly, not assumed.
- **Per-process CUDA context memory overhead** on this exact driver/torch combination (580.159.03 /
  2.11.0+cu130) — unmeasured. This number, not the 2.79 GB model-weight figure alone, determines the
  real *N* that fits in 32 GB.
- **Nested process-pool interaction with `detect_all_dots`'s own internal `joblib` parallelism**
  (`n_jobs=-1` today, per `bottleneck-list.md` ②). If any candidate above wraps a function that
  itself spawns `joblib` workers, an outer `ProcessPoolExecutor` risks CPU oversubscription
  (daemonic-process restrictions can also outright block nested pools in some `multiprocessing`
  configurations). Likely needs `n_jobs=1` inside multiprocessed workers, traded against running more
  tiles concurrently — net wall-clock is the only thing that decides this, per the playbook; measure,
  don't reason from theory.
- **`gc.freeze()`/`unfreeze` contract for a persistent worker pool** used from the long-lived API
  server path (`backend/api/hybrid.py`), as distinct from today's clean single-shot CLI process
  lifetime — needs an explicit per-call freeze/unfreeze design if workers are reused across requests
  (§1 item 4). Not needed if workers are spawned fresh per `run_batch` call and torn down after —
  simpler, but reintroduces per-request model-init cost (§1 item 7) that a persistent pool would
  avoid. This tradeoff needs a decision once request-latency requirements are known (adjacent to the
  never-measured `19-open-backlog.md` §1 item 8).

## 6. What this document is not

- **Not a recommendation to implement any specific candidate.** Per the playbook, "let the
  bottleneck define the solution space," and per this project's own discipline (docs 17/18),
  nothing here gets built before it's measured. This document is the menu; §3 is the order to work
  through it; §4 is where to stop if the evidence says stop.
- **Not reopening already-closed items.** The CUDA-stream/depth-2 bubble redesign
  (`19-open-backlog.md` §1 item 4) and the joblib-based `detect_all_dots` process-backend attempt
  (§1 "already closed" list) are both explicitly distinct from Candidates C and A above respectively
  (see each candidate's own note) — this document does not re-litigate either, it names two related
  but different experiments that haven't been tried.
- **Not a replacement for `19-open-backlog.md`.** When any experiment in §3 produces a result, record
  it back into that file's §1 item 1 row (or this doc's own future revision), per that file's own
  "update discipline" — don't let this document go stale the way a snapshot number would.
