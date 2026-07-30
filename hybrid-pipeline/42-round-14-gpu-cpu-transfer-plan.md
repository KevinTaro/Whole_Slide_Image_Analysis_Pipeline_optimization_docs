# 42 — Round 14: GPU↔CPU data-transfer plan under `workers>1` (multiprocess)

> **Design-only document — no pipeline code changed here.** Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose, cheapest-first. Scoped explicitly to `workers>1` (the shipped
> `workers=4` production default since round 13) — **not** `workers=1`. Pinned memory, shared
> memory, zero disk scratch, zero copy, PyTorch asynchronous transfer, and CUDA streams are the
> user's named directions, aimed at reducing time and memory/disk space specifically in the
> multiprocess regime.

## 0. Scope — why "focus on `workers>1`" changes what this document can cite

This supersedes the first draft of this document, which cited every ceiling from this project's
existing record without flagging which process regime each number was actually measured in. That
was a real gap: **every single existing measurement bearing on GPU↔CPU data movement in this
project's history was taken at `workers=1`**, either because it predates cross-tile multiprocessing
entirely or because the multiprocess re-run was explicitly skipped or its wall-clock verdict
withheld. Re-checked directly against each source:

| finding | where it was actually measured | workers>1 status |
|---|---|---|
| `B2r_tile_read` prefetch (Option L) mechanism | Wired into `_mp_tile_worker` round 10 (doc 35 §2) | Mechanism **confirmed** at `workers=4` (bucket moves as predicted); crop-scale **wall-clock claim explicitly withheld** ("−0.83%... inside the noise floor," doc 35) |
| `B2r_tile_read`'s "settled, 4.21% of wall, 99.8% hidden" verdict | Doc 37 §4.1's full-slide run | **`workers=1` only** (`--no-prefetch` full-slide re-run, doc 37 §2's own header) |
| CUDA-stream / pipeline-depth-2 bubble redesign | Doc 17 §3 (plan) → doc 18 §3 (measured), **round 4** | **Predates cross-tile multiprocessing entirely** — doc 20/21 (round 5) is where `workers>1` was introduced. The ≤1.065× ceiling describes a world with one process on the GPU |
| Cellpose `_from_device` — 24.2% of MAIN-arm samples | Doc 23 §7, `py-spy record` "over the medium crop at `workers=1`" (doc 23's own wording) | **`workers=1` only**, single process, single CUDA context, nothing else on the GPU |
| Depth-2 pipelining inside a worker (deepen the per-worker CPU/read overlap) | `m0_multiprocess.py:225-226`'s own comment cites doc 21 §6 | **Tested and rejected**: "single-process +2.8% slower, flat under multiprocessing (W=3)" |
| CUDA MPS (multi-context GPU sharing) | Doc 20 Candidate C / doc 21 §5 | **Is** a `workers>1`-relevant test — works on this GPU (+44% synthetic launch-bound microbenchmark) but **flat end-to-end**, because the real pipeline isn't serialization-limited at its knee |

None of this means the `workers=1` numbers are wrong — it means they answer a different question
than this round is asking. Four concurrent processes sharing one physical GPU and one physical disk
is a qualitatively different regime for exactly the mechanisms in play: PCIe bandwidth and GPU copy
engines are shared, not dedicated; each process has its own CUDA context and default stream, with no
built-in coordination between them; and the OS page cache sees four concurrent readers instead of
one. This document's Discover step treats every prior number as **"known at `workers=1`, open at
`workers=4`"**, not as something that transfers by assumption — the same discipline doc 30 §3 itself
flagged and never resolved ("Multiprocess compatibility... a real IPC design... never done").

### 0.1 Direct answer: does `workers=4` get zero benefit from these directions?

**No — and this document should not be read as concluding that.** Three of the six rows in §0's
table are genuinely closed by a real measurement, but each is closed for a specific, narrow reason
that does not generalize to "pinned memory / shared memory / CUDA streams can't help under
multiprocessing":

- The UNet++ `.to(device)` transfer is small **because the data volume is trivial** (~12.6 MB/tile
  against ~600 ms/tile of GPU compute on tissue tiles) — an arithmetic fact about tile size vs.
  compute cost, true at any worker count.
- The one CUDA-stream design already tried (overlap tile N's CPU tail with tile N+1's forward) is
  capped **because that specific bubble is narrow**, and even narrower once 4 processes are already
  keeping the GPU busy — this is one design choice being closed, not "streams don't help."
- CUDA MPS is flat **because this pipeline isn't bottlenecked on launch/context-switch overhead** —
  MPS answers a different question than data-transfer efficiency.

The other two rows — Cellpose's `_from_device` and the disk-scratch round trip — are **not shown to
be small**. They are the two places in this entire project's history where a real, potentially large
cost was identified (24.2% of one process's GPU-thread time, in `_from_device`'s case) and then
never measured in the actual `workers=4` production regime. If `_from_device` turns out to be
genuine, fixable transfer/sync cost, it is paid independently by all four worker processes — fixing
it helps all four at once, which is a case where the *absolute* payoff at `workers=4` could exceed
what a single-process measurement would suggest, not fall short of it. §2 and §3 below exist
precisely because this question is open, not because it's expected to close negative.

## 1. Discover — the multiprocess architecture, verified against current code

Traced via `codegraph_explore`, not assumed, against `m0_multiprocess.py`:

- **`_run_tiles_multiprocess`** (`m0_multiprocess.py:250`) spawns `workers` processes via
  `mp.get_context("spawn")` — never `fork` (Candidate E ruled fork-based reuse unsafe: CUDA contexts
  aren't fork-safe). The **parent process never touches CUDA** on this path (the code's own comment:
  "父行程在這條路徑上完全不碰 CUDA") — `PYTORCH_CUDA_ALLOC_CONF` is written to `os.environ` *before*
  spawning specifically because that is the only point where it can affect every child's first CUDA
  allocation.
- **`_mp_tile_worker`** (`m0_multiprocess.py:138-247`) is the entire per-process pipeline: each
  worker independently calls `_init_unet_inferencer` / `_init_cellpose_segmenter` /
  `_init_dish_cellpose_segmenter` (own model weights, own CUDA context, own default stream — nothing
  shared across workers), then runs its own two-thread-pool loop: a dedicated one-thread `tile-read`
  pool (`prefetch_tile_reads`, depth 1) and a dedicated one-thread `tile-cpu` pool
  (`_process_precut_tile_cpu`, the BG arm). **Four workers means four independent read pools, four
  independent CPU pools, and four independent CUDA contexts, all contending for the same physical
  GPU and the same physical disk simultaneously** — this is the concrete shape of "how data moves
  between GPU and CPU" once `workers>1` is the actual regime, and it is structurally different from
  anything the `workers=1` numbers in §0's table were measured against.
- Depth is capped at 1 deliberately: the code's own comment at `m0_multiprocess.py:225-226` cites
  doc 21 §6's finding that deepening this pipeline further was measured and rejected (+2.8% slower
  single-process, flat under multiprocessing at W=3) — already-closed territory, not reopened here.
- **VRAM headroom at `workers=4` is tight**: 92.2% of the 32,607 MB card in production (round 11,
  with `expandable_segments:True`, the shipped default since `b3fa47d`) — roughly **2.5 GB of
  headroom across all four workers combined**, not per worker. Any design that adds device-resident
  state per worker (as opposed to host-side pinned buffers, which don't count against this budget)
  has to be sized against a shared 2.5 GB ceiling, not treated as free.
- **Disk I/O is now four-way concurrent, not single-reader.** Doc 27 §6.4's diagnosis of the
  `workers=1` regression (process RSS competing with OS page cache for room to hold the ~49 GB
  precut scratch) was reasoned about **one** process's RSS against the page cache. At `workers=4`,
  four processes each hold their own model weights and per-tile state, and four independent
  `tile-read` threads issue overlapping random reads against the same scratch directory
  simultaneously. Whether this makes the read cost better (four-way parallel disk queueing) or worse
  (four-way contention for the same page-cache room, or IOPS-bound rather than throughput-bound
  access) than the `workers=1` figure is **not known** — doc 30 §3 N1's own caveat ("if the host's
  storage is latency/IOPS-bound rather than throughput-bound... unconfirmed") applies with more force
  once four processes are issuing reads concurrently instead of one.

## 2. Analyze — Amdahl ceilings, re-derived for the `workers=4` regime specifically

| direction | what's actually known at `workers=4` | verdict |
|---|---|---|
| "Zero disk scratch" (shared-memory bypass of the precut round trip) | Mechanism (prefetch) confirmed wired in; **wall-clock share never measured at `workers=4` full scale** — only the `workers=1` 4.21%/99.8%-hidden figure exists, and doc 30 §3 Option M's own item 1 (cross-process IPC for a `pyvips.Image`/numpy array, since file paths — not decoded pixels — are what crosses a `spawn` boundary cheaply) was never resolved. Four-way concurrent disk contention could make this bigger than 4.21%, not smaller | **Measure first (§3 item 1)** — this is the one "zero disk scratch" question that's genuinely open under `workers>1`, unlike the `workers=1` case (closed) |
| CUDA-stream / intra-tile bubble redesign | Ceiling (≤1.065×) measured **before multiprocessing existed**. Under `workers=4`, four processes concurrently issuing kernels to the same GPU very plausibly already fills the per-process idle bubble this redesign targeted — a different mechanism than the original stop-loss, pointing the same direction | **Stop, more strongly than before.** Not reopened by this document |
| Depth-2 pipelining inside one worker | Directly tested under multiprocessing (doc 21 §6, W=3): flat | **Stop.** Already closed under the exact regime this document is scoped to |
| Pinned memory for this project's own UNet++ `.to(device)` transfer, ×4 concurrent workers | Same ~12.6 MB/tile transfer, now potentially four such transfers overlapping in time (≤~50 MB in flight across all workers at once) — still a tiny fraction of any PCIe generation's bandwidth, and each worker's own `B1_unet_coremask`-equivalent bucket is bounded by the same per-tile GPU compute time as before | **Still low priority regardless of worker count** — concurrency across workers doesn't change the fundamental Amdahl math from the first draft's §1.1, it only clarifies that the "×4" doesn't turn a sub-1%-of-wall bound into something larger |
| Cellpose `_from_device` (GPU→CPU transfer inside mask/flow post-processing) | 24.2% of MAIN-arm samples, **`workers=1`, single CUDA context, nothing else on the GPU** (doc 23 §7). At `workers=4`, four processes each call this concurrently against one shared physical GPU — the transfer/sync could **inflate** (queued behind sibling processes' kernels on shared copy engines) or its **share of wall could shrink** (the measured 1.745× end-to-end speedup means the denominator itself is smaller, so the same absolute per-tile cost is a different percentage of a different wall) | **Unknown, and now the central question has an extra dimension**: not just "genuine transfer vs. GPU-wait" (§2 of the first draft) but "does 4-way GPU contention change either of those relative to the single-process trace." Must be measured under real `--mp-workers 4`, not inferred |
| CUDA MPS (explicit multi-context GPU sharing) | Already tested under multiprocessing (doc 20 Candidate C, doc 21 §5): genuinely faster on a synthetic launch-bound microbenchmark, **flat end-to-end** because this pipeline is not serialization-limited at its knee | **Stop.** Already closed, under the correct regime, no new evidence here reopens it |

The quickref's anti-pattern #2 ("optimizing something under ~10% of total time") still rules out the
UNet++ transfer and depth-2 pipelining rows without further debate. CUDA-stream bubble redesign and
CUDA MPS are both closed **and** were tested in — or, for the bubble redesign, are now argued to be
even less relevant under — the multiprocess regime specifically, so neither is reopened. The two
rows genuinely unresolved *for `workers>1`* are the disk-scratch question and `_from_device` — both
because the existing record is silent on what four-way GPU/disk contention does to them, not because
either was shown to be small under multiprocessing.

## 3. Plan — cheapest first, all steps run under `--mp-workers 4`

### Item 1 — measure `B2r_tile_read`'s real share under 4-way concurrent reads (cheap, do first)

Run a crop-scale `--mp-workers 4` measurement (not full-slide — too expensive to be the first step)
with `perf_measure.py --worker-timings` to get the per-worker `B2r_tile_read` bucket, and compare its
share of each worker's own wall against the `workers=1` reference (14.02 ms/tile tissue, 6.82 ms/tile
background, doc 33 §1). This settles whether four-way concurrent scratch reads behave like four
independent `workers=1` streams (the "per-worker-invariant" assumption doc 30 §3 made but never
verified) or whether they contend for disk/page-cache in a way that inflates the cost. Cheap: reuses
`perf_measure.py`'s existing `--worker-timings` instrumentation (doc 27 §4's 26 worker-side buckets),
no new tooling.

**Decision it feeds**: if the per-worker share stays close to the `workers=1` figure, Option M's
cross-process IPC redesign (shared memory instead of a file round trip) stays exactly as
deprioritized as it already is — the ceiling is still ~4% divided across four workers, i.e. small.
If it's materially worse under contention, that is the first time in this project's history the
"zero disk scratch" direction would have a real, `workers>1`-specific ceiling worth designing
against, and Option M's open IPC question (doc 30 §3 item 1) becomes worth resolving rather than
parking.

### Item 2 — trace `_from_device` under real 4-way GPU contention (decision-determining for the standout candidate)

Doc 23 §7's py-spy trace attached to a single `workers=1` process. Under `workers=4`, py-spy would
need to attach to one worker process among several spawned ones (each a separate PID) — this project
has already hit a wall with `py-spy` attach reliability on this host (doc 37 §2's `exit_latency_probe.py`
was built specifically because `py-spy` could not attach for a different investigation). **Nsight
Systems is the right tool here instead of py-spy**: it natively traces a full multi-process tree,
which is exactly the shape of a `--mp-workers 4` run, and it directly distinguishes actual
`cudaMemcpyAsync`/D2H byte transfer from `cudaStreamSynchronize`/kernel-wait — the same distinction
the first draft's Discover step needed, now with the added question of whether four concurrent
processes' transfers serialize against each other on shared GPU copy engines.

Run at crop scale, `--mp-workers 4`, comparing:

- `_from_device`'s share of one worker's own wall (does it still look like ~24.2% of that worker's
  MAIN-thread time, or does contention change it), and
- whether the *absolute* time in `cudaMemcpyAsync` vs. `cudaStreamSynchronize` shifts when three
  sibling processes are also active on the GPU, versus running the identical worker alone.

This is the measurement that decides item 3, exactly as it did in the single-process framing — just
run in the regime this round is actually scoped to.

### Item 3 — pinned-memory/`non_blocking` microbenchmark for UNet++'s own transfer (cheap, low expectation, bundle with item 2's harness)

Same as the first draft's §1.1/item 2, but run inside one spawned worker process (its own CUDA
context, not the parent, which never touches CUDA on this path) and under concurrent load from the
other three workers, not in isolation — an isolated microbenchmark would not reflect the actual PCIe
contention this pipeline runs under in production. Prediction unchanged: negligible, because the
transfer is small (~12.6 MB/tile) relative to each tile's own GPU compute budget regardless of how
many sibling workers are also transferring. This step exists to confirm that, not to assume it.

### Item 4 — only if item 1 or item 2 shows a real, `workers=4`-specific ceiling ≥10% of wall

- **If item 1 reopens the disk-scratch question**: design Option M's never-resolved cross-process IPC
  (doc 30 §3 item 1) — e.g., shared memory (`multiprocessing.shared_memory` or a memory-mapped
  region) so a worker's own `PrecutStream`-equivalent hands decoded pixels to itself without a file
  round trip, sized against the four-worker disk-contention number item 1 actually measures, not the
  `workers=1` figure.
- **If item 2 shows `_from_device`'s cost is genuinely transfer-bound and worsens under multiprocess
  contention**: design a per-worker fix inside `_mp_tile_worker`'s own process — pinned staging
  buffers, `non_blocking=True`, a dedicated copy stream *per worker* (each worker already has its own
  CUDA context, so this is naturally per-process, not shared). Any device-resident staging memory
  this adds must be checked against the shared **2.5 GB VRAM headroom across all four workers
  combined** (§1) — a constraint that does not exist in a `workers=1` framing and must be sized
  explicitly, not assumed away.
- **If item 2 shows `_from_device`'s cost is dominated by `cudaStreamSynchronize`/GPU-wait, worsened
  by contention rather than by transfer bytes**: this is not a data-movement fix at all — it would be
  evidence that four processes' kernels are queuing behind each other on the GPU, which is the same
  family of problem CUDA MPS was built to address and which doc 21 §5 already measured as flat
  end-to-end for this pipeline. Re-affirm that finding rather than propose a new fix.

### Item 5 — explicitly not scheduled (reaffirmed, not reopened)

- **CUDA-stream / intra-tile bubble redesign** — closed before multiprocessing existed, and §2
  argues contention makes its target smaller, not larger, under `workers=4`. Not reopened.
- **Depth-2 pipelining inside a worker** — directly tested under multiprocessing already (doc 21 §6,
  flat at W=3). Not reopened.
- **CUDA MPS** — already tested under multiprocessing, flat end-to-end. Not reopened.
- **Fork-based context/model reuse across workers** — architecturally unsafe (Candidate E), and any
  shared-memory design in item 4 must hand off *data*, not a CUDA context or model, across the
  `spawn` boundary.

## 4. Decision gates / stop-loss

- **VRAM**: any design from item 4 that adds device-resident memory must fit inside the ~2.5 GB
  headroom shared across all four workers (92.2% of 32,607 MB already committed, round 11). This is
  a hard constraint, checked before code is written, not after.
- **If item 1 shows disk contention does not materially worsen `B2r_tile_read`'s per-worker share**:
  stop — Option M's IPC redesign stays parked exactly as it already is, now confirmed rather than
  assumed for `workers>1`.
- **If item 2 shows `_from_device`'s cost (or its worsening under contention) is dominated by
  GPU-wait rather than genuine transfer**: stop, do not design a transfer fix — re-affirm the CUDA
  MPS / bubble-redesign stop-losses instead, per item 4's third bullet.
- **Correctness veto, unchanged and now explicitly per-worker**: any change to a worker's internal
  transfer pattern (Cellpose or otherwise) must produce bit-identical (or this project's established
  GPU-nondeterminism noise-floor-identical) masks against the unpatched baseline, checked
  independently — a bug that only manifests under real four-way concurrency would not necessarily
  show up in a `workers=1` correctness check.
- **Judge everything by share of the real `--mp-workers 4` wall**, never a `workers=1` number carried
  over by assumption, and never a microbenchmark of a single worker run in isolation from its three
  siblings.

## 5. What this document is not

- **Not a `workers=1` optimization plan** — every item above is scoped to run under `--mp-workers 4`
  specifically; a `workers=1` framing was this document's first draft and is superseded here.
- **Not a reopening of the CUDA-stream bubble redesign, depth-2 pipelining, or CUDA MPS** — all three
  were tested under conditions this document's own §2 argues are at least as unfavorable to them
  under `workers=4` as under `workers=1`, if not more so.
- **Not a commitment to building shared-memory IPC or a Cellpose transfer patch** — items 4's two
  branches are entirely gated on items 1 and 2's measurements; a large `workers=1` percentage (doc 23
  §7) is a lead into this round, not a green light for a `workers=4` fix.
- **Not a re-derivation of rounds 9–13's numbers** — §0/§1 cite them, with an explicit note on which
  process regime each was measured in, precisely because that regime tag is what changed in this
  rewrite.
