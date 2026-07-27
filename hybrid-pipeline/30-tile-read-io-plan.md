# 30 — `B2r_tile_read` / precut-scratch I/O: design plan

> **Design-only document — no pipeline code changed here.** Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose.

## 0. Discover — yes, there is an existing record; here is all of it

This bottleneck is not new and has been tracked since round 4:

| where | what it says |
|---|---|
| [`18-gpu-starvation-prerequisites-implementation.md`](./18-gpu-starvation-prerequisites-implementation.md) §6.3 | Original measurement and **stop-loss**: 5.87 s = 1.22% of wall on the 441-tile crop, ceiling 1.012×. "Nothing in the measured record supports building this." |
| [`19-open-backlog.md`](./19-open-backlog.md) item 6 | **Reopened (round 8)**: "the stop-loss was measured on a crop that doesn't hold at production scale." |
| [`DISCOVERED-NOT-IMPLEMENTED.md`](./DISCOVERED-NOT-IMPLEMENTED.md) #13 / #44 | Same reopening, filed as two entries (the original candidate and the round-8 finding that its stop-loss doesn't survive to scale). |
| [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md) `B2r_tile_read` row | Current state: 🟢 OPEN, 17.2% of wall (2,368.5 s) at full scale. |
| [`27-remaining-work-implementation.md`](./27-remaining-work-implementation.md) §6.4 | The round-8 full-slide measurement that reopened it: "the ~49 GB precut scratch no longer fits page cache at full-slide scale, so reads that were effectively free on a crop become real disk I/O." |

**So: nothing has fixed this yet.** Doc 18 closed it once, correctly, on the evidence it had; doc
27 reopened it with real full-slide evidence and explicitly left the ceiling "not yet re-derived."
This document is that re-derivation, plus a plan.

## 1. Analyze — what `B2r_tile_read` actually is, verified against the current code

Traced directly via `codegraph_explore` (not assumed from the bucket name):

```python
# hybrid_pipeline.py:261-288, _process_precut_tile_gpu — the GPU-front half of one tile
def _process_precut_tile_gpu(ihc_tile_path, dish_tile_path, abs_x, abs_y, geometry,
                              unet_inferencer, cellpose_segmenter, dish_cellpose_segmenter,
                              output_dir):
    ...
    try:
        ihc = _read_rgb(ihc_tile_path)     # <-- B2r_tile_read
        dish = _read_rgb(dish_tile_path)   # <-- B2r_tile_read
    except Exception as exc:
        ...
    ...
    chunk = _process_one_chunk_gpu(ihc, dish, abs_x, abs_y, ...)   # the 3 GPU forwards
```

`_read_rgb` (`hybrid_pipeline.py:603-610`) is `skimage.io.imread` + a channel-normalize +
`astype(uint8)`. It runs **synchronously on the MAIN thread**, called twice (IHC, DISH) at the top
of `_process_precut_tile_gpu`, **before** the GPU forward for that tile starts. `run_batch`'s tile
loop (`hybrid_pipeline.py:1161-1168`) calls `_process_precut_tile_gpu` directly, once per tile, in
its own `for` loop — there is no prefetch, no read-ahead, no thread hand-off for this specific
step. The tile files it reads were written moments earlier by `PrecutStream._cut`
(`m0_reader.py:137-148`), which runs on an **independent** 8-worker thread pool that cuts every
tile from the two source WSIs up front and streams completions via `as_completed` — the write side
of this round trip was already fixed once (doc 18 §4.2, "Precut A") by overlapping it with the
analysis loop; the **read** side (turning those written files back into numpy arrays for analysis)
was never separately addressed, because at crop scale it was too cheap to matter (doc 18 §6.3).

**One correction to the compact record, made because the code disagrees with it.**
`measurement/bottleneck-list.md`'s current row describes `B2r_tile_read` as sitting "on the arm
with slack." Read against the code above, this is not literally where the time is spent: the read
call is inline, on the MAIN thread, strictly before the GPU forward — it is structurally part of
the **MAIN/critical** arm, the same one `B1_m3b_cellpose` and `B1_unet_coremask` are on, not the
BG arm `B3_detect_dots`/`B2_png_encode` run on. The most charitable and almost certainly intended
reading of that phrase is forward-looking, not descriptive: *if* this read were moved onto a
background thread (§3 Option L), it could execute concurrently with the arm that has slack, the
same way Precut A's write-side already does. Read as a plan, not a measurement, that phrasing is
consistent with everything else in this document; read as a description of where the cost
currently lives, it is not, and this plan proceeds on the code-verified version.

## 2. Analyze — why this got 14× worse at full scale, and a connection to doc 28

Doc 27 §6.4's own explanation — "the ~49 GB precut scratch no longer fits page cache" — is correct
but underspecifies the mechanism, and the fuller version matters for sequencing this plan against
[`28-gc-collect-round2-plan.md`](./28-gc-collect-round2-plan.md):

- At the 441-tile crop scale, total scratch volume is small enough that the OS page cache holds
  essentially all of it, so `_read_rgb` almost never touches physical disk — it reads back pages
  the kernel still has cached from `PrecutStream`'s write moments earlier. This is why doc 18 §6.3
  measured 1.22% of wall: it was measuring page-cache-warm reads, not disk I/O.
- At full-slide scale, the scratch is ~49 GB, and — separately, but simultaneously — the pipeline
  process itself peaks at **61.13 GB of RSS** (doc 27 §6.3), driven mostly by
  `per_tile_owned`'s accumulation (doc 28's subject) and the final stitch holding 27,565 lazy
  pyvips handles open. On a host where the process's own working set competes for RAM with the OS
  page cache, growing RSS directly shrinks the cache room available to hold the precut scratch —
  so tiles written by `PrecutStream` get evicted before `_process_precut_tile_gpu` reads them back
  a few tiles later, turning a page-cache hit into a genuine cold disk read. **This is not
  primarily a "scratch is bigger than RAM" problem** (49 GB scratch vs. presumably far more than
  49 GB of host RAM, if the host can sustain 61 GB of process RSS at all) — it's a **process RSS
  vs. page cache competition** problem, and the process's own RSS growth is exactly what doc 28
  targets.
- **Practical consequence for sequencing**: doc 28's fix reduces the number of *tracked* objects,
  not necessarily the RSS they occupy (freezing pins memory, it doesn't free it) — so it should
  **not** be assumed to fix this bottleneck as a side effect. But it is free information: re-run
  the full-slide validation once doc 28 ships, and re-measure `B2r_tile_read`'s share before
  spending any engineering effort here specifically (Option K below). If RSS drops for other
  reasons in the future (e.g., Option J from doc 28, if it's ever picked up), the same
  cheap-re-measurement-first logic applies again.

## 3. Plan — options, cheapest first

### Option K — re-measure after doc 28, before building anything (free, do first)

Zero engineering cost: once doc 28's gc fix lands and the next full-slide validation run happens
for that reason anyway, read `B2r_tile_read`'s bucket off the same run before scoping anything
below. This is exactly the kind of "the bottleneck moves, re-measure after every fix" discipline
the playbook's own three-things-to-remember section states, and it costs nothing extra since the
run is happening regardless.

### Option L — prefetch/pipeline the read (cheap, recommended first build if a gap remains)

Move `_read_rgb`'s two calls off the synchronous, blocking position they occupy today and onto a
one-tile-ahead read-ahead, so tile N+1's disk read overlaps with tile N's GPU forward instead of
preceding tile N+1's own forward serially. This is architecturally the same move doc 18 §4.2
already made for the *write* side (Precut A) — "the tile grid is derivable from the image header
alone... so the geometry `run_batch` needs upfront does not require the cutting to have happened"
— applied one stage further downstream, to the read that consumes what Precut A produces.

**Why this is plausible, from the numbers already in doc 27 §6.4:**

| bucket | total | calls | per-call average |
|---|--:|--:|--:|
| `B1_m3b_cellpose` (2× Cellpose forward, tissue tiles only) | 6,358.9 s | 24,360 | ≈261 ms/tile |
| `B2r_tile_read` (2 reads, all tiles) | 2,368.5 s | 55,130 | ≈43 ms/tile |

On tissue tiles, the per-tile read cost (≈43 ms) is a small fraction of the per-tile Cellpose
forward cost (≈261 ms) it could hide behind, if prefetched one tile ahead. **This suggests a large
share of the 17.2% could be hidden almost for free on tissue tiles** — but this average conflates
tissue and background tiles, and background tiles (55.8% of the slide) skip the Cellpose forwards
entirely (`core_mask.sum() == 0` fast path, `hybrid_pipeline.py:486-487`), leaving only the cheap
UNet++ forward (`B1_unet_coremask`, 734.8 s / 27,565 tiles ≈ 27 ms/tile) for a prefetched read to
hide behind — likely not enough margin to fully absorb a 43 ms read. **The tissue/background split
of `B2r_tile_read`'s own cost is not currently measured** — doc 27's 2,368.5 s is an aggregate
across both populations, and this is the first thing to measure before sizing Option L's real
ceiling, not something to assume from the aggregate average.

**Design sketch, subject to the measurement above:**

- Add a one-deep read-ahead: while `_process_one_chunk_gpu` runs for tile N (the three GPU
  forwards, on MAIN), a background helper calls `_read_rgb` for tile N+1's two files. This can
  reuse the same single-thread `ThreadPoolExecutor` pattern `run_batch`'s tile loop already uses
  for the CPU back-end (`hybrid_pipeline.py:1159-1160`, `tile-cpu` pool) — or a second, dedicated
  one-worker pool, to avoid contending with that pool's own (unrelated) work.
- Ordering/correctness: `run_batch`'s tile loop already consumes `tiles` (the `PrecutStream`
  iterator) in whatever order `as_completed` yields them, and doc 18 §4.2 already established that
  processing order is safe to change (`_finish_batch` sorts by `(abs_y, abs_x, cell_id)` before
  renumbering, `_stitch_overlay_slide` reads by coordinate). A one-tile read-ahead doesn't change
  *which* tiles get processed, only *when* their bytes are decoded relative to the previous tile's
  GPU work, so this carries no additional ordering risk beyond what's already accepted.
- Fail-fast: today, a read failure inside `_process_precut_tile_gpu` is caught and turned into a
  `None` return, which `run_batch`'s loop treats as a real error and aborts the whole batch
  (`hybrid_pipeline.py:1169-1179`). A prefetched read's failure needs to surface at the point the
  prefetched tile is actually consumed (not silently dropped), preserving today's fail-fast
  semantics exactly — this is a real correctness requirement for the design, not just an
  implementation detail.
- No new dependency, no VRAM cost (this is a CPU/disk-I/O change), and it's confined to the
  `workers=1` **and** the multiprocess path equally, since each `_mp_tile_worker` runs its own
  independent copy of the same two-stage loop structure (doc 24 §0.5's "per-worker-invariant"
  reasoning applies here directly — a prefetch built once, at the loop level, benefits every
  worker without per-worker-specific code).

### Option M — eliminate the round trip through the precut scratch (`workers=1` only; larger change)

`PrecutStream._cut` already holds each tile as an in-memory `pyvips.Image` (`_crop_to_tile(...)`)
immediately before writing it to disk (`m0_reader.py:142-147`). The analysis loop then reads that
same tile back from disk via a **different** library (`skimage.io.imread` inside `_read_rgb`).
If `PrecutStream.__iter__` handed over the decoded pixel data directly — instead of, or alongside,
the file paths — the analysis loop would never need a disk round trip for its own consumption of a
tile at all.

This has a real ceiling (up to the full 17.2%, not just the hideable portion Option L targets) but
is flagged as the larger, second-priority option because of two open questions that must be
resolved before it's scoped, not assumed away:

1. **Multiprocess compatibility.** `_mp_tile_worker` (`hybrid_pipeline.py:673`, referenced from
   `_run_tiles_multiprocess`) pulls tiles from a shared work queue across `spawn`ed processes.
   File paths cross a process boundary cheaply; a `pyvips.Image` handle or a large in-memory numpy
   array does not, at least not without its own IPC design (shared memory, or serializing the
   pixel data through the queue, which reintroduces a copy cost this option is trying to remove).
   This option is very plausibly `workers=1`-only unless a separate IPC mechanism is designed for
   the multiprocess path — and multiprocess (`workers=4`) is the shipped production
   recommendation (doc 27 §6.6), so a `workers=1`-only fix has a narrower practical audience than
   it might first appear.
2. **The existing bit-exact regression guarantee is scoped to the disk round trip, not the
   in-memory path.** `m0_reader.py`'s own docstring states the precut tile format is verified
   "逐位元一致" (bit-identical) against `skimage.io.imread` for the on-disk file — that guarantee
   was established for *reading the file back*, not for *skipping the file and consuming the
   pyvips buffer directly*. Converting a `pyvips.Image` to a numpy array in-memory
   (`pyvips` exposes this via its buffer/array interface) must be independently re-verified
   bit-identical to what `_read_rgb`'s `skimage.io.imread` path produces today, covering dtype and
   channel-order handling specifically — it is likely fine given the file format was chosen
   precisely for lossless round-tripping, but "likely fine" is not the bar this project holds
   itself to elsewhere (doc 18 §4.2 ran an explicit sha256-over-242-tiles check before adopting
   the write-side streaming change; this option needs the equivalent check before adoption).

**Recommendation: scope this only if Option L's measured ceiling disappoints**, and even then,
build and ship it for `workers=1` first, independent of the multiprocess question — this project's
own precedent (doc 18 §4.2's Precut A) is to land the single-process win first and treat
multiprocess compatibility as a separate, explicit question rather than a blocking one.

### Option N — broader data-reading improvements (explicitly lower priority, future work)

Per the scope of this request specifically: these are recorded because they were asked for, not
because they are being recommended for near-term scheduling. Each is independent of Options K/L/M
and larger in either engineering cost, risk, or both.

- **N1 — cheaper intermediate compression for the precut scratch.** The scratch is written with
  lossless `deflate` (`m0_reader.py`'s `_TILE_COMPRESSION`), chosen for bit-exact round-tripping,
  not speed. `zlib`/`deflate` decode is not free CPU work; a faster lossless codec (or no
  compression at all) would cut `_read_rgb`'s decode cost at the price of a larger scratch
  directory (already ~49 GB; uncompressed could approach 2-4× that). Unlike doc 27 §3's `zstd`
  finding for the *final* stitch output, there is **no BioFormats/QuPath compatibility constraint**
  here — the scratch is pipeline-internal and never seen by a pathologist — so this tradeoff is
  purely disk-space-vs-CPU-decode-time, not gated by a correctness veto the way the final output
  is. Still needs measuring: if the host's storage is latency/IOPS-bound rather than
  throughput-bound (unconfirmed — doc 27 §6.2 notes the ~513 GB full-slide run has to land on
  `/home`'s filesystem, whose underlying device characteristics are not documented anywhere in
  this project), a smaller-but-slower-to-decode format could lose to a larger-but-faster one, or
  vice versa, and guessing either way would violate the playbook's "ignoring hardware/platform
  quirks" anti-pattern.
- **N2 — read scratch tiles with `pyvips` instead of `skimage.io.imread`.** The scratch tiles are
  *written* by pyvips (`PrecutStream._cut`); reading them back through a second, different image
  library (`skimage`) for `_read_rgb` costs a second TIFF-parsing implementation's overhead for no
  clear benefit, and diverges from `PrecutStream._open_rgb`'s own pyvips-based read path used for
  the *source* WSIs. Switching `_read_rgb`'s scratch-tile path to `pyvips.Image.new_from_file` +
  buffer-to-numpy would need the same bit-exact re-verification Option M's item 2 already
  requires, and is really a small piece of Option M rather than an independent option — recorded
  separately here because it could in principle be adopted even without the larger "skip the file
  entirely" change in Option M, as a smaller, standalone swap.
- **N3 — remove the precut scratch directory for `workers=1` entirely (largest, architecturally
  invasive).** The source WSIs are already opened once as random-access `pyvips.Image` handles
  (`PrecutStream._open_rgb`, held for the life of the stream). In principle, the `workers=1`
  analysis loop could crop tiles on demand directly from those two handles — the same
  `_crop_to_tile` call `PrecutStream._cut` already makes internally — and skip writing (and later
  reading) a ~49 GB scratch directory altogether. This would remove essentially all of both the
  write cost (already hidden by doc 18 §4.2's overlap, so removing it buys little on its own) and
  the read cost (this document's subject) for the single-process path, but it dissolves the
  current clean separation between "cut" and "analyze" stages — a separation the multiprocess
  path's shared file-based work queue and the round-8 checkpoint/resume feature
  (`output_dir/_resume/tile_x{ax}_y{ay}.pkl`, keyed by tile position, implicitly assuming tiles
  are independently addressable files or at least independently re-derivable) both lean on to
  varying degrees. This is explicitly **not** to be scheduled now — recorded per this plan's own
  request to note it rather than let it be rediscovered from scratch, in the same spirit
  `DISCOVERED-NOT-IMPLEMENTED.md`'s "out of scope, noted only" items (e.g. doc 14 §1 Option G) are
  kept. If it is ever revisited, it should be scoped together with Candidate F's storage argument
  (doc 25 §4.3/§11 — the ~157 GB/slide of duplicate blank-tile bytes) and doc 27 §6.2's ~513 GB
  full-slide disk footprint, since all three are really one question ("does this pipeline's
  intermediate/output disk footprint need to shrink") wearing three different hats.

## 4. Choose — sequencing and stopping rule

1. **Option K** (free) — re-measure after doc 28 ships, before scoping anything else here.
2. **Measure the tissue/background split of `B2r_tile_read`** (new, cheap instrumentation — extend
   the existing per-bucket worker-timing infrastructure from doc 27 §4 rather than building new
   tooling) to size Option L's real ceiling before building it. Don't skip this step: the
   aggregate 43 ms/tile average in §3 could be masking a much larger tissue-tile cost offset by a
   much smaller background-tile one, or vice versa, and the prefetch's payoff depends on which.
3. **Build Option L** if step 2 shows a worthwhile, hideable gap on tissue tiles. Cheap, no new
   dependency, mirrors an already-proven pipeline pattern (Precut A's overlap), applies to both
   `workers=1` and `workers=4` without extra design work.
4. **Only if L's measured ceiling disappoints**, scope Option M for `workers=1`, with the two open
   questions in §3 (multiprocess compatibility, bit-exact re-verification) resolved *before* any
   code is written — not discovered mid-build.
5. **Option N is deprioritized per this request's own framing** — record, do not schedule. Revisit
   only if disk footprint (not speed) becomes the driving concern, at which point N3 should be
   scoped alongside Candidate F and doc 27 §6.2's numbers together, not alone.

**Success criteria**: judged by `B2r_tile_read`'s share of wall on a real (or realistically
concatenated) multi-thousand-tile run, never a microbenchmark of `_read_rgb` in isolation.
Correctness veto: fail-fast semantics must be preserved exactly under prefetching (§3 Option L);
any in-memory bypass of the scratch file (Option M/N2) must pass an explicit bit-exact check
against today's `_read_rgb` output, the same discipline doc 18 §4.2 already applied to the
write-side change this plan extends. Ablate Options K → L → M individually — per the playbook's
anti-pattern #6/#7, a combined win across sequenced changes that isn't independently attributed to
each one doesn't clear this project's own bar for adoption.
