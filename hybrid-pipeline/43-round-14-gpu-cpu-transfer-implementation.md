# 43 — Round 14: GPU↔CPU data transfer at `workers>1` — implementation

> Executes [`42-round-14-gpu-cpu-transfer-plan.md`](./42-round-14-gpu-cpu-transfer-plan.md)
> §3 items 1, 2 and 3, under
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md).
> **Every measurement doc 42 asked for was taken. All of them close negative, and doc 42 §4's
> stop-loss therefore fires on both open questions.** Item 4 is gated on items 1/2 clearing 10% of
> wall; neither comes within an order of magnitude, so **no pipeline code was changed this round** —
> which is the plan executing as written, not the plan being abandoned.
>
> Raw metrics: [`measurement/_metrics_r14/`](./measurement/_metrics_r14/).

## 0. Result in one table

| doc 42 item | question it was opened to answer | measured at `--mp-workers 4` | gate |
|---|---|--:|---|
| **1** — `B2r_tile_read` under 4-way concurrent reads | do four readers contend on disk/page cache, inflating the read past its `workers=1` 4.21%? | **1.34–1.39%** of each worker's wall, *fully exposed* (prefetch OFF). Amdahl ceiling **1.014×** | **STOP** — Option M's IPC redesign stays parked |
| **2** — Cellpose `_from_device` under 4-way GPU contention | is doc 23 §7's 24.2% genuine D2H transfer, and does contention worsen it? | 24.2% **reproduces** (25.6% of a worker's wall) but is **96.3% GPU-wait**; real D2H copy is **0.7%** of it, **0.18%** of wall | **STOP** — GPU-wait branch (doc 42 §4 bullet 3) |
| **3** — pinned / `non_blocking` for the ~12.6 MB/tile transfer | does PCIe contention across 4 processes change the "it's negligible" prediction? | pinned is **14.1× faster** H2D in isolation, but worth **~0.4% of wall**; contention costs only **+4–8%** median | **STOP** — prediction confirmed, not assumed |

The one thing that *did* change under four-way contention is the thing doc 42 predicted would not be
a transfer problem: per-call GPU-wait inside `_from_device` went **38.7 ms → 101.8 ms** (2.63×) from
`workers=1` to `workers=4`. That is four processes' kernels queuing on one device — the failure mode
doc 21 §5 already measured CUDA MPS against and found flat end-to-end. §4 re-affirms rather than
reopens it.

## 1. Discover — what was actually run

All arms use the `match24` composition-matched crop (576 tiles, 55.9% background), the same anchor
doc 33 §1 and doc 39 used, with `--stream-precut` and the shipped
`cuda_alloc_conf=expandable_segments:True`. `config_hash` is **`d2ccc46b`** in every arm — identical
to round 9's, so the `workers=1` reference numbers this round compares against were produced by the
same configuration.

| arm | n | purpose |
|---|--:|---|
| `r14_m24_w1_r{1,2}` | 2 | `workers=1` control (playbook: never discard the dumb baseline) |
| `r14_m24_w4_r{1,2}`, `r14_m24_w4_pf_p{1,2}` | 4 | `workers=4`, prefetch ON (shipped default) |
| `r14_m24_w4_noprefetch_r{1,2}`, `r14_m24_w4_npf_p{1,2}` | 4 | `workers=4`, prefetch OFF — item 1's ablation arm |
| `r14_m24_w{1,4}_fromdev` | 2 | item 2's split probe |
| `transfer_probe.json` | 2 arms | item 3, 0 vs 3 sibling processes |

`workers=4` runs the crop in **104.7 s** against `workers=1`'s **188.3 s** — **1.80×**, consistent
with the 1.745×/2.138× band rounds 12–13 recorded, so this is the normal regime, not an outlier host
state.

### 1.1 A tool substitution, made explicit

Doc 42 §3 item 2 specified **Nsight Systems**, on the correct reasoning that `py-spy` cannot
separate `cudaMemcpyAsync` from `cudaStreamSynchronize` and had already failed to attach on this host
(doc 37 §2). **`nsys` is not installed here** (nor is `py-spy`), and installing a profiler was not in
scope. The same split was produced arithmetically instead, which for this specific question is not a
downgrade:

```python
torch.cuda.synchronize(X.device)   # drains what this worker's stream still owes  -> GPU-wait
host = X.detach().cpu()            # an already-idle device can only be moving bytes -> D2H copy
out  = host.to(torch.float32).numpy()   # host-side compute, not data movement at all -> cast
```

This lives in `scripts/perf_measure.py:wrap_from_device()`, gated behind
`HYBRID_PROBE_FROM_DEVICE=1` and **off by default** — the added synchronize serializes the very
GPU/CPU overlap the pipeline is built on, so no arm reporting wall-clock has it installed. Because
`spawn` children inherit the environment and re-import `perf_measure` through
`mp_worker_probe:install`, one env var reaches all four workers.

The synchronize **relocates** a wait that `.cpu()` would have paid anyway rather than adding one, and
the wall-clock confirms it: probed arms ran 189.5 s (w1) and 105.1 s (w4) against unprobed 186.8–189.8 s
and 103.3–107.2 s. The probe is free to within noise.

## 2. Item 1 — the read does not contend, and its ceiling is 1.4%

### 2.1 Per-read cost *falls* under four-way concurrency

| arm | tissue ms/read | bg ms/read | all ms/read |
|---|--:|--:|--:|
| r9 `workers=1` inline (doc 33 §1's 14.02/6.82 ms/**tile** = these ×2) | 7.010 | 3.409 | 4.997 |
| r9 `workers=1` prefetch | 8.852 | 4.075 | 6.182 |
| **r14 `workers=1`** (r1 / r2) | 9.276 / 9.377 | 4.588 / 4.683 | 6.655 / 6.753 |
| **r14 `workers=4`**, 4-worker aggregate (r1 / r2) | **6.864 / 7.065** | **4.488 / 4.720** | **5.536 / 5.754** |

Four concurrent readers are **~25% cheaper per read** than one, not more expensive. The
contention hypothesis doc 42 §1 raised ("four-way contention for the same page-cache room, or
IOPS-bound rather than throughput-bound access") is **not what happens on this host**.

The likely mechanism — offered as the explanation that fits, not as a separately verified claim — is
that it is not a disk effect at all but a GIL/CPU one. Each `workers=4` process spends much more of
its life blocked on the contended GPU (§3), which leaves its own single `tile-read` thread far more
uncontended CPU than the `workers=1` process's read thread gets while competing with a saturated
main thread and `detect_all_dots` arm. Note the r14 `workers=1` prefetch numbers are themselves
*worse* than r9's (9.28 vs 8.85 ms/read) — same `config_hash`, so this is host/kernel drift over
rounds 10–13, and exactly why the decision below rests on the within-round `w1`-vs-`w4` pair rather
than on a cross-round comparison.

### 2.2 The ablation — the only number that decides anything

Per the playbook's step 4, share-of-bucket is not evidence; end-to-end ablation is. Prefetch OFF
makes every read synchronous and **fully exposed** on the worker's main thread, so its cost is an
upper bound on what eliminating disk entirely (Option M) could ever buy.

| | wall (s) |
|---|--:|
| `workers=4`, prefetch ON — time-paired arms | 103.25, 104.67 → **103.96** |
| `workers=4`, prefetch OFF — time-paired arms | 105.66, 105.39 → **105.53** |
| delta | **+1.57 s (+1.51%)** |
| pooled over all 4+4 arms | ON 104.68 / OFF 104.99 → **+0.30%** |

The first two ON arms ran ~7 h before the OFF arms, so the pair was **re-run back-to-back** to keep
machine-state drift out of a 1%-scale delta; both framings are reported because they disagree by
more than either one's size, which is itself the finding — this is noise-floor territory.

The exposed read cost, read directly off the prefetch-OFF arms' own worker timings:

> **5.68 s and 5.86 s** of read work across 4 workers = **1.42 / 1.46 s per worker** against that
> worker's ~105 s wall = **1.34% / 1.39%**. Amdahl ceiling **1.0136× / 1.0141×**.

**Decision (doc 42 §4 bullet 2): STOP.** Disk contention does not materially worsen
`B2r_tile_read`'s per-worker share — it improves it. The `workers>1` ceiling for "zero disk scratch"
is **1.4% of wall**, below the `workers=1` figure of 4.21% (doc 37 §4.1) rather than above it, and
far below the quickref's ~10% threshold. Option M's cross-process IPC redesign (doc 30 §3 item 1)
stays parked — now **confirmed** for `workers>1` rather than assumed, which is precisely the gap
doc 42 §0 opened this round to close.

Note this also means the shipped prefetch is buying ~1.5% at `workers=4` on this crop, consistent
with doc 35's withheld "−0.83%… inside the noise floor." It is not doing harm, and doc 27 §6.4's
full-slide 17.2% is a different regime (RSS evicting page cache at 49 GB of scratch), so nothing here
argues for removing it.

## 3. Item 2 — the standout candidate is 96% GPU-wait

`cellpose.core._from_device`, split three ways:

| | `workers=1` | `workers=4` (4-worker aggregate) |
|---|--:|--:|
| calls | 1,016 | 1,016 |
| `_from_device` **TOTAL** | 42.171 s — 41.5 ms/call | 107.449 s — 105.8 ms/call |
| ├ **GPU-wait** (sync) | 39.302 s — **93.2%** | 103.437 s — **96.3%** |
| ├ **D2H copy** | 0.708 s — **1.7%** | 0.761 s — **0.7%** |
| └ host cast+numpy | 2.161 s — 5.1% | 3.250 s — 3.0% |
| bytes moved D2H | 7.195 GB → 10.16 GB/s | 7.195 GB → 9.45 GB/s |
| **D2H as share of a worker's wall** | 0.374% | **0.181%** |

Three things fall out.

**Doc 23 §7's headline number reproduces; its interpretation does not.** `_from_device` is
107.4 s / 4 workers = 26.9 s per worker against a 105.1 s wall = **25.6% of that worker's wall**, a
near-exact match for the 24.2% of MAIN-arm samples py-spy reported at `workers=1`. The percentage was
never wrong. But **96.3% of it is the worker blocking on its own Cellpose forward**, because `.cpu()`
on a CUDA tensor waits for the stream to drain before it moves a byte — and a sampling profiler
cannot tell those two apart. This is the quickref's red flag "the profiler's hot function turns out
to be a small slice of total time," caught by measurement rather than by intuition.

**The actual transfer barely notices the other three processes.** Per-call D2H went
0.697 → 0.749 ms (+7.5%) and effective bandwidth 10.16 → 9.45 GB/s (−7.0%) going from one process to
four. Doc 42 §2's worry that "the transfer could inflate, queued behind sibling processes' kernels on
shared copy engines" is measured and **does not happen** at this scale — the RTX 5090's copy engines
are nowhere near saturated by four workers moving ~7 MB at a time.

**What *does* inflate is GPU-wait: 38.7 → 101.8 ms per call, 2.63×.** That is the real, large,
`workers=4`-specific effect this round found — and it is not a data-movement cost. It is four
processes' kernels serializing on one device, and it is already priced in: the 1.80× end-to-end
speedup at `workers=4` is *net of* this queuing.

**Decision (doc 42 §4 bullet 3): STOP, do not design a transfer fix.** `_from_device`'s cost, and its
worsening under contention, are dominated by GPU-wait, not transfer bytes. Per doc 42 §3 item 4's
third branch this is re-affirmation territory, not new-fix territory: pinned staging buffers,
`non_blocking=True`, or a per-worker copy stream would all be optimizing the **0.7%** slice while the
**96.3%** slice is untouched — Amdahl ceiling **1.0018×** if D2H were made literally free. CUDA MPS
is the family of fix that addresses the 96.3%, and doc 21 §5 already measured it flat end-to-end.

## 4. Item 3 — pinned memory is 14× faster and still not worth it

`scripts/gpu_transfer_probe.py` (new this round), 1024×1024×3 float32 = 12.58 MB — one tile's UNet++
input. The measuring process is itself a `spawn`ed child, never the parent, because
`_run_tiles_multiprocess` keeps the parent off CUDA entirely and a parent-side microbenchmark would
be timing a context production never creates.

| transfer | isolation (median ms / GB/s) | 3 siblings (median ms / GB/s) | contention cost |
|---|--:|--:|--:|
| H2D pageable | 3.343 / 3.76 | 3.526 / 3.57 | +5.5% |
| **H2D pinned** | **0.237 / 53.16** | 0.256 / 49.13 | +8.2% |
| H2D pinned + `non_blocking` | 0.234 / 53.73 | 0.239 / 52.64 | +2.0% |
| D2H `.cpu()` (pageable) | 0.718 / 17.54 | 0.746 / 16.86 | +4.0% |
| D2H pinned | 0.323 / 38.94 | 0.341 / 36.95 | +5.4% |

Pinned memory is a genuine **14.1× win on H2D** in isolation, and it survives contention. It is still
not worth building, for exactly the reason doc 42 §2 gave in advance:

> 3.343 − 0.237 = **3.11 ms saved per tile**. At 144 tiles per worker that is **0.45 s** against a
> ~105 s wall = **0.42%**. Amdahl ceiling **1.004×**.

This is anti-pattern #5 in its purest form — a 14× microbenchmark win worth four-tenths of one
percent end-to-end — and the reason the plan required the number rather than the intuition.

Two secondary observations, neither decision-relevant:

- **Medians hold up under contention but tails do not**: D2H `.cpu()` p95 goes 0.735 → 1.420 ms
  (+93%) while its median moves 4%. Four-way contention adds jitter, not sustained bandwidth loss.
- `.cpu()` (0.718 ms) is *faster* than `copy_` into a pre-allocated pageable tensor (1.099 ms),
  so the pipeline is already on the better of the two pageable paths. The in-pipeline Cellpose D2H
  (§3, 0.697–0.749 ms/call at ~7 MB average) and this microbenchmark's `.cpu()` row agree to within
  the size difference — two independent measurements of the same thing landing together.

## 5. An incidental finding: `workers=4` VRAM is as tight as doc 42 §1 said

One of the eight `workers=4` ablation arms (`r14_m24_w4_pf_p2`, first attempt) died of
`torch.OutOfMemoryError` trying to allocate **48 MiB**, with `expandable_segments:True` active:

```
GPU 0 ... 31.36 GiB total, of which 8.44 MiB is free.
Process 1369473 has 9.30 GiB   Process 1369471 has 11.55 GiB   Process 1369474 has 9.30 GiB
```

Three siblings held 30.15 GiB between them; one had ballooned to 11.55 GiB against its peers' 9.30 GiB.
The re-run succeeded, so the rate is **1 failure in 10 `workers=4` crop-run attempts** this round
(8 ablation arms + the split-probe arm + the one retry). This is not something doc 42 asked
for and it is not a transfer question — but it is direct evidence for doc 42 §1's "~2.5 GB of headroom
across all four workers combined, not per worker," and it is a **production fail-fast risk at the
shipped default**, on a crop 48× smaller than a full slide. Recorded here, not fixed here; it belongs
in the backlog as a `workers=4` stability item, not in a data-transfer round.

## 6. What this round changed

**No pipeline code.** Two measurement-only additions, both outside `backend/`:

1. `scripts/perf_measure.py` — `wrap_from_device()` plus its call in `install_wrappers()`, gated on
   `HYBRID_PROBE_FROM_DEVICE=1`, default off. Adds buckets `X_fromdev_gpuwait`,
   `X_fromdev_copy_d2h` (with byte counts), `X_fromdev_cast_numpy`, `X_fromdev_TOTAL`.
2. `scripts/gpu_transfer_probe.py` — new, the item 3 harness.

The correctness veto in doc 42 §4 is not engaged: no worker's transfer pattern was altered, so there
is no output to diff. The probe's own replacement of `_from_device` performs the identical operations
in the identical order (`.detach().cpu()`, `.to(torch.float32)`, `.numpy()`), and never runs in a
shipped configuration.

## 7. Where this leaves the backlog

**Closed by measurement this round, for `workers>1` specifically:**

- "Zero disk scratch" / Option M cross-process IPC — ceiling **1.4%**, and reads get *cheaper*, not
  dearer, under four-way concurrency. This was the last of doc 42 §0's rows still genuinely open on
  the disk side.
- Cellpose `_from_device` as a transfer target — **96.3% GPU-wait**. Doc 23 §7's lead, carried across
  four rounds as "the standout candidate," is now closed with a direct measurement in the right
  regime.
- Pinned memory / `non_blocking` for this project's own transfers — **0.42%**, confirmed under real
  contention rather than predicted.

**Re-affirmed, not reopened** (doc 42 §3 item 5, all three now with one more piece of supporting
evidence): CUDA-stream intra-tile bubble redesign, depth-2 pipelining inside a worker, CUDA MPS. The
2.63× inflation of per-call GPU-wait at `workers=4` is the MPS-shaped problem, and MPS was already
measured flat end-to-end (doc 21 §5).

**The honest summary of round 14**: every named direction in doc 42 — pinned memory, shared memory,
zero disk scratch, zero copy, asynchronous transfer, CUDA streams — is now measured rather than
assumed at `workers=4`, and none of them is where the time is. The time at `workers=4` is in GPU
kernel execution and the queuing of four processes' kernels against one device. **There is no
GPU↔CPU data-transfer bottleneck in this pipeline at `workers>1`.** Round 15 should not look for one
again without new evidence; the next real lever is elsewhere, and doc 42 §0's discipline — tag every
number with the process regime it was measured in — is what made that answerable in one round instead
of five.

## 8. Reproduce

```bash
# §2 — item 1: the w1 control, the w4 arms, and the prefetch ablation pair.
# (run_item1.sh / run_item3.sh in the round-14 scratch drove these; the shape is:)
.venv/bin/python scripts/perf_measure.py \
  --ihc  test_picture/_roi_crops/match24_ihc.tiff \
  --dish test_picture/_roi_crops/match24_dish.tiff \
  --output /home/taro/r14/r14_m24_w4_pf_p1 --label r14_m24_w4_pf_p1 \
  --mp-workers 4 --stream-precut --worker-timings \
  --metrics-dir /home/taro/r14/_metrics
# ... same with --mp-workers 1 (control) and with --no-prefetch (ablation arm)

# §3 — item 2: the _from_device split. HYBRID_PROBE_FROM_DEVICE is what installs it,
# and it must never be set in an arm whose wall-clock you intend to quote.
HYBRID_PROBE_FROM_DEVICE=1 .venv/bin/python scripts/perf_measure.py \
  --ihc  test_picture/_roi_crops/match24_ihc.tiff \
  --dish test_picture/_roi_crops/match24_dish.tiff \
  --output /home/taro/r14/r14_m24_w4_fromdev --label r14_m24_w4_fromdev \
  --mp-workers 4 --stream-precut --worker-timings \
  --metrics-dir /home/taro/r14/_metrics

# §4 — item 3: pinned vs pageable, isolation vs 3 concurrent sibling processes.
.venv/bin/python scripts/gpu_transfer_probe.py --siblings 0,3 --iters 200 \
    --out /home/taro/r14/_metrics/transfer_probe.json
```
