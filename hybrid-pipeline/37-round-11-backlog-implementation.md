# 37 — Round 11: closing the round-10 backlog — implementation and results

> Implements [`36-round-11-backlog-plan.md`](./36-round-11-backlog-plan.md), which sequenced the four
> open items [`35-round-10-backlog-implementation.md`](./35-round-10-backlog-implementation.md) §8 left
> actionable. Follows [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose. Round 11.

## 0. Outcome in one paragraph

**Step 1 did not reproduce the regression it was sent to attribute, and found what caused it
instead.** On the composition-matched anchor, `B1_m3b_cellpose` moves **+1.24%** across the whole
commit range doc 36 suspected — not +11.8% — and with Option L's prefetch ablated on both sides the
range is **flat** (`B1_unet_coremask` +0.02%). What *does* move the bucket is Option L itself:
turning the prefetch on inflates `B1_unet_coremask` **+16.4%** and `B2r_tile_read` **+26.4%** *while
wall-clock falls 1.21%*. Round 10's +751 s is that inflation at full-slide scale, i.e. an accounting
artefact of concurrency, not new GPU work — a prediction step 4 was scheduled to test, and which it
**half confirmed**: the contention coefficient transfers from 576 tiles to 27,565 within 4%, but it
accounts for 221 s of the 751 s, not most of it.
**Step 2 found the hang, with a stack trace rather than a hypothesis**: the parent blocks forever in
`multiprocessing`'s atexit handler joining `task_q`'s feeder thread, which is itself blocked writing
to a pipe whose readers have just been terminated. Fixed, plus a second defect the probe exposed —
`PrecutStream` kept cutting the *entire* slide after the batch was abandoned. `workers=4` fail-fast
exit latency: **never → 0.32 s.** The `workers=1` hang doc 35 also reported **did not reproduce**, at
crop or full-slide scale. **Step 3** settled the allocator balloon: the control arm failed **4 times
in 12** with the byte-identical 24.76 GiB signature, `expandable_segments:True` **0 in 12**, for
+0.67% wall. **Step 4's clean full-slide arm settles Option K at 1.044×, not 1.208× — and explains
the gap**: with read placement identical to the doc 27 baseline, the read bucket is **449.3 s against
that baseline's 2,368.5 s, −81%**. The read was never expensive; the 275 GB of concurrent per-tile
writes that ran alongside it were. Option L now hides 448.4 s of a 449.3 s cost — **99.8% of a
ceiling that is 4.21% of wall, not the 17.2% doc 33 sized it against.**

Two things happened mid-round that the record has to carry: `main` was fast-forwarded from
`6b0286c` to `2b3ad68` by a `git pull` at 20:56 (§6.1), and the first two measurement sweeps had to
be thrown away for a design fault of this round's own making (§1.1).

## 1. Step 1 — attributing `B1_m3b_cellpose`'s +751 s

### 1.1 Two sweeps discarded first, and why that matters

Doc 36 §2.1 asked for the `match24` anchor run once on `1c0b31e` (the commit before the M0 split)
and once on `HEAD`, at `workers=1`. The obvious design — alternate the arms, A,B,A,B — produced this:

| | run 1 | run 2 | run 3 |
|---|--:|--:|--:|
| PRE `B1_m3b_cellpose` | 134.03 | 157.58 | 158.81 |
| HEAD `B1_m3b_cellpose` | 141.37 | 169.60 | 174.04 |

Every run is slower than the one before it, in **both** arms: the machine warms up over the sweep and
does not settle until about run 5. In strict A,B,A,B the second arm always occupies the *later* slot
of each pair, so that drift is handed entirely to one arm. The +11.53 s mean difference this sweep
reported is indistinguishable from the ~+11.8 s per-slot drift the same data shows. **It measured the
sweep order, not the code.** This is playbook anti-pattern #7 (changing more than one thing at once)
arriving through the back door — the second thing that changed was *when* the run happened.

A second sweep (`--no-prefetch` vs PRE) then stepped down a whole level partway through — walls
221/214 then 191/190/190/188 — so its arms were not comparable to each other either.

The sweep that is reported below is the third: **three arms in one 3×3 Latin square, repeated twice,
so every arm occupies every slot position equally often**, started from an already-warm machine. Its
within-arm standard deviations are **0.3–0.9 s on a ~189 s wall (0.2–0.5%)**, and the residual
position effect is 189.68 / 189.29 / 188.38 s across slots 1/2/3 — small, and balanced across arms by
construction. Doc 35 §2.3 concluded this anchor "could not resolve a 0.83% effect" from six
`workers=4` pairs; that is a statement about a *thermally unsettled, order-confounded* sweep, not
about the anchor. Settled and counterbalanced, it resolves 1.2% at `d/SE` = 7.

### 1.2 The three arms

| arm | code | reads |
|---|---|---|
| **PRE** | `1c0b31e` — before `d6592c3` (no M0 split, no Option H, no Option L) | inline, on MAIN |
| **HEAD** | `2b3ad68` — today, as shipped | one tile ahead, on the `tile-read` arm |
| **HEAD_NP** | `2b3ad68` with `--no-prefetch` (Option L ablated) | inline, on MAIN |

`HEAD_NP` is what makes the comparison honest. PRE has no Option L *at all*, so PRE→HEAD changes two
things at once — the refactor and where reads happen. PRE→HEAD_NP changes only the first.

n = 6 per arm; **`stats` identical (224 success / 352 skipped) in all 18 runs**, matching round 10's
twelve runs and round 9's exactly. Correctness veto passed.

| | wall | `B1_m3b_cellpose` | `B1_unet_coremask` | `B2r_tile_read` |
|---|--:|--:|--:|--:|
| PRE | 188.63 ± 0.42 | 135.93 ± 0.32 | 16.65 ± 0.05 | 5.42 ± 0.05 |
| HEAD_NP | 190.52 ± 0.87 | 137.48 ± 0.37 | 16.66 ± 0.05 | 5.58 ± 0.06 |
| HEAD | 188.22 ± 0.81 | 137.61 ± 0.43 | 19.39 ± 0.10 | 7.05 ± 0.05 |

Each cell is the arm-to-arm change, with the effect-to-standard-error ratio (`d/SE`, where about 2 is
needed to call an effect real) in brackets:

| comparison | wall | `B1_m3b_cellpose` | `B1_unet_coremask` | `B2r_tile_read` |
|---|--:|--:|--:|--:|
| **PRE → HEAD** (doc 36's question) | −0.22% (1.0) | **+1.24%** (6.97) | +16.44% (55.7) | +30.03% (49.6) |
| **PRE → HEAD_NP** (refactor alone) | +1.00% (4.4) | **+1.14%** (7.1) | **+0.02%** (0.08) | +2.85% (4.6) |
| **HEAD_NP → HEAD** (Option L alone) | **−1.21%** (4.3) | +0.10% (0.5) | **+16.42%** (54.3) | **+26.43%** (42.4) |

### 1.3 What this settles

**The +11.8% does not reproduce at crop scale.** The bucket moves **+1.24%** across the entire
commit range — 9.5× smaller than the number doc 36 sent this step to explain.

**`d6592c3` is exonerated by measurement, not by argument.** With read placement held fixed
(PRE → HEAD_NP), `B1_unet_coremask` moves **+0.02%** — dead flat — and `B1_m3b_cellpose` +1.14%. A
refactor that moved 1,065 lines of per-tile GPU orchestration into a new module did not move the cost
of the GPU forwards. Doc 36 §1 named it "the more plausible remaining candidate" after ruling out
`e806938` by diff inspection; both suspects are now excluded, one by reading and one by measuring.

**What does move the bucket is Option L, and it moves it without doing any more work.** Turning the
prefetch on (HEAD_NP → HEAD) raises `B1_unet_coremask` by 16.4% and `B2r_tile_read` by 26.4% while
**wall-clock falls 1.21%**. Buckets are wall-clock timers around calls on the main thread; a
concurrent reader thread contends with that thread, so every main-thread bucket reads longer even
though the device is doing exactly what it did before. Doc 35 §2.3 identified this signature at
`workers=4` ("`B2r_tile_read` rises +14.9% … while wall does not rise") and correctly called it
evidence the mechanism works. What it did not do is subtract the same effect from the *other* buckets
on the same run.

The accounting closes to within 0.4 s on the crop:

```
read leaving MAIN        −5.58 s
B1 inflation             +2.86 s   (154.14 → 157.00 s, m3b + unet)
                         ───────
predicted wall           −2.72 s     measured −2.30 s
```

i.e. **0.513 s of B1 inflation per second of read moved off the main thread.**

### 1.4 The number this predicts for round 10 — and for step 4

Doc 27's baseline ran with reads inline (2,368.5 s of `B2r_tile_read` on MAIN); round 10's run had
Option L on. Applying the crop's measured ratio:

| | |
|---|--:|
| read moved off MAIN at full slide (baseline inline total) | 2,368.5 s |
| × measured inflation ratio 0.513 | **≈ 1,214 s predicted B1 inflation** |
| round 10 observed: `B1_m3b_cellpose` | **+751.0 s** |
| round 10 observed: `B1_unet_coremask` + `BM1_*` + misc | +144 s |
| observed ÷ predicted | **0.62 – 0.74** |

Same order, with the observed a little under the crop-derived prediction — which is the direction to
expect, because the crop's reads are page-cache hits (a thread spinning in `skimage.io.imread` holds
the GIL more of the time) while the full slide's are real disk I/O at 61 GB RSS, and a thread blocked
in `read()` contends less per second than one decoding.

**So round 10's "largest single unexplained number" and its headline Option L win are the same
phenomenon with opposite signs.** Doc 35 §4.1 booked −2,368.5 s to Option L *and* +751.0 s as an
unexplained regression; if this attribution holds, Option L's net MAIN contribution is
**≈ −1,617 s, not −2,368.5 s**, and there is no unexplained regression to chase.

**This is a falsifiable prediction, and step 4 tests it directly.** Step 4 runs the full slide with
`--no-prefetch` on today's code, so:

- if the attribution is right, `B1_m3b_cellpose` falls back toward the baseline's **~6,359 s**;
- if `B1_m3b_cellpose` stays near round 10's **7,109.9 s** with the prefetch off, the attribution is
  wrong and the regression is real and still unexplained.

Recorded here **before** step 4 ran, so the prediction cannot be fitted after the fact.

## 2. Step 2 — the post-fail-fast hang

### 2.1 Measuring a defect that every benchmark is blind to

Every number a run reports is produced *before* the process refuses to exit, so a job that hangs for
an hour afterwards looks identical to one that exits cleanly. `scripts/exit_latency_probe.py` (new)
measures the only thing that distinguishes them, from outside:

```
exit latency = (child process exits) − (child's last line of output)
```

The work runs in a child process; the parent timestamps every line, and when the child overruns its
deadline the parent sends `SIGUSR1` to a `faulthandler` registered in the child, so **every thread's
stack lands in the same log**. (`py-spy` cannot attach on this host — `ptrace_scope=1`, no root — so
a signal the child registers for itself is the only way in.) Fail-fast is triggered through the
existing `HYBRID_MP_WORKER_PROBE` hook, so the abort is the pipeline's own path and no pipeline code
learns anything about the probe.

### 2.2 Both shapes, measured before any fix

| shape | workers | exit after work finished |
|---|--:|--:|
| clean completion | 1 | 0.69 s |
| clean completion | 4 | 0.29 s |
| fail-fast abort | 1 | 0.77 s |
| **fail-fast abort** | **4** | **never** — SIGKILLed at 65 s by the probe; an earlier unbounded attempt in this session sat for 14 minutes before being killed by hand |

Doc 36 §2.2 was right to treat the two reported shapes as two candidate mechanisms rather than one
bug. It turns out only one of them is a bug at all: **the `workers=1` post-completion hang does not
reproduce** — that shape exits in 0.69 s. It is recorded as not reproduced (§2.5), not as fixed.

### 2.3 The `workers=4` hang, from the stack dump

```
Current thread (main):
  File "multiprocessing/util.py", line 363 in _exit_function
  File "multiprocessing/util.py", line 303 in _run_finalizers
  File "multiprocessing/util.py", line 227 in __call__
  File "multiprocessing/queues.py", line 199 in _finalize_join
  File "threading.py", line 1119 in join
  File "threading.py", line 1139 in _wait_for_tstate_lock

Thread (task_q's feeder):
  File "multiprocessing/queues.py", line 250 in _feed
  File "multiprocessing/connection.py", line 200 in send_bytes
  File "multiprocessing/connection.py", line 384 in _send
```

This is exactly the mechanism doc 35 §6.2 named from the signature alone, now with the two blocked
frames in hand. `_kill_all()` terminates the workers; nobody is left reading `task_q`; its pipe fills;
its feeder thread blocks forever inside `write()`; and `multiprocessing`'s atexit handler joins that
thread. **A correct fail-fast becomes a process that never returns.**

The fix is one line, and the standard-library primitive is named for exactly this situation —
`Queue.cancel_join_thread()`: *"allow the background thread to exit immediately when the process
exits, without flushing"*. The batch has been abandoned, so the un-sent work should not be delivered
anywhere; discarding it is correct, not a workaround.

```python
stop_feeding.set()
task_q.cancel_join_thread()   # <- new
_kill_all()
raise
```

### 2.4 A second defect the probe exposed: the precut never stopped

`_run_tiles_multiprocess._feed` carries this comment: *"中止後必須真的停下來 … 若 fail-fast 之後它還在拉
PrecutStream，就會在整批已放棄的情況下繼續把整張玻片切到磁碟."* It does not achieve that.
`PrecutStream.__iter__` submitted **every** position to its thread pool before yielding the first
tile, so the consumer's `break` stopped the *pulling* but not one single unit of the *cutting*.
Measured on the hung run: the batch aborted at tile ~8 of 576 and the scratch directory still held
**576 / 576 cut tiles, 842 MB**.

At full-slide scale that is 27,565 tile pairs — tens of GB and minutes of I/O written after the batch
is abandoned, and, with the queue hang fixed, it would still hold the exit open far longer than doc
36's "within seconds of its last log line" bar.

`__iter__` now keeps a bounded window (`workers × 8`) of cuts in flight and cancels what has not
started when the generator is closed. A drained stream yields the same tiles in the same
completion-order semantics; only an *abandoned* one behaves differently. The window cannot become a
bottleneck: cutting one pair costs tens of milliseconds against hundreds per tile of analysis, so the
cutter still runs an order of magnitude ahead — it just no longer runs ahead by an entire slide.

`tests/test_precut_stream_bounded.py` (3 tests) pins the three properties, and it is a real guard,
not a decoration: against the pre-fix code its first test fails (`abandoning the stream took 2.51s`,
having cut all 500 positions), and passes after.

### 2.5 Result

| shape | workers | before | after |
|---|--:|--:|--:|
| clean completion | 1 | 0.69 s | 0.69 s |
| clean completion | 4 | 0.29 s | 0.28 s |
| fail-fast abort | 1 | 0.77 s | 0.67 s |
| **fail-fast abort** | **4** | **never** | **0.32 s** |

Cost of the bounded precut window at `workers=4`, on the same anchor: the first run of step 3's sweep
came back at **104.4 s** against round 10's six-run prefetch mean of **104.43 s**, with identical
`stats` and peak RSS 11.79 GB against round 10's 11.70–11.81 GB. Free, as designed.

**The `workers=1` two-hour hang doc 35 §8 also reported is not fixed, because it did not reproduce.**
Both `workers=1` shapes exit in under 0.8 s here. Doc 35's own framing — "an audit of what non-daemon
threads outlive `main()`" — was checked: on that path `joblib` runs with `prefer='threads'` and
`dot_detect_n_jobs=1` (no `loky` process pool), and every executor in `run_batch` is inside a `with`.
That leaves scale-dependent teardown (a 61 GB-RSS process on a 62 GB machine with 27 GB of swap) as
the remaining hypothesis, which no crop-scale anchor can test. **Step 4 is a full-slide `workers=1`
run and therefore the instrument for it** — its exit behaviour is reported in §4.

## 3. Step 3 — `expandable_segments:True` at `workers=4`

Doc 36 §2.3 asked for an existing tool pointed at an untested configuration, with enough repeats to
have a shot at an intermittent failure. `scripts/alloc_conf_probe.py`, unchanged, 12 interleaved
repeats per condition on the `match24` anchor at `workers=4` — 24 runs, ~45 minutes.

**The defect reproduced far more often than round 10 saw it**, which is the single most useful thing
about this sweep: a knob cannot be shown to fix something that never happens.

| | control (`""`) | `expandable_segments:True` |
|---|--:|--:|
| runs | 12 | 12 |
| **OOM failures** | **4 (33%)** | **0** |
| median wall (successful runs) | 104.59 s | 105.23 s |
| mean wall (successful runs) | 104.31 ± 0.70 (n=8) | 105.01 ± 0.91 (n=12) |
| median peak framebuffer | 15,303 MB | 20,621 MB |
| **max peak framebuffer** | 18,193 MB | **30,061 MB** (92.2% of the card) |
| mean peak RSS | 11.90 GB | 11.88 GB |

Three of the four control failures carry the **byte-identical 24.76 GiB** balloon doc 19 #7b, doc 23
§4.6 and doc 35 §6.2 all recorded (the fourth ballooned to 20.49 GiB). This is the same fingerprint,
at the shipped production worker count, on today's code.

**Result: the knob eliminates the failure in this sweep.** 4/12 → 0/12 is significant on its own terms
(Fisher one-sided **p = 0.047**), and doc 35's own 1-in-12 observation folded in only strengthens the
control-arm rate.

**It is not free, and the cost is not the one to worry about.** Wall rises 0.70 s (+0.67%,
`d/SE` = 1.83 — at the edge of this sweep's resolution, and in any case a rounding error against
losing a 3-hour batch). Peak RSS is unchanged. What does move is **peak framebuffer: the maximum
observed rises from 18.2 GB to 30.1 GB, i.e. 92.2% of the 32,607 MB card.** Expandable segments do
not use less memory — they stop the allocator fragmenting into a state where a 176 MB request cannot
be served, and they will happily grow into whatever headroom exists. Doc 27 §6 already set 32 GB as a
hard floor for `workers=4` at 93.3% occupancy; **this knob does not create headroom, it removes a
failure mode**, and those two things must not be confused when sizing hardware.

**Recommendation: turn it on for `workers>1`** — `config.cuda_alloc_conf = "expandable_segments:True"`.
It is not changed in `config_example.py` by this round: the sweep is one anchor on one card, the
default has shipped as `""` since doc 27 built the knob, and flipping a default that governs
production VRAM behaviour on 12 crop runs is a decision for whoever owns the deployment, not a
side-effect of a measurement round. The measurement it was waiting for now exists.

### 3.1 An unplanned confirmation of step 2

Control run 8 OOM'd, fail-fasted correctly, and **returned** — the sweep moved on to run 9 after
26.4 s of driver wall. That is precisely the situation that cost round 10 a 49-minute manual kill
(doc 35 §6.2: *"the OOM hit the `--no-prefetch` control arm"*), reproduced here four times, on a real
unplanned allocator failure rather than an injected one, with step 2's fix in place. Without it this
sweep would have stalled on its third run and never finished unattended.

## 4. Step 4 — the clean full-slide `--no-prefetch` arm

One `workers=1` pass over the whole 16.2 GP slide with `--no-prefetch`, no `--resume`, on today's
code — the run doc 35 §4.3 said was *"the only thing that would turn 1.208× from an upper bound into
a measurement."* **2 h 37 m of analysis + 20 m 39 s of stitch, 2 h 57 m 46 s end to end.**

`prefetch_ablated: true` in the metrics, so the control arm took effect.

**Correctness veto passed on every axis.** `stats` **10,800 / 16,765 — identical to round 10's**;
356,226 cell rows; peak RSS 60.17 GB against round 10's 60.04; 70 `gc.freeze()` calls, matching
round 10 exactly. `scripts/overlay_pyramid_audit.py` on the produced 7.50 GB overlay: **PASS at every
level**, Predictor=1 across all 12 IFDs, per-level means 240.71 → 241.14 and zero-fraction 0.0% —
the same values round 10's regenerated slides decode to.

| | doc 27 baseline | round 10 | **round 11** |
|---|--:|--:|--:|
| prefetch | off | **on** | **off** |
| per-tile intermediate writes | **275 GB** | none | none |
| **end-to-end wall** | 13,762.5 s | **10,217.7 s** | **10,666.1 s** |
| `B1_m3b_cellpose` (n=24,360) | 6,358.9 | 7,109.9 | **6,888.7** |
| `B1_unet_coremask` (n=27,565) | 734.8 | — | 738.7 |
| **`B2r_tile_read`** (n=55,130) | **2,368.5** | 1,581.2 | **449.3** |
| — tissue / background | — | 1,037.8 / 543.4 | 274.3 / 175.0 |
| `B4_gc_collect` | 2,218.4 | 58.8 | 19.5 |
| Phase D stitch | 1,185.4 | 1,182.4 | 1,239.2 |
| peak RSS | 61.13 GB | 60.04 GB | 60.17 GB |
| `gc.freeze()` count | 1 | 70 | 70 |
| `stats` | 10,801 / 16,764 | 10,800 / 16,765 | 10,800 / 16,765 |

### 4.1 Option K, settled — and the answer is a quarter of the claim

```
wall, prefetch off (this run)   10,666.1 s
wall, prefetch on  (round 10)   10,217.7 s
                                ──────────
Option L's contribution            448.4 s  =  1.044x
```

**Doc 33's Option K is 1.044×, not the 1.208× doc 35 §4.3 recorded as an upper bound.** Doc 35 was
right to refuse to bank that number, and right about which confound was doing the work.

**Why the upper bound was so far out is now measurable, and it is the more interesting result.** With
read placement identical to the doc 27 baseline — inline, on the main thread — the read bucket is
**449.3 s, down from 2,368.5 s: −81.0%.** Same code path, same slide, same 55,130 reads. The only
thing that changed between those two runs is that one of them was also writing 275 GB of per-tile
intermediates. **Doc 27 §6.4 diagnosed the read cost as page-cache eviction under RSS and write
pressure rather than as intrinsic I/O; that diagnosis is now confirmed by direct measurement**, and
the confounded number was not "slightly" inflated — it was **5.3× the real cost**.

The consequence for Option L is that its Amdahl ceiling collapsed under it before it ever shipped:

| | share of wall the read can hide | measured saving | recovered |
|---|--:|--:|--:|
| doc 33's premise (baseline read ÷ baseline wall) | 17.2% | — | — |
| **today (inline read ÷ this run's wall)** | **4.21%** | **4.20%** | **99.8%** |

**Option L is close to a perfect optimisation of a problem that had already shrunk by 5×.** It hides
448.4 s of a 449.3 s read cost. There is nothing left to win here — and equally, doc 33's "17.2% of
wall" framing, and everything downstream of it (including doc 35 §5's `depth=2/3` arithmetic), was
sized against a number that belonged to a superseded write pattern.

**Caveat, stated plainly**: this is one run against one run, three days apart, and the prefetch-on
arm is round 10's (its metrics JSON is no longer on disk, so only the figures doc 35 published can be
used — §6.3). The two arms also differ by this round's own bounded-precut change, which was verified
not to throttle the run: the cut-ahead gap sat pinned at its 64-tile ceiling for the whole sweep it
was sampled in, meaning the cutter was always full and the analysis was always the limiter.

### 4.2 The pre-registered prediction: mechanism confirmed, magnitude wrong

§1.4 committed, before this run started, to a falsifiable claim: with the prefetch off,
`B1_m3b_cellpose` should fall back toward the baseline's ~6,359 s.

**It fell to 6,888.7 s — recovering 221.2 s of the 751.0 s, i.e. 29.5%, not "most".** The prediction
as written is **not confirmed.**

What *is* confirmed, and confirmed tightly, is the mechanism and its coefficient. §1.3 measured on the
crop that moving read work onto a concurrent thread inflates the main-thread B1 buckets by **0.513 s
per second of read moved**. Applied to the read this run actually moves:

| | |
|---|--:|
| read moved off MAIN (this run's inline total) | 449.3 s |
| × crop-measured ratio 0.513 | **230.5 s predicted** |
| measured `B1_m3b_cellpose` inflation (6,888.7 → 7,109.9) | **221.2 s** |
| implied ratio at full slide | 0.492 (**within 4.1% of the crop's 0.513**) |

A contention coefficient measured on 576 tiles reproduces on 27,565 tiles to within 4%. **§1.4's
error was not the mechanism but the input**: it multiplied the ratio by 2,368.5 s, the baseline's
read total — a number §4.1 has just shown was 5.3× inflated by a write pattern that no longer exists.
Feed the ratio the real read cost and it lands.

**So round 10's +751 s decomposes as: 221 s of prefetch contention (measured, both here and on the
crop) + 530 s that remains unattributed.** That residual is +8.33% on the bucket against the doc 27
baseline, and it is *not* the M0 split (§1.3 measured that range flat at +1.14% on this bucket, ≈ 72 s
at full-slide scale) and *not* Option L (measured here). What is left is n=1-vs-n=1 full-slide
variance across runs on different days — which this round can, unusually, put a number on rather than
wave at: §1.1 measured this same GPU drifting **+17% on this very bucket** across a single warm-up
sweep. An 8.3% difference between two single runs on different days sits inside that band.
**Resolving it further requires repeated full-slide runs (~3 h each), and nothing currently depends on
the answer** — the bucket is on the critical arm, but no proposed change targets it.

### 4.3 What else this run settles

- **The `workers=1` post-completion hang did not reproduce at full-slide scale either.** This is the
  exact shape doc 35 saw sit for two hours: `workers=1`, 27,565 tiles, 60.17 GB peak RSS, final JSON
  printed. Last log line 02:35:41; metrics written 02:35:42; process gone. **Under 2 seconds.**
  Combined with §2.2's crop-scale result, that shape is now recorded as **not reproduced on the
  instrument doc 35 observed it on**, and the scale-dependent-teardown hypothesis §2.5 raised is not
  supported.
- **Doc 31's Option H holds at full scale for the second consecutive round**: `B4_gc_collect` 19.5 s
  (down even from round 10's 58.8 s), 70 freezes, peak RSS flat at 60.17 GB.
- **Phase D is now 11.6% of wall** (1,239.2 s of 10,666.1 s) and has grown its share again purely
  because everything around it got faster — the same effect doc 35 §8 item 5 flagged.
- **The read is no longer a bottleneck by any definition**: 449.3 s inline is 4.21% of wall, below the
  playbook's ~10% floor. The tissue : background per-read ratio is now 1.97 (274.3 s / 24,360 vs
  175.0 s / 30,770), against doc 35's 2.41 and doc 33's crop-measured 2.06.

## 5. Step 5 — no action, as planned

Doc 36 §2.5 scheduled nothing here and this round re-opened nothing: Phase D Phase 2 stays closed on
doc 35 §3.3's measured ceiling, `depth=2/3` stays closed on its 1.04%-of-wall ceiling, and the QuPath
pass on the Phase D candidates stays dropped from active tracking. Re-litigating a closed item
without new information is the anti-pattern this series exists to avoid.

## 6. Things that happened to this round

### 6.1 `main` moved mid-round

At **20:56**, between this round's second and third measurement sweeps, `git pull` fast-forwarded
`main` from `6b0286c` to `2b3ad68` (three commits: a numpy-2 constraint bump, Windows portability,
and an analysis UI with ROI support). The round noticed because a file it had just read came back
different.

What this does and does not touch:

- The **per-tile analysis hot path is unchanged.** The pull's hybrid-side edits are ROI plumbing
  (`PrecutStream(region=...)`, an `origin` argument threaded into `compute_tile_geometry` /
  `_validate_axis`) and a Windows guard around `import resource`. M1/M2/M3b, the reads, the gc
  cadence and the multiprocess loop are byte-identical.
- The **installed environment is unchanged**: `pyproject.toml` and `uv.lock` moved, but the venv
  still resolves `numpy 1.26.4`. No dependency was reinstalled during this round.
- **Sweep 2 straddles the pull** and is discarded for that reason as well as the one in §1.1.
- **The reported sweep (§1.2) ran entirely after it**, so all three of its arms saw one repo state.
  "HEAD" in this document means `2b3ad68`.

### 6.2 A pre-existing test-collection failure, not caused here

`backend/tests/{test_chunked_upload,test_module1_strips,test_resume}.py` fail to *collect* —
`ModuleNotFoundError: No module named 'tuspyserver'`, a dependency the pull's new
`backend/api/tus_compat.py` introduces and the venv does not have. Confirmed pre-existing by stashing
this round's changes and re-running. Not fixed here (installing dependencies mid-measurement-round
would change the environment under the runs); recorded so the next round does not read it as fallout
from this one.

The hybrid suite itself: **79 passed** (`tests/`), plus the two `backend/tests` modules that do
collect.

### 6.3 Round 10's full-slide metrics are gone

`_metrics/*_timings.json` survives on this machine for doc 27's two full-slide runs
(`/home/taro/full_wsi_validation/_metrics/`) but **not for round 10's** — `/home/taro/r10_fullwsi_w1/`
keeps the outputs and the 45 GB precut scratch, and no metrics directory. Step 4's comparison
therefore uses the figures doc 35 §4 published rather than the raw buckets, which is why §4 compares
the numbers doc 35 quoted (wall, `B1_m3b_cellpose`, `B2r_tile_read`, `B4_gc_collect`, RSS, `stats`)
and does **not** attempt to reproduce doc 35 §4.1's MAIN/BG arm decomposition — that would require
buckets nobody can read back. Recorded as follow-up 5.

This round's own artefacts are kept: `/home/taro/r11_step1{,b,c,d}/_metrics/` (36 crop runs),
`/home/taro/r11_step2/*.json` (exit-latency, pre and post fix, including the stack dump),
`/home/taro/r11_step3_alloc/alloc_conf_w4.json` (24 runs), and
`/home/taro/r11_fullwsi_w1_np/_metrics/`.

## 7. Disposition

| doc 36 item | status |
|---|---|
| Step 1 — `B1_m3b_cellpose` +751 s | **MEASURED — regression does not reproduce at crop scale (+1.24%, not +11.8%).** `d6592c3` exonerated by measurement (`B1_unet_coremask` +0.02% with read placement held fixed); `e806938` was already exonerated by diff. 221 s of the 751 s attributed to Option L contention, confirmed at both scales with a coefficient that transfers within 4%. **530 s remains unattributed** and is inside this machine's measured run-to-run drift band (§4.2) |
| Step 2 — post-fail-fast hang | **FIXED**, with a stack trace, not a hypothesis. `workers=4` fail-fast exit latency **never → 0.32 s** (`task_q.cancel_join_thread()`). A second defect found and fixed: `PrecutStream` cut the entire slide after the batch was abandoned (576/576 measured). 3 tests, verified to fail pre-fix. **The `workers=1` hang did NOT reproduce** at crop *or* full-slide scale and is not fixed |
| Step 3 — `expandable_segments` at `workers=4` | **SWEPT and SHIPPED (§9).** Control **4 OOM / 12** (three byte-identical 24.76 GiB balloons), `expandable_segments:True` **0 / 12**, Fisher p = 0.047. Costs +0.67% wall, nothing in RSS, and raises **peak framebuffer to 92.2% of the card**. Default flipped in `config_example.py` and the live `config.py`; doc 27 §6.6's hardware floor restated alongside it, not silently inherited |
| Step 4 — full-slide `--no-prefetch` | **DONE — 10,666.1 s. Option K settled at 1.044×, not 1.208×.** The read bucket is **449.3 s inline, down 81.0% from the baseline's 2,368.5 s at identical placement** — doc 27 §6.4's page-cache diagnosis confirmed, and doc 33's headline "17.2% of wall" shown to belong to a superseded write pattern. Option L hides 448.4 s of a 449.3 s cost: **99.8% of a ceiling that is now 4.21%** |
| Step 5 — Phase D Phase 2, `depth=2/3`, QuPath pass | **No action**, as doc 36 §2.5 specified. Note §4.1 undercuts the *inputs* to doc 35 §5's `depth=2/3` arithmetic, which strengthens rather than reopens its "not worth building" conclusion |
| Correctness veto | **PASSED everywhere.** `stats` identical in all 18 step-1 runs, all 20 successful step-3 runs, and the full-slide run (10,800 / 16,765, matching round 10). Overlay pyramid audits clean at every level. 79 tests pass |

## 8. Follow-ups this round creates

Ordered by what a next round should actually do first.

1. ~~Decide whether `cuda_alloc_conf = "expandable_segments:True"` becomes the default~~ —
   **done, see §9.**
2. **`B2r_tile_read` is 4.21% of wall and Option L captures 99.8% of it (§4.1).** Every read-side
   item in the backlog was sized against 17.2%. Re-read doc 30/33/35's read-side reasoning with
   449.3 s in hand before spending anything else there — including doc 35 §5's `depth=2/3`, whose
   1.04% ceiling was computed from the inflated numbers and is smaller still today.
3. **The 530 s residual on `B1_m3b_cellpose` (§4.2)** is the last unattributed number on the critical
   arm. It is *not* the refactor and *not* the prefetch. Settling it needs repeated full-slide runs
   (~3 h each) to establish an error bar this project has never had — every full-slide comparison in
   docs 27/35/37 is n=1 vs n=1. **That error bar, not the residual itself, is the thing worth
   buying**, and §1.1 shows it can be had far more cheaply at crop scale if sweeps are
   counterbalanced and run warm.
4. **Measurement hygiene, now demonstrated rather than argued (§1.1).** Two sweeps this round were
   discarded for order/thermal confounds that would have produced a confident wrong answer. Doc 35
   §2.3's conclusion that this anchor "cannot resolve a 0.83% effect" should be re-read as being
   about an unsettled, uncounterbalanced sweep: warm and Latin-squared, within-arm sd is 0.2–0.5% and
   it resolves 1.2% at `d/SE` = 7. Future A/B rounds should counterbalance by default.
5. **Round 10's full-slide metrics JSON is not on disk (§6.3)**, so step 4's comparison had to be
   made against the figures doc 35 published rather than against the raw buckets. Keep
   `_metrics/*_timings.json` for full-slide runs — they are three hours each and irreproducible in
   practice.
6. **`tuspyserver` is missing from the venv (§6.2)**, so three `backend/tests` modules cannot be
   collected. Trivial to fix, deliberately not fixed mid-round.
7. **`scripts/exit_latency_probe.py` is new and belongs in the pre-release checks.** Exit latency is
   invisible to every wall-clock benchmark this project runs, and the defect it found converted a
   correct abort into an unattended job that never returned. Cheap to run: ~90 s.

## 9. Decision — `cuda_alloc_conf` default flipped to `expandable_segments:True`

Follow-up 1 above is closed within this round rather than carried forward, on explicit direction:
§3's measurement is one-sided (4/12 → 0/12 failures for +0.67% wall) and what was holding the default
back was ownership of a production VRAM decision, not missing evidence. That ownership question is
now answered, so the change ships here rather than waiting for a round that would only re-read §3.

**What changed**: `cuda_alloc_conf: str = ""` → `"expandable_segments:True"` in both
`backend/algorithms/hybrid/config_example.py` and the live, gitignored `config.py` on this machine —
the latter matters because `config.py` already existed before this round (`cp`'d once, per
`CLAUDE.md`), so editing only the example would not change what `run_batch(workers>1)` actually does
here. `tests/test_config_parity.py` (8 tests) still passes: it checks structural parity between the
two files, not value equality, and both were edited identically. `cuda_alloc_conf` is already outside
`compute_config_hash`'s `_HASH_EXCLUDE` set (correctly — it changes allocator arena behaviour, never
output bytes), so this flip does not perturb any config hash or invalidate any `--resume` checkpoint.

**What did *not* change, deliberately**: nothing in `m0_multiprocess.py`. The knob was already wired
end to end by doc 27 §5 — `_run_tiles_multiprocess` reads `config.cuda_alloc_conf` and writes it to
`os.environ["PYTORCH_CUDA_ALLOC_CONF"]` immediately before the workers spawn, only if the caller
hasn't already set it. Flipping the default is a one-line, two-file config change, not a code change.

**The hardware floor argument, restated rather than inherited.** Doc 27 §6.6 set 32 GB as a hard
floor for `workers=4` on the strength of one full-slide run peaking at 30,439 MB (93.3%, ~2.2 GB
headroom) — measured with the knob **off**. That number does not carry over silently now that the
knob is on, because §3 measured the opposite of what a naive reading of "fixes an OOM" would suggest:
**`expandable_segments` does not reduce peak VRAM — it raises it**, from a max of 18,193 MB (control)
to 30,061 MB (expandable) on the very same 12-repeat sweep, at `workers=4`. This is consistent with
round 8's earlier reading at `workers=6` (doc 27 §5.1): the knob doesn't defragment memory *away*, it
lets the allocator use segments a stricter policy would refuse, which is exactly why a request that
previously died with "176 MB unavailable, 293 MB reserved but unallocated" now succeeds — and exactly
why peak occupancy climbs. **So: doc 27 §6.6's 32 GB floor is not relaxed by this change, it is
tightened.** With the knob on, `workers=4` production runs should be assumed to sit near 92% of a
32 GB card as routine behaviour, not as a rare peak. The existing rule — no new GPU library or workload
added inside the worker pool without re-measuring against this number — carries forward unchanged in
substance, but the number it must clear is now closer to the ceiling than doc 27 measured it.
`workers≥5` stays off the table for the same reason it always was, more firmly than before.

**Not re-opened**: doc 27 §5's own round-8 decision to leave the default off. That decision was
correct for the evidence it had (workers=6, n=6, p≈1.0) — round 11 didn't overturn it by finding a
mistake, it re-asked the question at the shipped worker count with double the sample and got a
different, resolvable answer. Both rounds' measurements stand as written; only the default changes.
