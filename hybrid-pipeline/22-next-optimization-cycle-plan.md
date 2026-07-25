# 22 — Next optimization cycle: research plan (no code changes this round)

> **Status: planning only.** Per [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)'s
> Discover → Analyze → Plan → Choose discipline, this document only plans what to try, how to
> measure it, and what would count as success or failure. **No implementation happens here.**
> Compiled 2026-07-25, after reading
> [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md) (rounds 1–5),
> [`measurement/current-status-comparison.md`](./measurement/current-status-comparison.md),
> and [`19-open-backlog.md`](./19-open-backlog.md) (current as of round 5, 2026-07-23).
>
> **Trigger for this round:** the team wants to keep shortening full-WSI wall-clock time, from
> two angles: (A) more multi-process/multi-thread parallelism, (B) moving CPU-loop work onto
> the GPU as batched/parallel compute rather than per-item Python loops. This doc scopes both.
> **Constraint, explicit and non-negotiable for this cycle: do not change `default_tile_size`,
> `window_overlap_px`, `window_dedup_iomin`, or any other geometry that defines the segmentation
> window/tile boundaries.** That geometry is validated, stitching-correctness-sensitive, and out
> of scope. **This constraint does not extend to `cellpose_batch_size`/`batch_size`** — those are
> GPU-dispatch parameters (how many patches a forward call processes together), not geometry, and
> §3 below found a concrete, previously-untested way to make them matter without touching a
> single tile's size or boundary. (An earlier draft of this plan conflated the two and wrongly
> closed out cross-tile batching entirely — that was a misreading of the constraint, corrected
> below after actually reading the `cellpose` batching source instead of assuming.) This doc also
> folds in a new fact: **a teammate has now run a full real-WSI slide end-to-end and confirmed
> the current code works** — this changes the gating status of open-backlog item 7 and is the
> first thing to formalize below, before anything else in this plan is sized against it.

## 0. Where we actually stand (recap — do not re-derive, re-verify only)

Current default is `run_batch(workers=1)`, single-process, two-arm background-thread overlap.
Cross-tile multiprocessing (`workers=3`) is **built, measured, adopted, but not shipped to
production** — gated on real full-WSI validation. The measured chain, most-recent first:

| stage | large/441 wall | vs previous | source |
|---|--:|--:|---|
| control (fully serial) | 848.0 s | — | bottleneck-list round 0 |
| ① two-stage overlap | 707.4 s | −16.6% | current-status-comparison §1 |
| Cellpose 4.2.1.1 / `cpdino` swap | 573.7 s | −18.9% | bottleneck-list round 3 |
| ⑧ CPU-prep off MAIN + precut streamed | 480.3 s | −16.3% | bottleneck-list round 4 |
| **cross-tile multiprocessing, `workers=3`** | **156.1 s** | **−67.5%** | bottleneck-list round 5 |

Full-WSI (35,700-tile) linear projection at `workers=3`: **~3.3 h**, still an **upper bound**
(crops are ~85% tissue-dense; a real slide is mostly background, which short-circuits cheaply —
this is exactly what item 7 below needs to pin down instead of assuming).

**Already closed — do not re-litigate without new evidence** (full ablation exists for each):
CUDA MPS (flat end-to-end at the real pipeline's knee), deeper CPU-pipeline depth (+2.8%
slower single-process), `detect_all_dots` process-backend swap, `gc.collect` relocation,
fixed-N `gc.collect` batching, CUDA-stream/pipeline-depth-2 bubble redesign (≤1.065x),
GPU-side tile/transform loading (1.012x, nothing to move). See
[`19-open-backlog.md`](./19-open-backlog.md) §1 for the one-line pointer to each. **Re-reading
this list before proposing a "new" idea is mandatory** — three of the ideas below turn out to be
one small step past something already tried, and the doc says exactly where that step is.

**One item on that closed list needs a caveat, not a re-close:** the round-4
`cellpose_batch_size` sweep (16/32/64, flat) is *not* voided, but its scope was narrower than it
looked — it only ever tested the config value in isolation, at a call site that structurally
could never exercise what the value controls. §3 below traces this precisely and reopens
cross-tile batching on that basis, without touching tile size or window geometry.

---

## 1. Step 0 (do this before anything else): formalize the full-WSI result

[`19-open-backlog.md`](./19-open-backlog.md) item 7 says full-WSI-scale validation "has never
been done, any round" and is the **explicit gate** ([`20-cross-tile-multiprocessing-plan.md`](./20-cross-tile-multiprocessing-plan.md)
§4) before `workers>1` ships to production. A teammate has now run a complete slide and
confirmed the pipeline works — but "it worked" is not the same evidence class as every other
number in this document, which is all measured with `scripts/perf_measure.py --gpu-dmon` and a
correctness veto against a same-code noise floor.

**Action, not a code change:** re-run (or pull logs from) that full-WSI pass through the
existing harness discipline so it becomes a citable measurement, not an anecdote:

1. Wall-clock, `--gpu-dmon`, peak RSS/VRAM, at `workers=1` (control) and `workers=3`
   (candidate), same slide, same machine, GPU verified idle before launch (doc 13 §0
   discipline). This is the number every projection in this doc has been extrapolating toward
   since round 2 — get the real one.
2. **Tissue-density-driven load balance**: log per-worker tile counts and per-worker idle time
   over the full grid. The crops used for every round so far are ~85% tissue-dense; a real
   slide is mostly white background. The dynamic work queue ([`21-cross-tile-multiprocessing-implementation.md`](./21-cross-tile-multiprocessing-implementation.md))
   was designed for this but never measured against it — confirm workers stay balanced when
   most tiles short-circuit near-instantly instead of doing full GPU+CPU work.
3. **VRAM/RSS at N=3 over 35,700 tiles**, not 441 — the per-process figures in round 5
   (2787/3117/4118/5167 MB per process) were measured on a 441-tile crop; confirm they don't
   creep with tile count (the RSS invariant elsewhere in this pipeline is cell-count-driven,
   not tile-count-driven, but that was proven for `workers=1` — re-confirm it holds per-process
   under N-way concurrency too).
4. Correctness veto: same standard as every other round — diff cell counts / dot counts /
   score against a `workers=1` run of the identical slide, judged against the same-code noise
   floor (doc 21 §establishes this method).

**Decision rule:** if (1)–(4) pass, item 7 closes and item 1 (cross-tile multiprocessing) is
cleared to ship — update [`19-open-backlog.md`](./19-open-backlog.md) accordingly. This is
almost pure measurement effort, is already-built code, and unblocks the single largest
already-realized speedup (3.09x) from sitting on the shelf. **Do this before spending any
engineering time on the tracks below** — it's the cheapest, highest-confidence item in this
document.

---

## 2. Track A — more multi-process / multi-thread parallelism

The project has already tried threads (①'s background-thread overlap), tested and rejected a
deeper thread pipeline (Candidate A, doc 20), tested and rejected CUDA MPS (Candidate C), and
adopted cross-tile **multiprocessing** (Candidate D, not threads — GIL contention between the
two arms is exactly why threads plateau and processes don't). "More multi-threading" as a raw
lever is therefore largely spent; what's left in this track is *tuning* and *auditing* the
multiprocessing solution, plus one genuinely unmeasured concurrency axis.

### A1. Re-tune worker count against real full-WSI tissue density
Round 5 picked `workers=3` as the sweet spot on a tissue-dense crop (VRAM: 2787/6233/12354/
20667 MB at N=1/2/3/4 — superlinear, caps N). A real slide's background tiles are far cheaper
per-tile, so the *effective* concurrency the GPU sees may differ from a dense crop. Once step 0
lands, sweep `workers ∈ {2,3,4,5}` on the real full-WSI run (not a crop) and re-plot wall vs.
VRAM peak — the N=3 recommendation might move up or down once background-tile cheapness is
counted honestly.
- **Ceiling:** unknown until measured — could be higher than 3.09x if background tiles let more
  workers fit in the same VROM envelope, or the same if VRAM-per-process dominates regardless
  of tile content (it's driven by resident models, not per-tile data).
- **Method:** same harness as step 0, `--gpu-dmon`, vary `--workers`, hold slide fixed.
- **Stop-loss:** if `workers=3` remains optimal (likely, since VRAM/process is dominated by
  fixed model residency, not tile content), record that and close — don't keep sweeping.

### A2. CPU-core contention audit under N-way multiprocessing (new, unmeasured)
`detect_all_dots` (`m3_module/m3_dot_detection.py:194`) already runs
`Parallel(n_jobs=-1, prefer='threads')` **inside** each process's background-CPU arm. At
`workers=3`, that means **3 processes each trying to fan out `n_jobs=-1` threads** on the same
machine — nobody has measured whether this oversubscribes CPU cores and turns the "GPU-idle
problem" the project spent five rounds solving into a **CPU-core-idle problem** instead once
production-scale concurrency lands.
- **Hypothesis:** with `workers=3` and `os.cpu_count()` cores, if `n_jobs=-1` fans out to all
  cores in every process, the machine oversubscribes 3x and `detect_all_dots`'s wall-time per
  process degrades — silently eating into the 3.09x by a mechanism the round-5 measurement
  (single slide, whatever core count that machine has) may already be hiding or may not show at
  a different core count.
- **Method:** `psutil`/`os.sched_getaffinity` core count on the target machine; measure
  `detect_all_dots` per-call wall time at `workers=1` vs `workers=3` (same tile, same core
  budget) — if it inflates >~10–15%, cap `n_jobs` per worker (e.g. `n_jobs = cpu_count // workers`)
  and re-measure end-to-end. This is a one-line config change if the audit finds a problem —
  the **audit is the research**, not a redesign.
- **Ceiling:** capped by how much slack the BG arm still has (BG/MAIN was 0.841 single-process,
  round 4) — if this is real it's a correctness-of-measurement issue for the 3.09x figure, not
  a wall-clock opportunity in itself, so treat it as a validation gap first.

### A3. Multi-request / concurrent-job behavior (open-backlog item 8, never measured)
Distinct axis: not tiles within one slide, but multiple **API requests** (different slides)
in flight together. `measurement/bottleneck-list.md` item ⑦ flags "concurrent analysis requests
each hold a threadpool worker but serialize on the single GPU/CUDA context" as a risk, never
sized. If the deployment target ever runs more than one slide at a time, this determines
real-world throughput more than any single-slide number in this document.
- **Method:** load-test `backend/api/hybrid.py`'s job endpoint with 2–3 concurrent slide
  submissions on the same GPU, measure per-job wall-clock inflation vs. solo.
- **Priority:** lower than A1/A2 unless the deployment model is known to serve concurrent
  requests — confirm with whoever owns production rollout before spending time here.

---

## 3. Track B — move CPU-loop work to GPU-parallel compute (not micro-loop kernels)

**Scope discipline up front:** the quickref's own case study is "9 versions of GPU-kernel
effort for 1.1% of total time, and the result was slower than a CPU loop." Doc 18/`19-open-backlog.md`
item 5 already stop-lossed patching Cellpose's *internal* kernel-launch loops
(`_extend_centers_gpu`, `get_masks_torch`, `steps_interp`, `get_rel_pos`) at ~1.23x ceiling for
third-party-patch risk. **Track B below is deliberately not that** — every item targets code
this project owns (the call sites around Cellpose/UNet++, and the pure-CPU M3 stages), and none
of it touches `default_tile_size` or the window/dedup geometry.

### B1. Cross-tile batching of the Cellpose forward call — root cause traced to the wrapper, not the GPU; reopened
**This directly answers the question raised this cycle: "did the batch-size sweep fail because
of how `segment_windowed` is wrappered, rather than a GPU limitation?" — yes, confirmed by
reading `cellpose`'s actual source (`.venv/lib/python3.11/site-packages/cellpose/{models,core}.py`),
not by assumption.**

**The call chain today** (`hybrid_pipeline.py:581,598` → `m2_segmentation.py`):
`segment_windowed` loops over windows (`_overlap_window_coords`) and calls
`segmenter.predict(image[y0:y1, x0:x1])` **once per window** (`m2_segmentation.py:133`);
`CellposeSegmenter.predict` calls `self.model.eval(image, batch_size=self.batch_size, ...)`
(`m2_segmentation.py:81`) on **one 2D image**. At the current 1024px tile size there is exactly
one window per tile (`_overlap_window_coords`'s `_starts` returns `[0]` when `length <= tile`),
so this call site never presents cellpose with more than one image at a time.

**What `cellpose.models.CellposeModel.eval` actually does with its input** (`models.py:230-243`):
it checks `isinstance(x, list)` — and if so, it **loops one image at a time**
(`for i in iterator: out = self.eval(x[i], ...)`), fully serially. **A Python list of tiles
gets zero batching benefit; this was the first thing worth ruling out and it's ruled out.**

**But if `x` is a genuine stacked array** (not a list — a real `(Lz, H, W, C)` ndarray with
`Lz > 1`), `eval` takes a different path: it calls `self._run_net(x, batch_size=batch_size, ...)`
→ `cellpose.core.run_net(net, imgi, batch_size, bsize=384, ...)` (`core.py:165-243`). **This
function batches across images by design** — read literally from the source:
```
ntiles = ny * nx                       # patches per single image (16, at bsize=384/1024px)
nimgs = max(1, batch_size // ntiles)   # <-- number of IMAGES packed into one batch
niter = ceil(Lz / nimgs)
for k in range(niter):
    inds = <nimgs images from the Lz-stack>
    IMGa = zeros((ntiles * len(inds), nchan, bsize, bsize))
    for i, b in enumerate(inds):
        IMGa[i*ntiles:(i+1)*ntiles] = make_tiles(imgi[b], bsize=384)   # each image's own patches
    for j in range(0, IMGa.shape[0], batch_size):
        ya0, stylea0 = _forward(net, IMGa[j:j+batch_size])            # <-- the actual GPU call
```
`run_net` already knows how to pack **multiple separate images'** patches into one `IMGa` buffer
and run them through `_forward()` together, exactly the "stack N tiles into one tensor" idea —
this machinery already exists in the pinned dependency, unmodified. It is simply **never
reached**, because our call site (`CellposeSegmenter.predict`) always hands it `Lz=1`.

**Why the round-4 sweep (16/32/64) was flat — now a traced fact, not a hypothesis:** with
`Lz=1` always, `nimgs = batch_size // ntiles` is irrelevant to how many *images* get grouped
(there's only ever one), and since `ntiles=16` already equals `batch_size=16`, the inner
`_forward` chunking loop also never had a partial batch to fill. The sweep changed a number
that the call site made structurally unreachable — **a wrapper limitation, exactly as
suspected, not evidence the GPU/model can't benefit from batching.**

**The reopened design (not a config sweep — a call-site restructure):**
1. Accumulate a small group of **N tiles'** already-read RGB images (`_read_rgb` output) before
   invoking M2/M3b, instead of handing `_process_one_chunk_gpu` one tile at a time.
2. Stack them into a real `(N, H, W, 3)` ndarray (not a list) and call
   `cellpose_segmenter.model.eval(stacked, batch_size=N*16, ...)` — raising `batch_size` here is
   a GPU-dispatch parameter, not a tile-size or window-geometry change; nothing about
   `default_tile_size`, `window_overlap_px`, or `window_dedup_iomin` changes.
3. `run_net`'s output `yf`/`styles` are already shaped `(Lz, Ly, Lx, 3)` / `(Lz, 256)` — **flows
   and cell probability are returned per input image**, so per-tile identity survives the
   batched forward pass intact; the downstream flow→mask reconstruction (`dynamics.py`) still
   runs per-tile exactly as today, unmodified. This is a materially lower correctness risk than
   the earlier (withdrawn) "concatenate patches ourselves" framing — `cellpose` already keeps
   tiles separate on the output side, we're only sharing the input-side GPU call.
4. Same mechanism applies to M3b (the DISH-nucleus segmenter) and, separately, to M1 (§B1b
   below) — each is its own model/call site and would need its own grouping.

**What this costs / risks (real, not hand-waved):**
- **Pipeline-structure change.** `run_batch`'s per-tile precut→GPU-front→CPU-back loop
  (`hybrid_pipeline.py:930`) would need an N-tile gather stage before the GPU front, changing
  the two-stage MAIN/BG overlap's granularity from "1 tile ahead" to "N tiles ahead" — this
  needs re-measurement against the existing overlap model (bottleneck-list's `wall ≈
  max(MAIN, BG) + outside`), not an assumption that it stays valid at N>1.
- **Interacts with `workers=N` cross-tile multiprocessing (Track A), not additively.** Batching
  M tiles per process on top of N worker processes multiplies in-flight tiles per process by M;
  unlike per-process VRAM duplication (which duplicates whole model weights), this only grows
  *activation* memory per group, so it should be cheaper per unit of concurrency than spinning
  up another worker — but "should be" needs a VRAM measurement, not an assumption.
- **Floating-point ordering:** batched matmul over `(N*16, 3, 384, 384)` vs. sequential
  `(16, 3, 384, 384)` calls can produce tiny numeric differences from op-scheduling/reduction
  order — expect this to sit inside the same-code noise floor doc 18/21 already calibrate
  against, but it must be checked with the same correctness veto as every other candidate here,
  not assumed identical.
- **Group-fill edge case:** the last group of a slide may have fewer than N tiles (e.g. slide
  tile count not divisible by N) — pad-or-flush, doesn't change geometry, just needs handling.
- **Third-party surface, but not the stop-lossed one.** This calls *into* `cellpose.core.run_net`
  through its existing public-ish batching path — it does not patch `dynamics.py`/`get_rel_pos`
  internals (item 5's stop-lossed target). Still worth pinning the exact `cellpose==4.2.1.1`
  version's `run_net` signature before building, since a future cellpose upgrade could change it.
- **Ceiling is still unmeasured — ceiling estimate depends on where the current per-call cost
  actually lives.** Doc 18 attributed a large share of device idle to "intra-forward,
  launch-bound" overhead, but never separated *fixed-per-call* Python overhead (image
  normalize/pad/resize, style-vector bookkeeping, `_forward`'s tensor transfer) from
  *per-patch-proportional* GPU/kernel-launch cost. If the fixed-per-call share dominates,
  grouping N tiles amortizes it and this wins; if the launch-bound cost is genuinely
  per-patch (e.g. inside the ViT/DINOv3 backbone's attention blocks), grouping tiles doesn't
  reduce total patch-processing work and gains only whatever a bigger single matmul saves over
  N smaller ones (real, but likely smaller). **This is exactly what step 1 below measures.**

**Method — cheap-first, no pipeline changes for step 1:**
1. **Microbenchmark** (no pipeline restructuring; a standalone script): call
   `cellpose_segmenter.model.eval()` directly with (a) 1 real tile's image → `Lz=1`, 16 patches,
   and (b) a hand-stacked array of 2/4 real tiles' images → `Lz=2/4`, `batch_size=32/64`, same
   GPU, warm cache, `--gpu-dmon`. Compare **per-tile** wall time. If grouping doesn't reduce
   per-tile time, the fixed-per-call-overhead hypothesis is wrong — stop here, cheaply, without
   ever touching `run_batch`.
2. Repeat for M3b's DISH segmenter (separate model, same mechanism, check independently — no
   reason to assume the two segmenters share the same fixed/proportional-overhead split).
3. Only if (1)/(2) show a real per-tile speedup, design the N-tile gather/dispatch restructure
   as its own follow-up doc (out of scope here), sized against `workers=3` as baseline (not
   `workers=1`), with the correctness veto and overlap-model re-validation called out above.
- **Priority:** highest in Track B — grounded in a traced architectural cause rather than a
  hypothesis, and step 1 is cheap enough to falsify or confirm before any real engineering
  starts.

### B1b. Same question for M1 (UNet++) — same pattern found, not yet benchmarked
`UNetPPInference` (`backend/algorithms/hybrid/unet_inference.py`) is called once per tile via
`generate_ihc_core_mask` → `predict_single` (`unet_inference.py:210`). Internally,
`_predict_sliding_window` (`unet_inference.py:271-303`) already batches multiple **windows
within one image** using `self.batch_size` — but, same as Cellpose, there is no cross-*tile*
batching: each tile is its own `predict_single` call. This is the same shape of limitation as
B1, on a different model.
- **Concrete research lead, not yet followed up:** a different, apparently older inference
  module in this repo, `cell_mask/unet_mask/inference.py`, already has a `predict_batch` method
  (`inference.py:279`) that is *not* used by the hybrid pipeline's `unet_inference.py`. Read it
  before designing anything — it may already contain a working multi-image batching
  implementation for this exact model family that can be ported/adapted, which would be cheaper
  than designing cross-tile batching for UNet++ from scratch the way B1 had to for Cellpose.
- **Ceiling caveat, size it before investing:** UNet++'s forward is currently ~13.68 s / 441
  calls (~0.4% of wall at the round-3 anchor) — even a large per-call speedup here is
  Amdahl-capped by that small share. Worth doing only bundled with B1's pipeline restructure
  (same N-tile gather stage would feed both models), not as a standalone effort.

### B2. GPU port of `detect_all_dots` (LAB + morphology) — conditional, gate on A2
`detect_all_dots` (`m3_module/m3_dot_detection.py:97`) does one global `_rgb_to_lab` (whole-tile,
already vectorized) then per-cell red/black dot detection via LAB thresholding + connected
components, fanned out over `joblib` threads. It is pure NumPy/skimage, no GPU, and no model —
unlike B1/B1b, there is no existing "batching" primitive to activate here at all; the only way
to get GPU parallelism into this stage is a genuine port (e.g. CuPy/cuCIM), not a dispatch fix.
Ceiling **today is 1.013x** (bottleneck-list ⑨/round 3 arm-ceiling table) because it's fully
hidden on the slack BG arm — a GPU port cannot move single-process wall-clock at all right now,
and doc 19 item 3 already says not to chase it for that reason. (Whole-tile vectorization via
`regionprops_table` was already tried and stopped-out per bottleneck-list item ② — a hidden
stage can't move wall regardless of how it's implemented; don't re-attempt that specific
approach, a GPU port is a different lever than the CPU-side vectorization already closed.)
- **Why it's still worth planning, not dropping:** two things could re-expose it — (a) if B1
  (or B1b) ever land and shrink the GPU front further, the BG arm's 15.9% margin
  ([`bottleneck-list.md`](./measurement/bottleneck-list.md) round 4) closes and
  `detect_all_dots` becomes critical-path; (b) if A2's audit finds CPU-core oversubscription
  under `workers=3`, the fix might be "do less CPU work per process," and a
  GPU port (CuPy/cuCIM `label`, LAB conversion, morphology) directly relieves CPU-core pressure
  even while its wall-clock ceiling stays ~1.0 — i.e. its payoff would show up as **freeing CPU
  cores for other processes**, not as a local speedup. That's a different Amdahl argument than
  the ones in this document so far and needs to be evaluated against A2's finding, not against
  today's single-process ceiling.
- **Method (only run after A2 lands):** if A2 shows CPU-core contention, prototype
  `_detect_one_cell`'s LAB threshold + morphology on GPU (CuPy `cucim.skimage.measure.label` is
  the closest drop-in for skimage connected components) for one cell's bbox patch, and compare
  (i) per-process CPU time freed vs. (ii) GPU forward slowdown from adding this work onto the
  same CUDA context that Cellpose/UNet++ use. **Correctness veto**: bit-for-bit or
  noise-floor-equivalent red/black dot counts vs. the CPU path, same standard as every other
  item.
- **Stop-loss:** if A2 finds no CPU contention (plausible — the machine may simply have enough
  cores), **do not build this**. It has no standalone wall-clock case at ceiling 1.013x.

### B3. GPU port of `enlarge_cell_instances` / `build_all_positive_results` — bundle-only, not standalone
`enlarge_cell_instances` (`m3_cells_generator.py:74`, `skimage.expand_labels`) and
`build_all_positive_results` (`m3_cells_generator.py:29`, `scipy.ndimage.center_of_mass`) are
already on the BG arm (moved there in round 4, [`18-...-implementation.md`](./18-gpu-starvation-prerequisites-implementation.md) §2).
Combined self-time was 28.4 s/4.96% at the round-3 anchor before the move; post-move they're
part of the 15.9%-margin BG arm, same slack-arm ceiling logic as B2.
- **Do not plan this as a standalone item.** If B2 is ever built (CPU-core-freeing rationale),
  bundle these two in at the same time — both are pure NumPy/SciPy morphology/labeling ops with
  direct CuPy/cuCIM equivalents, and the engineering cost of porting them alongside `detect_all_dots`
  is marginal once the CuPy plumbing exists. Standing up a separate GPU port effort for 28 s at
  a ~1.05x ceiling, alone, is exactly the kind of over-investment the quickref warns against.

### B4. Re-confirm the Cellpose-internals kernel-launch trace against the current version
Open-backlog item 5 explicitly flags that its evidence (`gil-contention-diag.md`'s
"追加深挖" trace) was taken against `cellpose==4.0.8`, and the project has since upgraded to
`4.2.1.1`'s `cpdino`/DINOv3 backbone (round 3) — function names/line numbers inside cellpose may
have moved, and the DINOv3 backbone's internal structure is not necessarily the same
kernel-launch shape as the old `cpsam` backbone. **Administrative, not engineering**: before
anyone reopens or re-closes item 5, re-run the py-spy trace against the current pinned version
to confirm the ~1.23x ceiling and "launch-bound, not placement-bound" diagnosis still hold.
Cheap (a few hours), and it's the one thing standing between "closed" and "actually verified
against what's running today."

---

## 4. Advanced-technique back-pocket list (not scheduled — cite before reaching for these)

These are real options but are **not ranked into the sequence below** because none has a sized
hypothesis yet, and the quickref's own case study is a warning against reaching for
sophistication before measuring. Listed so nobody "discovers" them mid-cycle without checking
here first:

- **`torch.compile` on `_init_unet_inferencer`'s model** — this is project-owned code (not a
  third-party internals patch like the stop-lossed Cellpose kernel work), so the patch-risk
  objection that killed item 5 doesn't apply here. UNet++ forward is currently only 13.68 s /
  441 calls (~0.4% of wall) — a compile win here is bounded by that share regardless of how
  much it helps per-call, so this is Amdahl-capped before it starts. Worth a cheap ablation only
  if someone is already touching that code path for another reason.
- **CUDA graph capture around the UNet++ inferencer call** — same Amdahl cap as above (~0.4%
  of wall); not worth standalone effort.
- **Pinned memory / async H2D for `_read_rgb` tile loads** — already covered by open-backlog
  item 6 ("GPU-side tile/transform loading," stopped out at 1.012x ceiling, no existing
  CPU→GPU transform pipeline to move). Don't reopen without new evidence per that item's status.
- **NVIDIA DALI / TensorRT for the UNet++ or Cellpose forward path** — both would require
  re-exporting/re-validating models through a new runtime, with the same ~0.4–~perhaps
  double-digit % ceiling questions as the `torch.compile` item above, plus new correctness risk
  from a different numerics path (fp16/int8 quantization). Not sized; flag only.
- **`cellpose_batch_size` as a bare config sweep, at `Lz=1` (i.e. not paired with B1's stacking
  change)** — already measured flat (round 4) for the reason B1 §3 traces; don't re-run the
  16/32/64 sweep in isolation again. It only becomes a meaningful lever together with B1's
  call-site restructure (genuine multi-tile `Lz>1` stacking), not on its own.

---

## 5. Suggested sequencing (cheapest-and-most-confident first, per playbook §3)

1. **Step 0 — formalize the full-WSI validation** (§1). Near-zero engineering cost (the code
   is built), unblocks the largest already-realized win (3.09x) from shipping. Do this first,
   unconditionally.
2. **B1 step 1 — Cellpose cross-tile batching microbenchmark** (§3). Cheap, standalone,
   no pipeline changes: directly call `model.eval()` with a stacked multi-tile array vs.
   sequential single-tile calls and compare per-tile wall time with `--gpu-dmon`. This is the
   traced-root-cause candidate — resolve whether it's real before anything else in Track B.
3. **A2 — CPU-core contention audit** (§2). Cheap to measure (a timing comparison, no code
   change unless a problem is found), and it's a correctness check on the very number step 0
   is trying to formalize — if `workers=3` oversubscribes CPU cores on the target machine, that
   changes step 0's own recommended worker count.
4. **A1 — worker-count re-tune against real full-WSI density** (§2), once step 0's real-slide
   harness exists — reuses that same infrastructure, so it's nearly free once (1) is done.
5. **B1/B1b full design + build**, only if step 2's microbenchmark confirms a real per-tile
   speedup — the N-tile gather restructure, sized against `workers=3`, with the correctness
   veto and overlap-model re-validation §3 calls out. Read `cell_mask/unet_mask/inference.py`'s
   existing `predict_batch` (B1b) before designing UNet++'s version from scratch.
6. **B4 — re-confirm the Cellpose kernel-launch trace against 4.2.1.1** (§3). Administrative,
   low cost, closes an open correctness-of-evidence gap regardless of what else happens.
7. **B2/B3 — GPU port of `detect_all_dots` + CPU-prep stages** (§3), **only if** A2 (item 3)
   finds real CPU-core contention. Otherwise these stay logged, not built — same treatment the
   project has already given every other slack-arm item.
8. **A3 — multi-request load test** (§2). Do this once someone confirms concurrent-request
   serving is actually part of the deployment plan; otherwise it's speculative infrastructure
   for a usage pattern that may not occur.

Every item above inherits the project's existing discipline: **measure before building,
correctness veto on every candidate, ablate before shipping, and update
[`19-open-backlog.md`](./19-open-backlog.md) / [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md)
the moment anything here is measured** — this doc only plans the next round; it is not itself a
result and should not be cited as one once real numbers exist.
