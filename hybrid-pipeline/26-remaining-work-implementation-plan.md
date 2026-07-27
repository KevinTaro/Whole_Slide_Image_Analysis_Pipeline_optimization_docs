# 26 — Remaining work: implementation plan for everything not yet shipped

> **Executed by [`27-remaining-work-implementation.md`](./27-remaining-work-implementation.md)
> (round 8, 2026-07-27)** — that document is what actually landed, what it measured, and what
> stayed open with the reason. This document remains the plan as originally compiled; read 27 for
> current status.
>
> Compiled 2026-07-27 by reading [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md),
> [`19-open-backlog.md`](./19-open-backlog.md), [`measurement/current-status-comparison.md`](./measurement/current-status-comparison.md),
> [`DISCOVERED-NOT-IMPLEMENTED.md`](./DISCOVERED-NOT-IMPLEMENTED.md), and
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md).
> **Purpose**: a single ranked, sequenced plan covering every item those four documents mark as
> open, gated, or never-executed — i.e. everything *not yet implemented* — so picking up work
> means reading this doc once instead of re-deriving priority from four overlapping sources every
> time. This document does not replace any of the four; it is a plan built from them, applying the
> quickref's Discover → Analyze → Plan → Choose method **to the backlog itself**, not to one
> bottleneck. When an item here ships, close it the same way 19 already asks: update the source
> doc's status, then delete or check off its row here.

---

## 0. Method

Per the quickref: measure before optimizing, rank by Amdahl ceiling not absolute time, cheapest
fix first (parallelize → move off critical path → eliminate), decide by end-to-end wall-clock,
correctness is a veto. Applied here at the backlog level, that means:

- Items with a **ceiling under ~5%** (single-digit, per the quickref's own "stop, don't bother"
  rule) are listed but explicitly **not** sequenced into the plan — re-proposing them without new
  evidence is the anti-pattern this project has already paid for once (the CuPy case study the
  full playbook doc is named after). See §4.2.
- Items that are **reliability or correctness gates**, not speed items, are ranked *above*
  performance work of any ceiling, because they block shipping what's already built
  (`workers>1`) — an unshippable 3.5x is worth less than a shippable 1x.
- Items that are **pure documentation drift** carry no performance risk and no urgency; they're
  sequenced last, batched, because they're cheap and independent of everything else.

---

## 1. Discover — full inventory, deduped

Every row below is cross-referenced to its entry in `19-open-backlog.md` (§1/§2/§3) and
`DISCOVERED-NOT-IMPLEMENTED.md` (#1–42) so status can be re-verified at the source before acting.

### 1a. Performance — open or gated (not stop-lossed)

| Item | 19-backlog | DISCOVERED # | Ceiling |
|---|---|---|---|
| Full real-WSI-scale end-to-end validation | #7 | #3 | n/a — binding gate |
| Cross-tile multiprocessing — ship decision | #1 | #1 | 2.06x–3.51x, gated on the row above |
| `workers≥6` allocator-fragmentation OOM | #7b | #2 | reliability, not sized |
| No partial-resume/checkpointing for `run_batch` | #1c | #42 | reliability, not sized |
| `_stitch_overlay_slide` missing `RLIMIT_NOFILE` check | (under #7) | #4 | reliability, not sized |
| Phase D stitch — cheap `tiffsave` knobs | #1b | #5 | 1.036x–1.078x |
| Phase D stitch — GPU port (nvImageCodec) | #1b | #6 | same ceiling, gated on #5 first |
| `clear_slide_edge_cells` CPU glue | #2 | #10 | ~1.2% of wall |
| `detect_all_dots` +22.3% regression cause | #3 | #14 | 1.013x, no payoff |
| `detect_all_dots` whole-tile vectorization | — | #26 | unsized, highest risk |
| Cellpose kernel-internals retargeting (`_from_device`, `_quantile`) | — | #31 | small, third-party risk |
| Multi-request / concurrent-job load test (Phase E) | #8 | #15 | unsized, never measured |
| Persistent worker pool for API path | (under #8) | #41 | gated on the row above |
| A1 — re-tune `workers` against real full-WSI density | (under #7) | #30 | gated on validation |
| Per-bucket timing inside multiprocess workers | — | #40 | measurement infra, not perf itself |
| `pip freeze` with every measurement round | #9 | #16 | process discipline |

### 1b. Correctness / clinical — blocking

| Item | 19-backlog | DISCOVERED # |
|---|---|---|
| Cellpose 4.0.8→4.2.1.1 checkpoint retrain needs pathologist sign-off | §2 | #39 |

### 1c. Documentation ↔ code drift

| Item | 19-backlog | DISCOVERED # |
|---|---|---|
| `generate_ihc_core_mask` param named `ihc_tile_path` but takes ndarray | §3 | #32 |
| `docs/sdd-elastic-dish-matching.md` dead reference in `m3_elastic_matching.py` | §3 | #33 |
| `docs/dish_dot_detection_spec.md` dead reference in `config.py`/`config_example.py` | §3 | #34 |
| `elastic_matching_v3_explainer.html` describes wrong (nucleus-centric) algorithm | §3 | #35 |
| No test guards `config.py`/`config_example.py` parity | §3 | #36 |
| `backend/algorithms/hybrid/` has zero pipeline-correctness tests | §3 | #37 |
| Codegraph phantom-file list not reconfirmed against current path | §3 | #38 |

**Not carried into this plan** (see §4.2 for the full list with reasons): every 🔴 stop-lossed
candidate in `DISCOVERED-NOT-IMPLEMENTED.md` (#7, #9, #11, #12, #13, #17–#25), and the three
⚪ never-executed back-pocket ideas with no sizing and a sub-1%-of-wall cap (#27 `torch.compile`,
#28 CUDA graph capture on UNet++, #29 DALI/TensorRT).

---

## 2. Analyze — what actually gates what

Two dependency chains dominate this backlog; almost everything else is independent and can be
picked up in any order.

**Chain A (validation → shipping → tuning):**
Full real-WSI validation (§1a row 1) → unblocks → cross-tile multiprocessing ship decision → and
separately unblocks → A1 worker re-tune. The `workers≥6` OOM investigation and the
partial-resume/checkpointing gap should land *before* or *alongside* the validation run, because
that run is exactly the kind of long unattended job that would eat the cost of both defects if hit
mid-run — validating first and hardening later means a failed validation run's root cause is
ambiguous (bad tuning vs. a defect this backlog already knows about).

**Chain B (measure → decide → maybe build):**
Phase D cheap `tiffsave` knobs (no new dependency) must be tried and ablated *before* the GPU port
— the GPU port is real engineering (pyramid levels + BigTIFF container assembly, currently
undesigned) and the cheap knobs might close enough of the 1.036x–1.078x ceiling to make that
engineering not worth it. This is the same cheapest-first ordering the quickref names explicitly.

**Everything else is independent:** the documentation-drift items, the `pip freeze` discipline,
per-bucket worker timing, `clear_slide_edge_cells`, the `detect_all_dots` regression isolation, and
the correctness sign-off can all be done in parallel with the two chains above and with each other.

---

## 3. Plan — sequenced tiers with concrete next steps

### Tier 0 — blocking gates (do first; everything else in performance either depends on these or is independent of them)

**0.1 Full real-WSI-scale end-to-end validation** ([19 #7](./19-open-backlog.md), [DISCOVERED #3](./DISCOVERED-NOT-IMPLEMENTED.md))
Never done in any round — every number so far is a crop extrapolation. Concrete steps:
1. Pick one real slide already used for composition measurement (`HER2_processed.tiff` /
   `DISH_processed.tiff` under `/data/nvmessd/storge_tsgh/<case>/output`, per the reproduce
   commands in [`current-status-comparison.md`](./measurement/current-status-comparison.md) §6).
2. Run the full 27,565-tile grid end-to-end at `workers=1` (production default) with
   `scripts/perf_measure.py`, capturing the same timing/VRAM/RSS instrumentation as every prior
   round.
3. Repeat at `workers=4` (the round-6/7 recommendation) once 0.2/0.3 below are addressed — don't
   validate the risky configuration before hardening it.
4. Compare real wall-clock against the ~2.6h/~1.25h projections in
   [`bottleneck-list.md`](./measurement/bottleneck-list.md) and record the delta; this is also the
   first real end-to-end check on `_stitch_overlay_slide`'s `RLIMIT_NOFILE` behavior (0.4 below) at
   its actual scale (27,565 open file handles).
5. **Definition of done**: one full-slide `workers=1` run and one full-slide `workers=4` run,
   both completing without fail-fast abort, with output correctness spot-checked against a known
   region. This closes 19 #7, DISCOVERED #3, and unblocks the cross-tile multiprocessing ship
   decision and A1 re-tune below.

**0.2 `workers≥6` allocator-fragmentation OOM** ([19 #7b](./19-open-backlog.md), [DISCOVERED #2](./DISCOVERED-NOT-IMPLEMENTED.md))
Untried fix already named twice in prior docs: set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and re-run the same `workers=6` sweep that
produced the 2-in-6 failure rate (doc 23 §4.6/§6). If it clears the OOM, re-open whether
`workers=6` becomes viable at scale (it's currently capped at 4–5 specifically because of this).
If it doesn't, root-cause the byte-identical 24.76 GiB balloon before touching the worker
recommendation again.

**0.3 No partial-resume/checkpointing for `run_batch`** ([19 #1c](./19-open-backlog.md), [DISCOVERED #42](./DISCOVERED-NOT-IMPLEMENTED.md))
Fail-fast currently discards the whole batch on any single-tile failure — at full-WSI scale an
OOM at tile 25,000 of 27,565 costs the entire run. Minimum viable fix: a persisted per-tile
completion log (tile id → success/failure) that `run_batch` checks on start and skips completed
tiles on retry. Scope this *before* 0.1's full-slide run so a mid-run failure during validation is
resumable rather than a wasted multi-hour run.

**0.4 `_stitch_overlay_slide` missing `RLIMIT_NOFILE` check** ([DISCOVERED #4](./DISCOVERED-NOT-IMPLEMENTED.md))
Currently opens all 27,565 overlay tiles as pyvips images at once with no guard; passes only
because this host's soft limit is 1,048,576. Add a check (`resource.getrlimit(RLIMIT_NOFILE)`)
that fails loudly with a clear message (or batches the open calls) before the full-slide
validation run (0.1) is attempted on any host that isn't this one.

### Tier 1 — cheap performance wins, no new dependency

**1.1 Phase D stitch — cheap `tiffsave` knobs** ([19 #1b](./19-open-backlog.md), [DISCOVERED #5](./DISCOVERED-NOT-IMPLEMENTED.md))
Try `tiffsave` tile-size and pyramid-depth parameters, and skip re-encoding constant-background
regions, against the same 16.22-gigapixel measurement `scripts/stitch_probe.py` already
established (322.7 s baseline). Ablate each knob independently per the quickref's rule ("every
optimization layer must justify itself"). Ceiling is small (1.036x–1.078x) but this is genuinely
free — no new dependency, and it's the prerequisite for deciding on 1.2.

**1.2 Phase D stitch — GPU port (nvImageCodec)** ([19 #1b](./19-open-backlog.md), [DISCOVERED #6](./DISCOVERED-NOT-IMPLEMENTED.md))
Only pick this up if 1.1 doesn't close enough of the ceiling to be worth stopping there. nvImageCodec
encodes lossless LZW TIFF 19.2x faster but produces no pyramid/BigTIFF — this is real engineering
(downsample-level generation + container assembly), currently undesigned. Scope a design first if
1.1 proves insufficient.

**1.3 Per-bucket timing inside multiprocess workers** ([DISCOVERED #40](./DISCOVERED-NOT-IMPLEMENTED.md))
`perf_measure.py`'s instrumentation is parent-process-only today, so nothing at `workers>1` is
measurable at the stage level. Fix: each worker emits its own `TIMINGS` dict over the existing
result queue. Do this before or alongside 0.1's `workers=4` validation run — without it, that run
can report overall wall-clock but nothing about *why*.

### Tier 2 — small-ceiling watch items (bundle opportunistically, don't schedule dedicated time)

**2.1 `clear_slide_edge_cells`** (~1.2% of wall, [19 #2](./19-open-backlog.md)) — only worth
touching if bubble-closing work resumes generally; not worth a dedicated round on its own.

**2.2 `detect_all_dots` +22.3% regression cause** ([19 #3](./19-open-backlog.md), [DISCOVERED #14](./DISCOVERED-NOT-IMPLEMENTED.md)) —
zero wall-clock payoff (1.013x ceiling), but cheap to settle: rerun `detect_all_dots` over already-saved
instance masks under both the pre- and post-round-3 dependency sets. Worth doing only to close the
open question, not for speed.

### Tier 3 — governance / escalation (not engineering work)

**3.1 Cellpose 4.0.8→4.2.1.1 checkpoint clinical sign-off** ([19 §2](./19-open-backlog.md), [DISCOVERED #39](./DISCOVERED-NOT-IMPLEMENTED.md))
This blocks nothing technically but is the most consequential open item in the whole backlog:
every performance win since round 3 (including everything in Tier 0/1 above) rides on an
unvalidated model swap. This is not something to implement — it needs to be raised with whoever
owns pathologist/clinical review, flagged explicitly as pending since round 3. Recommend
surfacing this to the user/team directly rather than scheduling it as engineering work.

### Tier 4 — documentation ↔ code drift (independent, batch together, low risk)

All seven items in §1c are cheap, independent, single-file edits with no perf or correctness risk.
Batch them into one pass:
1. Rename `generate_ihc_core_mask`'s `ihc_tile_path: Path` param to `ihc_image: Union[np.ndarray, Path]` (#32).
2. Strip the dead `docs/sdd-elastic-dish-matching.md` reference from `m3_elastic_matching.py`'s docstring (#33).
3. Strip the dead `docs/dish_dot_detection_spec.md` reference from `config.py`/`config_example.py` comments (#34).
4. Update or retire `docs/algo/elastic_matching_v3_explainer.html` to describe the current
   cell-centric + overlap-priority + reach algorithm, and correct the `dish_elastic_expand_factor`
   deprecation note (#35).
5. Add a test asserting `config.py` and `config_example.py` stay in parity (e.g. same keys via
   `compute_config_hash()` structure) (#36).
6. Start a `backend/algorithms/hybrid/` correctness test suite with `m0_stitch` (pure data
   reorganization, no model — already flagged as the best first candidate) (#37).
7. Re-run `codegraph_files` vs. `git ls-files` against the current `backend/algorithms/hybrid/`
   path to refresh the phantom-file list (#38).

### Tier 5 — gated on Chain A, revisit only after Tier 0 closes

- **Cross-tile multiprocessing ship decision** ([19 #1](./19-open-backlog.md), [DISCOVERED #1](./DISCOVERED-NOT-IMPLEMENTED.md)) — ship `workers=4` as
  a production option once 0.1–0.4 are closed and the real-scale numbers confirm the crop-based
  projection.
- **A1 — re-tune `workers` against real full-WSI density** ([DISCOVERED #30](./DISCOVERED-NOT-IMPLEMENTED.md)) — only meaningful
  once 0.1 produces a real tissue/background distribution over the whole slide, not a crop.

### Tier 6 — needs a product decision before it's worth sizing

- **Multi-request / concurrent-job load test (Phase E)** ([19 #8](./19-open-backlog.md), [DISCOVERED #15](./DISCOVERED-NOT-IMPLEMENTED.md)) — before
  spending effort here, confirm concurrent-slide serving is actually part of the deployment plan;
  it never has been measured because this was never confirmed as a requirement.
- **Persistent worker pool for the API path** ([DISCOVERED #41](./DISCOVERED-NOT-IMPLEMENTED.md)) — gated on the item above; would
  require redesigning the `gc.freeze()`/`unfreeze` contract per-call, never done.

### Tier 7 — open in principle, not recommended to schedule

- **`detect_all_dots` whole-tile vectorization** ([DISCOVERED #26](./DISCOVERED-NOT-IMPLEMENTED.md)) — highest ceiling of the
  remaining `detect_all_dots` work but also highest correctness risk, and the BG arm this stage
  sits on already has 47–53% slack (per `bottleneck-list.md` ②) — there is no wall-clock payoff
  until MAIN sheds ~50%. Don't pick this up before that precondition changes.
- **Cellpose kernel-internals retargeting** (`_from_device`, `_quantile`) ([DISCOVERED #31](./DISCOVERED-NOT-IMPLEMENTED.md)) — correctly
  never attempted; ceiling is small (~1.118x for the whole class of Cellpose-internal fixes) and
  the targets are pinned third-party internals.

### Ongoing process discipline (no scheduling needed, just don't drop it)

**`pip freeze` with every future measurement round** ([19 #9](./19-open-backlog.md), [DISCOVERED #16](./DISCOVERED-NOT-IMPLEMENTED.md)) —
already adopted since round 3; keep doing it. Round 2's missing snapshot can't be fixed
retroactively.

---

## 4. Choose — final priority queue and explicit exclusions

### 4.1 Recommended order of operations

1. Tier 0.2–0.4 (OOM fix attempt, checkpointing, RLIMIT_NOFILE guard) — hardening before the
   expensive validation run consumes them.
2. Tier 1.3 (per-worker timing) — instrument before generating data you'll want to break down.
3. Tier 0.1 (full real-WSI validation) — the single highest-leverage item in the whole backlog;
   everything in Tier 5 depends on it.
4. Tier 1.1 (Phase D cheap knobs) — independent of the above, cheap, no dependency; do in
   parallel with 1–3 if resourcing allows.
5. Tier 5 (ship decision, A1 re-tune) — immediately after Tier 0 closes.
6. Tier 4 (documentation drift) — batch anytime, doesn't block or get blocked by anything.
7. Tier 3.1 (clinical sign-off) — raise with the user now; don't wait for the rest of this
   sequence, since it has the longest external lead time and blocks nothing else technically but
   is the largest standing risk.
8. Tier 2, Tier 6, Tier 7 — opportunistic / conditional, as scoped above.

### 4.2 Explicitly excluded — do not re-propose without new evidence

Per the quickref's own discipline, these are recorded here so this plan doesn't imply they're
still live:

| Item | Why excluded |
|---|---|
| Cross-tile Cellpose/UNet++ batching | Measured negative at every group size incl. G=16; VRAM cost scales linearly | 
| CUDA MPS | Flat end-to-end — pipeline no longer serialization-limited at its knee |
| CPU-back-end-only process pool (Candidate A) | +2.8% slower single-process, flat under multiprocessing |
| Fork-based model reuse (Candidate E) | Architecturally unsafe — CUDA contexts aren't fork-safe |
| CUDA-stream / pipeline-depth-2 bubble redesign | Ceiling ≤1.065x, not worth the ordering/sync risk |
| CUDA graph capture / vectorize Cellpose internals | ~1.118x ceiling, requires patching pinned third-party code |
| GPU-side tile/transform loading | 1.012x ceiling, no existing CPU→GPU pipeline to move |
| Candidates B/D/E (`detect_all_dots`/`enlarge_cell_instances`/debug encode → GPU) | 1.00x ceiling at real composition (BG arm has slack); CuPy also can't run on this host today (numpy ABI conflict) |
| Candidate F (background-tile placeholder dedup) | Zero wall-clock payoff; it's a storage decision (~157 GB/slide), not a speed one — revisit only if someone decides storage matters |
| Candidate G (`mkdir` hoist) | Reverted — 0.056% of wall, ablation read slightly negative; patch preserved for one-edit revival only if storage backend moves to network filesystem |
| `torch.compile` on UNet++, CUDA graph capture on UNet++, DALI/TensorRT | Never sized, but Amdahl-capped at ~0.4% of wall before they start — not worth sizing |

### 4.3 What this plan deliberately does not include

Anything already shipped and adopted (`gc.freeze()`, precut streaming, CPU-prep relocation off
MAIN, `cellpose_batch_size` wiring, `dot_detect_n_jobs=1`, the core multiprocessing machinery) —
see `measurement/bottleneck-list.md` for what landed. This document is only the remainder.
