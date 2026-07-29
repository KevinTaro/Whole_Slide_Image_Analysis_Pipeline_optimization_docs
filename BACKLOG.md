# Open backlog — everything not done or only partially done

> Single place to look for "what's still owed" across the whole `docs/` tree, so it doesn't
> have to be reconstructed by reading 27+ hybrid-pipeline docs and the UI handoff packet
> every time. Compiled 2026-07-22, last updated 2026-07-27 for round 8
> ([`hybrid-pipeline/27-remaining-work-implementation.md`](hybrid-pipeline/27-remaining-work-implementation.md)):
> §1 item 7 (full real-WSI validation) is now **DONE and closed** — first complete slide this
> project has ever run, both `workers=1` and `workers=4`; item 1 (cross-tile multiprocessing)'s
> production gate is **satisfied**, ship decision now turns on a VRAM caveat (93.3% of the card at
> `workers=4`), not a speed one; item 1b (Phase D stitch)'s ceiling revised up ~3x after its cheap
> `tiffsave`-knob line closed negative; item 1c (partial resume) is built. Two new items (6, 6b)
> record that the full-slide run found `gc.collect` and tile read regressed back to their
> pre-stop-loss cost at production scale — both were measured "solved" on crops that don't hold at
> real scale. Previously updated 2026-07-26 for round 7 (§1 item 7 —
the slide's tissue fraction is now measured over every grid cell, replacing round 6's
brightness estimate; item 7b — new evidence at `workers=4`; new item 1b; four round-7
stop-losses), and re-synced 2026-07-26 after a documentation-only pass split the hybrid-pipeline
`measurement/` docs: `bottleneck-list.md` and `current-status-comparison.md` were compacted to
current-state-only (round-by-round narrative moved to `*-history.md` companions), and
[`hybrid-pipeline/DISCOVERED-NOT-IMPLEMENTED.md`](hybrid-pipeline/DISCOVERED-NOT-IMPLEMENTED.md)
was added as a full one-time audit of every candidate ever discovered but not shipped, by
> reading every file under `docs/hybrid-pipeline/` and
> `docs/UI/` (plus spot-checking the referenced code) — not from memory. Each item links back
> to its source of truth; **when you actually pick one up, re-read that source doc first**,
> this list only summarizes.
>
> Update discipline: when an item here is finished, move it out (or mark it done with a
> one-line pointer to the record) instead of leaving stale entries — this file is only useful
> if it stays accurate.

## 1. Hybrid pipeline — performance, ranked by what the measurement record says is left

> **Canonical, more detailed version of this section now lives in
> [`hybrid-pipeline/19-open-backlog.md`](hybrid-pipeline/19-open-backlog.md)** (hybrid-pipeline-only
> scope, references only documents inside that folder). Keep this section as a short mirror;
> if the two drift apart, `19-open-backlog.md` wins. For the full one-time audit of every
> candidate this project has discovered but never shipped (including minor items this short
> mirror doesn't carry), see
> [`hybrid-pipeline/DISCOVERED-NOT-IMPLEMENTED.md`](hybrid-pipeline/DISCOVERED-NOT-IMPLEMENTED.md).

Full current numbers and per-item status live in
[`hybrid-pipeline/measurement/bottleneck-list.md`](hybrid-pipeline/measurement/bottleneck-list.md)'s
"All bottlenecks — status and result" table (compact, current-state-only). The round-by-round
history that used to live there is now in
[`hybrid-pipeline/measurement/bottleneck-list-history.md`](hybrid-pipeline/measurement/bottleneck-list-history.md).
Summary of what's actually open:

| # | Item | Status | Ceiling | Why it's still open |
|---|---|---|---|---|
| 0 | **`detect_all_dots` joblib fan-out removed (`dot_detect_n_jobs=1`)** | **BUILT, MEASURED, ADOPTED (round 6). Ships on today's production default — no gate needed.** | **1.60x measured** at `workers=1` (large 484.7 → 302.7 s); ~0% at `workers=6` | The `Parallel(n_jobs=-1, prefer='threads')` fan-out over ~35 tiny per-cell tasks was **slower than serial at every process count** (2.77x at 20 threads, identical dot counts), and its surplus threads starved the MAIN arm of the GIL — removing them made the **GPU forwards 43.4% faster with no GPU code change**. Record: [`23-next-optimization-cycle-implementation.md`](hybrid-pipeline/23-next-optimization-cycle-implementation.md) §4. |
| 1 | **Cross-tile multiprocessing** | ✅ **SHIPPED (round 8)** — production gate satisfied | **2.216x measured** on the real 27,565-tile full slide (in the 2.06x–2.17x predicted band) | Built as `run_batch(..., workers=N)`: `spawn` workers, per-process model reload, dynamic work queue, global renumbering still a single parent-side pass. Correctness veto passed at every anchor tested, incl. the real full slide (356,255 vs 356,221 rows, −0.01%). Round 6 revised the recommendation down to `workers=4` (unattended) / `workers=5` (restart-tolerant) after a `workers≥6` OOM rate (item 7b). **Round 8 ran the full-slide validation that gated production** ([`27-...`](hybrid-pipeline/27-remaining-work-implementation.md) §6): `workers=1` 3.82h, `workers=4` 1.73h. **Recommendation: ship `workers=4`**, but it peaked at **93.3% of the card (30,439/32,607 MB, ~2.2 GB headroom)** — treat 32 GB VRAM as a hard floor, add no GPU library inside the worker pool without re-measuring, and keep `workers≥5` off the table until item 7b's allocator balloon is root-caused. |
| 1b | **Phase D slide stitch (`_stitch_overlay_slide`)** — the only fully serial block left | **Cheap knobs CLOSED, negative (round 8); ceiling revised up ~3x, GPU port reopened** | ~~1.036x/1.078x~~ → **measured 8.6% of wall at `workers=1`, 19.3% at `workers=4`** on the real slide → ceiling **1.094x/1.239x** | Round 7 measured 322.7 s at the real 16.22-gigapixel grid via a synthetic probe (1.8× the previous extrapolation, superlinear). Round 8 ablated all 13 `tiffsave` knobs doc 25 named: all dead (tile size monotonically worse, pyramid depth a no-op) except `zstd` level 1 (1.2331x, 13.8% smaller) — **vetoed on correctness, QuPath/BioFormats cannot open it**. The real full-slide stitch (not the synthetic probe) took **1,185.4 s — 3.7× the probe's figure** — because real tiles don't compress/cache as well as the probe's replicated pool. GPU port (nvImageCodec, 19.2× faster lossless-LZW encode, but no pyramid/BigTIFF — real engineering) is now the largest single remaining lever in the pipeline. See [`27-...`](hybrid-pipeline/27-remaining-work-implementation.md) §3, §6.4, §6.6. |
| 1c | **No partial-resume/checkpointing for `run_batch`** | ✅ **BUILT (round 8), opt-in** | reliability, not speed | Fail-fast discards the *whole* batch on any single-tile failure — an OOM at tile 25,000 of 27,565 costs the entire run. `run_batch(checkpoint=True)` pickles each completed tile's `owned` result list (tmp+rename, config-hash-guarded); resumed output byte-identical to a cold run. Fail-fast is unchanged. Default off; on via `--resume` / always in `full_wsi_validate.py`. See [`21-...`](hybrid-pipeline/21-cross-tile-multiprocessing-implementation.md) §10 follow-up #7, [`27-...`](hybrid-pipeline/27-remaining-work-implementation.md) §2. |
| 2 | **`clear_slide_edge_cells`** — last CPU glue between M2/M3b forwards | Watch item, not actioned | 1.2% of wall today | It's the only thing left in the M2→M3b device-idle gap after item ⑧ moved out (gap closed 10.06 s → 1.66 s). Too small alone to justify a redesign; natural next candidate *if* someone revisits bubble-closing. See doc 18 §5 follow-up #5. |
| 6 | **GPU-side tile/transform loading (`B2r_tile_read`)** | 🟢 **REOPENED (round 8)** — stop-loss was measured on a crop that doesn't hold at production scale | ~~1.012x~~ → **17.2% of wall on the real full slide** (2,368.5 s) | Stopped out at 1.22% of wall on a 441-tile crop (round 4). The ~49 GB precut scratch no longer fits page cache at full-slide scale, so reads that were free on a crop become real disk I/O. Ceiling needs re-deriving, not re-assuming. See [`27-...`](hybrid-pipeline/27-remaining-work-implementation.md) §6.4. |
| 6b | **`gc.collect` — the `gc.freeze()` win does not survive to full-slide scale** | 🟢 **NEW (round 8), highest-value open item** | ~~1.083x, cost ~0~~ → **16.1% of wall on the real full slide** (2,218.4 s, back to the pre-freeze 80.5 ms/call) | `run_batch` accumulates `per_tile_owned` (356,255 `CellAnalysisResult` dataclasses by the end of a slide), created *after* `gc.freeze()`, fully tracked, rescanned on every one of 27,565 collections — invisible on a 441-tile crop (~6,000 objects). Plausible cheap fix (freeze again periodically, or keep accumulating results out of GC's reach) not yet built. See [`27-...`](hybrid-pipeline/27-remaining-work-implementation.md) §6.4. |
| 3 | **Isolate the `detect_all_dots` +22.3% regression (⑨)** | Optional, not done | 1.013x (no wall-clock payoff) | Leading hypothesis is the round-3 numpy/scikit-image/opencv downgrade, competing with retrained-checkpoint cell-geometry change — not isolated. Matters only because it eats the shrinking BG-arm margin, not because it can move wall today. Cheap to settle: rerun `detect_all_dots` over saved instance masks under both dependency sets. |
| 4 | **CUDA-stream / pipeline-depth-2 bubble redesign** | **Closed, do not reopen without new evidence** | ≤1.065x after item ⑧ landed | Sized with `torch.cuda.Event` instrumentation (doc 18 §3) and found not worth the ordering/thread-safety/CUDA-stream-sync risk. Reopen only if the intra-forward launch-bound idle (the larger half of device idle) is ever fixed upstream. |
| 5 | **CUDA graph capture / vectorize Cellpose's internal Python loops** (`_extend_centers_gpu`, `get_masks_torch`, `steps_interp`, `get_rel_pos` — already GPU-resident, kernel-launch-bound) | **Stop-lossed backlog — re-confirmed round 6** | **~1.118x** (was ~1.23x at 4.0.8) | Requires patching pinned third-party `cellpose`/`segment_anything` internals. **Re-traced against the running 4.2.1.1/`cpdino` ([`23-...`](hybrid-pipeline/23-next-optimization-cycle-implementation.md) §7): `get_rel_pos` is gone entirely (SAM ViT backbone no longer used), the other four survive at shifted line numbers, and their combined critical-arm share is 10.58% → 1.118x. If ever reopened, the real targets are now `_from_device` (24.2%) and `_quantile` (10.1%), not the four named loops. Only `fill_holes_and_remove_small_masks` (`cellpose/utils.py`) has no GPU path at all — everything else is a launch-overhead problem, not a device problem. See [`gil-contention-diag.md`](hybrid-pipeline/measurement/gil-contention-diag.md) "追加深挖". |
| 7 | **Full real-WSI-scale validation** | ✅ **DONE (round 8) — CLOSED.** First complete slide this project has ever run, both `workers=1` and `workers=4` | n/a | Every round (1–7) measured only 25/121/441/576-tile crops of a real WSI, never a complete slide end-to-end. Round 8 ran it: **`workers=1` 13,762 s = 3.82h (+47% vs the 2.6h projection), `workers=4` 6,211 s = 1.73h (+38% vs 1.25h), measured speedup 2.216x** (inside round 7's predicted 2.06x–2.17x band), correctness veto passed (356,255 vs 356,221 `report.csv` rows, −0.01%). The composition prediction was right to within one tile (15,385 vs predicted 15,386 background tiles), so the overrun is entirely per-tile-rate, not composition — traced to three numbers this project had already measured and closed on crops that don't hold at production scale: `gc.collect` (16.1% of wall, not ~0), tile read (17.2%, not 1.22%), Phase D stitch (8.6%/19.3%, not 3.5%/7.3%). Peak RSS **61.13–61.67 GB** — a new, previously-unstated host requirement (no prior round exceeded ~4 GB). Preflight also found the registration stage emits a **different canvas per modality** (HER2 141818×114366 vs DISH 141658×114415), which `PrecutStream` fail-fasts on; `scripts/full_wsi_validate.py --conform` crops to the intersection (99.86% retained). The `RLIMIT_NOFILE` sub-item below is closed. Full record: [`27-...`](hybrid-pipeline/27-remaining-work-implementation.md) §6. |
| 7a | **`_stitch_overlay_slide` opens all 27,565 tiles at once; no `RLIMIT_NOFILE` check** | ✅ **CLOSED (round 8)** | Would have failed on a host with the common 1,024 limit, after the whole slide had been analysed | `_ensure_nofile_limit()` raises the soft limit itself when the hard limit permits, else fails loudly before opening anything; 7 tests. Exercised for real on the full-slide run (12,027 open fds observed mid-stitch) and passed silently. See [`27-...`](hybrid-pipeline/27-remaining-work-implementation.md) §1. |
| 7b | **`workers≥6` allocator balloon: one worker takes exactly 24.76 GiB and OOM-kills the batch** | 🟢 **OPEN — candidate fix tried (round 8), did not clear it** | reliability, not speed | 2 failures in 6 `workers=6` runs (round 5b saw 0/6). Same byte-identical 24.76 GiB balloon both times, different victim tiles — the victim looks like fragmentation, the balloon does not. **Round 8**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` swept (12 runs, interleaved) — **does not reduce peak VRAM** (median 24,040 vs 22,968 MB control — evidence *against* the fragmentation hypothesis), 0-in-6 vs 1-in-6 OOM is statistically indistinguishable (p≈1.0), and costs +2.0% wall. Default stays off. Also found: `workers=6` is not faster than `workers=4` on this crop (65.46 vs 65.55 s) — removes the only reason to raise the cap. Next step is root-causing the balloon directly, not more allocator flags. On the real full slide, `workers=4` peaked at **93.3% of the card** (~2.2 GB headroom), worse than round 7's crop-based ~6 GB estimate. See [`23-...`](hybrid-pipeline/23-next-optimization-cycle-implementation.md) §4.6/§6, [`27-...`](hybrid-pipeline/27-remaining-work-implementation.md) §5. |
| 8 | **Multi-request / concurrent-job behavior of the API layer** | **Never measured** | Flagged, not sized | `bottleneck-list.md` ⑦ notes "concurrent analysis requests each hold a threadpool worker but serialize on the single GPU/CUDA context" as a *future* risk, not a measured one. No load test exists. Relevant once the UI (Phase 3, already shipping) sees real multi-user usage. |
| 9 | **Record `pip freeze` with every future measurement round** | Partially adopted | process, not perf | Rounds 3–5 do this; round 2 (`_metrics_current/`) still has no snapshot and can't be fixed retroactively — it's why ⑨'s cause can't be cleanly attributed. Keep doing it going forward. |

**Already closed, don't re-litigate** (kept here only so nobody re-proposes them — full ablation evidence in the linked docs): `detect_all_dots` → process backend (negative, `gil-contention-diag.md`), `gc.collect` relocation to background thread (negative, same doc), fixed-N `gc.collect` batching (built, measured, added nothing beyond `gc.freeze()`, deleted — [`16-gc-collect-frequency-result.md`](hybrid-pipeline/16-gc-collect-frequency-result.md)), `cellpose_batch_size` sweep at the current 1024px tile size (wired but flat — doc 18 §6.1).

**Closed in round 7** ([`25-...`](hybrid-pipeline/25-gpu-encode-decode-loop-acceleration-implementation.md)): **`mkdir` hoist out of `_save_tile_array`** — built, ablated at `workers=1` and `4`, **reverted**: 0.056% of wall, end-to-end effect inside the noise, contention real but ~2 s per slide (re-measure if output ever moves to a network filesystem). **Background-tile placeholder writes** — measured (24 ms/tile, ~6 min per slide) and not built: it is BG-arm work, and the BG arm finishes with 47–53% of its capacity unused, so removing it moves wall by zero; the `os.link` alternative is 272–407× cheaper but the real case for it is **storage** (~157 GB of constant TIFF per slide). **`detect_all_dots` / `enlarge_cell_instances` / per-tile debug encode → GPU** — closed on measurement: all BG-arm, 1.00× wall ceiling at real composition. **Cross-tile Cellpose batching at G=16** — reopened by doc 24, closed again: +6.6% / +5.9% slower than G=1 at 15.8 GB peak. **GPU codec ecosystem (nvTIFF / cuCIM / CuPy)** — environment gate failed for everything except nvImageCodec: nvTIFF ships no Python module, cuCIM cannot write, and CuPy cannot run on this host (v14 needs numpy≥2 against the project's `numpy<2` pin; v13 cannot JIT without a CUDA toolkit).

**Closed in round 8** ([`27-...`](hybrid-pipeline/27-remaining-work-implementation.md)): **Phase D `tiffsave` knob ablation** — 13 configs, all dead except `zstd` (1.2331x, vetoed on correctness — unreadable by QuPath/BioFormats); see item 1b above. **CUDA allocator config vs. `workers≥6` OOM** — knob built and swept, did not clear the defect; see item 7b above. All seven documentation↔code drift items in §3 below except the orphaned-HTML one. See §0/§1 for what round 8 *reopened*: `gc.collect` (item 6b) and tile read (item 6), both previously "closed" on crop-scale measurements that don't hold at production scale — a reminder that "closed" here means closed against the evidence available at the time, not closed forever.

## 2. Correctness / clinical validation — blocking, not a performance task

| Item | Status | Source |
|---|---|---|
| **Round-3 Cellpose checkpoint retrain needs pathologist/clinical sign-off** | **Pending, unresolved through round 7.** Cell counts shifted +1.8–2.5% and one tile flipped success→skipped between round 2 and round 3; this is a segmentation-quality change, not noise. All performance wins from round 3 onward — including cross-tile multiprocessing and the round-7 composition correction — ride on top of this unvalidated model swap. | [`13-next-optimization-plan.md`](hybrid-pipeline/13-next-optimization-plan.md) §3, `bottleneck-list-history.md` round-3 section |
| **UI Phase 1–3 shipped while the algorithm is still mid-iteration** | Not a defect, but explicitly noted as a departure from the original two-condition gate ("physician validation passed" + "algorithm in maintenance mode") — see [`UI/07-phase-roadmap.md`](UI/07-phase-roadmap.md) "啟動 Phase 1 的條件". Worth knowing before assuming the pipeline's I/O contract is stable. | `UI/07-phase-roadmap.md` |

## 3. Documentation ↔ code drift not yet fixed

These are real, verified gaps between what a doc says and what the code does — checked against
the current repo, not assumed from the docs. Fixed drift from this pass (path migration to
`backend/algorithms/hybrid/`, `config_example.py`'s missing tail, phase-roadmap staleness) has
already been corrected in-place; **round 8 additionally closed all six items below marked ✅** — see
[`hybrid-pipeline/27-remaining-work-implementation.md`](hybrid-pipeline/27-remaining-work-implementation.md)
§7; what's still open is the orphaned-HTML item.

| Item | Detail | Source |
|---|---|---|
| **`generate_ihc_core_mask` parameter named `ihc_tile_path: Path` but always receives an ndarray** | ✅ **CLOSED (round 8)** — renamed to `ihc_image: Union[np.ndarray, Path, str]`; the now-redundant `# pyright: ignore` at the call site removed too. | `backend/algorithms/hybrid/m1_overlay.py:54`; [`07-gotchas-appendix.md`](hybrid-pipeline/07-gotchas-appendix.md) G5 |
| **`docs/sdd-elastic-dish-matching.md` referenced but never exists** | ✅ **CLOSED (round 8)** — dead reference stripped from `m3_elastic_matching.py`'s docstring; a **second, unlisted instance** in `m3_module/m3_dot_detection.py` was found and stripped too. | `07-gotchas-appendix.md` G4 |
| **`docs/dish_dot_detection_spec.md` referenced but never exists** | ✅ **CLOSED (round 8)** — stripped from both `config.py` and `config_example.py`, replaced with a pointer to `m3_dot_kernels.py`/`m3_dot_detection.py`. | `07-gotchas-appendix.md` G4 |
| **`docs/algo/elastic_matching_v3_explainer.html` describes the wrong matching algorithm** | ✅ **CLOSED (round 8)** — updated to v4: Part A rewritten cell-centric + overlap-priority + reach, the obsolete "多核排除" outcome removed, and the parameter table corrected (it listed two params that don't exist and struck through `dish_elastic_expand_factor`, which the code still uses). | `04-optimization-roadmap.md` "為什麼某些嘗試被棄用"; `07-gotchas-appendix.md` G4 |
| **`docs/algo/frontend_backend_split_architecture.html`** (added 2026-07-21, "API 傳圖片還是傳路徑") | Still open — orphaned, not linked from any nav table (`hybrid-pipeline/README.md`, `UI/README.md`). Content overlaps `UI/04-guardrails-red-lines.md`'s "boundary is always file path + JSON" guardrail; worth either linking it in or folding its content into that guardrail doc. Not part of round 8's scope. | this pass's own file survey |
| **`Config` dataclass has no automated test guarding `config.py`/`config_example.py` parity** | ✅ **CLOSED (round 8)** — `tests/test_config_parity.py`, 6 cases over field names/order/types/non-site-local defaults plus the module tail G2 lost. | inferred from `07-gotchas-appendix.md` G2's fix history |

## 4. UI — not started or partially started

Full phase table: [`UI/07-phase-roadmap.md`](UI/07-phase-roadmap.md) (status corrected in this
pass — Phases 1–3 are done, table previously said "not started" for all of them).

| Item | Status |
|---|---|
| **Phase 4 — ROI drawing + live parameter tuning** | **Partially done (2026-07-28)** — ROI selection for the hybrid analysis is built and shipped (`HybridPanel.tsx` + `PrecutStream(region=...)` + `HybridTileIn.roi_*`); it takes the ROI from the viewer's current viewport, not a drawn rectangle, and live parameter tuning is still not started. Record: [`UI/13-numpy2-migration-and-analysis-ui.md`](UI/13-numpy2-migration-and-analysis-ui.md) §5. |
| **Analysis (hybrid) UI progress + cancellation** | **Open, blocked in the algorithm/framework layer (2026-07-28)** — the mockup asks for per-tile progress and a cancel button. `run_batch` reports nothing until it returns `{success, skipped}`, and FastAPI `BackgroundTasks` cannot be cancelled; both need a design decision, not just frontend work. See [`UI/13-numpy2-migration-and-analysis-ui.md`](UI/13-numpy2-migration-and-analysis-ui.md) §7. |
| **Phase 5 — desktop packaging (pywebview + PyInstaller)** | Not started. `backend/launcher.py` (the planned pywebview entry point) does not exist yet; currently dev-server only (`uvicorn` + Vite dev server). |
| **Automatic viewer-copy generation after a pipeline run** | Currently a manual one-off conversion (`scripts/make_viewer_copy.py` run by hand). "Pipeline finishes → auto-produce a viewer copy" is explicitly pending a PM decision on which stage owns it. | [`10-viewer-ui-implementation.md`](UI/10-viewer-ui-implementation.md) §4 |
| **8 open decisions blocking Phase 4/5 scoping** | Physician OS distribution (Windows/macOS split); whether physician machines have a GPU (decides CPU-only fallback need); image format range (.svs/.ndpi/.tiff shares); multi-case/patient-list management; UI language (zh/en/bilingual); classification of `module3_roi_evaluation.py` in the Phase-1 migration map; whether an in-UI model-(re)training GUI is in scope (default: no); `ruff` vs `black` for Python formatting. | [`08-pitfalls-open-decisions.md`](UI/08-pitfalls-open-decisions.md) |
| **`backend/tests/` doesn't cover the hybrid pipeline** | ✅ **CLOSED (round 8)** — `backend/algorithms/hybrid/`'s own `tests/` now has 48 automated tests: `test_m0_stitch.py` (21 cases, incl. a full-slide coverage assertion that core crops partition the slide exactly once), `test_config_parity.py` (6), `test_stitch_nofile_guard.py` (7), `test_run_batch_resume.py` (12), plus `scripts/verify_mp_failfast.py` (unchanged, still passes at `workers=1`/`3`). | [`hybrid-pipeline/27-remaining-work-implementation.md`](hybrid-pipeline/27-remaining-work-implementation.md) §7.3, §1, §2 |

## 5. Where to look next, by role

- **Picking up hybrid-pipeline perf work** → start at §1 above, then
  [`hybrid-pipeline/measurement/bottleneck-list.md`](hybrid-pipeline/measurement/bottleneck-list.md)
  for the current numeric record, then the specific doc 10–25 for the item you're taking. For the
  round-by-round "how we got here" or a candidate this short list doesn't track, see
  [`hybrid-pipeline/measurement/bottleneck-list-history.md`](hybrid-pipeline/measurement/bottleneck-list-history.md)
  and [`hybrid-pipeline/DISCOVERED-NOT-IMPLEMENTED.md`](hybrid-pipeline/DISCOVERED-NOT-IMPLEMENTED.md).
- **Picking up UI work** → §4 above, then [`UI/07-phase-roadmap.md`](UI/07-phase-roadmap.md).
- **Doing a documentation pass** → §3 above lists the known-stale cross-references; re-run the
  same verification method used to build this file (`grep` the referenced path/symbol, confirm
  with `git ls-files`/`find`, don't trust a doc's claim about another file without checking).
