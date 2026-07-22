# Open backlog — everything not done or only partially done

> Single place to look for "what's still owed" across the whole `docs/` tree, so it doesn't
> have to be reconstructed by reading 18+ hybrid-pipeline docs and the UI handoff packet
> every time. Compiled 2026-07-22 by reading every file under `docs/hybrid-pipeline/` and
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
> if the two drift apart, `19-open-backlog.md` wins.

Full current numbers and the arm model live in
[`hybrid-pipeline/measurement/bottleneck-list.md`](hybrid-pipeline/measurement/bottleneck-list.md)
("第 4 輪後重新排序的優先順序" section). Summary of what's actually open:

| # | Item | Status | Ceiling | Why it's still open |
|---|---|---|---|---|
| 1 | **Cross-tile multiprocessing** | **Not built.** Only remaining lever with a real ceiling. | ~1.23x–1.7x | Blocked by fork-under-CUDA (3 GPU models share one CUDA context in the main process — see `backend/algorithms/hybrid/CLAUDE.md`). Needs per-process model reload (VRAM 2.79 GB × N, init 2.4 s × N) or a multi-context design. Highest correctness risk of anything left. Scope only after re-measuring the residual idle_frac post round-4. See [`18-gpu-starvation-prerequisites-implementation.md`](hybrid-pipeline/18-gpu-starvation-prerequisites-implementation.md) §6.2. |
| 2 | **`clear_slide_edge_cells`** — last CPU glue between M2/M3b forwards | Watch item, not actioned | 1.2% of wall today | It's the only thing left in the M2→M3b device-idle gap after item ⑧ moved out (gap closed 10.06 s → 1.66 s). Too small alone to justify a redesign; natural next candidate *if* someone revisits bubble-closing. See doc 18 §5 follow-up #5. |
| 3 | **Isolate the `detect_all_dots` +22.3% regression (⑨)** | Optional, not done | 1.013x (no wall-clock payoff) | Leading hypothesis is the round-3 numpy/scikit-image/opencv downgrade, competing with retrained-checkpoint cell-geometry change — not isolated. Matters only because it eats the shrinking BG-arm margin, not because it can move wall today. Cheap to settle: rerun `detect_all_dots` over saved instance masks under both dependency sets. |
| 4 | **CUDA-stream / pipeline-depth-2 bubble redesign** | **Closed, do not reopen without new evidence** | ≤1.065x after item ⑧ landed | Sized with `torch.cuda.Event` instrumentation (doc 18 §3) and found not worth the ordering/thread-safety/CUDA-stream-sync risk. Reopen only if the intra-forward launch-bound idle (the larger half of device idle) is ever fixed upstream. |
| 5 | **CUDA graph capture / vectorize Cellpose's internal Python loops** (`_extend_centers_gpu`, `get_masks_torch`, `steps_interp`, `get_rel_pos` — already GPU-resident, kernel-launch-bound) | **Stop-lossed backlog** | ~1.23x even at zero | Requires patching pinned third-party `cellpose==4.0.8`/`segment_anything` internals (round 3 changed the Cellpose version but this trace predates it — needs re-confirming against 4.2.1.1). Only `fill_holes_and_remove_small_masks` (`cellpose/utils.py`) has no GPU path at all — everything else is a launch-overhead problem, not a device problem. See [`gil-contention-diag.md`](hybrid-pipeline/measurement/gil-contention-diag.md) "追加深挖". |
| 6 | **GPU-side tile/transform loading** | **Stopped out, not backlog** | 1.012x | `B2r_tile_read` is 1.22% of wall post-precut-streaming, and there's no existing CPU→GPU transform pipeline to move — would be new construction for a ceiling that doesn't justify it. |
| 7 | **Full real-WSI-scale validation** | **Never done, any round** | n/a | Every round (1–4) measures only 25/121/441-tile crops of a real WSI, never a complete 156k×134k slide end-to-end. `09-measurement-analysis-plan.md` §3.6 explicitly asked for this ("需要規劃一次完整 WSI 的跑批") and it has not happened by round 4. All full-WSI hour estimates (currently ~10.5h) are linear extrapolations from crops, explicitly flagged as upper bounds. |
| 8 | **Multi-request / concurrent-job behavior of the API layer** | **Never measured** | Flagged, not sized | `bottleneck-list.md` ⑦ notes "concurrent analysis requests each hold a threadpool worker but serialize on the single GPU/CUDA context" as a *future* risk, not a measured one. No load test exists. Relevant once the UI (Phase 3, already shipping) sees real multi-user usage. |
| 9 | **Record `pip freeze` with every future measurement round** | Partially adopted | process, not perf | Rounds 3 and 4 do this; round 2 (`_metrics_current/`) still has no snapshot and can't be fixed retroactively — it's why ⑨'s cause can't be cleanly attributed. Keep doing it going forward. |

**Already closed, don't re-litigate** (kept here only so nobody re-proposes them — full ablation evidence in the linked docs): `detect_all_dots` → process backend (negative, `gil-contention-diag.md`), `gc.collect` relocation to background thread (negative, same doc), fixed-N `gc.collect` batching (built, measured, added nothing beyond `gc.freeze()`, deleted — [`16-gc-collect-frequency-result.md`](hybrid-pipeline/16-gc-collect-frequency-result.md)), `cellpose_batch_size` sweep at the current 1024px tile size (wired but flat — doc 18 §6.1).

## 2. Correctness / clinical validation — blocking, not a performance task

| Item | Status | Source |
|---|---|---|
| **Round-3 Cellpose checkpoint retrain needs pathologist/clinical sign-off** | **Pending, unresolved as of round 4.** Cell counts shifted +1.8–2.5% and one tile flipped success→skipped between round 2 and round 3; this is a segmentation-quality change, not noise. All performance wins from round 3 onward ride on top of this unvalidated model swap. | [`13-next-optimization-plan.md`](hybrid-pipeline/13-next-optimization-plan.md) §3, `bottleneck-list.md` round-3 section |
| **UI Phase 1–3 shipped while the algorithm is still mid-iteration** | Not a defect, but explicitly noted as a departure from the original two-condition gate ("physician validation passed" + "algorithm in maintenance mode") — see [`UI/07-phase-roadmap.md`](UI/07-phase-roadmap.md) "啟動 Phase 1 的條件". Worth knowing before assuming the pipeline's I/O contract is stable. | `UI/07-phase-roadmap.md` |

## 3. Documentation ↔ code drift not yet fixed

These are real, verified gaps between what a doc says and what the code does — checked against
the current repo, not assumed from the docs. Fixed drift from this pass (path migration to
`backend/algorithms/hybrid/`, `config_example.py`'s missing tail, phase-roadmap staleness) has
already been corrected in-place; what's below is what's still open.

| Item | Detail | Source |
|---|---|---|
| **`generate_ihc_core_mask` parameter named `ihc_tile_path: Path` but always receives an ndarray** | Not a bug (the callee accepts `Union[ndarray, Path, str]`, caller has `# pyright: ignore`), but misleading to a new reader. Cosmetic rename (`ihc_image`, `Union[np.ndarray, Path]`) never landed. | `backend/algorithms/hybrid/m1_overlay.py:54`; [`07-gotchas-appendix.md`](hybrid-pipeline/07-gotchas-appendix.md) G5 |
| **`docs/sdd-elastic-dish-matching.md` referenced but never exists** | `m3_elastic_matching.py`'s own docstring points at it. Confirmed missing (`find` turns up nothing). Not worth creating — the recommendation is to strip the dead reference from the docstring, which hasn't happened. | `07-gotchas-appendix.md` G4 |
| **`docs/dish_dot_detection_spec.md` referenced but never exists** | Same pattern, referenced from `config.py`/`config_example.py` comments (`see docs/dish_dot_detection_spec.md v0.2`). Confirmed missing. | `07-gotchas-appendix.md` G4 |
| **`docs/algo/elastic_matching_v3_explainer.html` describes the wrong matching algorithm** | Describes a "nucleus-centric" expansion variant; current code (`m3_elastic_matching.py`) is "cell-centric + overlap-priority + reach". The HTML was never updated after the rewrite. Same for `dish_elastic_expand_factor`, which the HTML marks deprecated but the code still uses to compute reach radius. Treat `m3_elastic_matching.py` as the sole source of truth until someone updates or retires the HTML. | `04-optimization-roadmap.md` "為什麼某些嘗試被棄用"; `07-gotchas-appendix.md` G4 |
| **`docs/algo/frontend_backend_split_architecture.html`** (added 2026-07-21, "API 傳圖片還是傳路徑") | Orphaned — not linked from any nav table (`hybrid-pipeline/README.md`, `UI/README.md`). Content overlaps `UI/04-guardrails-red-lines.md`'s "boundary is always file path + JSON" guardrail; worth either linking it in or folding its content into that guardrail doc. | this pass's own file survey |
| **`Config` dataclass has no automated test guarding `config.py`/`config_example.py` parity** | The tail-mismatch bug (G2, now fixed) was caught by manual reading, not a test. Nothing currently prevents the two files drifting apart again the same way. | inferred from `07-gotchas-appendix.md` G2's fix history |

## 4. UI — not started or partially started

Full phase table: [`UI/07-phase-roadmap.md`](UI/07-phase-roadmap.md) (status corrected in this
pass — Phases 1–3 are done, table previously said "not started" for all of them).

| Item | Status |
|---|---|
| **Phase 4 — ROI drawing + live parameter tuning** | Not started. Depends on the coordinate-conversion contract in [`05-dataflow-api-contract.md`](UI/05-dataflow-api-contract.md). |
| **Phase 5 — desktop packaging (pywebview + PyInstaller)** | Not started. `backend/launcher.py` (the planned pywebview entry point) does not exist yet; currently dev-server only (`uvicorn` + Vite dev server). |
| **Automatic viewer-copy generation after a pipeline run** | Currently a manual one-off conversion (`scripts/make_viewer_copy.py` run by hand). "Pipeline finishes → auto-produce a viewer copy" is explicitly pending a PM decision on which stage owns it. | [`10-viewer-ui-implementation.md`](UI/10-viewer-ui-implementation.md) §4 |
| **8 open decisions blocking Phase 4/5 scoping** | Physician OS distribution (Windows/macOS split); whether physician machines have a GPU (decides CPU-only fallback need); image format range (.svs/.ndpi/.tiff shares); multi-case/patient-list management; UI language (zh/en/bilingual); classification of `module3_roi_evaluation.py` in the Phase-1 migration map; whether an in-UI model-(re)training GUI is in scope (default: no); `ruff` vs `black` for Python formatting. | [`08-pitfalls-open-decisions.md`](UI/08-pitfalls-open-decisions.md) |
| **`backend/tests/` doesn't cover the hybrid pipeline** | Real pytest tests exist for chunked upload / alignment strips / resume / OpenSlide, but `backend/algorithms/hybrid/` still has zero test files (only `scripts/verify_gc_freeze.py`, a standalone invariant guard, not a pytest suite covering pipeline correctness). | verified this pass; [`hybrid-pipeline/05-dev-testing-guide.md`](hybrid-pipeline/05-dev-testing-guide.md) |

## 5. Where to look next, by role

- **Picking up hybrid-pipeline perf work** → start at §1 above, then
  [`hybrid-pipeline/measurement/bottleneck-list.md`](hybrid-pipeline/measurement/bottleneck-list.md)
  for the full numeric record, then the specific doc 10–18 for the item you're taking.
- **Picking up UI work** → §4 above, then [`UI/07-phase-roadmap.md`](UI/07-phase-roadmap.md).
- **Doing a documentation pass** → §3 above lists the known-stale cross-references; re-run the
  same verification method used to build this file (`grep` the referenced path/symbol, confirm
  with `git ls-files`/`find`, don't trust a doc's claim about another file without checking).
