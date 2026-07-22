# 19 — Open backlog: what's not done or only partially done

> Single place to look for "what's still owed" in this pipeline, so it doesn't have to be
> reconstructed by reading all of 01–18 + `measurement/*.md` every time. Compiled 2026-07-22
> by reading every document in this folder (`docs/hybrid-pipeline/`) and spot-checking the
> claims against the current code — not from memory. **Every reference below is to another
> document in this same folder.** When you pick an item up, re-read its source doc(s) first —
> this file only summarizes and ranks.
>
> Update discipline: when an item here ships or is stopped-out with new evidence, move it out
> of the "open" tables below (or into the "closed" list at the bottom with a one-line pointer)
> instead of leaving it stale — this file is only useful if it stays accurate. Follow the same
> Discover → Analyze → Plan → Choose discipline as every other doc here
> ([`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md))
> before touching any item: re-measure, don't assume a number from this list is still current.

## 1. Performance — ranked by what the measurement record says is left

Full current numbers, the two-arm model, and the per-round history live in
[`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md) (see its "第 4 輪後重新排序的優先順序" section, which this table summarizes and re-links). **Re-read that document's latest round before acting on any ceiling number quoted here** — it moves every round.

| # | Item | Status | Ceiling | Why it's still open |
|---|---|---|---|---|
| 1 | **Cross-tile multiprocessing** | **Not built.** Only remaining lever with a real ceiling. | ~1.23x–1.7x | Blocked by fork-under-CUDA (3 GPU models share one CUDA context in the main process — `CLAUDE.md` in this pipeline's source directory states this explicitly). Needs per-process model reload (VRAM 2.79 GB × N, init 2.4 s × N) or a multi-context design. Highest correctness risk of anything left. Scope only after re-measuring the residual idle_frac at the current HEAD. See [`18-gpu-starvation-prerequisites-implementation.md`](./18-gpu-starvation-prerequisites-implementation.md) §6.2, and the design constraints in [`17-gpu-starvation-prerequisites-plan.md`](./17-gpu-starvation-prerequisites-plan.md) §4.4. |
| 2 | **`clear_slide_edge_cells`** — last CPU glue left between the M2/M3b GPU forwards | Watch item, not actioned | ~1.2% of wall today | It's the only thing left in the M2→M3b device-idle gap after item ⑧ (CPU prep) moved off the MAIN arm — that move closed the gap from 10.06 s to 1.66 s. Too small alone to justify a redesign; natural next candidate *if* bubble-closing work resumes. See [`18-...-implementation.md`](./18-gpu-starvation-prerequisites-implementation.md) §5 follow-up #5 and §3's bubble map. |
| 3 | **Isolate the `detect_all_dots` +22.3% regression (⑨)** | Optional, not done | 1.013x (no wall-clock payoff) | Leading hypothesis is the round-3 numpy/scikit-image/opencv downgrade, competing with retrained-checkpoint cell-geometry change as the cause — not isolated. Matters only because it eats the shrinking BG-arm margin, not because it can move wall today. Cheap to settle: rerun `detect_all_dots` over saved instance masks under both dependency sets. See `measurement/bottleneck-list.md` item ⑨ and its round-4 note that the number already moved on its own once (thread-placement side effect), so any isolation attempt must control for that. |
| 4 | **CUDA-stream / pipeline-depth-2 bubble redesign** | **Closed, do not reopen without new evidence** | ≤1.065x after item ⑧ landed | Sized with `torch.cuda.Event` instrumentation ([`18-...-implementation.md`](./18-gpu-starvation-prerequisites-implementation.md) §3) and found not worth the ordering/thread-safety/CUDA-stream-sync risk. Reopen only if the intra-forward launch-bound idle (the larger half of all device idle) is ever fixed upstream. |
| 5 | **CUDA graph capture / vectorize Cellpose's internal Python loops** (`_extend_centers_gpu`, `get_masks_torch`, `steps_interp`, `get_rel_pos` — already GPU-resident, kernel-launch-bound, not a CPU-vs-GPU placement problem) | **Stop-lossed backlog** | ~1.23x even at zero | Requires patching pinned third-party `cellpose`/`segment_anything` internals. Traced against `cellpose==4.0.8` in [`measurement/gil-contention-diag.md`](./measurement/gil-contention-diag.md) "追加深挖"; round 3 upgraded to `cellpose==4.2.1.1` (`cpdino`) afterward, so **this trace needs re-confirming against the current version before anyone reopens it** — the function names/line numbers may have moved. Only `fill_holes_and_remove_small_masks` (`cellpose/utils.py`) has no GPU path at all; everything else in that trace is a launch-overhead problem, not a device problem. |
| 6 | **GPU-side tile/transform loading** | **Stopped out, not backlog** | 1.012x | `B2r_tile_read` was 1.22% of wall post-precut-streaming (round 4), and there's no existing CPU→GPU transform pipeline to move — building one would be new construction for a ceiling that doesn't justify it. See [`18-...-implementation.md`](./18-gpu-starvation-prerequisites-implementation.md) §6.3. |
| 7 | **Full real-WSI-scale validation** | **Never done, any round** | n/a | Every round (1–4) measures only 25/121/441-tile crops of a real WSI, never a complete 156k×134k slide end-to-end. [`09-measurement-analysis-plan.md`](./09-measurement-analysis-plan.md) §3.6 explicitly required this ("需要規劃一次完整 WSI 的跑批") and it has still not happened as of round 4. All full-WSI hour estimates (currently ~10.5h, see `18-...-implementation.md` §5) are linear extrapolations from tissue-dense crops, explicitly flagged as upper bounds in every round. |
| 8 | **Multi-request / concurrent-job behavior of the API/job layer (Phase E)** | **Never measured** | Flagged, not sized | `measurement/bottleneck-list.md` item ⑦ notes "concurrent analysis requests each hold a threadpool worker but serialize on the single GPU/CUDA context" as a *future* risk, not a measured one — see also [`09-measurement-analysis-plan.md`](./09-measurement-analysis-plan.md) §3.5, which scoped this as a measurement item that was never executed. No load test exists. |
| 9 | **Record `pip freeze` with every future measurement round** | Partially adopted | process, not perf | Rounds 3 and 4 do this; round 2 (`_metrics_current/`) still has no snapshot and can't be fixed retroactively — it's the reason item ⑨ above can't be cleanly attributed to a cause. Keep doing it going forward (`13-next-optimization-plan.md` §0 and Priority 6). |

**Already closed, don't re-litigate** (kept here only so nobody re-proposes them — full ablation evidence in the linked docs): `detect_all_dots` → process backend (negative, [`measurement/gil-contention-diag.md`](./measurement/gil-contention-diag.md)), `gc.collect` relocation to a background thread (negative, same doc), fixed-N `gc.collect` batching (built, measured, added nothing beyond `gc.freeze()`, deleted — [`16-gc-collect-frequency-result.md`](./16-gc-collect-frequency-result.md)), `cellpose_batch_size` sweep at the current 1024px tile size (wired but flat — [`18-...-implementation.md`](./18-gpu-starvation-prerequisites-implementation.md) §6.1).

## 2. Correctness / clinical validation — blocking, not a performance task

| Item | Status | Source |
|---|---|---|
| **Round-3 Cellpose checkpoint retrain needs pathologist/clinical sign-off** | **Pending, unresolved as of round 4.** Cell counts shifted +1.8–2.5% and one tile flipped success→skipped between round 2 and round 3 in the measured crops; this is a segmentation-quality change, not GPU-nondeterminism noise. Every performance win from round 3 onward (the Cellpose swap itself, plus everything built on top of it in rounds 4+) rides on top of this unvalidated model swap. | [`13-next-optimization-plan.md`](./13-next-optimization-plan.md) §3 "Correctness caveat"; `measurement/bottleneck-list.md` round-3 section |

## 3. Documentation ↔ code drift not yet fixed

Real, verified gaps between what a doc in this folder says and what the code does — checked
against the current repo, not assumed. (Path-migration drift — `cell_mask/hybrid/` →
`backend/algorithms/hybrid/` — and the `config_example.py` missing-tail bug were already fixed
in a prior documentation pass; what's below is what's still open.)

| Item | Detail | Source |
|---|---|---|
| **`generate_ihc_core_mask` parameter named `ihc_tile_path: Path` but always receives an ndarray** | Not a bug (the callee accepts `Union[ndarray, Path, str]`, caller has `# pyright: ignore`), but misleading to a new reader. Cosmetic rename (`ihc_image`, `Union[np.ndarray, Path]`) never landed. | [`07-gotchas-appendix.md`](./07-gotchas-appendix.md) G5 |
| **`docs/sdd-elastic-dish-matching.md` referenced but never exists** | `m3_elastic_matching.py`'s own docstring points at it. Confirmed missing. Not worth creating — the actual recommendation is to strip the dead reference from the docstring, which hasn't happened. | [`07-gotchas-appendix.md`](./07-gotchas-appendix.md) G4 |
| **`docs/dish_dot_detection_spec.md` referenced but never exists** | Same pattern, referenced from `config.py`/`config_example.py` comments (`see docs/dish_dot_detection_spec.md v0.2`). Confirmed missing. | [`07-gotchas-appendix.md`](./07-gotchas-appendix.md) G4 |
| **`docs/algo/elastic_matching_v3_explainer.html` describes the wrong matching algorithm** | Describes a "nucleus-centric" expansion variant; current code (`m3_elastic_matching.py`) is "cell-centric + overlap-priority + reach". The HTML was never updated after the rewrite. Same for `dish_elastic_expand_factor`, which the HTML marks deprecated but the code still uses to compute reach radius. Treat `m3_elastic_matching.py` as the sole source of truth until someone updates or retires the HTML. | [`04-optimization-roadmap.md`](./04-optimization-roadmap.md) "為什麼某些嘗試被棄用"; [`07-gotchas-appendix.md`](./07-gotchas-appendix.md) G4 |
| **No automated test guards `config.py`/`config_example.py` parity** | The tail-mismatch bug (G2 — `config_example.py` used to be missing `compute_config_hash()`/`config = Config()`) was caught by manual reading, not a test, and has since been fixed by hand. Nothing currently prevents the two files drifting apart again the same way. | [`07-gotchas-appendix.md`](./07-gotchas-appendix.md) G2 |
| **`backend/algorithms/hybrid/` still has zero automated pipeline-correctness tests** | The only automated check in this directory's scope is [`15-gc-collect-frequency-implementation.md`](./15-gc-collect-frequency-implementation.md) §4's `scripts/verify_gc_freeze.py` — a standalone invariant guard for one specific optimization, not a general pipeline-correctness suite. `m0_stitch` (pure data reorganization, no model) is explicitly flagged as the best candidate for a synthetic-numpy unit test and still has none. | [`05-dev-testing-guide.md`](./05-dev-testing-guide.md) "現有測試方式" |
| **Codegraph index staleness never reconfirmed against the current `backend/algorithms/hybrid/` path** | G1's phantom-file list was recorded against the old `cell_mask/hybrid/` path before the directory moved. Nobody has re-run the same `codegraph_files` vs `git ls-files` diff against the current path to produce an up-to-date phantom list. | [`07-gotchas-appendix.md`](./07-gotchas-appendix.md) G1 |

## 4. How to use this file

- **Picking up perf work**: start at §1, confirm the ceiling/status against the latest round in
  [`measurement/bottleneck-list.md`](./measurement/bottleneck-list.md) (numbers here are a
  snapshot, that document is the live source), then open the specific doc 10–18 for the item.
- **Picking up a documentation fix**: §3 lists what's known-stale; verify with `grep`/`find`
  against the current repo before editing (per this repo's own `codegraph-first` /
  `git ls-files`-over-trust discipline) — don't assume a doc's claim about another file without
  checking.
- **Closing an item**: update its status in the relevant table (or move it to "already closed"
  in §1) and link the record, the same way every other document in this folder cites its own
  evidence.
