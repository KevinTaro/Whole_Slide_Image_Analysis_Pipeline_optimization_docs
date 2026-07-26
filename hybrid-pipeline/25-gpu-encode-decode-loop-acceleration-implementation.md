# 25 — GPU/CUDA encode-decode & per-item-loop candidates: implementation & measurement record (round 7)

> Executes [`24-gpu-encode-decode-loop-acceleration-plan.md`](./24-gpu-encode-decode-loop-acceleration-plan.md)
> §4's ranked list, in its own order, under
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)'s
> Discover → Analyze → Plan → Choose discipline: cheapest signal first, every candidate
> ablation-proved, **correctness is a veto**. Doc 24 was survey-only and deliberately
> proposed nothing to build; this round measures every item on its list. Complete diff
> surface is §10.
>
> Round-7 anchors, git `025f9a5` + the tooling in §10, RTX 5090 / driver 580.173.02,
> 20 CPU cores, torch 2.11.0+cu130, cellpose 4.2.1.1, numpy 1.26.4, scikit-image 0.24.0,
> joblib 1.5.3. Config hash **`3d1087f2` — unchanged; no config field was added or altered
> this round.** Raw artifacts in `measurement/_metrics_r7/` (incl. `env_stamp_r7.txt`,
> `pip_freeze_r7.txt`).
>
> **Headline.** Doc 24 §4 item 1 was scoped to *refine* a composition estimate. It
> **overturned** it: measured with the pipeline's own rule — a tile is background iff
> UNet++'s core mask comes back empty — the slide is **55.8% background, not the ~39%**
> docs 23/24 carried, and that number is what every full-WSI projection in docs 19–24 is
> built on. On top of that measurement: **Candidate G is stop-lossed** (0.075 s of a 135 s
> run; the end-to-end ablation cannot see it), **Candidate F is real but worth zero wall**
> (24 ms per background tile, on an arm with 47–53% slack), and **Candidate A is the one item
> that survives** — Phase D at real 16.2-gigapixel scale measures **322.7 s**, 1.8× doc
> 24's extrapolation, fully serial, in the parent process — and the GPU encoder that could
> attack it (nvImageCodec, lossless LZW, 19.2× on the encode step) is the one library in
> doc 24's list that both exists for CUDA 13 and runs here (§8).

---

## 0. Protocol

Same protocol as doc 18 §0 / doc 21 §0 / doc 23 §0, so this round stays comparable:

- GPU confirmed idle before **every** launch — the drivers block on `memory.used < 200 MiB`
  rather than assuming it.
- `--gpu-dmon --workers 8 --stream-precut` on every end-to-end run; env stamp +
  `pip freeze` beside the metrics.
- **n=2 per configuration**, control and candidate **interleaved** (`w1 control, w1
  candidate, w4 control, w4 candidate`, twice) so machine drift hits both arms.
- Microbenchmarks report the per-process/per-rep spread, so a difference smaller than the
  spread is never read as a result.
- Correctness veto against the **same-code noise floor**, not exact equality (§3.4).
- **New and load-bearing this round:** the anchor is no longer the 441-tile tissue-dense
  crop rounds 1–6 used. §1 is why that crop cannot answer any question doc 24 asked.

---

## 1. The composition premise — measured against the pipeline's own rule, and it does not hold

### 1.1 What was carried into this round

Doc 23 §1 measured a stride-768 brightness thumbnail of the aligned slide, found the
background level to be ~213, and reported that only **~39% of grid cells** sit at/above
it — i.e. "**~61% tissue-bearing**". Doc 24 built its entire §0 on that: the composition
correction (§0.1), the corrected arm model (§0.4), the sizing of Candidate F (§2.5), and
the priority order in §4 all take 39%/61% as given. [`19-open-backlog.md`](./19-open-backlog.md)
item 7 records it as fact.

### 1.2 The brightness map reproduces — but only at a threshold nobody stated

`scripts/composition_crop.py` rebuilds that map from pyramid level 3 (one mean per
stride-768 cell), giving the same **149 × 185 = 27,565**-cell grid doc 23 reports. The
share of background cells is extremely sensitive to where the threshold is put:

| rule | background share |
|---|--:|
| cell mean ≥ 213 (doc 23's stated background level) | **2.4%** |
| cell mean ≥ 200 | **39.0%** |
| cell mean ≥ 188 | 55.1% |

Doc 23's 39% is reproducible — at ≥200, not at the ≥213 it names. The distribution is
dense in 180–213 (p50 = 191, p90 = 211, max = 226), so a ±13 grey-level choice moves the
answer from 2% to 39%. **A quantity that swings by 16× on an unstated threshold cannot
carry a full-WSI projection**, which is what it has been doing since round 6.

### 1.3 The rule that actually governs the pipeline, measured directly

Brightness is not the pipeline's rule. `_process_one_chunk_gpu`
(`hybrid_pipeline.py:571-577`) runs the UNet++ core-mask forward on every tile and returns
`None` when `core_mask.sum() == 0`; that, and only that, decides whether the two Cellpose
forwards and the whole BG arm run. `scripts/tissue_calibrate.py` samples grid cells, cuts
each tile exactly as `PrecutStream` does, runs the **real** M1 forward, and labels it:

| sample | n | background share | 95% CI | best brightness threshold (accuracy) |
|---|--:|--:|---|---|
| seed 0 | 250 | **58.0%** | 51.9–64.1% | ≥191 (88.4%) |
| seed 1 | 800 | **55.3%** | 51.8–58.7% | ≥188 (88.3%) |

**By the pipeline's own definition the slide is ~55% background / ~45% tissue-bearing —
not 39% / 61%.** The 39% figure is outside the 95% CI of both samples. The brightness
proxy, at its *best* threshold, still mislabels ~12% of tiles, which is why it cannot be
patched by picking a better number: it is measuring a different thing.

Sampling leaves a ±3.5 pp confidence interval, so `scripts/core_mask_map.py` then ran the
same rule over **every cell of the grid** — 27,565 real M1 forwards, tile cutting
prefetched on a thread pool, 1,043.7 s wall. This is no longer an estimate:

| | tiles | share |
|---|--:|--:|
| **background** (`core_mask.sum() == 0`) | **15,386** | **55.82%** |
| **tissue-bearing** | **12,179** | **44.18%** |
| total (185 × 149) | 27,565 | 100% |

The n=800 sample was accurate to 0.6 pp (55.25% vs 55.82%), and the brightness proxy's
39% is not close. **The slide is 55.8% background.** Everything below uses this number.

### 1.4 What this changes

| affected claim | as carried in docs 19/23/24 | measured round 7 |
|---|---|---|
| slide composition | 39% background / 61% tissue | **55.8% background / 44.2% tissue** |
| background tiles at full scale | ~10,750 | **15,386** |
| tissue tiles at full scale | 16,815 | **12,179** |
| Candidate F's population | 10,750 tiles / 64,500 encode calls | **15,386 tiles / 92,316 encode calls** (+43%) |
| everything gated on tissue-tile count (Cellpose forwards, `detect_all_dots`, per-cell crops, PNG encode) | sized on 16,815 tiles | **sized on 12,179 tiles — 28% smaller** |

The direction matters: **every MAIN-arm cost shrinks by ~28% and the one BG-arm cost that
scales with background tiles grows by ~43%.** That makes the full-WSI wall estimate
smaller (§6) and Candidate F relatively larger — which is exactly why doc 24 asked for
this measurement before ranking anything.

---

## 2. The round-7 anchor, and the arm ceiling it establishes

### 2.1 The crop, and an honest note on how it was selected

`scripts/composition_crop.py` slides a 24×24 window over the cell map (integral image) and
takes the one whose background share is closest to the target, then cuts it at full
resolution through the same pyvips path `scripts/crop_roi.py` uses. Selected window:
**(96768, 1536), 18,688² px, 24×24 = 576 tiles** — predicted 55.2% background by the
brightness proxy.

**Realized: 423 of 576 tiles (73.4%) were background**, not 55%. The proxy is ~88%
accurate per tile, but its errors are spatially correlated, so window-level error is much
larger than tile-level error. This is recorded rather than hidden because it changes how
the run may be used: **the run is a valid measurement of per-population *rates* (153
tissue tiles, 423 background tiles, each timed separately), and those rates are what §6
re-weights to the real slide.** It is not a claim that this crop *is* the slide.
§7 re-runs the anchor on a crop selected from the exact core-mask map instead of the
proxy.

For orientation against the rounds it replaces:

| anchor | tiles | background share | source |
|---|--:|--:|---|
| small / medium / large (rounds 1–6) | 25 / 121 / 441 | 12.0% / 14.9% / 14.1% | `measurement/bottleneck-list.md` |
| **comp24** — the A/B anchor (round 7) | **576** | **73.4%** | this round, proxy-selected |
| **match24** — composition-matched (round 7, §2.4) | **576** | **55.9%** | this round, selected from the exact map |
| **the real slide** | **27,565** | **55.8%** | §1.3, measured over every cell |

### 2.2 Anchor numbers

576 tiles, `dot_detect_n_jobs=1`, n=2 interleaved, GPU idle before each launch.

| config | wall (r1 / r2) | mean | success / skipped |
|---|--:|--:|--:|
| `workers=1` | 135.43 / 134.39 s | **134.91 s** | 137 / 439 |
| `workers=4` | 65.55 / 65.53 s | **65.54 s** | 137 / 439 |

`workers=4` buys **2.06×** end-to-end here (2.10× on the run-batch part alone), against
**2.35×** on the tissue-dense large crop in round 6 (302.7 → 128.8 s). Multiprocessing
gains *less* on a background-heavy workload — background tiles are cheap and their
per-tile fixed costs do not parallelise away.

### 2.3 The two-arm model at this composition — and the ceiling it puts on Candidates B/D/E/F

`scripts/arm_report.py`, `workers=1` runs (the only mode where per-function buckets are
populated — the harness monkeypatches the parent, and `workers>1` spawns re-import the
module):

| label | wall | MAIN | BG | outside | pred | err | **BG/MAIN** | MAIN must shed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| control_w1_r1 | 135.4 | 127.4 | 67.2 | 6.9 | 134.3 | −0.8% | **0.527** | 47.3% |
| control_w1_r2 | 134.4 | 127.3 | 66.4 | 6.9 | 134.2 | −0.1% | **0.522** | 47.8% |
| candidate_w1_r1 | 136.6 | 127.2 | 67.1 | 7.2 | 134.3 | −1.7% | 0.527 | 47.3% |
| candidate_w1_r2 | 134.9 | 125.7 | 65.1 | 6.7 | 132.4 | −1.9% | 0.518 | 48.2% |

MAIN arm (127.2 s): Cellpose M2+M3b 85.5 s (62.6% of wall), UNet++ 16.0 s, `enlarge_cell_instances`
7.1 s, `_read_rgb` 5.1 s, M1 overlay glue 4.7 s, `build_all_positive_results` 4.1 s.
BG arm (67.1 s): `detect_all_dots` 26.0 s, PNG encode 25.6 s, **blank-tile writes 10.1 s**,
`render_overlay_image` 3.2 s, per-cell crops 1.7 s, TIFF encode 0.4 s.

**This is the single most consequential number in the round: BG/MAIN ≈ 0.52.** The
background arm runs concurrently with MAIN and finishes with **47% of its capacity
unused**. Doc 24 §0.4 predicted the direction (more background ⇒ more BG slack) and this
confirms it with a measurement — at round 6's tissue-dense anchor the same ratio was
~0.74.

### 2.4 Confirmation on a crop selected from the exact map — the ratio holds, with more slack

Once §1.3's full-grid map existed, the crop selection was redone against it: no proxy, the
window whose *measured* background share is closest to the slide's 55.82%. Selected
window **(46080, 768), 18,688², 576 tiles**, predicted 55.9% background — and the pipeline
realized **exactly 322/576 = 55.9%** (254 tissue tiles), i.e. the exact-map selection is
accurate where the brightness proxy was 18 pp off.

| run | wall | MAIN | BG | outside | **BG/MAIN** | MAIN must shed | tissue / background |
|---|--:|--:|--:|--:|--:|--:|--:|
| `match_w1_r1` | 188.8 s | 187.4 | 88.0 | 7.5 | **0.470** | **53.0%** | 254 / 322 |
| `match_w4_r1` | 88.3 s | — | — | 4.6 | — | — | 254 / 322 |

`workers=4` buys **2.14×** here (2.17× on the run-batch part). Two things this confirms:

1. **The BG-arm slack is not an artifact of the over-background comp24 crop** — at the
   slide's true composition it is *larger* (53.0% vs 47.3%), because more tissue tiles
   load MAIN's Cellpose forwards faster than they load BG.
2. **The per-population rates reproduce across two independently selected crops**: the
   blank-tile write rate is 0.0237 s/tile here vs 0.0241 on comp24 (−1.7%), which is what
   makes §6's re-weighting trustworthy.

Consequence, stated as the Amdahl arithmetic the playbook asks for **before** touching
anything: **any candidate whose work sits on the BG arm has a wall-clock ceiling of
1.00× until MAIN sheds ~50% first.** That is Candidate B (per-tile debug PNG/TIFF encode
+ per-cell crops), Candidate D (`detect_all_dots` → GPU), Candidate E
(`enlarge_cell_instances`/`build_all_positive_results` — partly MAIN, see the table
above), and Candidate F (§4). Doc 24 already declined to build B/D/E on a 1.013×–1.05×
estimate; **the measured ceiling at real composition is worse than the estimate, not
better**, and §0.5's VRAM gate (§8) applies on top. Disposition for B/D/E: **do not
build, confirmed on measurement.**

---

## 3. Candidate G (`mkdir` hoist) — prototyped, measured at both worker counts, stop-lossed

Doc 24 §2.6 called this "**measure, then very likely build** … the highest-confidence,
lowest-risk item in this survey". It was built and measured. The measurement does not
support shipping it.

### 3.1 The redundant syscalls exist, exactly as described — and cost 0.06% of wall

`_save_tile_array` ran `path.parent.mkdir(parents=True, exist_ok=True)` before every
write: 6 per tissue tile and 6 per background tile, against six fixed directories that
exist after the first tile. Instrumented (`perf_measure.py`'s `G_mkdir_*` buckets, which
time `pathlib.Path.mkdir` process-wide and split by target):

| run | mkdir calls (fixed out-dirs) | total time | share of wall | Amdahl ceiling |
|---|--:|--:|--:|--:|
| control_w1_r1 (per-call mkdir) | 3,462 | 0.075 s | **0.056%** | **1.0006×** |
| control_w1_r2 | 3,462 | 0.078 s | 0.058% | 1.0006× |
| candidate_w1_r1 (hoisted) | 6 | 0.0001 s | 0.00007% | — |
| candidate_w1_r2 | 6 | 0.0001 s | 0.00007% | — |

3,462 = 6 × 576 + 6, i.e. the count is exactly what doc 24 predicted. Extrapolated to the
full slide: 165,390 calls ≈ **0.83 s** total — spread across workers, on the arm that
already has 47% slack.

### 3.2 The concurrency question, answered directly

Doc 24's strongest argument was not speed but stability: "`workers=4` means 4 processes
issuing `mkdir()` against the *same* directory inodes concurrently". `scripts/fs_write_probe.py`
runs P processes sharing one output tree, barrier-synchronised:

| P | mkdir µs/call (mean) | per-process spread | blank-tile write ms | hardlink ms |
|--:|--:|---|--:|--:|
| 1 | 2.60 | — | 20.25 | 0.050 |
| 2 | 2.58 | 2.57 / 2.59 | 20.70 | 0.056 |
| 4 | **5.11** | 2.79 / 2.86 / 2.89 / **11.90** | 23.15 | 0.085 |
| 6 | **5.33** | 2.79–2.85 ×4, **8.43**, **12.25** | 23.81 | 0.084 |

**The contention doc 24 hypothesised is real** — at P≥4 one or two processes see 8–12 µs
per call while the rest stay at ~2.8 µs. It is also **bounded and tiny**: even at the
worst observed 12 µs, the full slide's 165,390 calls total ≈ 2 s across all workers, on
the slack arm. There is no stability failure mode here — no error, no unbounded wait, just
a few microseconds.

### 3.3 End-to-end ablation — negative, i.e. no effect

n=2 interleaved, both worker counts, same crop, GPU idle before each launch:

| config | control (per-call mkdir) | candidate (hoisted) | Δ | run-to-run spread |
|---|--:|--:|--:|---|
| `workers=1` | 135.43 / 134.39 → **134.91 s** | 136.64 / 134.88 → **135.76 s** | **+0.63%** | control 1.04 s, candidate 1.76 s |
| `workers=4` | 65.55 / 65.53 → **65.54 s** | 66.33 / 65.79 → **66.06 s** | **+0.80%** | control 0.02 s, candidate 0.55 s |

The candidate is nominally *slower* at both worker counts, by less than the spread. The
honest reading is **no measurable effect in either direction**, which is what §3.1's
0.056% predicts. This is the playbook's red flag #1 ("your optimized version is about
equal to the dumbest baseline") behaving exactly as designed: the Amdahl arithmetic was
computed first, and the end-to-end run confirmed it.

### 3.4 Correctness veto — passed (and it could not have failed)

Per-cell reddot / blackdot / score matched by nearest centroid against `control_w1_r1`:

| run | cells (ref 5,687) | reddot max\|Δ\| | blackdot max\|Δ\| | score max\|Δ\| | X-flips |
|---|--:|--:|--:|--:|--:|
| control_w1_r2 (same-code control) | 5,688 | 0 | 0 | 0 | 0 |
| candidate_w1_r1 | 5,688 | 0 | 0 | 0 | 0 |
| candidate_w1_r2 | 5,686 | 0 | 0 | 0 | 0 |
| candidate_w4_r1 | 5,687 | 0 | 0 | 0 | 3 |
| candidate_w4_r2 | 5,686 | 0 | 0 | 0 | 0 |

Every measured field is bit-identical; the ±1–2 cell-count and 3 X-flip residue is the
same-code noise floor rounds 4–6 documented (upstream GPU non-determinism in
segmentation). Expected: the change moves directory creation, it cannot touch numerics.

### 3.5 Disposition — **not shipped**

Doc 24 expected to build this. The measurement says: 0.056% of wall, an end-to-end
ablation that reads slightly negative, and a contention effect that is real but ~2 s over
a whole slide. Against that, hoisting the `mkdir` introduces a **new invariant** —
`_save_tile_array` stops being self-sufficient and requires `_ensure_output_dirs` to have
run first, so any future call site that writes to a directory outside the fixed six fails
with `FileNotFoundError` instead of just working.

Per the playbook's step 4 ("every optimization layer must justify itself via ablation;
zero-contribution layers get cut no matter how clever") the change is **reverted**; the
pipeline is left byte-identical to `025f9a5`. The patch is recorded in §10 so it can be
revived in one edit if the arm balance ever changes — but on today's numbers it would be
adopting an invariant for a benefit no measurement in this project can detect.

---

## 4. Candidate F (background-tile placeholder writes) — first measurement: real cost, zero wall

Doc 24 §2.5 called this "the one genuinely new finding in this doc", unmeasured because at
14%-background crops it was invisible. It is now measured
(`perf_measure.py`'s `F_write_blank_tile` bucket wraps `_write_blank_tile`, and the six
writes it issues are split into `B2_*_encode_blank`).

### 4.1 What a background tile costs

| bucket | calls | total | per call | share of wall |
|---|--:|--:|--:|--:|
| `F_write_blank_tile` | 423 | 10.18 s | **24.07 ms/tile** | **7.5%** |
| ├ `B2_png_encode_blank` | 1,269 | 8.91 s | 7.02 ms | 6.6% |
| └ `B2_tiff_encode_blank` | 1,269 | 0.89 s | 0.70 ms | 0.7% |
| (for comparison) `B2_png_encode`, tissue tiles | 459 | 25.62 s | 55.82 ms | 18.9% |

So doc 24's §0.4 footnote — "assumes background-tile writes were a small, fast-compressing
share … reasonable but **unverified**" — is **half right**: a blank PNG is ~8× cheaper
than a real one (7.0 vs 55.8 ms), but not free, and there are **4.0× more of them** on a
real slide than any crop suggested (55.8% vs 14.1%). Note also the disk side: the three
TIFFs a blank tile writes (two int32 label masks + the RGB overlay) are **uncompressed** —
`skimage.io.imsave` writes no compression — so 423 background tiles produced **4.3 GB**,
**10.2 MB per blank tile of entirely constant data**. At full-WSI scale that is **~157 GB**
of identical bytes, i.e. well over half of the ~270 GB a full slide run was estimated to
produce (doc 23 §1).

### 4.2 The non-GPU alternative doc 24 proposed, measured

Doc 24 §2.5 argued the fix, if any, is not a CUDA library but writing one blank tile set
once and hard-linking it. Measured in `fs_write_probe.py` (same process, same directory
tree, real `_write_blank_tile` vs `os.link` of a pre-encoded template):

| P | `_write_blank_tile` | 6 × `os.link` | speedup |
|--:|--:|--:|--:|
| 1 | 20.25 ms | 0.050 ms | **407×** |
| 4 | 23.15 ms | 0.085 ms | **272×** |
| 6 | 23.81 ms | 0.084 ms | **283×** |

The alternative is real and enormous *in relative terms*: it would remove ~99.6% of the
blank-tile write cost, ~364 s of BG-arm work at full-WSI scale, and ~**157 GB** of
identical uncompressed TIFF bytes (§4.1).

### 4.3 Why it still buys nothing, and is therefore not built

The entire cost sits on the BG arm, which has **47% slack at the comp24 anchor and 53% at
the composition-matched one** (§2.3, §2.4). Removing all 364 s of
it moves BG from 70.3 → 64.2 min against a MAIN arm of 149.7 min (§6): the critical path
does not move, at any worker count, because the per-worker arm structure is identical
(doc 24 §0.5's per-worker-invariance, confirmed by the arm table being reproducible at
`workers=1` and the `workers=4` speedup tracking it).

**Disposition: measured, sized, and not built.** Recorded here with the numbers so it can
be picked up immediately *if* MAIN ever sheds enough for BG to become the critical path —
and if that happens, the fix is `os.link`, not nvImageCodec. The disk-space argument
(~157 GB of constant TIFF per slide, §4.1) is a separate, non-performance reason someone may want
it; that is a storage decision, not an optimization, and is logged in §11 rather than
decided here.

---

## 5. Candidate A (Phase D slide stitch) — measured at real scale, and it is the one survivor

Doc 24 §2.1 ranked this strongest and asked for one thing before any engineering: convert
"~3 minutes" from extrapolation into a fact.

### 5.1 Method

`scripts/stitch_probe.py` measures the **real** `_stitch_overlay_slide` at a chosen slide
size without running any inference. It rebuilds the exact grid `chunk_offsets` produces,
computes every tile's core-crop size with the pipeline's own `core_crop_bounds` (so edge
columns/rows keep their real, different sizes — 896 / 768 / 250 px classes), and
materialises `overlay_annotated/` from **real** overlay tiles produced by an actual
pipeline run: one encoded template per (size class × pool member), every other position a
hard link. Templates are written uncompressed, matching what `skimage.io.imsave` produces.

The dominant variable is composition — a blank placeholder is constant fill and compresses
almost for free — so the annotated share is set to the slide's tissue share as measured
when the probe ran (44.75%; the full map later refined it to 44.18%, §1.4 — a 0.6 pp
difference), rather than inherited from whatever crop the source tiles came from.

### 5.2 Results

| slide | gigapixels | tiles | annotated share | **stitch** | s/gigapixel | output |
|---|--:|--:|--:|--:|--:|--:|
| 35,840 × 28,928 | 1.04 | 1,786 | 44.7% | **14.63 s** | 14.11 | 0.37 GB |
| 71,168 × 57,344 | 4.08 | 6,975 | 44.8% | **57.80 s** | 14.16 | 1.44 GB |
| **141,818 × 114,366 (the real slide)** | **16.22** | **27,565** | **44.8%** | **322.7 s** | **19.90** | **5.76 GB** |

Two findings:

1. **The real cost is 322.7 s = 5.4 minutes, not ~3.** Doc 24's extrapolation (5.05 s at
   0.46 GP ⇒ ~11 s/GP) understated it by **1.8×**, partly because the crop it extrapolated
   from is 86% tissue and partly because of finding 2.
2. **It is superlinear at full scale.** The 1 GP and 4 GP points agree to within 0.4%
   (14.11 vs 14.16 s/GP), then the real slide jumps to 19.90 s/GP — **+40% per pixel**.
   Whatever the mechanism (pyramid level count, allocator/page-cache pressure at
   16 gigapixels), a linear extrapolation from any crop *systematically underestimates*
   this stage, which is precisely the error doc 24 was trying to avoid.

A robustness point worth recording for backlog item 7: the current implementation opens
all 27,565 tiles as pyvips images before saving, and completed with `RLIMIT_NOFILE` at
1,048,576. On a host with the common 1,024 soft limit this would fail, and nothing in the
pipeline checks it.

### 5.3 What it is worth

At `workers=1` (§6): 5.4 min of a 2.59 h wall = **3.5%** — ceiling 1.036×, below the
playbook's bar on its own.

At `workers=4`, using this round's measured 2.17× on the parallel part: full-WSI wall
≈ 149.7/2.17 + 5.4 ≈ **74 min**, of which the stitch is **7.3%** — ceiling 1.078×. The
share **doubles** going from `workers=1` to `workers=4`, exactly as doc 24 §2.1 predicted:
it is the one cost that does not parallelise, so it grows as everything else shrinks. At a
hypothetical perfect-parallel limit it would approach 100%.

**Disposition: the only candidate still worth engineering — but not yet.** 1.078× at the
recommended worker count does not clear the playbook's bar on its own, and §8 shows the
GPU route is real but not a drop-in: nvImageCodec encodes lossless LZW TIFF **19.2×**
faster than the pipeline's pyvips call, but produces no pyramid and no BigTIFF, so the
saving is bounded by the container/pyramid work that this measurement cannot separate
from the encode. What *is* settled is the number — **322.7 s measured, not ~180 s
assumed, and superlinear** — and the order to attack it in: §11's cheaper non-GPU knobs
(`tiffsave` tiling/pyramid parameters, not re-encoding constant regions) come before any
new dependency.

---

## 6. Full-WSI projection, rebuilt from measured rates and measured composition

`scripts/wsi_projection.py` splits every bucket by the population it actually runs on
(all tiles / tissue tiles / background tiles), takes the rate from the
**composition-matched** `match_w1_r1` run (§2.4), and re-multiplies by the slide's
**measured** populations (§1.3): 27,565 tiles = 12,178 tissue + 15,387 background.

| bucket | arm | population | rate | full-WSI |
|---|---|---|--:|--:|
| `B1_m3b_cellpose` (2× Cellpose) | MAIN | tissue | 0.5259 s/tile | **106.7 min** |
| `B2_png_encode` | BG | tissue | 0.1463 s/tile | 29.7 min |
| `B3_detect_dots` | BG | tissue | 0.1398 s/tile | 28.4 min |
| `B1_unet_coremask` | MAIN | **all** | 0.0289 s/tile | 13.3 min |
| `B3_enlarge_cells` | MAIN | tissue | 0.0430 s/tile | 8.7 min |
| **`F_write_blank_tile`** | **BG** | **background** | **0.0237 s/tile** | **6.1 min** |
| `B3_build_results` | MAIN | tissue | 0.0275 s/tile | 5.6 min |
| everything else | — | — | — | 19.2 min |
| **MAIN total** | | | | **149.7 min** |
| **BG total** | | | | **70.3 min** (BG/MAIN = 0.470) |
| Phase D stitch (§5, measured) | outside | pixels | — | **5.4 min** |
| **wall = max(MAIN, BG) + outside** | | | | **≈ 2.59 h at `workers=1`** |
| **wall at `workers=4`** (2.17× on the parallel part) | | | | **≈ 1.24 h** |

The same arithmetic on the comp24 run (73.4% background, different tiles, different cell
density) lands at **2.70 h / 1.33 h** — a 4% spread between two independently selected
crops, which is a fair indication of the extrapolation's own uncertainty.

Against the record this replaces:

| estimate | value | why it differed |
|---|--:|---|
| round 5 (35,700-tile grid) | ~10.5 h | wrong tile count |
| round 6 official (`23-...` §4.4) | ~5.3 h | blended per-tile rate from an 86%-tissue crop |
| doc 24 §0.4 back-of-envelope | 3.1–3.4 h | right method, wrong composition input (39%) |
| **round 7, measured rates × measured composition** | **~2.6 h** (`workers=1`) / **~1.25 h** (`workers=4`) | composition-matched crop, full-grid composition, measured Phase D |

Same caveat as every prior round, stated plainly: these are still *rates measured on one
crop*, extrapolated. Cells per tissue tile vary across the slide, so the MAIN term carries
real uncertainty. It is not a substitute for backlog item 7's actual full-slide run — it
is a better-founded estimate than the one it replaces, built on a measured composition
instead of an assumed one.

---

## 7. Cross-tile Cellpose batching at G=16 — the one item doc 24 reopened, closed again

Doc 24 §1 reopened exactly one closed item, cheaply and with low expectation: round 6
tested G = 1, 2, 4, 8 and found per-patch-proportional cost, but never tested the
G ≈ 16–20 range that `workers=1` VRAM headroom would allow. Re-run with
`scripts/cellpose_batch_probe.py` on a **tissue-dense** crop (8×8 grid, 100% tissue by the
§1.3 rule — the round-6 anchor's character), 26 real per-model inputs, 3 reps, min
reported:

| model | G | full `eval` ms/tile | Δ vs G=1 | `run_net`-only ms/tile | Δ | peak alloc |
|---|--:|--:|--:|--:|--:|--:|
| **M2** (cell) | 1 | **238.5** | — | 204.1 | — | 1,167 MB |
| | 16 | 254.3 | **+6.6%** | 207.4 | +1.6% | **15,840 MB** |
| **M3b** (DISH nucleus) | 1 | **268.5** | — | 193.4 | — | 1,165 MB |
| | 16 | 284.3 | **+5.9%** | 211.9 | +9.6% | **15,838 MB** |

**Worse on both models, and worse on the batchable part itself** (`run_net`, measured with
`compute_masks=False`) — which is the mechanism round 6 identified: cost inside the
DINOv3 backbone is per-patch proportional, so grouping images does not reduce patch work.
Extending the sweep from G=8 to G=16 does not change the shape; it extends it.

The price confirms doc 24's own VRAM objection quantitatively: **15.8 GB, 48.6% of the
32.6 GB card, for one process.** At the recommended `workers=4` (steady state ~10 GB total
this round, §11) it is arithmetically impossible — 4 × 15.8 GB = 63 GB.

Correctness note, recorded because it differs from round 6: at G ≤ 8 round 6 found masks
**bit-identical**; at G=16 they are not quite — cell counts move by ≤0.15% (M2: one tile
81 → 80; M3b: 4,610 → 4,608 and 6,452 → 6,458 of ~6,500). The probe's raw pixel-diff
figures (3.9% / 53%) overstate this badly, because a single added or dropped instance
renumbers every subsequent label; the cell counts are the meaningful comparison. Moot
given the timing, but it means G=16 would also have needed its own correctness argument.

**Disposition: closed again, now with the G=16 datum doc 24 asked for.** `cellpose_batch_size`
stays 16 (per-image patches), no cross-tile gather stage.

---

## 8. Environment + VRAM gate (doc 24 §3 / §4 item 4) — what is actually installable and usable here

Run in a **throwaway venv** (`uv venv`, one package at a time), never the project venv —
doc 24 §3's own constraint, because round 3's bundled `uv sync` produced a +22.3%
regression that is still unexplained. `scripts/gpu_codec_spike.py` records import status,
device usability, encode throughput and VRAM footprint.

### 8.1 Installability — all four ecosystems now ship CUDA-13 wheels

| package | version | installs | imports | usable here |
|---|---|---|---|---|
| `cupy-cuda13x` | 14.1.1 | ✅ | ❌ | **no** — needs the numpy ≥ 2 ABI; the project is pinned `numpy<2` (`pyproject.toml` `override-dependencies`) |
| `cupy-cuda13x` | 13.6.0 | ✅ | ✅ (reports **cc 120**, i.e. Blackwell recognised) | **no** — first kernel JIT fails: `Failed to auto-detect CUDA root directory`; no CUDA toolkit on this host, only driver + torch's bundled libs |
| `nvidia-nvimgcodec-cu13` | 0.9.0 / 0.8.0 | ✅ | ✅ | **yes** (see §8.2) — nvJPEG/nvJPEG2000 extensions need separate wheels, irrelevant for TIFF |
| `nvidia-nvtiff-cu13` | 0.8.0.82 | ✅ | ❌ **no Python module at all** (`import nvtiff` → `ModuleNotFoundError`) | **no** — the wheel ships shared libraries only |
| `nvidia-nvcomp-cu13` | 5.3.0 | ✅ | ✅ | usable, but its codec list is LZ4/GDeflate/Zstd/ANS-class — **no LZW**, as doc 24 said |
| `cucim-cu13` | 26.06.00 | ✅ | ✅ | **read-only** — `CuImage` exposes no `write`; confirms doc 24's suspicion that cuCIM has no write path |

Two of doc 24's open questions resolve to "no": **nvTIFF has no Python binding** (using it
means writing a ctypes/C++ layer), and **cuCIM cannot write**. And CuPy — the dependency
every one of Candidates B/D/E would need — **does not run on this host today**: the CUDA-13
line requires numpy ≥ 2 that the project's pins forbid, and the numpy-1-compatible line
cannot JIT without a CUDA toolkit. That is not "CuPy is broken on Blackwell" (it detects
sm_120 correctly); it is a real, concrete environment obstacle that would have to be paid
for — a numpy 2 migration across valis/opencv/scikit-image, or a system CUDA toolkit —
before any CuPy-based candidate could even be benchmarked.

### 8.2 The one GPU encode path that works — and what it can and cannot produce

nvImageCodec **can** encode TIFF here. Same 4096² buffer, 3 reps, min:

| encoder | configuration | throughput | output |
|---|---|--:|--:|
| pyvips (CPU) | `lzw` only | 129.3 MP/s | 1.12 MB |
| pyvips (CPU) | `lzw` + `tile` | 136.3 MP/s | 1.24 MB |
| pyvips (CPU) | `lzw` + `tile` + `pyramid` + `bigtiff` — **what `_stitch_overlay_slide` calls** | **93.0 MP/s** | 2.65 MB |
| **nvImageCodec (GPU)** | TIFF, default | **1,789.6 MP/s** | 1.61 MB |

Verified on the output rather than assumed: nvImageCodec's TIFF is **compression tag 5 =
LZW**, photometric RGB, and round-trips **bit-identical** to the input. So the GPU path
produces exactly the codec this pipeline requires, **13.8× faster** than pyvips's
equivalent (strip LZW) and **19.2×** faster than the full pipeline configuration.

What it does **not** produce, checked against `EncodeParams` (`tile_width`/`tile_height`,
`quality_*`, `jpeg*_params` — nothing else): **no pyramid, no BigTIFF**, single page. The
pipeline's output is tiled + pyramidal + BigTIFF because that is what a 16-gigapixel
viewer-friendly slide needs. So the honest reading of the 19.2× is: **the entropy-coding
step is not the floor** — the pyramid/container work is a large part of the 322.7 s, and a
GPU port means generating downsample levels (`torch`/`cupyx`) and assembling a
tiled/pyramidal BigTIFF container around GPU-encoded strips. Doc 24 called this "real
engineering, not a drop-in call"; that is confirmed, with the upside now quantified.

### 8.3 VRAM footprint — the `workers>1` gate, with numbers

| stage | device memory in use |
|---|--:|
| baseline (idle card) | 41 MB |
| after `import pyvips` / `import cupy` (failed) | 41 MB |
| after constructing `nvimgcodec.Encoder()` | **557 MB** |
| after one 4096² TIFF encode | **1,261 MB** |

**~0.5 GB per process for the codec context, ~1.2 GB with working buffers.** Candidate A
is exempt — `_stitch_overlay_slide` runs once, in the parent, outside the worker pool
(`_finish_batch` is the single convergence point for both process models). Anything that
would run *inside* the workers pays this ×N: at `workers=4` that is ~5 GB against a card
whose measured transient peak this round already touched **26.7 GB of 32.6 GB** (§11).
Doc 24 §0.5's gate stands, and it is tighter than the steady-state numbers suggest.

**Disposition:** the environment gate **clears for Candidate A only** (nvImageCodec, parent
process, LZW verified lossless), and **does not clear** for anything CuPy-based
(B/D/E/F) — which is moot anyway, since §2.3/§2.4 show those have a 1.00× wall ceiling.

---

## 9. Dispositions — doc 24 §4's list, answered

| doc 24 item | disposition after round 7 |
|---|---|
| **1. composition-matched measurement at `workers=1` and `4`** | **Done — and it overturned the premise** (§1): the slide is 55.8% background, not 39%. Arm model at real composition: **BG/MAIN = 0.47–0.53**, i.e. 47–53% BG slack (§2.3, §2.4). |
| **2. size Candidate A at real scale** | **Done** (§5): **322.7 s** at 16.22 GP, 1.8× the extrapolation, superlinear. Strongest remaining candidate; 1.036× at `workers=1`, 1.078× at `workers=4`. |
| **3. prototype + measure Candidate G** | **Done, then stop-lossed** (§3): 0.056% of wall, ablation reads −0.6/−0.8% (i.e. nothing), contention real but ~2 s/slide. **Not shipped**; patch preserved in §10. |
| **4. environment + VRAM spike** | **Done** (§8): all four ecosystems ship CUDA-13 wheels, but **nvTIFF has no Python binding**, **cuCIM cannot write**, and **CuPy cannot run on this host** (numpy≥2 ABI vs the project's `numpy<2` pin; the numpy-1 line cannot JIT without a CUDA toolkit). nvImageCodec does encode **lossless LZW TIFF at 1,789.6 MP/s — 19.2× the pipeline's pyvips path** — but with no pyramid/BigTIFF, and costs ~0.5–1.2 GB VRAM per process. |
| **5. Candidates B, D/E — do not build** | **Confirmed on measurement** (§2.3): every one of them is BG-arm work with a **1.00× wall ceiling** at real composition. Stronger than doc 24's estimate, not weaker. |
| **6. cross-tile Cellpose batching at G=16** | **Done, closed again** (§7): +6.6% (M2) / +5.9% (M3b) **slower** than G=1, and 15.8 GB peak — 48.6% of the card for one process. |
| **(new) Candidate F** | **Measured** (§4): 24 ms/background tile, 6.1 min at full scale, all of it on the slack arm ⇒ zero wall. The `os.link` alternative is 272–407× cheaper and would also drop **~157 GB** of identical bytes per slide — a storage argument, not a speed one. |

---

## 10. Code changes this round (complete diff surface)

**Pipeline code: none.** `backend/algorithms/hybrid/` is byte-identical to `025f9a5`, and
`config.py` / `config_example.py` are untouched (config hash stays `3d1087f2`). The one
pipeline change this round built — Candidate G — was measured and reverted (§3.5).

New measurement tooling (standalone; imports the pipeline, never modifies it):

| script | purpose |
|---|---|
| `scripts/tissue_calibrate.py` | §1.3 — samples grid cells, runs the **real** M1 forward, reports the true background share and how well a brightness proxy reproduces it |
| `scripts/core_mask_map.py` | §1.4 — the same rule over the **whole** grid; saves the per-cell core-pixel map |
| `scripts/composition_crop.py` | §2.1 / §7 — picks and cuts the crop whose tile grid matches a target composition (from the exact map, or the brightness proxy) |
| `scripts/fs_write_probe.py` | §3.2 / §4.2 — mkdir + `_write_blank_tile` + `os.link` cost at P concurrent processes sharing one output tree |
| `scripts/stitch_probe.py` | §5 — `_stitch_overlay_slide` at a chosen slide size, from real overlay tiles, without inference |
| `scripts/wsi_projection.py` | §6 — re-weights a measured run to the slide's real tissue/background populations |
| `scripts/gpu_codec_spike.py` | §8 — nvTIFF / nvImageCodec / nvCOMP / CuPy / cuCIM availability, sm_120 usability, encode throughput and VRAM footprint |

Modified tooling:

| file | change | why |
|---|---|---|
| `scripts/perf_measure.py` | new buckets `F_write_blank_tile` (wraps `_write_blank_tile`), `B2_{png,tiff}_encode_blank` (writes issued from it), `G_mkdir_fixed_outdir` / `G_mkdir_other` (times `Path.mkdir` process-wide, split by target) | §3.1 / §4.1 — neither Candidate F nor G was separately measurable before |
| `scripts/arm_report.py` | `F_write_blank_tile` added to the BG arm | §2.3 — it is a background tile's whole BG-arm cost; its inner `_blank` encode buckets are its breakdown and are deliberately left out to avoid double counting |

**The stop-lossed Candidate G patch**, recorded so it can be revived in one edit if the
arm balance or the storage backend ever changes (§11):

```python
# hybrid_pipeline.py — add next to _save_tile_array
_TILE_OUTPUT_SUBDIRS = ("core_mask", "masked_ihc", "dish_mask_overlay",
                        "instance_mask", "dish_nucleus_mask", "overlay_annotated")

def _ensure_output_dirs(output_dir: Path, merge_dir: Optional[Path] = None) -> None:
    for name in _TILE_OUTPUT_SUBDIRS:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    if merge_dir is not None:
        (output_dir / "merge_overlay").mkdir(parents=True, exist_ok=True)

# _save_tile_array: drop `path.parent.mkdir(parents=True, exist_ok=True)`
# run_batch: call _ensure_output_dirs(output_dir, merge_dir) once, after
#            compute_tile_geometry and before the workers>1 branch (covers both paths)
# process_precut_tile: same call at the top (it is a standalone public entry point)
```

---

## 11. What this leaves open

| item | state after this round |
|---|---|
| **Full-WSI validation (backlog item 7)** | **Still open** — but two of its inputs are now measured rather than assumed: composition (55.8% background, §1) and Phase D's real cost (322.7 s, §5). Its wall estimate drops to ~2.6 h (`workers=1`) / ~1.25 h (`workers=4`). New sub-item: the stitch opens 27,565 files at once and no code checks `RLIMIT_NOFILE` (§5.2). |
| **Docs 19/23/24's 39%/61% composition claim** | **Superseded by measurement.** [`19-open-backlog.md`](./19-open-backlog.md) item 7 and doc 24 §0.1/§0.4 quote the brightness-proxy number; anything derived from it should be re-derived from §1.3. |
| **Candidate A (Phase D)** | Sized, not built. Before any GPU work, the cheaper question this measurement raises: the stage is superlinear and single-threaded, so **`tiffsave`'s own knobs** (bigger tile size, `pyramid` level control) and simply *not* re-encoding blank regions are untested and free of new dependencies. |
| **Candidate F (blank-tile writes)** | Not built for speed (zero wall). Separately worth **~157 GB of disk per slide** — background tiles write 10.2 MB each of uncompressed, entirely constant TIFF (§4.1). If anyone wants that back, the fix is `os.link`, and it is a storage decision rather than an optimization. |
| **Candidates B / D / E** | Closed on measurement, not estimate (§2.3/§2.4). Reopen only if MAIN sheds ~50%. |
| **`workers≥6` allocator balloon (item 7b)** | **New evidence, no fix.** At `workers=4` this round, one dmon sample hit **26,687 MB** of 32,607 while the median was 9,137 MB — the same one-worker balloon signature, at a worker count that has never OOM'd. It did not fail here, but it means the true headroom at `workers=4` is transiently ~6 GB, not the ~12 GB a steady-state reading suggests. |
| **Clinical sign-off on the round-3 checkpoint retrain** | Still pending and still blocking, unchanged by this round. |

One deployment-dependent caveat on §3.5's revert, recorded rather than assumed away: the
mkdir measurement was taken on local NVMe, where a redundant `mkdir` on an existing
directory costs ~2.6 µs. On a network filesystem the same call is typically **three
orders of magnitude** more expensive, and 165,390 of them would stop being noise. If
slide output is ever written to NFS/SMB, re-measure §3.1 before assuming this conclusion
still holds — the patch in §10 is the fix, unchanged.

---

## 12. Reproducing this round

```bash
M=docs/hybrid-pipeline/measurement/_metrics_r7
SLIDE=/data/nvmessd/storge_tsgh/<case>/output

# §1.3 -- is the slide really 39% background? (real M1 forward, ~45 s for n=800)
.venv/bin/python scripts/tissue_calibrate.py --ihc $SLIDE/HER2_processed.tiff \
    --n 800 --seed 1 --out $M/tissue_calibration_n800.json

# §1.4 -- the same rule over the whole 27,565-tile grid (~25 min)
.venv/bin/python scripts/core_mask_map.py --ihc $SLIDE/HER2_processed.tiff \
    --out $M/core_mask_map.npz

# §2.1 / §7 -- cut a crop with a chosen composition (exact map, or brightness proxy)
.venv/bin/python scripts/composition_crop.py --ihc $SLIDE/HER2_processed.tiff \
    --dish $SLIDE/DISH_processed.tiff --grid 24 --map $M/core_mask_map.npz \
    --out-ihc test_picture/_roi_crops/match24_ihc.tiff \
    --out-dish test_picture/_roi_crops/match24_dish.tiff --report $M/match24_crop.json

# §2.2 / §3.3 -- the A/B matrix (control vs candidate x workers 1,4, n=2, interleaved).
#   NOTE: the arms differ by ONE line in _save_tile_array, and `workers>1` spawns
#   re-import the module from disk, so the file itself must be the switch -- a runtime
#   monkeypatch cannot reach the workers.
.venv/bin/python scripts/perf_measure.py --ihc <crop>_ihc.tiff --dish <crop>_dish.tiff \
    --output output/r7_run --label control_w1_r1 --workers 8 --stream-precut \
    --gpu-dmon --mp-workers 1 --metrics-dir $M
.venv/bin/python scripts/arm_report.py --metrics-dir $M --detail
.venv/bin/python scripts/gc_ablation_report.py --metrics-dir $M --runs-dir $M/runs \
    --reference $M/runs/control_w1_r1/report.csv --baseline control_w1_r1   # §3.4 veto

# §3.2 / §4.2 -- mkdir + blank-write + hardlink at P concurrent processes
.venv/bin/python scripts/fs_write_probe.py --procs 1,2,4,6 --iters 80 \
    --out $M/fs_write_probe.json

# §5 -- Phase D at real scale (needs one run's overlay_annotated/ as the source pool)
for dims in 35840x28928 71168x57344 141818x114366; do
  .venv/bin/python scripts/stitch_probe.py --overlay-src <run>/overlay_annotated \
      --slide-w ${dims%x*} --slide-h ${dims#*x} --out $M/stitch_probe_${dims}.json
done

# §6 -- full-WSI projection from measured rates + measured composition
.venv/bin/python scripts/wsi_projection.py --timings $M/control_w1_r1_timings.json \
    --background-share <measured> --stitch-s 322.7 --out $M/wsi_projection.json

# §7 -- cellpose G=16 (needs a tissue-dense crop; --tiles must exceed the tissue count)
.venv/bin/python scripts/cellpose_batch_probe.py --ihc <dense>_ihc.tiff \
    --dish <dense>_dish.tiff --tiles 32 --groups 1,16 --reps 3 \
    --out $M/b1_cellpose_batch_probe_g16.json

# §8 -- environment gate. NEVER into the project venv (doc 24 §3: round 3's bundled
#       `uv sync` regression is still unexplained).
.venv/bin/uv venv /tmp/spike-venv --python 3.11
.venv/bin/uv pip install --python /tmp/spike-venv/bin/python \
    numpy pyvips cupy-cuda13x nvidia-nvimgcodec-cu13 nvidia-nvtiff-cu13 nvidia-nvcomp-cu13
.venv/bin/uv pip install --python /tmp/spike-venv/bin/python \
    --extra-index-url https://pypi.nvidia.com cucim-cu13
/tmp/spike-venv/bin/python scripts/gpu_codec_spike.py --out $M/gpu_codec_spike.json
```
