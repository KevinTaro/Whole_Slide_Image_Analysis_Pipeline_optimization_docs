# 41 — Round 13: shipping Phase D's pipelined stitch — implementation

> Executes [`40-round-13-phase-d-pipelined-stitch-plan.md`](./40-round-13-phase-d-pipelined-stitch-plan.md)
> §3 items 2 and 3. Doc 40's items 0 and 1 were already done when this round started (commit
> `244cca1`); item 4 was **not run** (§6); item 5 stays closed and this document argues it should
> now stay closed on its own merits (§5.2). **Candidate B is built, measured at full slide scale,
> and is the shipped default as of this round.**

## 0. Result in one table

| | round 12 (`pyvips`) | round 13 (`tifffile`) | |
|---|--:|--:|--:|
| **End-to-end wall, `workers=4`** | 5,854.9 s | **4,877.4 s** | **1.200x** |
| Phase D stitch (`D_stitch_overlay`) | 1,889.8 s | **987.8 s** | **1.913x** |
| Peak RSS | 45.56 GB | **16.95 GB** | −62.8% |
| `overlay_slide.tiff` | 7.50 GB | 5.85 GB | 0.78x |
| `report.csv` rows | 356,220 | 356,225 | **+0.0014%** |
| `config_hash` | `d2ccc46b` | `d2ccc46b` | identical |

Doc 40 §1.1 projected **1.135x** end-to-end with a perfect-overlap floor of **1.184x**. The measured
number is **1.200x — above both**. Doc 38 §4's stop-loss therefore does not trigger, and the
default was flipped to `"tifffile"` as a separate, explicit decision after this number existed
(§4), not as part of the build.

## 1. What was built (doc 40 §3 item 2)

`config.stitch_backend` selects between two encoders. `_stitch_overlay_slide` is now a dispatcher;
the shipped body moved **verbatim** into `_stitch_overlay_slide_pyvips`, so the old path is
byte-for-byte preserved as a fallback and as a measurement control.

`_stitch_overlay_slide_tifffile` is the port of `scripts/stitch_probe.py`'s
`cand_tifffile_streamed()`: `_band_source` (one row of source tiles at a time, joined the same way
`_join_overlay_tiles` does — `arrayjoin` mis-pads non-uniform grids), `_prefetch_bands` (single
thread, depth 1) hiding the read behind the encode, `_shrink2_cpu` for the pyramid,
`_encode_tile_row` applying TIFF Predictor 2 itself, and `tifffile.TiffWriter` for the container.

Doc 40 §1.2 was right that this is a replacement, not a patch. Two deliberate deviations from the
spike code, both because the spike was measurement scaffolding and this is not:

1. **The per-phase timers, `keep_levels`, and the GPU-pyramid branch are gone.** Candidate C is a
   separate decision (§5.2); `read_s`/`pyramid_s`/`encode_s` exist to attribute a spike, and
   `perf_measure.py` already times Phase D from outside.
2. **`_slide_dims` replaces the spike's size probe.** `cand_tifffile_streamed` called
   `_join_overlay_tiles` purely to learn the finished height/width — which opens all 27,565 tiles
   at once. That call sat *before* the spike's `t_all`, so its cost was never in the 1.365x figure;
   importing it would have added unmeasured cost to the shipped path. `_slide_dims` reads one
   header per column and per row instead (332 opens at full scale), which is exact because same
   column ⇒ same width and same row ⇒ same height, by `core_crop_bounds`'s construction. The
   tifffile path consequently needs `_ensure_nofile_limit(cols + 256)`, not `cols × rows + 256`.

`stitch_backend` is in `config._HASH_EXCLUDE`. It *does* change the output bytes, so this needs
justifying rather than asserting: neither hash consumer reads `overlay_slide.tiff`. The worker
guard compares parent-vs-child algorithm settings, and the resume checkpoint stores each tile's
`owned` cell list — both are fixed by M1–M3, which finish before Phase D starts. Including it would
discard a multi-hour full-slide checkpoint for swapping an encoder and buy no correctness. §2's two
runs confirm it behaves as intended: both report `config_hash = d2ccc46b`.

### 1.1 Tests

`backend/tests/test_stitch_scratch_cleanup.py` and `tests/test_stitch_pyramid_levels.py` were
**extended over both backends**, not replaced — the pyvips coverage they already had is intact.
3 tests → 8. Two are new:

- **`test_backends_agree_on_layout_and_level0_pixels`** — doc 32 §3's non-comparability guard as a
  test: same page count, same container tile size, and **every level pixel-identical**. That last
  part is stronger than it looks, because it pins `_shrink2_cpu` to pyvips's own `region_shrink`
  kernel. It is not an assumption: §3.2 measured on the real shipped 141658×114366 overlay that
  every level L3..L11 is a bit-exact 2×2 box shrink of the level above.
- **`test_unknown_backend_fails_loudly`** — a typo'd backend raises rather than silently falling
  through to pyvips, which would make every measurement of the switch a lie.

## 2. The full-slide confirmation (doc 40 §3 item 3)

Same protocol as doc 39 §2, same conformed input pair, same host, `workers=4`, `--stream-precut`.
`scripts/perf_measure.py` gained `--stitch-backend` so the run does not require editing the
gitignored `config.py` and leaving it flipped; the chosen value is recorded in the run's
`_timings.json`.

| | r12 `pyvips` | r13 `tifffile` |
|---|--:|--:|
| `end_to_end_total_s` | 5,854.9 | 4,877.4 |
| `runbatch_BCD_s` | 5,854.8 | 4,877.3 |
| `D_stitch_overlay` | 1,889.8 | 987.8 |
| `peak_rss_gb` | 45.558 | 16.953 |
| tiles success / skipped | 10,801 / 16,764 | 10,800 / 16,765 |

**Phase D's 1.913x exceeds its own 1.365x spike figure.** That is not a contradiction: doc 39 §4.4
measured the cold read at **51.9% of Phase D at full scale against 48.7% at crop scale**, and the
lever hides the read — more read to hide is more win. Doc 39 flagged that transfer as the thing the
projection depended on; it transferred, and then some.

Phase D is now **20.3% of the `workers=4` wall** (987.8 / 4,877.4), down from 32.3%. It is no longer
the pipeline's largest single term.

### 2.1 Two effects nobody predicted

- **Peak RSS fell 45.6 → 17.0 GB.** Band-streaming replaces a pyvips lazy join holding 27,565 files
  open. Doc 27 §6 recorded 61–62 GB as "a new, previously-unstated host requirement"; this
  materially relaxes it. Not tuned for, not asked for, and worth stating plainly because it changes
  a documented deployment constraint.
- **The resolution tag is now exact.** pyvips/libtiff stores `XResolution` as a C `float`, so 25.4
  becomes `float32(25.4) = 25.399999618530273`, written as the rational `13316915/524288`; QuPath
  reads 1000.000015 µm/px and reports the slide as 141,658,002.13 µm wide. tifffile writes `127/5`
  = exactly 25.4 → exactly 1000 µm/px → 141,658,000.00 µm. A 1.5×10⁻⁸ relative difference, no
  effect on anything. **Both are placeholder calibration** — 1 px = 1 mm, roughly 4000× off a real
  40× pixel — so any µm measurement taken in QuPath off these overlays is equally meaningless in
  both. Unchanged by this round, recorded here because the difference is visible in QuPath and will
  otherwise be re-discovered.

## 3. Correctness veto

Unchanged from every prior round, plus doc 40 §1.4's QuPath gate (cleared 2026-07-29).

### 3.1 Tables

- `report.csv`: **356,225 vs 356,220 rows, +0.0014%** — tighter than round 12's −0.002% and round
  8's −0.01%.
- `summary.txt`: mean black dots 3.24 → 3.25, black total 148,385 → 148,417, red total 108,458 →
  108,462, valid cells 45,732 → 45,735. Verdict split identical to one decimal: ratio <2
  75.3%, ≥2 24.7% in both. Inside this project's own directly-measured run-to-run drift band.

Phase D cannot affect these — it runs after the tables are written. The deltas are the GPU
nondeterminism this project has measured every round; they are reported to show the run was a real
independent run, not to attribute anything to the encoder.

### 3.2 The artifact

- `scripts/overlay_pyramid_audit.py`: **PASS at every level**, `Predictor=2` consistent across all
  12 IFDs.
- **Pyramid self-consistency**: every level L3→L11 is a **bit-exact** 2×2 box shrink of the level
  above (`maxdelta=0`). This is a check the audit script does *not* do — it only tests per-level
  brightness — and it is what proves the ported `_shrink2_cpu` builds a mathematically correct
  pyramid at real scale rather than merely a bright-looking one.
- **No tile displacement**: 27,565 level-0 container tiles compared against the r12 artifact;
  the worst tile differs in 4.28% of its pixels (annotation nondeterminism). A misplaced tile would
  differ in ~100%.

## 4. The default flip

Doc 40 §3 item 2 specified the round-11 `expandable_segments` pattern: measure behind a flag, flip
the default as a **separate** deployment decision once the number exists. Both halves happened, in
that order. The number (§0) came back above projection with every gate passed, and the flip was
then made explicitly:

- `config_example.py` / `config.py`: `stitch_backend: str = "tifffile"`.
- `pyvips` is retained, not deleted — it is the fallback if this path misbehaves in the field, and
  it is the control arm any future Phase D measurement needs (playbook: never discard the control).

`compute_config_hash` still returns `d2ccc46b` after the flip, so existing resume checkpoints
remain valid across the change — the `_HASH_EXCLUDE` decision in §1 doing its job.

## 5. What this round did not do

### 5.1 Item 4 — the error-bar campaign — was not run

Doc 40 §3 item 4 (2–3 repeats each of `workers=1` and `workers=4`, 11–16 h of exclusive GPU) is
**still open and still n=1**. Doc 40 §1.5's reasoning is unchanged by this round's result: the
payoff was sourced from the spike and corroborated by an independent cold-read measurement, and
neither depends on the 32.3% figure being exact. This round adds one more full-slide data point
almost for free but does **not** put a variance band on anything. Every full-slide comparison in
this project's history remains n=1 vs n=1 (doc 37 §5, doc 39 §2.4).

### 5.2 Item 5 — candidate C — should stay closed

Doc 40 §3 item 5 gated candidate C on "item 3's confirmed number leaves headroom worth the extra
risk." It does not. Phase D is now 20.3% of the wall, and C's incremental over B at spike scale was
1.581/1.365 = 1.158x on Phase D alone — roughly 2–3% end-to-end, against taking the **first CUDA
allocation the `workers>1` parent has ever made** (doc 40 §1.3). The ratio of payoff to
architectural risk got materially worse, not better, because B took most of the win. Recommend
closing item 5 rather than leaving it as a stretch.

### 5.3 The QuPath report is parked, not resolved

A separate report arrived mid-round: some tiles rendering wrong on some slides, attributed to the
round-9 pyramid fix. It could not be reproduced in `r12_fullwsi_w4/overlay_slide.tiff`, and one
finding contradicts the attribution: **level 0 of the fixed file is byte-identical to
`overlay_slide.tiff.predictor-broken`** across 27,565 sampled container tiles (`maxdelta=0`). The
pre-fix file has `L0 pred=2, L1..L11 pred=1`, so its level 0 always decoded correctly and the fix
only ever touched the reduced levels — it cannot explain anything visible at full zoom. Also
checked clean: all 27,565 source tiles exactly their expected core-crop size (so
`join(expand=True)` never silently mis-padded), grid summing exactly to 141658×114366, and the
`workers=1`/`workers=4` stitches agreeing to 0.0002%. One false alarm worth recording: a 45×
pixel discontinuity across every core seam is the **intentional** blue dashed grid
(`config.draw_window_grid = True`), drawn on the left tile's last column. Parked pending a slide
that reproduces the report.

**One gap this exposed, worth closing regardless:** `overlay_pyramid_audit.py` tests per-level
brightness only, so it passes a file with a tile-placement defect. §3.2's level-vs-downsample check
and a source-tile-size check are both cheap and strictly stronger. Not done this round.

## 6. Reproduce

```bash
# §2 — the full-slide workers=4 run with the ported backend (1 h 22 m on the reference host)
.venv/bin/python scripts/perf_measure.py \
  --ihc  /home/taro/full_wsi_validation/_conformed/ihc_conformed_141658x114366.tiff \
  --dish /home/taro/full_wsi_validation/_conformed/dish_conformed_141658x114366.tiff \
  --output /home/taro/r13_fullwsi_w4_tifffile --label r13_fullwsi_w4_tifffile \
  --mp-workers 4 --stream-precut --stitch-backend tifffile \
  --metrics-dir /home/taro/r13_fullwsi_w4_tifffile/_metrics

# §3.2 — correctness veto on the produced overlay
.venv/bin/python scripts/overlay_pyramid_audit.py \
  /home/taro/r13_fullwsi_w4_tifffile/overlay_slide.tiff

# §1 — the ablation that says the PORT kept the spike's win, at doc 35 §3.2's 4.055 GP scale.
# --stitch-backend times the shipped _stitch_overlay_slide once per backend, rebuilding the
# input between arms (a successful stitch deletes it). --candidates measures the spike code;
# this measures the code that actually ships.
.venv/bin/python scripts/stitch_probe.py \
    --overlay-src /home/taro/full_wsi_validation/fullwsi_w1/overlay_annotated \
    --stitch-backend pyvips,tifffile --slide-w 70909 --slide-h 57183 \
    --root /tmp/probe_4gp --out /tmp/port_4gp.json
```

The 4.055 GP ablation, run before spending 1.6 h of GPU on §2:

| backend | wall | output | audit |
|---|--:|--:|:--:|
| `pyvips` | 63.30 s | 2.44 GB | PASS |
| `tifffile` | 46.52 s | 1.89 GB | PASS |

**1.361x against doc 39 §4's 1.365x for the spike — reproduced inside 0.3%**, with the same
2.44 → 1.89 GB Predictor-2 size ratio doc 35 §3.4 recorded. That is what said the port had carried
the win before any full-slide time was committed.

## 7. What is still owed

1. **Item 4, the full-slide error-bar campaign** (§5.1) — untouched, still the honest gap.
2. **Harden `overlay_pyramid_audit.py`** with §3.2's pyramid-consistency and source-tile-size
   checks (§5.3) — cheap, and it closes a real hole in the shipped audit.
3. **The QuPath render report** (§5.3) — needs a reproducing slide.
4. **Real resolution calibration** (§2.1) — both backends write placeholder 1 px/mm. Out of scope
   here, but it is now written down.
