# 16 — `gc.collect` frequency reduction: measurement result and Choose decision

> Executes the experiment matrix in [`14-gc-collect-frequency-plan.md`](./14-gc-collect-frequency-plan.md) §4
> and records the Choose-phase outcome per
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md).
> Implementation record: [`15-gc-collect-frequency-implementation.md`](./15-gc-collect-frequency-implementation.md).
>
> **Outcome: Option C (`gc.freeze()`) adopted — −37.0 s of main-thread work at the 441-tile anchor,
> ≈1.07x end-to-end, landing exactly on this priority's Amdahl ceiling. Option A (fixed-N
> `gc.collect()` batching) was implemented, measured, found to add nothing on top of C, and deleted.**

## 1. Headline

| anchor | baseline | adopted (`gc.freeze()`) | delta | speedup |
|---|--:|--:|--:|--:|
| large / 441 tiles (n=3 each) | 571.5 s | **530.4 s** | **−41.1 s** | **1.077x** |
| — attributable component only | 109.7 s | **72.7 s** | **−37.0 s ± 1.5** | 1.069x |
| medium / 121 tiles (n=2 each) | 166.4 s | **157.3 s** | −9.1 s | 1.058x |

`gc.collect()` itself, large anchor: **36.71 s → 0.52 s** (−98.6%), from **83.2 ms → 1.2 ms per
call**, with the call count unchanged at 441 — once per tile, exactly as before.

Read the second row, not the first, as the size of the effect. §2.2 explains why: end-to-end wall at
the large anchor carries ±3% noise from the Cellpose GPU stage that has nothing to do with this
change, so the 41.1 s mean difference contains ~4 s of luck. The attributable, reproducible number is
**37.0 s**, and it matches the 36.2 s of `gc.collect()` removed almost exactly.

## 2. Method and its limits

Harness `scripts/perf_measure.py --gpu-dmon --workers 8`, both ROI anchors from
`backend/algorithms/hybrid/test_picture/_roi_crops`, GPU confirmed idle before launch, all cells run
back-to-back on one otherwise-idle machine. Raw artifacts in `measurement/_metrics_gc/` (timings
JSON, 0.5 s resource CSV, `nvidia-smi dmon`, `pip freeze`, per-run stdout); kept `report.csv` per cell
in `measurement/runs_gc/`. Analysis: `scripts/gc_ablation_report.py`.

### 2.1 Two plan assumptions that had to be dropped

- **Exp 0 had to be re-measured, and it mattered.** Round 3's numbers (707.4 s large / 208.2 s
  medium) do not reproduce: today's baseline is 571.5 s / 166.4 s on unmodified code. The environment
  has drifted since round 3 — exactly the risk plan §4 Exp 0 exists to cover. **Every delta here is
  against today's Exp 0, never round 3.**
- **Round 3's `report.csv` is no longer a valid correctness reference.** Plan §4 says to check cell
  counts against it, but that assumed HEAD == round-3 code. It does not: `9e618d3` (cached
  morphological disk footprint — dot detection) and `5ee788c` (remove legacy code) landed afterwards.
  Today's baseline differs from round 3's CSV by 89 cells and 462 excluded-flag flips — real
  computational drift, not GC. The bar used here is **today's Exp 0 `report.csv`**, which is the
  correct control anyway: it isolates the GC change and nothing else.

### 2.2 The noise floor — and a mistake this caught

Medium is quiet: two identical Exp 0 runs differed by **0.2 s (0.1%)**. That number does not
transfer to the large anchor, and initially assuming it did produced a wrong conclusion.

Three baseline and three freeze runs at the large anchor, decomposed:

| condition | e2e s | `B1_*` GPU stage | residual (everything else, incl. gc) | gc s |
|---|--:|--:|--:|--:|
| baseline ×3 | 572.9 / 577.6 / 564.0 | 443.0 / 446.7 / 434.4 | 109.6 / 110.5 / 109.0 | 36.71 / 36.70 / 36.71 |
| freeze ×3 | 512.2 / 540.3 / 538.7 | 419.7 / 446.4 / 445.4 | 71.9 / 73.4 / 72.8 | 0.52 / 0.53 / 0.52 |

- **`B1` (Cellpose GPU forward) is pure noise here**: baseline spans 434.4–446.7 s, freeze spans
  419.7–446.4 s. The ranges overlap almost completely. Freeze cannot make CUDA kernels faster, and
  the data agrees it doesn't.
- **The residual is the clean signal**: 109.0–110.5 → 71.9–73.4. Non-overlapping, ±1.5 s within each
  condition, **−37.0 s** between them.

An earlier draft of this document reported 1.118x and claimed the win *beat* its Amdahl ceiling via a
second-order GIL/automatic-GC effect, because the first freeze run happened to draw a low `B1`
(419.7 s) and was the only one taken. Two further replicates refuted it. **There is no second-order
effect: the win is exactly the explicit `gc.collect()` cost removed from the main thread**, and it
lands on the ceiling rather than above it, which is the best a single-cost removal can do. Recorded
here rather than quietly fixed, because "one run, theory-beating result" is the playbook's red flag
#3 and it very nearly went into the codebase as fact.

## 3. Full ablation table

`N` = `gc.collect()` every N tiles; `frz` = `gc.freeze()` after init.

**Medium anchor (121 tiles)** — the cheap sweep that located the mechanism:

| cell | N | frz | e2e s | gc calls | gc s | **ms/call** | peak RSS GB |
|---|--:|:--|--:|--:|--:|--:|--:|
| exp0 base | 1 | off | 166.3 | 121 | 9.91 | 81.9 | 3.096 |
| exp0 repeat (noise) | 1 | off | 166.4 | 121 | 9.92 | 81.9 | 3.066 |
| **exp1 — C alone** | 1 | **on** | **157.0** | 121 | **0.09** | **0.7** | 3.073 |
| exp2a — A alone | 4 | off | 159.4 | 31 | 2.54 | 82.0 | 3.061 |
| exp2b — A alone | 8 | off | 159.1 | 16 | 1.31 | 82.0 | 3.066 |
| exp2c — A alone | 16 | off | 156.6 | 8 | 0.66 | 83.0 | 3.077 |
| exp4 — A+C | 8 | on | 156.9 | 16 | 0.10 | 6.2 | 3.023 |
| exp4 — A+C | 16 | on | 156.2 | 8 | 0.09 | 11.2 | 3.023 |
| adopted (shipped) | 1 | on | 157.5 | 121 | 0.08 | 0.7 | 3.040 |

**Large anchor (441 tiles)**, all cells doing identical work (378 success / 63 skipped):

| cell | N | frz | e2e s | residual s | gc s | peak RSS GB |
|---|--:|:--|--:|--:|--:|--:|
| baseline (mean of 3) | 1 | off | 571.5 | 109.7 | 36.71 | 3.881 |
| **adopted (mean of 3)** | 1 | **on** | **530.4** | **72.7** | **0.52** | 3.925 |
| exp4 — A+C | 8 | on | 535.1 | 70.4 | 0.16 | **3.991** |

## 4. What the numbers say

### 4.1 The cost driver was per-call scan volume, not call count

The `ms/call` column with freeze **off** is the whole story: **81.9 / 82.0 / 82.0 / 83.0 ms** at
N = 1 / 4 / 8 / 16. Batching changes how many times you pay; it never changes the price. Freeze
changes the price: **83.2 ms → 1.2 ms, a 69x cut**, because what every full collection was walking is
the resident model graph — UNet++ (EfficientNet-B4) plus two Cellpose SAM-ViT models, held alive for
the entire batch by design.

Option A was aimed at the wrong variable. That is only visible once both are measured, which is
precisely why plan §4 insisted on isolating C before combining it.

### 4.2 Option A contributes nothing on top of C — deleted

With freeze on, total explicit gc cost is 0.52 s of a 530 s run: **0.10% of wall**, an Amdahl ceiling
of 1.001x. Batching can recover at most 0.36 s of that. Measured:

- medium, A+C vs C alone: −0.1 s at N=8 — inside the ±0.2 s noise floor.
- large, A+C (535.1 s) vs adopted range (512.2–540.3 s) — **indistinguishable**.
- peak RSS: A+C at N=8 gives **3.991 GB, the highest of any configuration measured**, against 3.881
  baseline and 3.925 adopted.

So Option A buys nothing measurable and costs a little memory, plus a config field, a hot-loop
branch, an end-of-loop special case, and an RSS-generalisation risk that plan §2 would have required
re-validating per slide density. Per the playbook's Choose step — *zero-contribution layers get cut
no matter how clever* — it is gone.

Note it is **not** a regression: an earlier draft called it "+22.9 s slower", which was the same
single-run `B1` artifact described in §2.2. It is simply zero.

### 4.3 Correctness veto: passed

`adopted` vs `baseline` at the large anchor: 13148 vs 13149 cells, 13139 matched by nearest centroid
(<3 px), and among matched cells **max|Δ| = 0 for reddot, blackdot, and score**; 18 excluded-flag
flips against a floor of **15 flips between two identical runs**. `gc.freeze()` changes only which
objects the collector *scans*, never what is reachable, and the output confirms it.

### 4.4 The memory-bounded invariant: untouched by construction

Plan §2 required the two-part RSS check for every option except C, because C alone does not change
collection frequency. That is exactly what shipped: `gc.collect()` still runs **once per tile**.

Measured anyway — peak RSS 3.881 → 3.925 GB (**+0.044 GB, +1.1%**), and the sawtooth growth shape is
intact (272 reclamation events >50 MB vs 204 at baseline; no monotonic ramp). The small rise is
expected: the frozen generation is never swept during the batch. It is 0.14% of the 32 GB machine,
and since garbage accumulation itself is unchanged, there is **no full-WSI re-validation burden** —
the concern plan §0 raised about cadence tuned on a 441-tile crop does not apply to a change that
does not touch cadence.

## 5. Decision (Choose phase)

**Adopted — Option C, unconditional.** `run_batch` wraps its tile loop in `_frozen_gc_generation()`.
No config knob: plan §3 asked for it to be unconditional if adopted, and no measured setting favours
turning it off.

**Rejected and deleted — Option A.** `config.gc_collect_every_n_tiles`, its validation, the
`if idx % N == 0` guard, the trailing end-of-loop sweep, and the `--gc-every-n` / `--gc-freeze`
harness flags are all removed. `config.py` / `config_example.py` are byte-identical to their
pre-change state (config hash unchanged at `db2b7e6a`), and `perf_measure.py` is back to its original
content.

**Not built — Options D, E, F, G.** Every one is a variation on *reducing collection frequency*, and
§4.1 shows frequency was never the cost driver; Option F (`gc.disable()`) is the most dangerous
version of the same wrong idea. The plan's escalation rule terminates here: escalate only if cheaper
options underperform, and C captured the ceiling.

## 6. Reproducing

```bash
# adopted code — no flags, freeze is unconditional
.venv/bin/python scripts/perf_measure.py \
  --ihc  backend/algorithms/hybrid/test_picture/_roi_crops/large_ihc.tiff \
  --dish backend/algorithms/hybrid/test_picture/_roi_crops/large_dish.tiff \
  --output <out> --label large --workers 8 --gpu-dmon --metrics-dir <m>

.venv/bin/python scripts/gc_ablation_report.py --metrics-dir <m> --baseline <label> \
  [--runs-dir <runs> --reference <baseline report.csv>]

# invariant guard — no GPU or slide data, ~1 min
.venv/bin/python scripts/verify_gc_freeze.py --control-ref <pre-change SHA>
```

To re-measure the baseline, revert the `_frozen_gc_generation()` wrapper in `run_batch`; git history
is the control group now that the flag is gone. **Use n≥3 at the large anchor** (§2.2) — a single run
there is not interpretable.

## 7. Follow-ups this raised

1. **`B1` needs n≥3 at the large anchor.** Cellpose GPU-forward wall varies ±3% run to run
   (434.4–446.7 s across identical baselines). Any future single-run large-anchor comparison claiming
   less than ~15 s of effect is not measurable. Decompose to the affected bucket instead of trusting
   end-to-end, as §2.2 does.
2. **The bottleneck has moved further toward Cellpose.** At the large anchor `B1_*` is now 437 s of a
   518 s `run_batch` — **84%**, up from 80%. Everything else combined is 81 s. Doc 13's remaining
   priorities should be re-ranked against today's Exp 0, and anything that is not the Cellpose GPU
   forward now has a ceiling below 1.19x.
3. **Round-3 references are stale** for both timing and correctness (§2.1). Regenerate reference CSVs
   at HEAD before using them as a bar.
4. **`perf_measure.py`'s `B4_gc_collect` counts explicit calls only.** It wraps `gc.collect`, so
   automatic generational collections are invisible to it. That did not matter here (§2.2 showed the
   explicit bucket fully accounts for the win), but a future GC question could need `gc.callbacks`.
