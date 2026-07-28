# 32 — Phase D (slide-level TIFF stitch) GPU port: Phase 1 spike results

> Implements [`29-phase-d-gpu-port-plan.md`](./29-phase-d-gpu-port-plan.md) — **Phase 1 only**,
> which doc 29 §3 makes the entire gate for whether Phase 2 is ever built. Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md).
> Round 9. Companions: [`31-gc-collect-round2-implementation.md`](./31-gc-collect-round2-implementation.md),
> [`33-tile-read-io-implementation.md`](./33-tile-read-io-implementation.md).

## 0. Outcome

Doc 29 §2.2 named two questions that had to be answered before any build estimate could be
trusted. **Both are now answered, and they point in opposite directions.** The container-assembly
problem everyone feared turns out to be *already solved* by a library the project already
depends on — `tifffile` accepts externally pre-compressed tile bytes, so nobody has to hand-write
BigTIFF offset tables. But **nvImageCodec cannot produce those bytes**: its TIFF encoder refuses
tiled output outright, and its strip layout is internal. The headline 19.2× GPU-encode route from
doc 25 §8.2 is therefore **dead** — not on speed, on output shape. What replaced it is a result
nobody predicted: doc 29 §1.2 attributed ≈55% of Phase D's encode cost to irreducible
"container/tile-buffer/TIFF-structure assembly," and that figure is **an artefact of pyvips, not
a property of the problem** — `tifffile`'s container write is ~1% of the same work. The
measured winner is a **CPU codec + `tifffile` container with GPU-generated pyramid levels**, and
it lands **just above** doc 29 §1.3's corrected ceiling (1.109× vs 1.093× at `workers=4`) — which
is §2.3's stated condition for justifying more engineering, though by a thin enough margin that
§6's unmeasured half could erase it.

**The QuPath check has since been run, and it earned its keep** (§5). First pass: A and B opened,
**C crashed**. Diagnosing that crash found a predictor-encoding bug in this spike *and* a
pre-existing, clinical-facing defect in the shipped pipeline's own output that had survived eight
rounds — see §5.1. Both are fixed, the timings in §3 are post-fix, and on the second pass **all
three render identically, so candidate C's correctness veto is cleared.** What remains open for C
is sizing (§4, §6), not correctness.

**The most consequential result of this round is not the spike.** It is §5.1(b): every
`overlay_slide.tiff` this pipeline has written decodes its pyramid levels as noise, so the
zoomed-out view a pathologist sees was broken. That is fixed, tested, and confirmed in QuPath.

## 1. §2.2 question 1 — pre-compressed tile passthrough: **YES**

Doc 29 §2.2 asks whether any maintained Python TIFF writer accepts externally pre-compressed tile
bytes, and warns that if not, *"the alternative is writing tile bytes into a BigTIFF structure by
hand (offsets, tile byte-count arrays, IFD chaining) — a materially larger and riskier
undertaking … and would change this plan's recommendation."*

It does. `tifffile.TiffWriter.write` accepts `data` as `Iterator[bytes]`, documented as *"Iterator
bytes must be compatible with the `compression`, `predictor`, `subsampling`, and `jpegtables`
arguments."* Verified rather than trusted (`scripts/phase_d_container_spike.py
--verify-passthrough`): LZW-compressed tiles with horizontal differencing applied, handed
straight to `tifffile`, produce a tiled BigTIFF that reads back **pixel-identical**.

`tifffile` (2026.3.3) is **already a project dependency**, so this route adds nothing to the
dependency surface — which matters given doc 24 §3's standing rule about installing into the
project venv.

One trap worth writing down because it fails silently: `tifffile` only *declares* the
`predictor` tag for pre-compressed input, it does not apply the transform. Passing raw LZW bytes
while declaring `predictor=True` yields a file that opens fine and decodes to garbage. The
spike applies the differencing itself.

## 2. §2.2 question 2 — can nvImageCodec produce those bytes? **No**

This is new; no prior document checked it. Three findings, in the throwaway venv doc 24 §3
requires (never the project venv):

**(a) nvImageCodec's TIFF encoder refuses tiling.** Setting `EncodeParams(tile_width=...,
tile_height=...)` — the only tiling control the API exposes — produces:

```
[WARNING] [nvtiff_cuda_encoder] Tiling is not supported with TIFF encoder.
[WARNING] [opencv_tiff_encoder]  Tiling is not supported with OpenCV encoder.
[WARNING] [pynvimgcodec] Something went wrong during encoding image #0 ...
```

and the encode returns `None`.

**(b) Its strip layout is internal and fine-grained**, targeting roughly 8 KB per strip:

| buffer | RowsPerStrip | segments |
|---|--:|--:|
| 256² | 10 | 26 |
| 512² | 5 | 103 |
| 1024² | 2 | 512 |
| 4096² | 1 | **4,096** |

**(c) Those segments cannot be merged into one tile.** A TIFF tile must be exactly one LZW
stream, and independent LZW streams cannot be concatenated — each ends in an EOI code, so a
decoder stops at the first. Measured, not asserted: concatenating two independent LZW streams and
decoding yields **3,072 of the expected 6,144 bytes**, i.e. only the first substream.

So there is no path from nvImageCodec's output into a tiled container's tile bytes. The 19.2×
figure remains real for what it measured (doc 25 §8.2's encode-throughput microbenchmark, which
this round independently reconfirmed produces LZW / compression tag 5, photometric RGB,
predictor 2, lossless) — it just cannot be spent on this pipeline's required output shape.

### 2.1 A correction to doc 25 §8.1's environment table

Doc 25 §8.1 records `nvidia-nvtiff-cu13` as **"no Python module at all … the wheel ships shared
libraries only"** and marks it unusable. That is accurate about direct import and misleading
about consequence: **that wheel is exactly what gives nvImageCodec its TIFF codec.** Without it
installed, `nvimgcodec` loads but its nvTIFF extension fails to register and TIFF encoding is
unavailable. The row should read "no Python binding, but required for nvImageCodec TIFF support"
rather than "unusable".

## 3. The spike — what was measured

Because §2 removes the GPU codec, the remaining live question is doc 29 §2.2's own conservative
fallback: *"is it faster to let `tifffile` (or another CPU library) do both compression and
container writing on data whose pyramid levels were generated on GPU?"* Three candidates, all
producing the format §2.1 requires (**LZW + tiled + pyramidal + BigTIFF**), on an overlay-like
synthetic slab (the `gpu_codec_spike.py` content model, so LZW compresses as it does on real
annotated tiles):

- **A `pyvips_tiffsave`** — literally what `_stitch_overlay_slide` calls today. The baseline.
- **B `tifffile_cpu`** — `imagecodecs` LZW per tile (8 threads) → `tifffile` pre-compressed
  passthrough; pyramid downsampled on CPU. **Uses no GPU at all.**
- **C `tifffile_gpu_pyramid`** — same container path, pyramid levels downsampled on GPU (torch).

Every candidate's output shape is **checked against the baseline's**, not assumed: page count and
tile size must match, or a candidate that wrote a shallower pyramid would post a speedup for
having done less work. (The first run of this spike did exactly that — 256px tiles and a pyramid
stopping at 512px, against pyvips's 128px/full-depth — and the numbers below are from the
corrected, shape-matched configuration.)

16,384² slab = 0.268 GP, best of 3. **All outputs verified: 8 pages / 128×128 tiles / LZW /
BigTIFF / resolution tags matching the baseline / and every pyramid level pixel-identical** —
the last of those being the check that §5.1 added after QuPath caught what its absence hid:

| candidate | wall | vs pyvips | pyramid | encode | **container** | output size |
|---|--:|--:|--:|--:|--:|--:|
| **A** `pyvips_tiffsave` (today) | 2.84 s | 1.000× | — (internal) | — | — | 42.38 MB |
| **B** `tifffile_cpu` (no GPU at all) | 2.028 s | **1.4004×** | 1.01 s | 0.996 s | **0.022 s** | 42.10 MB |
| **C** `tifffile_gpu_pyramid` | **1.323 s** | **2.1466×** | **0.081 s** | 1.222 s | **0.021 s** | 42.15 MB |

Two things fall out of the breakdown:

- **Container assembly is 0.021 s — ~1.6% of candidate C's total.** Writing the TIFF structure
  around pre-compressed tiles is nearly free.
- **GPU pyramid generation is ~12.5× faster than CPU** (1.010 s → 0.081 s), and that single
  substitution is the entire difference between B and C.

## 4. What this means for the ceiling — doc 29 §1.3, revisited

Doc 29 §1.2 decomposed doc 27's ablation to conclude that pyramid ≈22% and LZW ≈23% of encode,
leaving *"≈55% — the container/tile-buffer/TIFF-structure assembly itself — as a cost neither
bound touches"*, and §1.3 built its 1.05–1.09× corrected ceiling on that residual being
irreducible.

**The residual is not irreducible; it is pyvips-specific.** The spike times container writing
separately, and `tifffile`'s pre-compressed container write is ~1% of candidate C's total. The
≈55% doc 29 attributed to "TIFF structure assembly" is overwhelmingly pyvips's own per-tile
pipeline overhead, and a different writer simply does not pay it.

This is precisely the case doc 29 §2.3's decision point anticipated:

> *"If Phase 1 lands meaningfully above both estimates (e.g., because `tifffile`'s own
> tile-writing overhead turns out cheaper than pyvips's regardless of compression codec, which is
> possible and not something §1.3's analysis could predict from the ablation data alone), that is
> new information justifying more engineering."*

Translating the measured 1.400× / 2.147× onto Phase D's real cost (doc 27 §6.4/§6.5:
1,185.4 s = 8.6% of a 13,762 s `workers=1` wall; 1,200.8 s = 19.3% of a 6,211 s `workers=4`
wall), with `join()` ≈5% of Phase D left untouched:

| | Phase D after | saving | `workers=1` ceiling | `workers=4` ceiling |
|---|--:|--:|--:|--:|
| **B** CPU codec + `tifffile` | ~863 s | ~322 s | 1.024× | **1.055×** |
| **C** + GPU pyramid | ~583 s | ~602 s | 1.046× | **1.109×** |
| doc 29 §1.3's estimates | — | — | — | 1.052× / 1.093× |

**Candidate C clears the higher of doc 29 §1.3's two estimates (1.109× vs 1.093×) — but by 1.6
percentage points, and B lands essentially on top of the lower one.** So §2.3's condition for
"new information justifying more engineering" is met on paper, and only just. Set against §5.2's
outstanding QuPath re-check and §6's unmeasured read/join half, the honest reading is that this
justifies a **scale-accurate Phase 1.5**, and nothing more. It is not a green light for Phase 2,
and the margin is small enough that Phase 1.5 could plausibly erase it.

## 5. The QuPath check — run twice, and it earned its veto

Doc 29 §2.3 step 5 and §3 are unambiguous: *"correctness is a veto, not a tiebreaker."* It was
run, it failed, the failures were root-caused, and it was run again:

| candidate | QuPath, first pass | QuPath, after the §5.1 fixes |
|---|---|---|
| A `pyvips_tiffsave` | opens | opens |
| B `tifffile_cpu` | opens | opens |
| C `tifffile_gpu_pyramid` | **crashes the application** | **opens** |

The first pass also showed A and B/C **displaying at different scale**. All three now render
identically. Both original observations were correct and neither was a QuPath quirk — between
them they exposed two separate real defects, one of which had been shipping for eight rounds.

**Consequence for candidate C: its correctness veto is cleared.** C is a valid encode path, not
a broken one. What is *not* cleared is its adoption into Phase D, which still turns on §6's
unmeasured read/join half and the thin margin in §4 — those are sizing questions, not
correctness ones.

### 5.1 Two real defects, one of them pre-existing and shipped

**(a) The spike's predictor encoding was wrong — my bug.** `_tile_bytes` computed horizontal
differencing as `np.diff(t, axis=1, prepend=t[:, :1])`, which sets each row's first output value
to `0` instead of the pixel's absolute value, destroying the row's base. Correct predictor 2 is
`out[0] = in[0]; out[k] = in[k] - in[k-1]`.

**Why every automated check missed it:** the synthetic image draws black grid lines every 64 px
and the container uses 128 px tiles, so *every tile's first column was exactly 0* — which makes
the buggy and correct encodings identical at level 0. `verify()` only checked level 0. The
reduced levels, where the grid lines no longer land on tile boundaries at value 0, were corrupt:
candidate C's pyramid degenerated to **pure black by level 7** (min = max = mean = 0), which is
what crashed QuPath. B was corrupt too and merely survived being opened.

Fixed two ways: the encoder now prepends zero, and `verify()` now checks **every** level against
what was handed to the writer. Post-fix, all levels are pixel-identical for B and C. (This is why
the corrupt files were also *smaller* — degenerate levels compress to nearly nothing. A candidate
posting both a speedup and a smaller output should have prompted a closer look than it got.)

**(b) `_stitch_overlay_slide` has been writing broken pyramid levels all along — pre-existing,
and the more serious finding.** Chasing the scale difference exposed it:

- libvips 8.15.1's `tiffsave` default is `predictor="horizontal"`. It writes tag **317
  Predictor=2 on the full-resolution IFD only**, while still horizontally differencing the
  **reduced** levels' pixel data.
- Any reader that honours each IFD's own tag therefore fails to un-difference levels 1..N.
  Measured on a **real `overlay_slide.tiff` produced by the pipeline** (not a synthetic file):

  | level | Predictor tag | decoded mean | exactly-zero pixels |
  |---|---|--:|--:|
  | L0 | 2 | 242.3 | 0.5% |
  | L1 | **absent** | 12.8 | **90.8%** |
  | L2–L8 | **absent** | 16.3 → 83.0 | 88% → 37% |

- This is **not** reader strictness: `tifffile` and libtiff (via Pillow) both decode it as noise.
  Proof the data itself is fine: un-differencing L1 manually, per tile, reproduces a 2×2 box
  shrink of the source **exactly** (max |Δ| = 0).

**Impact — confirmed in QuPath, not merely predicted.** Level 0 is correct, so the file opens and
looks right at full resolution, which is why this survived eight rounds and every existing test.
It goes wrong **when a pathologist zooms out** — the first thing they do. Checked directly on a
slide written before the fix: *"the old picture actually not normal."*

**Fix**, applied to `_stitch_overlay_slide`: `predictor="none"`. Nothing is differenced, so every
IFD is self-consistent and every level decodes correctly. Measured on a **real** overlay
(18,688² level 0), not a synthetic one:

| | file size | encode time | pyramid levels |
|---|--:|--:|---|
| `predictor="horizontal"` (previous default) | 93.35 MB | 4.14 s | **8 of 9 decode as noise** |
| `predictor="none"` (now) | 102.08 MB | 4.18 s | **all 9 correct** |

**+9.4% disk, time-neutral.** Doc 27 §3.2 had already measured this knob as speed-neutral, which
is confirmed here; what that round did not check was whether it was *correctness*-neutral, and it
was not. Paying ~9% disk to make the zoomed-out view exist is the same trade doc 27 §3.1 made
when it vetoed `zstd`.

A cheaper fix exists in principle — the reduced levels' *data* is already correct, only tag 317
is missing, so injecting it would keep the 9.4% — but that means rewriting BigTIFF IFD entries by
hand, exactly the fragile work doc 29 §2.2 wanted to avoid. Rejected on risk, not on cost. If
Phase 2 ever lands, `tifffile` writes the tag correctly on every level and the question disappears.

`tests/test_stitch_pyramid_levels.py` (2 tests) guards it: every pyramid level of a real
`_stitch_overlay_slide` output must decode bright and non-degenerate, and the Predictor tag must
be consistent across IFDs. Confirmed to **fail without the fix** (level 1 decodes to mean 6.9).

### 5.2 Still outstanding

- **Any `overlay_slide.tiff` produced before this fix has broken pyramid levels** and should be
  regenerated if it is still in use. This does **not** require recomputing a batch: with
  `_stitch_scratch/` still present the stitch re-runs on its own, and `run_batch(checkpoint=True)`
  short-circuits straight to `_finish_batch` when the checkpoint covers every tile. Where the
  scratch has already been cleaned up, the batch does have to be re-run.
- The fix is guarded going forward by `tests/test_stitch_pyramid_levels.py`, but **no
  full-slide** output has been regenerated and re-opened yet — only the crop-scale one.

## 6. Scale gap

The spike runs on a contiguous in-memory slab, where doc 29 §2.3 step 1 asked for the 4.055 GP /
6,900-tile scale doc 27 §3 screened at. Real Phase D never holds its input in memory: it joins
27,565 tile files lazily through pyvips so a 16.22 GP slide never materialises (the ≈400 GB
full-canvas failure mode the architecture exists to avoid). A `tifffile`-based Phase 2 needs an
equivalent lazy source — reading tiles from `_stitch_scratch/` in the container writer's tile
order — which is **real, unscoped work not covered by this spike**. The spike measures
encode + pyramid + container given data in memory; it does not measure the read/join half.

`scripts/stitch_probe.py`, which doc 29 §2.3 wanted reused for exactly this, was **broken by the
M0 split** and has been repaired this round (see doc 33 §5). That unblocks a scale-accurate
Phase 1.5 without new tooling.

## 7. Disposition

| doc 29 item | status |
|---|---|
| §2.2 Q1 — pre-compressed tile API exists? | **ANSWERED: yes**, `tifffile`, already a dependency, verified lossless |
| §2.2 Q2 — nvImageCodec can produce tiled bytes? | **ANSWERED: no** — tiling unsupported, strips internal, LZW streams unmergeable |
| Candidate A (nvImageCodec GPU encode) | **CLOSED — measured, sized, and killed on output shape**, in the manner doc 29 §3 asks for. Recorded here so it is not rediscovered |
| §2.3 Phase 1 spike | **DONE** — `scripts/phase_d_container_spike.py`, JSON in `measurement/_metrics_r9/phase_d_container_spike.json` |
| §2.3 step 5 — QuPath/BioFormats open | **DONE — failed, root-caused, fixed, re-checked.** C crashed on the first pass; after the §5.1 fixes all three open and render identically. **C's correctness veto is cleared** |
| §2.4 Phase 2 — full port | **NOT STARTED.** Correctness is no longer the blocker; the gate is now §6's unmeasured read/join half and §4's thin margin |
| `zstd` veto | Unchanged — still dead, still on correctness |
| **`_stitch_overlay_slide` pyramid levels (pre-existing, found here)** | **FIXED** — `predictor="none"`; every shipped `overlay_slide.tiff` before this had pyramid levels that decode as noise (§5.1b). 2 tests |

**Recommended next steps, in order:**

1. **Regenerate any `overlay_slide.tiff` still in use that predates the §5.1b fix** — its
   pyramid levels are broken. Re-stitching alone is enough where `_stitch_scratch/` survives
   (§5.2); no batch recompute is needed in that case.
2. Re-run the spike at 4.055 GP via the repaired `stitch_probe.py`, sourcing real overlay tiles
   from disk rather than a memory slab, so the read/join half (§6) is included.
3. Only then scope Phase 2 against `_join_overlay_tiles` / `_stitch_overlay_slide`.
4. Note that candidate **B needs no GPU at all**. If it holds up at scale and through QuPath, it
   is a materially smaller and lower-risk change than any GPU port — and doc 29 §3's own rule
   ("prefer the simplest solution that clears the bar") would favour it over C unless C's margin
   justifies the extra dependency on torch inside Phase D.
