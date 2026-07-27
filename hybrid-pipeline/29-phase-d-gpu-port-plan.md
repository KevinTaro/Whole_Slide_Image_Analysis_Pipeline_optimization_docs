# 29 — Phase D (slide-level TIFF stitch) GPU port: design plan

> **Design-only document — no pipeline code changed here.** Follows
> [`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md):
> Discover → Analyze → Plan → Choose. Picks up the item
> [`26-remaining-work-implementation-plan.md`](./26-remaining-work-implementation-plan.md) Tier 1.2
> named and deferred pending Tier 1.1's cheap-knob ablation, which
> [`27-remaining-work-implementation.md`](./27-remaining-work-implementation.md) §3 has since run
> to completion — all cheap knobs are closed negative except `zstd`, which is vetoed on
> correctness (QuPath/BioFormats cannot read it). This document is the "scope a design first"
> step doc 26 asked for before any nvImageCodec engineering starts. Related evidence, all cited
> rather than re-derived: [`24-gpu-encode-decode-loop-acceleration-plan.md`](./24-gpu-encode-decode-loop-acceleration-plan.md)
> §2.1 (Candidate A, the original survey entry), [`25-gpu-encode-decode-loop-acceleration-implementation.md`](./25-gpu-encode-decode-loop-acceleration-implementation.md)
> §5 and §8 (real-scale stitch measurement, nvImageCodec spike), doc 27 §3 (the 13-config
> `tiffsave` ablation) and §6.6 (the ceiling revision this plan responds to).

## 0. Where this stands today

| fact | value | source |
|---|---|---|
| Real Phase D cost (16.22 GP, real 27,565 distinct tiles) | **1,185.4 s** at `workers=1` (8.6% of wall), **1,200.8 s** at `workers=4` (19.3% of wall) | doc 27 §6.4/§6.5 |
| Synthetic probe's own estimate of the same stage | 322.7 s — **3.7× optimistic**, because its inputs are hard-linked duplicates that compress/cache far better than 27,565 genuinely distinct tiles | doc 27 §6.4 |
| Cheap `tiffsave` knobs (tile size, pyramid depth, predictor, `region_shrink`, `subifd`, `deflate`) | **All dead** — tile size monotonically worse, others no-ops or worse | doc 27 §3.2 |
| `zstd` level 1 | 1.2331× faster, 13.8% smaller, verified lossless — **but QuPath/BioFormats cannot open it** | doc 27 §3.1 — vetoed on correctness |
| Naive Amdahl ceiling if Phase D were eliminated entirely | 1.094× (`workers=1`) / **1.239×** (`workers=4`) | doc 27 §6.6, `19-open-backlog.md` #1b |
| nvImageCodec raw encode throughput vs. pyvips's actual call | **1,789.6 MP/s vs. 93.0 MP/s = 19.2×**, verified bit-identical LZW output | doc 25 §8.2 |
| What nvImageCodec does **not** produce | No pyramid, no BigTIFF, single page | doc 25 §8.2 |
| VRAM cost | ~0.5 GB (codec context) + ~0.7 GB (working buffer) per process; **exempt from the `workers>1` VRAM gate** because Phase D runs once, in the parent process, outside the worker pool | doc 25 §8.3 |

This document's job is narrow: **is the 19.2× encode number actually worth the "real engineering,
currently undesigned" container-assembly work every prior document has flagged but never scoped?**
§1 shows the honest answer is "less than the headline number suggests, but plausibly still worth a
cheap spike" — and §2 designs that spike.

## 1. Analyze — the 19.2× figure does not transfer to a 19.2× (or even 1.239×) Phase D win

This is the step no prior document has done, and it changes the priority of everything after it.

### 1.1 The 19.2× and the 1.239× ceiling are two different, incompatible framings

- **19.2×** (doc 25 §8.2) is a **microbenchmark** ratio: nvImageCodec encoding a single 4096²
  buffer to LZW TIFF, vs. pyvips's `tiffsave(..., compression="lzw", tile=True, pyramid=True,
  bigtiff=True)` on the same buffer. It measures *encode-call throughput*, and pyvips's call
  includes pyramid generation that nvImageCodec's does not — so part of that 19.2× is "GPU codec
  vs. CPU codec" and part of it is "no pyramid vs. pyramid," conflated into one number.
- **1.239×** (doc 27 §6.6) is the Amdahl ceiling for **eliminating Phase D entirely** — i.e., what
  the pipeline would gain if the stitch stage cost zero seconds. No GPU port makes an operation
  free; it makes the *compressible* part of the operation faster. Nothing in doc 24/25/27
  computes what fraction of Phase D's cost a GPU codec swap can actually reach.

### 1.2 Decomposing Phase D's own ablation table to find that fraction

Doc 27 §3's 13-config ablation, screened at 4.055 GP (92×75 = 6,900 tiles), gives exactly the
numbers needed — they were measured for a different purpose (ranking `tiffsave` knobs) but
contain the answer to this question too:

| config | encode s | vs. baseline |
|---|--:|--:|
| `baseline` (LZW + tile + pyramid + BigTIFF — what the pipeline calls) | 57.02 | — |
| `BOUND_no_pyramid` (LZW, no pyramid) | 44.23 | **−12.79 s** ⇒ pyramid generation ≈ 22.4% of encode |
| `BOUND_no_compression` (no LZW, still pyramid) | 43.96 | **−13.06 s** ⇒ LZW compression ≈ 22.9% of encode |

Neither bound removes *both* pyramid and compression at once, so the two deltas cannot simply be
summed without a caveat — but treating them as roughly independent (a reasonable first
approximation; nothing in the data suggests a strong interaction, since both bounds land within
0.6% of each other), **the compression + pyramid work together account for ≈45% of the encode
step**, leaving **≈55% — the container/tile-buffer/TIFF-structure assembly itself — as a cost
neither bound touches.** That residual is what pyvips's `tiffsave` call spends regardless of
compression codec or pyramid depth: marshaling pixel data into tiles, writing the TIFF directory
structure, and the Python/C++ call boundary overhead of doing this ~6,900 times (once per tile, at
every pyramid level).

**Consequence.** A GPU port that only replaces the *compression* step (which is all nvImageCodec,
as measured, actually does — §0's table) can plausibly reach at most the ≈23% of encode time that
compression accounts for, **not** the ≈45% that includes pyramid generation, and certainly not the
≈100% a naive reading of "19.2× faster" implies. Generating pyramid levels on GPU (a separate,
unmeasured piece of engineering — resizing kernels via `torch`/`cupyx`, not encoding) could add
the other ≈22%, but that is additional scope beyond what doc 25 §8 actually spiked.

### 1.3 A corrected, honest ceiling

Using doc 27's own real-scale numbers (§6.4/§6.5) and the fraction derived above:

- Encode is the overwhelming majority of Phase D's own time (the `join()` step is ~3s of the
  ~59-76s screening runs, ≈5%, per doc 27's own table — small and not addressed by this plan
  either way, matching doc 18 §4.1's earlier finding that `join()` is metadata-cheap and lazy).
- If a GPU port replaces **compression only** (the piece doc 25 §8 actually verified works on
  this host): addressable share of Phase D ≈ 23% × 95% ≈ **22%**. At `workers=4`, Phase D is
  19.3% of wall → new Phase D ≈ 0.78 × 19.3% ≈ 15.1% of wall → end-to-end ceiling ≈
  1/(1 − (19.3% − 15.1%)) ≈ **1.052×**.
- If a GPU port also moves **pyramid generation** to GPU (compression + pyramid, the full scope
  doc 24/25 originally imagined): addressable share ≈ 45% × 95% ≈ **43%**. New Phase D ≈
  0.57 × 19.3% ≈ 11.0% of wall → end-to-end ceiling ≈ 1/(1 − (19.3% − 11.0%)) ≈ **1.093×**.
- The **1.239×** figure everyone has been citing assumes the remaining ≈55-57% container-assembly
  cost also vanishes, which no candidate in any document proposes.

**This is the single most important finding in this plan, and it argues for scoping down, not up:**
the realistic ceiling is **≈1.05–1.09×**, not 1.24×, unless the container-assembly cost itself can
also be reduced (§2.3) — which is a genuinely open question, not yet measured by anyone. Per the
playbook's own anti-pattern #3 ("real speedup far below what theory predicted"), this should be
written down and used to size the engineering *before* it starts, not discovered after a
multi-week build.

## 2. Plan — what to build, in order, and what to verify before building it

### 2.1 The hard constraint, unchanged from doc 27 §3.2

Whatever this produces must be **BioFormats-readable**: LZW (or another codec BioFormats
supports — not `zstd`, per the round-8 veto), tiled, pyramidal, BigTIFF. This rules out using
nvImageCodec's own encoder defaults or any container format modern GPU codecs are fastest at; the
19.2× figure is specifically for the one codec (LZW) that clears this bar, which is doc 25 §8.2's
one genuinely useful finding.

### 2.2 The unresolved technical question this plan does not answer by assertion

Every prior document (24 §2.1, 25 §8.2, 27 §3.2) describes the missing piece as "generating
downsample levels and assembling a tiled/pyramidal BigTIFF container around GPU-encoded strips" —
correctly identified as real engineering, but never scoped to the level of "which library, which
API." This plan does not invent an answer here either — asserting one without checking would
violate the same "no unverified 'should work' claims" discipline doc 14 §1 Option C held itself
to. What needs to be checked, concretely, before any build estimate is trustworthy:

- **Does any maintained Python TIFF-writing library accept externally pre-compressed tile bytes**
  (i.e., let nvImageCodec do the LZW encode, and have the container writer place those bytes into
  a BigTIFF tile directory without re-compressing them)? `tifffile` is the most plausible
  candidate — it is already the closest thing to a Python-native BigTIFF/OME-TIFF pyramid writer
  in wide pathology-adjacent use — but **this project's own docs contain no confirmation either
  way**, and it must not be assumed. If no such API exists in a maintained library, the
  alternative is writing tile bytes into a BigTIFF structure by hand (offsets, tile byte-count
  arrays, IFD chaining) — a materially larger and riskier undertaking than "call a library
  function," and would change this plan's recommendation.
- **If no pre-compressed-tile API exists, is it faster to let `tifffile` (or another CPU library)
  do both compression and container writing on data whose pyramid levels were generated on GPU**
  — i.e., GPU only touches the ≈22% pyramid-generation share (§1.3's smaller estimate), and the
  compression + container work stays exactly as it is today? This is a strictly smaller, more
  conservative port and should be the fallback answer if the harder question above comes back
  negative.

### 2.3 Phase 1 — standalone container-assembly spike (do this before any pipeline code)

A throwaway script, in the spirit of `scripts/gpu_codec_spike.py` (doc 25 §8) and
`scripts/stitch_probe.py` (doc 25 §5) that this plan reuses rather than reinvents:

1. Build a synthetic input at the same scale doc 27 §3 already screened at (4.055 GP, 6,900 tiles,
   real-composition annotated share) — reuse `stitch_probe.py`'s existing tile-generation method
   directly rather than rebuilding it.
2. Encode every tile's LZW bytes via nvImageCodec (already proven to work on this host, sm_120,
   bit-identical output — doc 25 §8.2).
3. Assemble those bytes into a pyramidal BigTIFF using whichever answer §2.2 finds — either the
   pre-compressed-tile API (if one exists) or a GPU-pyramid + CPU-container hybrid (fallback).
4. Time the **whole** thing (encode + assembly), not just the encode step in isolation — doc 25
   §8.2's number is encode-only and must not be quoted again as if it were the end-to-end figure.
5. Verify losslessness and BioFormats/QuPath readability with the **exact same protocol** doc 27
   §3.1 already used for the `zstd` candidate: pixel-identical spot checks at tile and ~1 GP slide
   scale, and an actual open in the QuPath build pathologists use. This is not optional — it is
   the step that turned `zstd`'s 1.2331× into a zero, and this port carries the identical risk
   class (a new encode path whose readability has not been checked by anyone yet).

**Decision point.** Compare Phase 1's measured end-to-end number against §1.3's two estimates
(≈1.05× if compression-only, ≈1.09× if compression+pyramid). If Phase 1 lands meaningfully above
both (e.g., because `tifffile`'s own tile-writing overhead turns out cheaper than pyvips's
regardless of compression codec, which is possible and not something §1.3's analysis could
predict from the ablation data alone), that is new information justifying more engineering. If it
lands at or below §1.3's conservative estimate, or if BioFormats can't open the result, stop here
and record it exactly like `zstd` was recorded — measured, sized, and closed — rather than
proceeding to Phase 2 on hope.

### 2.4 Phase 2 — full port (only if Phase 1 clears a real bar)

Only after Phase 1 produces a verified, correct, faster-than-today standalone result:

1. Wire the chosen encode+assembly path into `_join_overlay_tiles` / `_stitch_overlay_slide`
   (already split by doc 27 §1 specifically so the join and the encode can be worked on
   independently — this plan is exactly the kind of follow-up that split was made for).
2. Preserve `_ensure_nofile_limit()`'s guard (doc 27 §1) — a GPU-assembled container still needs
   to read all 27,565 source overlay tiles' pixel data from disk, so nothing about this port
   removes that requirement.
3. Re-run the correctness protocol from §2.3 step 5 at full slide scale (141,818×114,366, the real
   16.22 GP grid), not just the 4.055 GP screening scale — the same "screening ranks correctly,
   confirm the winner at full scale before switching" discipline doc 27 §3's own caveat already
   states for the `tiffsave` knobs.
4. Re-measure end-to-end wall at both `workers=1` and `workers=4`, against the real numbers this
   plan is trying to improve (1,185.4 s / 8.6% and 1,200.8 s / 19.3% — doc 27 §6.4/§6.5), not the
   4 GP screening numbers.

## 3. Choose — decision rule

- **Bar to clear**: end-to-end wall reduction that matches or beats §1.3's corrected estimate
  (≈1.05–1.09×), verified at real scale, with output that opens in the QuPath build pathologists
  use. Anything that clears this bar but falls short of the old 1.239× figure is still a real win
  and should not be discarded for missing a ceiling that was never achievable in the first place.
- **Do not build Phase 2 speculatively.** Phase 1's spike is cheap (a standalone script, reused
  tooling, no pipeline code) and is the entire gate. This mirrors exactly how doc 25 §8 spiked the
  encode-only question before this plan existed, and how doc 27 §3 spiked the cheap knobs before
  concluding the GPU port was the only route left.
- **Correctness is a veto, not a tiebreaker.** If §2.3 step 5 fails at either scale, the candidate
  is dead regardless of speed, exactly as `zstd` was — a 1.2331× (or better) that produces an
  overlay a pathologist cannot open is worth zero, and this port is not exempt from that standard
  just because it clears the BioFormats-readability bar on paper (LZW) — it must be verified to
  clear it in the specific container shape this produces, since BigTIFF assembly bugs (bad offset
  tables, malformed IFDs) are a distinct failure mode from codec incompatibility and would also
  show up as "cannot open," not as a wrong pixel value.
- **If Phase 1 stops here**, record the finding the same way doc 27 §3.2 recorded the cheap-knob
  line: measured, sized, closed, with the specific number that killed it (not just "didn't work").
  That is more useful to whoever reads this next than leaving the item marked merely "open."
