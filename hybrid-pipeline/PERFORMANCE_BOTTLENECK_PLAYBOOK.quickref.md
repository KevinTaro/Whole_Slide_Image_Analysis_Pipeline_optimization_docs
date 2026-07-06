# Performance Bottleneck Playbook — Quick Reference

> Condensed from `PERFORMANCE_BOTTLENECK_PLAYBOOK.md` (full case study: this repo's
> CuPy/GPU pipeline work, v0–v32, where GPU kernel optimization got 9 versions of effort
> for a piece that was 1.1% of total time — and the "fully optimized" GPU version was
> *slower* than a single-core CPU loop). Read the full doc for the numbers; this is the
> operating checklist distilled from it.

## TL;DR
Measure before optimizing. Amdahl: if X is `p%` of total time, perfecting X only buys
`1/(1-p)` overall speedup. Optimizing something that's 1% of runtime is wasted effort
no matter how well it's done.

## Workflow

1. **Discover** — build an honest baseline *and* a naive control (dumbest possible
   version, e.g. single-threaded). Measure end-to-end wall-clock first. Then break time
   down: put timers on both sides of each suspect stage → a `%` table. The candidate
   bottleneck is the stage with the **biggest %**, not the biggest absolute time.

2. **Analyze** — before touching the candidate, compute its Amdahl ceiling
   (`1/(1-p)`). Single-digit % → stop, don't bother. Then measure both ends of the
   pipeline (supply throughput vs. consume throughput) to see who's starving whom.
   Keep "is it fast" and "is it the bottleneck" separate — a genuinely fast component
   can still be irrelevant if it's hidden behind something slower upstream/downstream.

3. **Plan** — let the bottleneck define the solution space, not your toolbox of known
   tricks. Cheapest-first:
   (a) parallelize the bottleneck (often a one-line `Pool`),
   (b) move it off the critical path (prefetch/pipeline/overlap),
   (c) eliminate it (algorithm or hardware change — strongest but most expensive).
   Fix **one** bottleneck, then re-measure — the bottleneck moves.

4. **Choose** — decide by end-to-end wall-clock only, never by micro-benchmark. Every
   optimization layer must justify itself via ablation (remove it, does total time
   change?); zero-contribution layers get cut no matter how clever. Correctness is a
   veto — faster-but-different output is broken, not better. Prefer the simplest
   solution that clears the bar; don't reward complexity that can't prove its worth.

## Red flags — stop and re-measure

- Your "optimized" version is slower than, or about equal to, the dumbest baseline.
- You stacked several optimizations and total time barely moved.
- Real speedup is far below what theory predicted.
- The profiler's "hot function" turns out to be a small slice of total time.
- You're chasing the part that *looks* compute-heavy instead of the boring stuff
  (I/O, serialization, locks, allocation, sync points) that's actually costing time.

## Anti-pattern checklist (ask before you touch code)

1. Optimizing before measuring (assumption-driven).
2. Optimizing something under ~10% of total time.
3. No control group/baseline — you can't detect a negative optimization.
4. Trusting intuition over profiler data for where the hot spot is.
5. Treating a micro-benchmark win as an end-to-end win.
6. Stacking techniques without ablation proof each one helps.
7. Changing several things at once — can't attribute the result.
8. Ignoring hardware/platform quirks (e.g. fp16 on an old GPU can be 1/64 the
   throughput of fp32 → a "speedup" that's actually a regression).
9. Trading correctness for speed without noticing or reporting it.
10. Mistaking "looks parallel" for "is parallel" (e.g. shared fd + lock serializes
    reads) — measure the achieved throughput, don't trust the code's shape.

## Three things to remember

1. One line of Amdahl arithmetic beats a month of optimization intuition.
2. A dumb control group is your mirror for negative optimization — never discard it.
3. Fast ≠ bottleneck. Slow ≠ worth fixing first. Bottlenecks are measured, and they
   move — re-measure after every fix.
