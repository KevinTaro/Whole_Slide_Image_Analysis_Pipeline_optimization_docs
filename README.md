# docs/ — project documentation index

This tree holds the working documentation for the WSI (whole-slide image)
analysis pipeline: performance optimization history for the hybrid pipeline,
the UI handoff packet, and algorithm explainers. It is **not** a polished
manual — it's the running record a person picks up mid-project from.

## Repo layout (important — two git repos overlap here)

`docs/` is split across two repositories:

- **Main repo** (`tsgh`, the one this file lives in from the outside) tracks
  only `docs/UI/**` and `docs/BACKLOG.md` (see the root `.gitignore`'s
  `docs/*` / `!docs/UI/` / `!docs/BACKLOG.md` whitelist).
- **Nested repo** at `docs/.git` tracks everything under `docs/`, including
  `hybrid-pipeline/**` and `algo/**`, which the main repo never sees. It has
  its own remote (`origin`, a separate private GitHub repo) — commit and push
  changes to `hybrid-pipeline/` or `algo/` from inside `docs/` (`git -C docs
  ...`), not from the main repo.

**Never** `git add docs/hybrid-pipeline` or `docs/algo` from the main repo —
the whitelist is intentional, not a gap to fix.

## Where to start

| You want to... | Start here |
|---|---|
| See what's still open, across the whole project | [`BACKLOG.md`](BACKLOG.md) — single current list; everything else below is detail behind an item there |
| Pick up hybrid-pipeline performance work | [`hybrid-pipeline/README.md`](hybrid-pipeline/README.md) — round-by-round optimization history, current bottleneck ranking, navigation table |
| Pick up UI work (FastAPI + React + pywebview) | [`UI/README.md`](UI/README.md) — architecture, guardrails, phase roadmap, dev setup |
| Read an algorithm design explainer (HTML, open in browser) | [`algo/`](algo/) — DISH nucleus matching, sliding-window stitch, frontend/backend image-transfer tradeoff |

## Subdirectories

- **`hybrid-pipeline/`** — 40+ numbered docs tracking the IHC(HER2)+DISH dual-stain
  tile pipeline's optimization history (plan → implementation → result, per
  round), plus `measurement/` for the current bottleneck ranking and raw
  metrics. Read its own `README.md` first — it has a reading order and flags
  which docs are stale.
- **`UI/`** — handoff packet for the FastAPI + React + pywebview UI: architecture,
  API contract, phase roadmap (Phases 1–3 shipped, 4–5 open), runbook, open
  decisions.
- **`algo/`** — standalone HTML explainers for specific algorithm design
  questions (DISH dot matching, stitch seams, API image-transfer boundary).
  Not phase-tracked; read `BACKLOG.md` §3 before trusting one against current
  code — at least one has known drift from what's shipped.

## Before trusting any specific number in here

This project has repeatedly found that numbers measured on crops, or on the
wrong canvas, didn't hold at full-slide scale (see `BACKLOG.md`'s intro and
round 15). Prefer whichever doc is linked from `BACKLOG.md` or
`hybrid-pipeline/measurement/bottleneck-list.md` as current over an older doc
making the same claim.
