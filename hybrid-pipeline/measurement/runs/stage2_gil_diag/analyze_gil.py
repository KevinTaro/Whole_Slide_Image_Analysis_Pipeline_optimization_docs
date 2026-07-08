#!/usr/bin/env python3
"""Parse py-spy raw (collapsed) stacks → GIL-contention distribution.

Two inputs: a --gil run (samples that HELD the GIL) and a wall-time run
(all active samples). Answers doc 11 §4(a)'s two questions:
  Q1: is GIL contention in the background CPU stage concentrated in a few
      Python-level functions, or spread out?
  Q2: how much of the main thread's time is GIL-held pure-Python (the hard
      overlap ceiling) vs GIL-released CUDA/C?
"""
import gzip
import sys
from collections import defaultdict


def _open(path):
    """Open plain or .gz collapsed-stack file transparently."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)

# --- stage classification by marker substrings anywhere in the stack line ---
# order = priority; first hit wins.
STAGE_MARKERS = [
    ("detect.regionprops", ["regionprops", "_regionprops"]),
    ("detect.red_black",   ["_detect_red_dots", "_detect_black_dots",
                             "_detect_one_cell", "_compute_ring_stats"]),
    ("detect.merge",       ["_merge_close_dots"]),
    ("detect.morphology",  ["h_maxima", "h_minima", "binary_dilation",
                             "grayreconstruct", "reconstruction", "skimage/morphology"]),
    ("detect.label",       ["skimage/measure", "ndimage", "find_objects", " label "]),
    ("detect.lab",         ["rgb2lab", "_rgb_to_lab"]),
    ("detect.owner_mask",  ["_build_nucleus_owner_mask",
                             "elastic_dish_nucleus_matching",
                             "_filter_dish_nucleus_by_core_mask"]),
    ("detect.other",       ["detect_all_dots", "m3_dot"]),
    ("io.png_write",       ["_save_tile_array", "imsave", "_write_blank_tile",
                             "pil_", "PngImagePlugin", "Image.save"]),
    ("io.overlay_render",  ["render_overlay_image", "export_per_cell_images",
                             "_export_chunk_merge_overlay", "m4_module"]),
    ("gpu.unet",           ["generate_ihc_core_mask", "unet_inference",
                             "_run_m1_overlay_stage", "UNetPP"]),
    ("gpu.cellpose",       ["segment_masked_dish", "segment_windowed",
                             "run_net", "cellpose", "_run_cp"]),
    ("gpu.torch",          ["torch/", "torch\\", "aten", "cuda"]),
    ("joblib.machinery",   ["joblib", "Parallel", "_dispatch", "retrieve",
                             "ThreadPoolExecutor", "concurrent/futures"]),
    ("io.tiff",            ["tifffile", "tiffsave", "pyvips"]),
]


def classify_stage(line):
    for name, marks in STAGE_MARKERS:
        for m in marks:
            if m in line:
                return name
    return "other"


def is_background(line):
    # spawned threads (tile-cpu ThreadPoolExecutor worker + joblib thread pool)
    # all go through threading._bootstrap_inner; MainThread does not.
    return "_bootstrap_inner" in line or "_bootstrap (threading" in line


def parse(path):
    """Return (total, by_thread, by_thread_stage, by_thread_leaf)."""
    total = 0
    by_thread = defaultdict(int)
    by_thread_stage = defaultdict(int)
    by_thread_leaf = defaultdict(int)
    with _open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # split trailing count
            sp = line.rsplit(" ", 1)
            if len(sp) != 2 or not sp[1].isdigit():
                continue
            stack, cnt = sp[0], int(sp[1])
            total += cnt
            th = "background" if is_background(stack) else "main"
            stage = classify_stage(stack)
            leaf = stack.split(";")[-1].split(" (")[0]
            by_thread[th] += cnt
            by_thread_stage[(th, stage)] += cnt
            by_thread_leaf[(th, leaf)] += cnt
    return total, by_thread, by_thread_stage, by_thread_leaf


def pct(n, d):
    return f"{100.0*n/d:5.1f}%" if d else "   -  "


def report(tag, path):
    total, by_thread, by_thread_stage, by_thread_leaf = parse(path)
    print(f"\n{'='*70}\n{tag}: {path}\n  total samples = {total}\n{'='*70}")
    for th in ("main", "background"):
        print(f"\n[{th} thread]  {by_thread[th]} samples "
              f"({pct(by_thread[th], total)} of all)")
        stages = sorted(((s, c) for (t, s), c in by_thread_stage.items() if t == th),
                        key=lambda x: -x[1])
        for s, c in stages:
            print(f"    {s:22s} {c:8d}  {pct(c, total)} all "
                  f"| {pct(c, by_thread[th])} of {th}")
        print(f"  -- top leaf functions ({th}) --")
        leaves = sorted(((l, c) for (t, l), c in by_thread_leaf.items() if t == th),
                        key=lambda x: -x[1])[:12]
        for l, c in leaves:
            print(f"    {c:8d}  {pct(c, total)}  {l}")
    return total, by_thread, by_thread_stage


if __name__ == "__main__":
    gil_path = sys.argv[1]
    wall_path = sys.argv[2] if len(sys.argv) > 2 else None
    gtot, gthread, gstage = report("GIL-HOLDING", gil_path)
    if wall_path:
        wtot, wthread, wstage = report("WALL-TIME (active)", wall_path)
        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        print(f"main-thread GIL share (of all GIL samples): "
              f"{pct(gthread['main'], gtot)}")
        print(f"background-thread GIL share (of all GIL samples): "
              f"{pct(gthread['background'], gtot)}")
        # main-thread GIL-released fraction = main wall - main gil, over main wall
        main_wall = wthread["main"]
        main_gil = gthread["main"]
        print(f"\nmain-thread wall samples={main_wall}, "
              f"main-thread gil samples={main_gil}")
        print("(main-thread GIL-released ~ CUDA/C = wall_main - gil_main, "
              "normalized separately since the two runs have independent sample totals)")
