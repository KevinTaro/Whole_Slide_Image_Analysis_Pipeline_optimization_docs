# 44 — The overlay shift: wrong input files, and the crop that hid it

## What happened

The full-slide runs (rounds 8–13) fed the hybrid pipeline
`HER2_processed.tiff` / `DISH_processed.tiff`. Those are **Module 1 output — VALIS's
input**, not registered images (`thriple_image_layer/config_example.py:26`:
*"Module 1 輸出 / Module 2 輸入"*; `:48` uses `HER2_processed.tiff` as the registrar's
`reference_img_f`). They are raw CZI→BigTIFF conversions, each at its own mosaic bounding
box, sharing no coordinate frame. Doc 27 §6.1's own preflight shows it — three
modalities, three different sizes:

```
IHC  HER2_processed.tiff   141818 x 114366
DISH DISH_processed.tiff   141658 x 114415
HE   HE_processed.tiff     141717 x 116400
```

`PrecutStream.__init__` (`m0_module/m0_reader.py:130-138`) rejects unequal IHC/DISH
dimensions — which would have caught this. `conform_to_intersection()` in
`scripts/full_wsi_validate.py`, added for the round-8 perf validation, cropped both to
`min(width), min(height)` and made them *equal in size without making them aligned*,
so the guard passed and the pipeline analysed unregistered input.

That is the whole defect. The crop's own trim was only a 160px/49px sliver (0.14%) from
a shared `(0,0)` origin, so it never shifted anything itself — it just disabled the check.
The reported gap (141658×114366 vs the correct 156222×134028) is the difference between
raw and registered canvases, not something the crop did.

The orange circles confirm it: they are drawn from `cr.dish_nucleus_mask`
(`m0_tile_runner.py:501-507`) at whatever position each tile's DISH crop landed. An
unwarped DISH puts every tile's content at the wrong absolute coordinates, so every
circle is displaced, while cells *within* a tile still segment the same local tissue and
look locally fine — exactly the reported symptom.

## The fix — use the files the original flow already produces

`module4_thumbnail.py` already warps both modalities at full resolution and writes them
out, as intermediates for its merge:

```python
level = config.thumbnail.level                      # default 0 = full resolution
dish_temp = temp_dir / f"dish_warped_lv{level}.tiff"
her2_temp = temp_dir / f"her2_warped_lv{level}.tiff"
dish_warped = dish_obj.warp_slide(level=level, non_rigid=True, crop="overlap")
her2_warped = her2_obj.warp_slide(level=level, non_rigid=True, crop="overlap")
```

Both use `crop="overlap"`, so both land on the **same** canvas — identical size, which
means `PrecutStream`'s guard passes with no conforming at all.

So: point the hybrid pipeline at `<temp_dir>/her2_warped_lv0.tiff` and
`<temp_dir>/dish_warped_lv0.tiff` instead of the `*_processed.tiff` pair.

```bash
python hybrid_pipeline.py \
  --ihc  <temp_dir>/her2_warped_lv0.tiff \
  --dish <temp_dir>/dish_warped_lv0.tiff \
  --workers 4
```

No algorithm change, no new code, no change to the original flow.

## Changed

- **Removed** `conform_to_intersection()` and its `--conform` flag from
  `scripts/full_wsi_validate.py` (69 lines). That function was added for the round-8
  perf validation; it is not part of the original pipeline. `preflight()`'s existing
  "IHC size != DISH size" check is untouched and now the only guard — a mismatched pair
  fails loudly instead of being silently equalised.
- Nothing else. In particular `core_crop_bounds` / `_join_overlay_tiles` /
  `m0_module/m0_stitch.py` (the tile seam-trim) are untouched and unrelated — round 13
  (doc 41 §3.2) verified they cause no tile displacement.

Also delete the stale `_conformed/{ihc,dish}_conformed_141658x114366.tiff` so nothing
reuses them.

## Note on the perf history

Rounds 8–13's wall-clock numbers remain valid as *timing* data — seconds, tile counts,
GB and RSS don't depend on whether tissue is aligned. They were simply never valid as
clinical output. Re-measuring on the registered pair would change the tile grid (larger
canvas → more tiles) and so the absolute walls, but not the ratios those rounds compared.
