# 02 — 逐模塊參考

> 每個模塊：檔案/入口 → I/O → 關鍵演算法 → 瓶頸 → 套件（**實測 venv 版本**）→ 可優化點。
> 全部路徑相對 `backend/algorithms/hybrid/`（舊稱 `cell_mask/hybrid/`，該路徑已隨 UI Phase 1
> 目錄重構移除，見 [README](./README.md) 頂部提醒）。Invariant：模塊間傳遞的影像一律 RGB `uint8 (H,W,3)`；core mask `uint8{0,1}(H,W)`；instance mask `int32(H,W)` 背景 0、細胞 1..N。
>
> ⚠️ **本文件描述的是 pre-precut 架構**（M0 讀取為逐 chunk 迭代器）。目前 HEAD 已改成
> `precut_paired_tiles`/`PrecutStream` 先落地/串流磁碟 tile，詳見 [09](./09-measurement-analysis-plan.md) §0 的落差表與 [18](./18-gpu-starvation-prerequisites-implementation.md) §4.2。M2/M3 的模組級 I/O、參數、瓶頸分類本身大致仍成立，但引用的入口函式（如 `segment_masked_dish`）部分已在後續 commit 移除，改由 `segment_windowed` 統一承接——見 [`current-status-comparison.md`](./measurement/current-status-comparison.md) 開頭的 harness caveat。

---

## M0 讀取 — `m0_reader.py`

- **入口**：`iter_paired_chunks(ihc_path, dish_path, tile_size=1024, overlap=256)` → `Iterator[Chunk]`；輔助 `read_size()`、`chunk_offsets()`。
- **輸入**：IHC/DISH 檔路徑（tile/ROI/WSI，須同尺寸、任一邊 ≥ `tile_size`，否則 `ValueError`）。
- **輸出**：`Chunk(ihc, dish, abs_x, abs_y)`，`ihc`/`dish` 為 `(1024,1024,3) uint8`（越界白底補滿）。
- **演算法**：`pyvips.Image.new_from_file(access="random")` 隨機存取開檔；視窗格線沿用 `m2_segmentation._overlap_window_coords`（`stride=tile_size-overlap`，最後一格貼邊）；`_crop_to_tile` 越界用 `gravity(extend="white")` 補白；`_to_numpy` 只取前 3 通道並確保連續。頂層 `pyvips.cache_set_max(0)` 關快取避免 RAM 單調成長。
- **瓶頸**：IO-bound；WSI 每塊 decode。不在 cProfile Top（量測時多為小 tile）。
- **套件**：`pyvips 2.2.3`、`numpy 1.26.4`。
- **可優化**：ROI-only 掃描（跳過空白塊）、prefetch 下一塊。

## M0 縫合 — `m0_stitch.py`

- **入口**：`StitchAccumulator(positions, full_h, full_w, overlap).add(ChunkResult) / .finalize()`；`clear_slide_edge_cells(...)`。
- **輸入**：逐塊 `ChunkResult`（局部座標 int32 mask + RGB artifacts + M3 結果）。
- **輸出**：`StitchedTile`（slide-level：instance/nucleus mask + core_mask + 3 張 RGB + results/all_dots/per_cell_dots）。
- **演算法**：質心 core-ownership 去重（`_cut_lines` 切核心區、`bisect_right` 判質心歸屬）；全域重編號（細胞/核 + `assigned_dish_ids` 改寫）；座標絕對化 `+(abs_x,abs_y)`；`_paint_unmatched_core_nuclei` 把未配對但質心在核心區的 DISH 核也畫入（僅供輪廓）。純資料重組、無模型。詳見 [01](./01-architecture-dataflow.md)。
- **瓶頸**：整圖畫布 allocate（6 張 full-size numpy）是 WSI 記憶體天花板；`center_of_mass` 對未配對核有額外掃描。
- **套件**：`numpy 1.26.4`、`scipy 1.17.1`（`scipy.ndimage.center_of_mass`）。
- **可優化**：畫布改「只縫 ROI 範圍」而非 full-WSI（見 [04](./04-optimization-roadmap.md) 長期）。

## M1 疊合 — `m1_overlay.py`

- **入口**：`generate_ihc_core_mask(ihc_tile_path, unet_inferencer, close_kernel)`、`apply_mask_to_ihc_image()`、`overlay_ihc_mask_on_dish()`、`fuse_masked_ihc_with_dish()`；批次配對 `find_paired_tiles()`、`parse_tile_coords()`。
- **輸入**：IHC/DISH `(H,W,3) uint8` + core mask `(H,W){0,1}`。
- **輸出**：`masked_ihc` / `dish_mask_overlay` / `overlay_image`（皆 `(H,W,3) uint8`），overlay 即 M2 輸入。空 core mask → 短路成空 CSV。
- **演算法**：UNet++ 推論出腫瘤 core mask → 分別遮罩 IHC/DISH（非 ROI 填 `background_fill_value=255`）→ `fuse` 做 `dish×(1-alpha)+ihc×alpha`。
- **⚠️ 命名誤導**：`generate_ihc_core_mask` 形參叫 `ihc_tile_path: Path`，但呼叫端傳的是 `chunk.ihc`（ndarray）。非 bug（`predict_single` 接受 `Union[ndarray,Path,str]`，呼叫端已標 `# pyright: ignore`），但會誤導新人。見 [07](./07-gotchas-appendix.md)。
- **瓶頸**：主要成本在其呼叫的 UNet++（見下）；overlay 本身是廉價 numpy 運算。
- **套件**：`opencv(cv2) 4.8.1`、`numpy 1.26.4`、`scipy 1.17.1`（gaussian_filter）。
- **關鍵參數**：`overlay_alpha` **config.py=0.65 / example=0.5**；`background_fill_value=255`；`mask_blur_sigma=0.0`；`core_close_kernel=7`。

## M2 分割 — `m2_segmentation.py`

- **入口**：`CellposeSegmenter`（封裝 `cellpose.models.CellposeModel`）、`segment_masked_dish(remove_border=False,...)`、`segment_windowed()`；內部 `_overlap_window_coords()`、`_dedup_instances()`、`_remove_border_cells()`、`_relabel_sequential()`。
- **輸入**：IHC-DISH 疊合圖 `(H,W,3) uint8`。
- **輸出**：`instance_mask (H,W) int32`（背景 0、細胞 1..N）。
- **演算法**：重疊視窗逐塊 `CellposeModel.eval` → 每個 instance 換算全圖座標收集 → **面積由大到小貪婪 IoMin 去重**（`交集/min(面積) ≥ dedup_iomin` 視為同一顆，保留較大者；用 256 粗格空間雜湊把比對限在 bbox 相鄰者，避免全域 O(n²)）→ 依序 paint 成 1..N。清邊移到 M0（`remove_border=False`）。
- **瓶頸**：**這是全流水線最大瓶頸** —— `CellposeModel` 用 ViT-SAM backbone：`run_net` GPU 前向 32.3%、`get_rel_pos`（SAM attention 相對位置）25.1%、`compute_masks` 8.1%、`flow_error` 5.3%。
- **套件**：`cellpose 4.0.8`（內含 `segment-anything 1.0`）、`torch 2.11.0+cu130`、`scikit-image 0.24.0`（clear_border）、`scipy 1.17.1`（find_objects）。
- **關鍵參數**：`cellpose_diameter=None`（自動）；`cellpose_flow_threshold` **config.py=0.6 / example=0.4**；`cellpose_cellprob_threshold` **config.py=-0.8 / example=0.0**；`window_overlap_px=256`；`window_dedup_iomin=0.5`。**`cellpose_batch_size` 硬編 16 —— Config dataclass 無此欄位，`getattr(...,16)` 永遠回 16**（見 [07](./07-gotchas-appendix.md)）。
- **可優化**：換非 SAM cellpose 模型（省 ~60%，需驗證精度）、batch size 16→32/64、多 tile 平行。

## M3 細胞/點位 — `m3_module/`（+ `m3_cell_detection.py` 相容 shim）

`m3_cell_detection.py` 只是 `from m3_module.* import *` 的相容再匯出；正式匯入走 `m3_module`（`__init__.py` 定義公開 API）。

### `m3_cells_generator.py`
- **入口**：`build_all_positive_results(cell_instance_mask)` → `List[CellAnalysisResult]`（每細胞一筆質心，全標陽性）；`enlarge_cell_instances(mask, cfg)`。
- **演算法**：一次 `center_of_mass` 取全部質心（避免逐細胞建 bool mask）；`enlarge` 用 skimage `expand_labels` 做 Voronoi 式外擴，把細胞面積放大 `cell_enlarge_area_factor` 倍（換算外擴距離 `d=r*(√factor−1)`，取全體中位半徑估 `d`），放大版**只供配對/點偵測**，醫師看到的綠框不變。
- **套件**：`scipy 1.17.1`、`scikit-image 0.24.0`。
- **參數**：`cell_enlarge_area_factor=1.5`。

### `m3_elastic_matching.py`
- **入口**：`elastic_dish_nucleus_matching(dish_nucleus_mask, strict_instance_mask, cfg)` → `({cell_id:[dish_id]}, drop_out_ids)`。
- **演算法**：**以細胞為中心 + 重疊優先 + reach 候選**。候選來源 (a) 與綠框像素重疊的核（`_overlap_pairs` 單次掃描編碼取 unique）；(b) 質心落在 `reach=max(sqrt(factor*area/π), min_reach_px)` 內的核（`cKDTree.query_ball_point`）。排序鍵 `(is_reach_only, dist)` 讓重疊對永遠先配；貪婪一對一 + lock，落敗細胞往後找。每細胞至多 1 核。0 核細胞再分：曾有候選=drop-out（打 X）、從無候選=照常計入(0/0)。
- **⚠️ 版本漂移**：docstring 引用的 `docs/sdd-elastic-dish-matching.md` 已不存在；且該 doc（與 `elastic_matching_v3_explainer.html`）描述的是「以核為中心」變體，與現行 code（以細胞為中心）不同 —— **以 code 為準**。見 [07](./07-gotchas-appendix.md)。
- **套件**：`scipy 1.17.1`（`cKDTree`、`center_of_mass`、`find_objects`）。
- **參數**：`dish_elastic_expand_factor=1.5`；`dish_elastic_min_reach_px` **config.py=20.0 / example=0.0**；`dish_elastic_exclude_zero=True`。

### `m3_dot_detection.py`
- **入口**：`detect_all_dots(dish_image, instance_mask, config, dish_nucleus_mask, core_mask, n_jobs=None)` → `(all_dots, per_cell_results, filtered_nucleus_mask)`；`merge_dot_results_to_cell_analysis()`。
- **演算法**：先 `_filter_dish_nucleus_by_core_mask`（核需 ≥ `dish_nucleus_core_min_inside_ratio` 落在 core mask 內，否則整顆丟）；`_build_nucleus_owner_mask` 把配對核區標成擁有者 id → 逐細胞在**自己贏得的核區域內**偵測紅/黑點（呼叫 `m3_dot_kernels`）；`_finalize_per_cell` 算 Score。**`n_jobs=None` → joblib 用滿全部核心 (-1) 平行逐細胞偵測**。
- **瓶頸**：`detect_all_dots` 累計 17.1%（M3 主成本）；joblib memmap 暫存清理 `delete_folder` 另佔 4.8%。
- **套件**：`joblib 1.5.3`、`scikit-image 0.24.0`、`scipy 1.17.1`、`numpy 1.26.4`。
- **參數（config.py / example 有多處差異）**：`dish_nucleus_core_min_inside_ratio` **0.97 / 1.0**；`dot_red_h` **5.0 / 12.0**；`dot_red_a_min` **17.0 / 25.0**；`dot_red_min_area` **5 / 7**；`dot_red_min_contrast` **7.0 / 10.0**；`dot_black_h` **5.0 / 12.0**；`dot_merge_distance` **2.0 / 3.0**；`dot_assignment_min_overlap_ratio` **0.10 / 0.20**；`dot_assignment_boundary_margin` **0.05 / 0.5**。

### `m3_dot_kernels.py`
- **入口**：`DetectedDot`（dataclass，見 `hybrid_data_types.py`）；`_detect_red_dots()` / `_detect_black_dots()` / `_compute_ring_stats()` / `_merge_close_dots()`（union-find）。
- **演算法**：RGB→LAB，紅點看 a* 通道 H-maxima、黑點看 L* 通道 H-minima；多準則閘控（面積/圓度/solidity/環形對比/chroma）；ring 統計判前景-背景對比；近距離點 union-find 合併。
- **套件**：`opencv 4.8.1`、`scikit-image 0.24.0`、`scipy 1.17.1`。

## M4 匯出 — `m4_export.py`（facade）+ `m4_module/`

`m4_export.py` 是穩定公開 API（`__init__` 式 re-export），呼叫端只 import 它。

| 子模塊 | 入口 | 職責 |
| --- | --- | --- |
| `m4_module/csv.py` | `export_tile_csv`、`DotStatsSummary`、`export_summary_statistics`、`write_summary_csv` | CSV 欄位 `cell_id/centroid_x/centroid_y/reddot/blackdot/score`（excluded → `NaN`）；`compute_config_hash` 寫入追溯。 |
| `m4_module/overlay.py` | `render_overlay_image`、`export_overlay_visualization`、`export_dot_only_visualization`、`stamp_grid_on_overlays` | 綠框 + 粉色核輪廓 + 飄移箭頭 + 細胞標號 + 紅黑點；`draw_dashed_grid` 畫視窗參考格。 |
| `m4_module/cell_crops.py` | `export_cell_dot_annotations`（統一入口：CSV+overlay+crop）、`export_per_cell_images` | 固定尺寸逐細胞裁切（`_fit_to_fixed_canvas`）。 |

- **瓶頸**：M4 匯出本身輕（2.35–2.94s/tile, ~6–7%）；**真正的 IO 大戶是 M1 debug PNG**（`_write_m1_artifacts` 每 tile 寫 7 張 PNG，`_write_m1_artifacts` 總 13.4% + PIL encode 13.0%）—— 這段在 `hybrid_pipeline.py`，非 M4。
- **套件**：`opencv 4.8.1`、`Pillow 12.2.0`、`scikit-image 0.24.0`。
- **參數**：`cell_crop_size` **config.py=100 / example=256**；`draw_window_grid=True`。

## UNet++ 推論 — `unet_inference.py`

- **入口**：`UNetPPInference(model_path, encoder_name, num_classes, image_size, batch_size, device)`；`.predict_single(image_or_path, return_proba=False)`；後處理 `postprocess_membrane_mask(raw, close_kernel_size=7, min_area=550)`。
- **輸入**：`(H,W,3)` RGB（或 2D/RGBA 自動轉）或路徑。**≤1024² 直接推論；>1024² 自動無重疊滑動視窗**（`_predict_sliding_window`，batch 依 `batch_size`）。
- **輸出**：core mask `(H,W) uint8{0,1}`。
- **演算法**：`smp.UnetPlusPlus`（encoder 無 ImageNet 權重、載訓練 checkpoint）；FP32 + `cudnn.benchmark` + `torch.inference_mode`；前處理 `albumentations`（PadIfNeeded 白底 + ImageNet Normalize + ToTensorV2）；後處理形態學閉合 + 移除 <550px 碎片。
- **瓶頸**：GPU 前向，但遠小於 Cellpose；不在 cProfile Top。
- **套件**：`torch 2.11.0+cu130`、`segmentation-models-pytorch 0.5.0`、`albumentations`、`opencv 4.8.1`。
- **關鍵參數**：`unet_encoder_name` **config.py="timm-efficientnet-b4" / example="efficientnet-b4"**；`unet_image_size=(1024,1024)`（最小強制 1024）；`batch_size`（用 config.batch_size）**config.py=4 / example=8**。

## 資料型別 — `hybrid_data_types.py`

中立位置定義三個 dataclass，讓 `m3_module` 與 `m4_module` 都從這裡 import（避免 m4 依賴 m3）：
- `DetectedDot`（y,x,radius,dot_type,cell_id,area,circularity,solidity,contrast,score）
- `CellAnalysisResult`（cell_id,centroid_x/y,is_her2_positive,hematoxylin_ratio + M3b 點位欄位 her2/cep17 count、ratio、is_amplified、score、excluded…）
- `CellDotResult`（含 `her2_dots`/`cep17_dots`/`assigned_dish_ids`、`exclude_reason`）

## 設定 — `config.py`（gitignored）/ `config_example.py`（範本）

- `config.py` **未進 git**（gitignored），必須 `cp config_example.py config.py` 才能跑；且 example 尾端缺 `config = Config()` 與 `compute_config_hash()`，直接 cp 會 ImportError —— 詳見 [07](./07-gotchas-appendix.md)。
- `compute_config_hash(cfg)` 取 dataclass 全欄位的 SHA-256 前 8 碼，寫進每張 CSV 供追溯。
- **config.py 實際值與 example 預設值的差異**已在上方各模塊「關鍵參數」標注（格式：**config.py 值 / example 值**）。完整差異表另見 [05](./05-dev-testing-guide.md)。
