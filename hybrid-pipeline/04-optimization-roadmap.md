# 04 — 優化路線圖

> 依「投報比 + 風險」排序。數字引自 [03](./03-benchmarks-bottlenecks.md) 的 perf_report.html。
> 先讀底部「設計決策的深層原因」再動 M0，很多看似可優化的點是刻意設計。
>
> **⚠️ 本文件數字是 pre-refactor（`m0_reader`/`m0_stitch` 改成 precut-to-folder 架構之前）量的，
> 已知與目前 HEAD 不符**——最新一輪實測見 [measurement/bottleneck-list.md](./measurement/bottleneck-list.md)。
> **M1 節（`ProcessPoolExecutor` 跨 tile 平行）尤其不能直接採用**：目前 `run_batch()` 明確寫死
> 序列迴圈，且原始碼註解與 `backend/algorithms/hybrid/CLAUDE.md` 都說明三個 GPU 模型共用同一個
> CUDA context，跨 tile 用 process 平行是不安全的（fork-under-CUDA）。針對①（GPU 序列/閒置）的
> 下一步方案，改看 [10-gpu-serial-pipeline-plan.md](./10-gpu-serial-pipeline-plan.md)（單 process
> 內 pipeline/overlap，不觸碰這個限制）。M2（換非 SAM Cellpose backbone）與 L1（GPU daemon 常駐）
> 兩項長期提案不受此限制影響，仍是有效方向，見該文件第 3.(c) 節的定位。

## 短期（perf_report 已建議，可直接採納）

### S1. 關閉 M1 debug PNG（production 模式）— 省 ~16s/tile（13.4%），WSI 快 ~22%
- **現況**：`hybrid_pipeline._write_m1_artifacts` 每 tile 寫 7 張中間 PNG（`_ihc_core_mask` / `_masked_ihc` / `_ihc_tumor` / `_dish_mask_overlay` / `_dish_tumor` / `_ihc_dish_overlay_raw` / `_m2_input_overlay`）。
- **這些是中間可視化，醫師看不到**，production 完全不需要。
- **做法**：加 `--no-debug-png` flag 或 `config.save_m1_artifacts=False`，在 `process_single_tile` 內條件跳過 `_write_m1_artifacts`。
- **風險**：極低（純 IO，不影響 CSV/結果正確性）。**這是第一個該做的。**

### S2. joblib backend 調整 — 省 ~5.8s/tile（4.8%）
- **現況**：`m3_dot_detection.detect_all_dots` 用 joblib 平行逐細胞偵測（`n_jobs=None`→-1 全核）；每次呼叫結束 `delete_folder` 清 memmap 暫存，佔 4.8%。
- **做法**：`backend='loky'` 搭 `prefer='threads'`，或加大 `max_nbytes` 避免 memmapping（小陣列不落磁碟就沒有 temp 清理開銷）。
- **風險**：低，但要保留 `n_jobs=-1` 全平行（符合專案「平行化一律全開」慣例）。

### S3. Cellpose batch size 16→32/64 — 省 ~10–20% GPU 推論
- **現況**：`CellposeModel.eval(batch_size=...)` 目前吃 `getattr(config, "cellpose_batch_size", 16)` → **永遠是 16**，因為 Config dataclass **沒有 `cellpose_batch_size` 欄位**（見 [07](./07-gotchas-appendix.md)）。
- **做法**：① 先在 `Config` 補上 `cellpose_batch_size: int = 32`（否則 config 怎麼設都沒用）；② RTX 5090 VRAM 只用 4.7GB/32GB，試 32–64。1024² sub-window 切 tiles 送 batch，大 batch 提高 GPU 吞吐。
- **風險**：低（VRAM 充裕），但**必須先補 Config 欄位**否則無效。

## 中期

### M1. 多 tile 平行（ProcessPoolExecutor）— 理論 4× 加速 ⚠️ **已知不可行，見文件頂部說明**
- **依據**：GPU 平均利用率僅 29%、VRAM 只用 15% → 卡是被餓著的，不是滿載（見 [03](./03-benchmarks-bottlenecks.md)）。
- **做法**：每 tile 是獨立 Python process，用 `concurrent.futures.ProcessPoolExecutor(max_workers=4)` 並行掃不同 tile；模型 weight 用 shared memory 避免每 process 重載 ~582MB。
- **估算**：並行 4 → WSI 2h10m；配合 S1 關 PNG → 1h42m。
- **風險**：中。要處理 (a) 每 process 的 CUDA context / VRAM 佔用（4×4.7≈19GB < 32GB OK）；(b) 模型權重共享（否則 4×582MB 重載吃滿啟動）；(c) `run_batch` 目前的單模型實例、GPU/GC 清理邏輯要改成 per-process。

### M2. 換非 SAM 的 Cellpose 模型 — 估省 ~30s/tile（~60%），需精度取捨
- **依據**：`get_rel_pos`（ViT-SAM attention）每 tile 呼叫 12,672 次、佔 25.1%；`run_net` 佔 32.3%。**這是最大單項**。
- **做法**：Cellpose 3.x 標準 CNN（cyto3 / 自訓 resnet）沒有 SAM 開銷。若換得動，是最大加速。
- **取捨**：**必須驗證精度**。目前 M2/M3b 兩顆 Cellpose 都是在醫師標註的 IHC-DISH 疊合圖上重訓的 —— 換 backbone 等於重訓 + 重新驗證。屬「架構級」改動，別當短期做。教授端偏好「不用太老的模型」（見專案記憶），換模型前要對齊這條線。

## 長期

### L1. GPU daemon 常駐 — 避免每 batch 重載 582MB 模型
- **現況**：`run_batch` 在 batch 開頭載一次三顆模型、整個 batch 重用（已避免 per-tile 重載）。但**每次啟動 process 仍要重載一次**（~582MB）。
- **做法**：把三顆模型放進常駐 GPU daemon（例如小型推論服務），CLI/UI 端只送影像、收結果，模型全程不卸載。這也是 [next-phase-ui-architecture.md](../next-phase-ui-architecture.md) 的 FastAPI 後端可以順手承接的角色。
- **風險**：架構級，等 UI Phase 1 一起做較划算。

### L2. 縫合畫布只縫 ROI，不整片 full-WSI
- **現況**：`StitchAccumulator.__init__` 為整張 slide 分配 6 張 full-H×full-W numpy（instance/nucleus mask + core + 3 RGB）。對 156k×134k 的 WSI，**這張輸出端畫布本身**就是新的記憶體天花板（不是讀取端）。
- **關鍵洞察**（來自專案記憶 `hybrid_cucim_wsi_reader`）：曾評估用 cuCIM 換 reader 省記憶體，**結論是不省** —— 400GB 級的壓力來自 `m0_stitch` 的整片縫合畫布（**輸出端**），換 reader（輸入端）解不了。真正的解是 **ROI-only + 停止整片縫合**。
- **做法**：讓縫合只在「有組織的 ROI 外接矩形」內 allocate，或直接串流輸出（逐塊寫 pyramidal TIFF / 逐塊出 CSV），不在記憶體保留整張 slide-level 影像。

## 設計決策的深層原因（動 M0 前必讀）

### 為什麼 core-ownership 去重比 IoMin 好（跨 chunk）
- 視窗**內部**用 IoMin（`m2_segmentation._dedup_instances`）合理：同一塊、幾何未知，只能兩兩比交集。
- 但**跨 chunk** 若也用 IoMin，成本隨細胞數 O(n²)（重疊帶所有 instance 兩兩比）。
- core-ownership 把每塊沿 `overlap/2` 切出互不重疊核心區，一顆細胞**只算在其質心所在的那一塊** —— 用「質心落點」這個 **O(n) 單一判斷** 取代兩兩比對。質心唯一、確定屬於某塊核心，天然無重複，故**跨塊完全不需要 IoMin**。這是把跨塊去重從 O(n²) 降到 O(n)。

### 為什麼分塊讀，而不整片先載入
- 原始痛點（見 `cell_mask/docs/6_30_report.md`）：整張 WSI 的 mask 一次載入，大圖爆記憶體（20k² ROI 峰值 ≈31GB），「撞了一個禮拜搞不出來」。
- M0 分塊讀（pyvips `access="random"` 逐 1024² 塊）讓峰值只由「單塊 + 整圖畫布」決定，不隨輸入面積線性成長。
- 代價是輸出端縫合畫布仍是 full-size（見 L2）—— 讀取端已解、輸出端待解。

## 為什麼某些嘗試被棄用

- **cuCIM 換 reader**：理論上 2026-07-01 的 warped 檔轉 JPEG(comp=7) 後 cuCIM 可當 reader，但**不省整片縫合的瓶頸**（400GB 是 `m0_stitch` 的輸出端縫合畫布，換輸入端 reader 無效）。→ 保留 pyvips 分塊讀，改攻 ROI + 停止整片縫合（L2）。
- **細胞膨脹搶核（舊 elastic matching）**：舊版讓細胞依面積等向膨脹去壓核，導致大細胞跨界搶遠核、小細胞搆不到最近核（不對稱）。已重寫為「以細胞為中心 + 重疊優先 + reach」（`m3_elastic_matching.py`）。注意 `docs/elastic_matching_v3_explainer.html` 描述的是另一個「以核為中心」變體 —— 文件與現行 code 有漂移，**以 code 為準**（見 [07](./07-gotchas-appendix.md)）。
- **`dish_elastic_expand_factor`（舊膨脹倍數）在 v3 explainer 標記已棄用**，但現行 code 仍用它算 reach 半徑 —— 又是文件/code 漂移的例子。
