# 03 — 效能實測與瓶頸

> 資料來源：`cell_mask/hybrid/output/perf_report.html`。
> 量測條件：`--batch --test`，**3 tiles（4096×4096）**，GPU **RTX 5090 32GB**，日期 **2026-06-29**。

## ⚠️ 量測時點（先看這個）

- 量測於 **2026-06-29 18:42**，約在 **M0 相關 commit 前 17 分鐘**。
- 也就是：這份數字**尚未包含後續的 chunked stitching / GC 記憶體優化**（見 recent commits `0d5c774`、`b51ce6a`）。
- **結論**：瓶頸**排名**在架構上仍然有效（Cellpose ViT-SAM 前向、debug PNG、dot detection 的相對占比不會因記憶體優化而翻轉）；但**絕對秒數**可能已隨後續 commit 略有改善。要精準數字請重跑（見 [05](./05-dev-testing-guide.md)，目前無自動腳本、是手動量測）。

## 每 tile 處理時間

| Tile | 總耗時 | M2 Cellpose | M3 Analysis | M4 Export | PNG 寫出 | 備註 |
| --- | --- | --- | --- | --- | --- | --- |
| tile_x61440_y36864 | **48.7s** | 22.8s (47%) | 12.1s (25%) | 2.94s (6%) | 5.39s | 首次 GPU 暖機 +1.4s |
| tile_x86016_y40960 | **37.8s** | 18.2s (48%) | 10.2s (27%) | 2.49s (7%) | 5.38s | 穩定狀態 |
| tile_x98304_y20480 | **31.5s** | 15.2s (48%) | 8.2s (26%) | 2.35s (7%) | 5.38s | 穩定（少細胞） |
| **合計 / 平均** | **120.4s → 39.3s/tile** | ~48% | ~26% | ~7% | ~5.4s | — |

- 每 tile **~一半時間在 M2 Cellpose**，~1/4 在 M3，M4 匯出很輕（~7%）。
- PNG 寫出恆定 ~5.4s（與細胞數無關，是固定 7 張 M1 debug 圖）。
- 暖機成本只在首 tile（+1.4s），穩定態約 31–38s/tile。

## cProfile Top 函數（3 tiles 合計 120.4s）

| # | 函數 | 累計 | % | 分類 |
| --- | --- | --- | --- | --- |
| 1 | `cellpose: run_net`（GPU 前向） | 38.9s | **32.3%** | Cellpose M2+M3 |
| 2 | `segment_anything: get_rel_pos`（ViT-SAM attention 相對位置） | 30.2s | **25.1%** | Cellpose M2+M3 |
| 3 | `m3_dot_detection: detect_all_dots` | 20.6s | **17.1%** | M3 Analysis |
| 4 | `_write_m1_artifacts`（total, debug PNG） | 16.2s | **13.4%** | I/O |
| 5 | `PIL/PNG encode: ImagingEncoder`（_write_m1_artifacts 內） | 15.7s | 13.0% | I/O |
| 6 | `cellpose: compute_masks`（後處理） | 9.7s | 8.1% | Cellpose |
| 7 | `cellpose: flow_error`（mask 品質過濾） | 6.4s | 5.3% | Cellpose |
| 8 | `joblib: delete_folder`（temp 清理） | 5.8s | 4.8% | Overhead |

> #4 與 #5 高度重疊（PIL encode 是 `_write_m1_artifacts` 的主要內容）；把兩者視為「同一件事：debug PNG 寫出 ≈13%」，不要相加成 26%。

### 三大瓶頸群（依可操作性排序）

1. **Cellpose ViT-SAM GPU 前向**（#1+#2+#6+#7 ≈ 70%）：`get_rel_pos` 每 tile 被呼叫 **12,672 次**（decomposed relative position embedding）。這是模型架構決定的固定開銷，換非 SAM backbone 才能根治。
2. **Debug PNG 寫出**（#4/#5 ≈ 13%）：`_write_m1_artifacts` 每 tile 寫 7 張中間可視化 PNG，醫師看不到，production 可直接關 → 最低風險的即時收益。
3. **M3 dot detection + joblib overhead**（#3+#8 ≈ 22%）：偵測本身 17%，joblib memmap 暫存清理額外 4.8%（可換 backend 消除）。

## 系統資源利用率

| 資源 | 平均 | 峰值 | 解讀 |
| --- | --- | --- | --- |
| CPU | **15.2%** | 76.8% | 大量時間在等 GPU/IO；只有 PNG encode / joblib 時衝高 |
| GPU | **29.3%** | 99% | **關鍵訊號**：GPU 大部分時間閒置 |
| GPU 記憶體 | **4.7GB / 32GB (15%)** | — | 32GB 卡只用 15%，還有 6× 餘裕 |
| 系統 RAM | 17.2GB | 18.7GB | M0 分塊後已可控 |

### 為什麼 GPU 平均只有 29%（peak 卻 99%）

- **peak 99%**：Cellpose 真正前向那幾個瞬間，GPU 是打滿的 —— 不是 GPU 太弱。
- **avg 29%**：但單 tile 是**序列**跑的 —— M1 UNet→M2 Cellpose→M3→M4 匯出→**寫 7 張 PNG（CPU/IO）**→下一 tile。GPU 在 M4/PNG/joblib/讀圖那些 CPU-bound 段落**整段閒置**，把平均拉低到 29%。
- **代表的瓶頸類型**：這不是「GPU 算力不足」，而是「**GPU 供給不足（starvation）**」—— pipeline 沒讓 GPU 一直有活幹。
- **兩個直接推論**：
  1. VRAM 只用 4.7GB/32GB → **同時塞多個 tile 的空間綽綽有餘**（理論可並行 ~6 tile）。
  2. 把 CPU-bound 的 PNG/匯出移出關鍵路徑、或多 tile 平行讓 GPU 不空轉，是把 29% 拉高、縮短 wall-clock 的正解。

→ 具體怎麼做見 [04-optimization-roadmap.md](./04-optimization-roadmap.md)。

## WSI 全圖估算

- **解析度**：156,222 × 134,028 px = 20,938 Mpx（`her2_warped_lv0.ome.tiff`）。
- **tile 數**：4096² tile → 39 × 33 = **1,287 tiles**。

| 情境 | tile 數 | 估算時間 | 說明 |
| --- | --- | --- | --- |
| 最佳（穩定態 100% 組織） | 1,287 | **12h 23m** | 全有腫瘤，34.7s/tile |
| 平均（含暖機 100% 組織） | 1,287 | 14h 03m | 39.3s/tile |
| **推薦估算（70% 組織）** | 900 / 1,287 | **8h 43m** | 空白 tile ~0.5s 快速跳過 |
| 並行 4 tiles（VRAM 允許） | — | 2h 10m | 假設線性加速 |
| PNG 優化 + 並行 4 | — | 1h 42m | 跳過 debug PNG 省 ~22% |

> 單機單 GPU、單線程下 WSI 全圖約 **8–12 小時**。並行 + 關 PNG 後理論可壓到 ~1.7 小時 —— 這是優化的目標區間。
