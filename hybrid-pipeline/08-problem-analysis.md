# 08 — 問題與瓶頸分析（僅問題定位 + 調查方法）

> 本文件只做兩件事：**1. 問題/瓶頸是什麼**、**2. 打算怎麼找出瓶頸的實際位置**。
> 不包含任何解法或結論性建議（解法請見 [04-optimization-roadmap.md](./04-optimization-roadmap.md)）。

---

## 1. 問題／瓶頸是什麼

### 1.1 已知的效能瓶頸排名（來自 2026-06-29 的舊量測）

依 [03-benchmarks-bottlenecks.md](./03-benchmarks-bottlenecks.md)：

- **Cellpose ViT-SAM GPU 前向**：`run_net`(32.3%) + `get_rel_pos`(25.1%) + `compute_masks`(8.1%) + `flow_error`(5.3%) ≈ 累計 **70%** 的量測時間。
- **Debug PNG 寫出**：`_write_m1_artifacts` + PIL encode ≈ **13%**。
- **M3 dot detection + joblib overhead**：偵測本身 17.1% + memmap 暫存清理 4.8% ≈ **22%**。
- **GPU 平均利用率僅 29%（peak 99%）**：文件判讀為「GPU 供給不足（starvation）」而非算力不足 —— 單 tile 內 M1→M2→M3→M4→PNG 是序列執行，GPU 在 CPU/IO 段落整段閒置。
- **輸出端整片縫合畫布**：`StitchAccumulator` 為整張 slide 配置 6 張 full-H×full-W numpy，被文件判定為真正 WSI 規模（156k×134k）下的新記憶體天花板，與讀取端無關。

### 1.2 這份瓶頸排名目前的可信度問題（核心待釐清點）

- 量測本身早於同一天稍後的 M0 chunked stitching / GC 記憶體優化 commit（03 文件已自行註記此點，並說「排名有效、絕對秒數可能已改善」）。
- **更關鍵**：目前 HEAD（`46e9c8d`，2026-07-05，比量測晚 6 天）是一次直接改動 `m0_reader.py`、`m0_stitch.py`、`m2_segmentation.py`、`hybrid_pipeline.py` 核心邏輯的重構 —— 把「chunked reading」換成「pre-cut tiling」、引入 `ThreadPoolExecutor` 做平行寫檔、重寫了 dedup 邏輯的組織方式。這代表：
  - 01/02/03/04 文件描述的巢狀迴圈、資料流、瓶頸佔比，可能已經與目前 HEAD 的實際行為不一致。
  - 03 文件「單 tile 序列跑、GPU 因此 starvation」的判讀，在新架構已經引入 `ThreadPoolExecutor` 平行寫檔之後是否仍然成立，尚未被驗證過。
  - 04 文件提出的「多 tile 平行（ProcessPoolExecutor）」是否與這次重構的 pre-cut tiling / ThreadPoolExecutor 有重疊或衝突，也未知。

### 1.3 文件自身承認的量測侷限

- `perf_report.html` 只用 3 個 4096² tile 手動跑一次 cProfile + 資源取樣，**沒有自動化腳本**，無法重跑取得目前 HEAD 的真實數字。
- WSI 全圖（156k×134k）的 8h43m 等估算，全部是由 3 個 tile 的單 tile 秒數外推，**沒有任何一次完整 WSI 的實測**。
- `cellpose_batch_size` 從未真正接線到 `Config`（`getattr(config, "cellpose_batch_size", 16)` 永遠回 16）—— 代表所有過去關於 batch size 的效能觀察，實際上都是在 batch=16 固定值下量出來的。
- 已知多處 `config.py` / `config_example.py` / docstring / HTML 說明文件彼此漂移（見 [07-gotchas-appendix.md](./07-gotchas-appendix.md)），任何依賴這些文件描述得出的效能結論都有「以為改了其實沒改」的風險。

---

## 2. 打算怎麼找出瓶頸的實際位置

### 2.1 先校準：確認舊量測基準在目前 HEAD 上是否還成立

- 用 `git show 46e9c8d` 逐檔讀 `m0_reader.py` / `m0_stitch.py` / `m2_segmentation.py` / `hybrid_pipeline.py` 的實際改動內容，對照 01/02 文件描述的巢狀迴圈與資料流，標出文件已經對不上目前 code 的地方，先重建目前 code 的正確心智模型，而不是直接沿用舊文件的架構圖。
- 讀目前 HEAD 的 `run_batch` / `process_single_tile` 實作，確認「單 tile 序列處理、GPU 在 CPU/IO 段落整段閒置」這個假設，在新增 `ThreadPoolExecutor` 平行寫檔之後是否還成立。

### 2.2 重新量測，而非沿用舊數字

- 依 [05-dev-testing-guide.md](./05-dev-testing-guide.md) 的方法（`--batch --test`，3 tiles）搭配 cProfile 重跑一次，拿目前 HEAD 的函數 Top 排名，逐項對照 03 的舊排名，找出哪些函數佔比因重構而改變（尤其是 `m0_reader` / `m0_stitch` 相關函數，因為它們正是這次重構的對象）。
- 同時用 `nvidia-smi dmon` 或 `torch.profiler` 取樣 GPU/CPU/RAM，重建資源利用率 timeline，確認「平均 29% GPU 利用率 / starvation」的結論在新架構下是否依然成立，或者平行寫檔已經改變了這個數字。

### 2.3 對「GPU starvation」假說做更細粒度的驗證

- 用 `torch.profiler`（或 Nsight Systems）在單一 tile/chunk 的處理過程中抓 GPU timeline，精確標出每次 GPU forward pass 之間的間隔各自對應到哪個 CPU/IO 階段（PNG 寫出、joblib、M4 export、M0 讀取），取代目前 03 文件僅靠一個聚合的「平均利用率 29%」數字所做的推論。
- 確認 `get_rel_pos` 每 tile 呼叫 12,672 次這個數字的來源（是否隨輸入尺寸 / pre-cut tile 大小而變化），並在新的 pre-cut tiling 機制下重新統計一次，確認呼叫次數是否仍然一致。

### 2.4 交叉確認 config / 程式碼是否真的如文件所述在運作

- 檢查 `cellpose_batch_size`（G3 提到的死 config）在目前 HEAD 是否已被接線；若尚未接線，任何「batch size 造成的效能差異」的觀察都必須先排除這個固定值 16 的干擾。
- 用 `codegraph_context` / `codegraph_callers` 追蹤目前 `_process_one_chunk`、`StitchAccumulator.add`、`CellposeSegmenter.eval` 等關鍵函式的呼叫鏈之前，先用 `git ls-files` 核對 codegraph 索引沒有把幻影檔案（見 07 的 G1）混進來，避免根據不存在的檔案推出錯誤的呼叫關係。

### 2.5 在真正 WSI 規模上驗證，而非只靠 3-tile 外推

- 03 文件的 8h43m 等估算全部由 3 個 4096² tile 外推得出，尚無人在完整 156k×134k WSI 上實測過。需要規劃一次完整 WSI 的跑批，同時做 cProfile + 資源監控，確認：
  - 瓶頸排名在 regime 變大後是否依然成立；
  - 輸出端整片縫合畫布（在 3-tile 小規模測試中不會顯現）在真正 WSI 尺度下，記憶體佔用曲線實際長什麼樣子。
