# 09 — 深度瓶頸量測與分析計畫

> 本文件是**任務與需求規格**：定義「要量什麼、怎麼量、量完怎麼歸類」。
> **不包含任何解法、不做任何解法猜測**——找到瓶頸後要往哪個方向解，只列**分類方向**，
> 具體怎麼改留待量測結果出來後另案決定。
> 也**不寫任何程式碼檔案**——本文件只規劃量測本身該怎麼做。

---

## 0. 為什麼需要重新規劃（現況核對）

寫這份計畫前，已對照 [08-problem-analysis.md](./08-problem-analysis.md) 的待釐清點，直接讀了目前 HEAD（`46e9c8d`）的原始碼，發現 **03/04 文件描述的架構已經整個換掉**，舊的效能數字不能直接沿用：

| 舊文件（01–07）描述 | 目前 HEAD 實際狀況（已讀原始碼確認） |
| --- | --- |
| M0 是「讀取」內嵌在逐 tile 迴圈裡的 chunk 迭代器（`iter_paired_chunks`），讀取與分析交錯進行 | `m0_reader.py` 已完全改成 `precut_paired_tiles()`：**先把整張 ROI/WSI 一次性預切成磁碟上的重疊 tile 檔**（`ThreadPoolExecutor` 平行寫檔），分析階段再逐檔讀。讀取與分析變成**兩個分開的階段**，不再交錯。 |
| `StitchAccumulator` 配置 6 張 full-H×full-W 的整圖 numpy 畫布（真實 WSI 下 ≈400GB 記憶體天花板，04 文件 L2 列為「長期優化」） | 整圖畫布**已完全移除**。改為：每塊獨立落地到磁碟（`core_mask/` `masked_ihc/` `instance_mask/` 等資料夾）、只有「細胞表格」全域合併（`compute_tile_geometry` + `filter_and_absolutize`，純函式、O(n)）、最終 overlay 用 `pyvips` 逐列/逐欄惰性 join 成 pyramid TIFF。**舊文件的「輸出端記憶體天花板」問題已經被解掉**，不用再列為現況瓶頸。 |
| 04 文件 M1 提案「多 tile 平行（ProcessPoolExecutor）」 | `hybrid_pipeline.run_batch()` 原始碼**明確寫死是序列迴圈**，且**明確註解說明原因**：三個 GPU 模型只在主行程載入一次、共用同一個 CUDA context，跨 process 平行會有 fork-under-CUDA 的問題；註解也明講「這輪重構只求正確性、不碰吞吐，跨 tile 批次化 GPU 推論是刻意延後、另案追蹤的優化」。→ **GPU 供給不足（starvation）這個舊瓶頸有沒有解掉、解到什麼程度，完全沒有被驗證過**——這是本計畫最優先要重新量測的項目。 |
| perf_report.html 的 cProfile Top 函數表（run_net/get_rel_pos/...） | 量測對象是舊架構（chunked reading + 全圖畫布）跑 3 tiles 的結果，且 M2 分割 (`m2_segmentation.py`) 本身也在最新 commit 被重構（dedup 邏輯拆成獨立函式）。**排名可能仍大致有效（Cellpose 前向本身沒換），但沒有在目前 HEAD 上重新量過，不能當作現況數字使用**。 |
| `cellpose_batch_size` 沒接線到 Config（G3） | 已重新核對 `config_example.py`：**依然沒有 `cellpose_batch_size` 欄位**，`hybrid_pipeline.py` 依然是 `getattr(config, "cellpose_batch_size", 16)` 硬 fallback 16。這個死 config 問題原封不動。 |
| 06 文件的 venv 路徑 `/home/sec312/project/tsgh/.venv` | 目前機器（`/data/taro_Projects/tsgh`）**沒有這個 venv**；本機找到的候選是 `/data/tsgh/.venv`。requirements.txt 仍是舊文件描述的衝突狀態（numpy 2.2.6 vs pyproject `numpy<2`；pyvips 3.1.1；scikit-image 0.25.2 vs pyproject `<0.25`；opencv 三個版本並列）——這部分沒變,但**要先確認本機真正在跑的 venv 是哪一個、裡面實際版本是什麼**，不能直接沿用 06 文件的版本表。 |

**結論（現況，非解法）**：新架構已經解決「輸出端記憶體天花板」，但「GPU 序列跑、平均利用率偏低」這個舊瓶頸的現況是**未知**——是本計畫的第一優先量測目標。此外，新架構引入了兩個**舊 perf_report 完全沒量過的新階段**：預切（precut，Phase A）與最終 overlay 縫合（`_stitch_overlay_slide`，Phase D），兩者都是全新的、與舊分析無關的 I/O 密集步驟，必須從零開始量測。

---

## 1. 任務目標與產出要求

### 1.1 目標
在目前 HEAD 的實際程式碼與實際硬體（RTX 5090）上，**重新產生一份可信、有時間戳、有版本戳的效能量測**，精確定位：
1. 每個階段（precut / per-tile M1-M2-M3 / 全域合併 / overlay 縫合）各佔多少 wall-clock。
2. 每個階段內部，時間花在哪些函式 / 哪個資源（GPU 前向、CPU、磁碟 IO）。
3. GPU 在整個流程中的真實利用率時間軸（不是單一平均數字），標出每一段空閒對應到哪個 CPU/IO 步驟。
4. 記憶體（RAM + VRAM）隨時間的真實曲線，而非單點峰值。
5. 目前 regime（3-tile 測試 vs 真實 WSI 1287-tile）下，排名是否一致；如果不一致，落差在哪。

### 1.2 產出要求（deliverables）
- 一份「量測結果報告」（含原始數據 + 圖表/timeline + 函式級 Top 表），格式比照現有 `perf_report.html` 但需標註：量測日期、git commit hash、venv 實際套件版本、硬體型號、輸入規模。
- 一份「瓶頸清單」，每條包含：現象描述、量測依據（哪份 profile/trace 的哪個數字）、**分類方向**（見第 6 節，只分類不開藥方）、信心等級（實測 vs 外推）。
- 若量測過程中發現任何「文件/程式碼/config 對不上」的新落差（如 G1-G5 那類坑），另行記錄，不要混進效能結論裡。

### 1.3 不在本計畫範圍內
- 不產出任何程式碼改動或 patch。
- 不對「該怎麼修」下結論或建議優先序（那是量測完之後的下一份文件的工作）。
- 不驗證醫學正確性 / 精度（本計畫只談效能，不談 Cellpose/UNet 輸出品質）。

### 1.4 量測執行順序（Discover 步驟，必須依序執行，不可跳過）

依「先量總數、再拆分」的原則，順序不可顛倒——不然會重蹈 CuPy 案例的覆轍（對只佔
1.1% 總時間的部分做了 9 版優化）：

1. **先量一個端到端 wall-clock 總數**（同一組輸入、同一次跑，從 CLI/API 呼叫開始到
   `report.csv`/`overlay_slide.tiff` 全部落地為止），作為本輪量測的**唯一錨點數字**。
   在還沒有這個總數之前，不要開始拆解任何子階段——子階段的秒數若沒有除以這個總數，
   毫無意義。
2. **這次端到端跑本身就是「笨版本」控制組**：目前 `run_batch()` 是明確寫死的序列迴圈、
   無任何跨 tile 平行化（見第 0 節），架構上已經是「最笨版本」。因此本輪測到的端到端
   總數**同時扮演兩個角色**：(a) 現況基準、(b) 未來任何優化嘗試的**負向優化偵測鏡子**
   ——之後不管做了什麼「優化」，只要總時間沒有明顯低於這個數字（或反而更高／持平），
   就要停下來重新量測，不能只信任局部 micro-benchmark。這個數字必須被**原樣保存**
   （見第 2 節第 7 點），不可事後用「理論上應該更快」的說法覆蓋。
3. **拆解時一律先算佔比，才看絕對秒數**：每個 Phase/子項的耗時都要先除以第 1 步量到的
   端到端總數，得到 `%`；候選瓶頸的認定標準是「**佔比最大**」，不是「絕對秒數最大」
   （見第 5 節的 % 排名表要求）。

---

## 2. 環境與基準前置需求（量測前必須先確認，否則數字不可信）

1. **確認本機實際可跑的 venv**：候選 `/data/tsgh/.venv`（本機找到的路徑，需人工確認是否為此專案的 venv，而非其他專案共用同名目錄）。要求：`source` 後跑 `pip freeze` 存一份快照，跟 `requirements.txt` / `pyproject.toml` 對照差異（延續 06 文件的比對方法，但用**這台機器的真實輸出**，不要沿用舊文件的版本表）。
2. **確認 GPU/CUDA 現況**：`nvidia-smi`（已確認本機為 RTX 5090, driver 580.159.03, CUDA 13.0）、`torch.cuda.is_available()`、`torch.cuda.get_device_capability()`（應為 `(12, 0)`），並實跑一次 Cellpose 前向確認真的吃得到 GPU（不要只看 import 成功）。
3. **確認 `config.py` 存在且可用**：`config.py` 是 gitignored；需 `cp config_example.py config.py` 並檢查是否仍缺尾端 `compute_config_hash()` / `config = Config()`（G2 舊坑，在目前 HEAD 上重新核對一次是否還在）。
4. **確認模型檔與測試資料位置**：本機目前找不到 `models/`、`test_picture/`、`perf_report.html` 等（可能在 gitignore 或其他路徑）——量測前必須先定位或補齊：
   - UNet++ / Cellpose (M2 IHC-DISH) / Cellpose (M3b DISH) 三顆模型權重。
   - 至少一組小規模測試 tile（供快速迭代）+ 至少一份完整 WSI 或大 ROI（供 regime 驗證，見第 3.6 節）。
5. **確認 `cellpose_batch_size` 現況**：已核對 `config_example.py` 無此欄位——量測時**必須先意識到 batch_size 固定是 16**，任何「調 batch size 觀察差異」的量測都要先在程式碼裡臨時加這個欄位才有意義（本計畫只指出這個前置條件，不在此文件內做程式碼改動）。
6. **建立乾淨的量測輸出目錄**，與正式 `output_dir` 分開，避免量測用的中間檔（precut tile、per-tile artifacts）污染正式輸出或反之。
7. **保存第 1.4 節第 1 步量到的端到端總數與其完整原始 log/trace 檔**（不只是摘要數字），
   標好 git commit hash、輸入規模、環境快照，永久留存為「控制組」。這是本輪唯一的
   笨版本基準，之後任何優化嘗試都要拿它做 ablation 對照（移除/加入某個改動，總時間
   是否真的變化）——遺失這份原始基準，之後就沒有東西可以拿來偵測負向優化。

---

## 3. 分階段量測計畫

新架構把流程拆成清楚的 4-5 個階段，各自的成本結構完全不同，必須**分開量測**，不能像舊 perf_report 一樣只看一次 cProfile 聚合總數。

### 3.1 Phase A — 預切（`m0_reader.precut_paired_tiles`）

- **性質**：I/O bound，`ThreadPoolExecutor` 平行讀（`pyvips access="random"`）+ 裁切 + 無失真 deflate 壓縮寫檔，每個 tile 位置寫 2 個檔（IHC + DISH）。
- **量測項目**：
  - 整體 wall-clock（從呼叫到所有 tile 寫完）。
  - 依 `workers` 數量（目前呼叫端有的傳 8、有的用預設）掃過幾組不同 thread 數，觀察 wall-clock 是否隨 thread 數線性下降（判斷是否真的被 IO 而非 GIL/CPU 限制住）。
  - 磁碟寫入量（tile 數 × 每 tile 檔案大小）與磁碟實際吞吐量（`iostat` 或等效工具）比較，確認是否吃滿磁碟頻寬。
  - 來源 WSI 檔案大小 vs 讀取解碼時間的關係（21GB 級 warped 檔本身開檔/隨機存取的開銷）。
- **量測規模**：至少跑一次「中型 ROI（觸發 10+ tile）」與一次「完整 WSI（1287 tile 等級）」，因為 I/O 瓶頸常常在小規模量不出來。

### 3.2 Phase B — 逐 tile 序列分析迴圈（`run_batch` 主迴圈 → `process_precut_tile` → `_process_one_chunk`）

這是舊 perf_report 唯一量過的部分，但現在的檔案 I/O 內容已經不同（見下），需要重新拆解。

- **子項 B1：GPU 計算本體**（M1 UNet++ 前向、M2/M3b 兩顆 Cellpose 前向）
  - 用 cProfile（延續舊方法）重跑，拿目前 HEAD 的函式 Top 表，直接對照舊 03 文件的排名（`run_net` / `get_rel_pos` / `compute_masks` / `flow_error` 等），確認佔比是否仍然一致，或因 m2_segmentation 的 dedup 重構而改變。
  - 額外用 `torch.profiler`（或 Nsight Systems）在單一 tile 處理期間抓 GPU timeline，取代「平均利用率」這種聚合數字，直接看每次 forward 之間的間隔對應哪個 CPU 階段。
- **子項 B2：每 tile 落地的檔案 I/O**（**與舊文件不同、需要全新量測的部分**）
  - 目前 `process_precut_tile` 對每個 tile **無條件**寫出：`core_mask/*.png`、`masked_ihc/*.png`、`dish_mask_overlay/*.png`、`instance_mask/*.tiff`（int32）、`dish_nucleus_mask/*.tiff`（int32）、`overlay_annotated/*.tiff`、`cell_crops/tile_x{x}_y{y}/`（多張逐細胞裁切）、以及可選的 `merge_overlay/*.tiff`。這與舊文件的「7 張 debug PNG」在檔案類型、格式、數量上都不同（多了 int32 TIFF、per-cell crop 資料夾），**必須重新量測每一類檔案各自的寫入耗時佔比**，不能延用舊的 13.4% 數字。
  - 需要分別量：PNG encode（`skimage.io.imsave`）、TIFF encode（int32 label 圖）、per-cell crop 迴圈（`export_per_cell_images`，數量隨細胞數變動，需確認是否隨細胞數線性成長）。
- **子項 B3：M3 分析**（`build_all_positive_results` / `enlarge_cell_instances` / `elastic_dish_nucleus_matching` / `detect_all_dots` + joblib）
  - 沿用舊方法但在目前 HEAD 重跑，確認 `detect_all_dots`（17.1%）與 joblib `delete_folder`（4.8%）的佔比是否還成立。
- **子項 B4：tile 間 GC / CUDA cache 清理**（`torch.cuda.empty_cache()` + `gc.collect()`，每 tile 呼叫一次）
  - 量測這兩行本身的耗時，確認是否隨 batch 拉長而變成不可忽略的固定開銷（舊文件沒有單獨量過這個）。
- **量測規模**：3-tile 快速迴圈（延續舊方法，供快速迭代對照）+ 至少一次數十 tile 等級的中型批次（觀察 tile 間 GC/cache 清理是否隨批次長度累積出可觀成本）。

### 3.3 Phase C — 全域合併（`compute_tile_geometry` + 攤平排序 + `export_tile_csv` / `export_summary_statistics`）

- **性質**：純 Python/numpy 表格運算，理論上應該很輕（O(n) 細胞數），但**從未被量過**（舊 perf_report 沒有這個階段，因為舊架構是邊跑邊縫）。
- **量測項目**：
  - 攤平 + `sort` + 重編號這段的耗時，隨全片總細胞數（可能數十萬顆）增長是否仍可忽略。
  - `compute_tile_geometry` 對 1287 個 tile 位置建幾何表的耗時（純 Python，需確認是否有隱藏的 O(n²) 比對，尤其 `_validate_axis` 與去重驗證邏輯）。

### 3.4 Phase D — Overlay 縫合（`_stitch_overlay_slide`）

- **性質**：**全新階段，舊 perf_report 完全沒有涵蓋**。在所有 tile 分析完成後，才逐列水平 join、再垂直 join 所有 `overlay_annotated/*.tiff`，最後以 lzw 壓縮寫出一張 pyramid bigtiff。
- **量測項目**：
  - 整體 wall-clock，並拆解「讀檔（`pyvips.Image.new_from_file`, access="sequential"）」vs「join 運算」vs「lzw 壓縮 + tiffsave 寫出」三段各自佔比。
  - 隨 tile 數量（列數 × 欄數）增長的 scaling 行為——1287 tile（39×33）下這個**單一序列步驟**（無平行化）本身可能就是可觀的一段 wall-clock，需要精確數字而非猜測。
  - 這段是否為「阻塞在最後」的單點瓶頸：如果 Phase B 已經有某種平行化空間，Phase D 目前是完全序列、發生在所有分析都結束後——量出它在整體 wall-clock 中的絕對秒數與佔比。

### 3.5 Phase E — API / Job 排程層（`backend/api/hybrid.py` + `backend/api/jobs.py`）

- **性質**：新增的 FastAPI 包裝層，用 `BackgroundTasks.add_task` 執行整個 `run_batch`（或先 `precut_paired_tiles` 再 `run_batch`）。
- **量測項目**：
  - 從 API 收到請求到背景任務實際開始執行之間的延遲（排隊/排程開銷）。
  - 確認 `BackgroundTasks` 的執行緒是否與主 FastAPI event loop 競爭資源（尤其若未來有多個並發請求）。
  - 這層目前是否可能是可忽略的固定開銷（相對於數小時等級的分析時間），或是否在**多個並發請求**情境下才會顯現問題——本計畫只列為量測項，不預設它「一定不重要」。

### 3.6 Regime 驗證：小規模 vs 真實 WSI

- 03 文件的 WSI 全圖估算完全來自 3-tile 外推，**從未有人在完整 156k×134k WSI 上實測過**。
- 本計畫要求至少完整跑一次真實規模（或至少數百 tile 等級的大 ROI），同時做 Phase A-D 的全流程 profiling + 資源監控，確認：
  - 各階段佔比在小規模與大規模下是否一致（尤其 Phase A 的預切與 Phase D 的縫合，兩者都與 tile 總數強相關，小規模測試可能完全看不出它們的真實成本）。
  - 記憶體曲線（RAM）是否真的如新架構設計「不隨 tile 數線性成長」（這是新架構的核心宣稱，需要用真實大規模跑驗證，不能只信程式碼註解）。

---

## 4. 橫切量測維度（跨所有 Phase 都要做）

1. **GPU 利用率時間軸**：不要只取一個平均值。用 `nvidia-smi dmon`（取樣間隔精細到秒級以下）或 `torch.profiler` 對整個 Phase B 迴圈取樣，畫出完整 timeline，標出每段對應哪個子步驟（B1 GPU / B2 IO / B3 CPU / tile-boundary GC）。
2. **CPU 利用率與 thread/process 拓撲**：確認 Phase A 的 `ThreadPoolExecutor`、Phase B3 的 joblib（`n_jobs=-1`）、Phase B 主迴圈是否有互相搶核心的情況（例如 precut 的 thread pool 還沒完全結束、下一階段就開始跑，兩者是否重疊執行）。
3. **RAM / VRAM 曲線**（非單點峰值）：整段 batch 跑完的完整記憶體時間序列，確認新架構「記憶體不隨 tile 數線性成長」的宣稱在大規模下是否成立，以及 Phase A 的 pyvips cache（`cache_set_max(0)`）在新的「一次性預切」流程下是否仍然必要/生效。
4. **磁碟 I/O 吞吐與容量**：新架構把大量中間產物落地到磁碟（precut tile 檔 + per-tile 5 種陣列 + overlay + per-cell crop），需要量測：
   - 總磁碟佔用量（隨 tile 數 × 細胞數增長的曲線）。
   - 讀寫吞吐是否在大規模下成為新瓶頸（尤其如果輸出目錄與來源 WSI 在同一顆磁碟上競爭頻寬）。
5. **Config/死變數稽核**：量測前後都要確認哪些 config 欄位是「設了沒用」（如 `cellpose_batch_size`），避免把死變數的固定行為誤判成某種效能特性。
6. **版本/環境戳記**：每一次量測輸出都必須記錄 git commit hash、venv 套件版本快照、硬體型號與 driver 版本、輸入規模與檔案來源——這是為了讓未來任何人重跑量測時能立刻判斷數字是否還適用（吸取 03 文件「量測時點模糊、事後無法精確判斷是否還有效」的教訓）。
7. **相鄰 Phase 邊界的供給 vs 消耗吞吐量比對**：不要只在 Phase B 內部看 GPU vs CPU/IO
   誰餓誰（第 4.1 點），**每一個相鄰 Phase 邊界都要各自量一次**：
   - Phase A（precut 寫檔速率，tiles/sec）vs Phase B（逐 tile 分析消耗速率，tiles/sec）
     ——確認目前是「A 完全跑完才開始 B」（純序列疊加）還是有任何重疊/pipeline 化；
     哪一段「supply」跟不上另一段「consume」（或反之）。
   - Phase B（逐 tile 分析完成速率）vs Phase D（overlay 縫合的讀檔/join 速率）——同理。
   - 這一步是為了避免「看起來平行/看起來重疊」但實際上被某個共享資源（單一磁碟、
     GIL、單一 CUDA context）序列化掉（playbook anti-pattern #10：mistaking "looks
     parallel" for "is parallel"）——要量實際達成的吞吐量，不能只看程式碼形狀
     （例如 `ThreadPoolExecutor` 存在不代表真的平行 I/O）。

---

## 5. 量測結果的記錄格式（給下一步分析用，不含解法）

### 5.1 必要的第一份輸出：% 排名表（Discover 的綜合產出）

在填第 5.2 節的瓶頸清單之前，**必須先產出一張表**：把第 3 節所有 Phase/子項
（A、B1–B4、C、D、E）的量到秒數，全部除以第 1.4 節第 1 步量到的端到端總數，
得到各自佔比，由高到低排序。這張表本身就是「候選瓶頸」的認定依據——**排名看 %，
不看絕對秒數**。

### 5.2 Analyze：Amdahl 天花板與停損規則

對排名表中**每一個**候選（不只是排第一的），計算 Amdahl 天花板 `1/(1-p)`
（`p` = 該項佔總時間比例）：
- **佔比落在個位數 %（<~10%）者，標記為「已達 Amdahl 天花板下限，本輪不再深入分析」**
  ——即使該項有明顯可優化的空間（例如演算法明顯是 O(n²)），也先如實記錄現象與佔比，
  不要投入更多量測資源去挖它的細節（呼應 playbook 案例：CuPy GPU 核心優化耗了 9 個
  版本卻只佔總時間 1.1%）。
- 佔比較高者，才進入第 5.3 節逐條記錄，並在該條額外附上 Amdahl 天花板數字。
- **「快」與「是瓶頸」要分開記**：某個 Phase/子項即使自身耗時很短（測起來很快），
  只要它佔總時間比例不低就仍是候選瓶頸；反之某個階段"看起來慢"但被上下游完全遮蔽
  （在其他資源被佔用的空窗期發生），也不能只憑直覺判為瓶頸——一切以第 5.1 節的
  % 排名表為準。

### 5.3 逐條瓶頸記錄格式

僅對第 5.2 節篩選後、佔比不可忽略的候選，記錄格式：

| 欄位 | 說明 |
| --- | --- |
| 現象 | 具體量到的數字（秒數、%、次數），來源 profile/trace 的精確位置 |
| 佔總時間比例 | 依第 5.1 節排名表，該項的 % |
| Amdahl 天花板 | `1/(1-p)`，理論上限——只是算給後續決策參考，本文件不據此下解法結論 |
| 所在 Phase | A/B1/B2/B3/B4/C/D/E（對照第 3 節） |
| 量測規模 | 3-tile / 中型 ROI / 完整 WSI，及對應 git commit |
| 是否為新現象 | 舊文件（03/04）是否已提過，或本次新發現（尤其 Phase A/D/C 是全新階段） |
| 分類方向 | 見第 6 節分類——只填分類，不填具體做法 |
| 信心等級 | 實測直接量到 / 由小規模外推 / 理論推論尚待驗證 |

---

## 6. 解法方向分類（僅分類，不含具體解法或猜測）

找到瓶頸後，先歸類它「屬於哪一種性質的問題」，作為之後決定要不要處理、由誰處理的依據。以下**只是分類框架**，本文件不對任一具體瓶頸下結論該歸哪類、也不建議怎麼改。

1. **演算法/模型複雜度類**：成本本質來自演算法或模型架構本身的計算複雜度（例如某個模型的前向計算量、某段去重邏輯的漸進複雜度）。
2. **硬體限制規避類**：成本受限於特定硬體特性或驅動/框架版本限制（例如新架構 GPU compute capability 的相容性限制、精度/資料型別造成的計算或搬移開銷）。
3. **平行/併發執行類**：目前是序列執行、但理論上有平行化空間的部分（不限於 GPU 平行，也包含 CPU 執行緒/行程、I/O 併發）。
4. **記憶體佔用/生命週期類**：與 RAM 或 VRAM 的配置、釋放時機、成長曲線有關的成本。
5. **I/O 與儲存佈局類**：與磁碟讀寫、檔案格式、壓縮方式、資料落地策略有關的成本。
6. **軟體架構/框架開銷類**：與 API 包裝層、任務排程、序列化、跨層呼叫有關、非核心運算本身的成本。
7. **設定/死程式碼正確性類**：不是真正的效能問題，而是「設定沒接線」或「程式碼與文件/設計不一致」導致量測或行為失真的問題（需先修正才能讓其他分類的量測可信）。

同一個瓶頸現象可能同時落在多個分類（例如「GPU 序列跑」可能同時是分類 3 與分類 6 的交集）——分類的目的是幫助後續決策分工，不是強制唯一歸屬。
