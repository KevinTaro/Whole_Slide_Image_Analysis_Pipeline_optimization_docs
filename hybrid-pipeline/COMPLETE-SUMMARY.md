# Hybrid Pipeline 優化文檔 — 完整彙整（單檔版）

> 本檔把 `hybrid-pipeline/` 資料夾內所有文件（README、01–46、DISCOVERED-NOT-IMPLEMENTED、PERFORMANCE_BOTTLENECK_PLAYBOOK、`measurement/` 下各報告與原始量測資料夾）彙整成一份文件。
> 彙整原則：保留每份文件的**結論、關鍵數字、設計決策、停損理由**，並標出文件之間的**修正/推翻關係**（後面的輪次常常推翻前面的數字）。逐字細節請回到原文件。
> 彙整時間：2026-10-05；文件內容截至 round 15（2026-08-01）。

## 目錄

| 章 | 內容 | 對應原檔 |
| --- | --- | --- |
| §0 | 全域背景、三個修正提醒、資料夾結構、輪次對照速查表 | README.md |
| §1 | README 現況總結（round 15 後） | README.md |
| §2 | 基礎文件：架構、模塊、舊 benchmark、舊路線圖、dev 指南、依賴、踩坑、問題分析 | 01–08 |
| §3 | Round 1–2：重新規劃量測、GPU 序列瓶頸、GIL、`detect_all_dots` | 09–12、pipeline-overlap-result、gil-contention-diag、detect-all-dots-result |
| §4 | Round 3：Cellpose 換模型後重新排序、`gc.freeze()`、GPU starvation 前置條件 | 13–17 |
| §5 | Round 4 落地（doc 18）與開放待辦總表（doc 19） | 18、19 |
| §6 | Round 5：跨 tile 多行程 | 20、21 |
| §7 | Round 6：下一輪優化週期（`dot_detect_n_jobs=1`） | 22、23 |
| §8 | Round 7：GPU encode/decode 與逐項迴圈調查 | 24、25 |
| §9 | Round 8：剩餘待辦計畫與第一次完整 WSI 端到端 | 26、27 |
| §10 | Round 9：gc round 2、Phase D GPU 化 spike、tile-read I/O | 28–33 |
| §11 | Round 10：關閉 round 9 的待辦 | 34、35 |
| §12 | Round 11：cellpose 迴歸歸因、hang 修復、allocator 掃描、Option K | 36、37 |
| §13 | Round 12：`workers=1→4` 只有 ~2x 的完整成因 | 38、39 |
| §14 | Round 13：Phase D 管線化縫合上線 | 40、41 |
| §15 | Round 14：GPU↔CPU 資料搬運；doc 44 畫布位移調查 | 42、43、44 |
| §16 | Round 15：ETA 估算基準 | 45、46 |
| §17 | 「已發現但從未落地」全稽核 | DISCOVERED-NOT-IMPLEMENTED.md |
| §18 | 方法論 playbook | PERFORMANCE_BOTTLENECK_PLAYBOOK(.quickref).md |
| §19 | `measurement/` 現況報告 | bottleneck-list.md、current-status-comparison.md |
| §20 | `measurement/` 其餘：歷史版、翻譯版、HTML、原始資料 | `*-history.md`、`*-zh.md`、perf_report.html、pipeline-flow.html、`_metrics_r*` |
| §21 | 跨文件綜合：主線敘事、關鍵數字、方法論教訓、目前仍開著的事項 | — |

> **一句話現況（2026-08-01，round 15）**：在**正確（配準後）畫布**（156222×134028、35,700 tile、20.938 GP、65.92% 背景）上，整片分析 `workers=1` 3.023 h、`workers=4` **1.478 h**，peak RSS 13.66 GB；相對 round 1 當年的 18.87 h 投影快 12.77x（其中 ~2.85x 是工程優化、~2.19x 是組成修正、~2.05x 是多行程）。Phase D 縫合已用 `tifffile` 管線化上線。仍欠：Round-3 Cellpose checkpoint 的臨床簽核、Phase D 冷讀模型、跨玻片驗證、ETA 接進 UI。

---

## 0. 先讀這裡：全域背景與閱讀地圖

### 0.1 這個專案是什麼

- **對象**：`backend/algorithms/hybrid/` —— IHC(Her2)＋DISH 雙染色 WSI（Whole Slide Image）的逐 tile 分析流水線。
- **流程**：讀圖 → 疊合 → Cellpose 分割 → 逐細胞 HER2/CEP17 點位計數與擴增判定 → 縫回整圖 → 匯出 CSV/視覺化。
- **硬體**：RTX 5090（Blackwell, sm_120, 32 GB），**只能跑 cu130 的 torch**（`torch 2.11.0+cu130`），不可降版。
- **目標**：把整片 WSI（原先推估 8–18 小時）壓到可接受時間；本資料夾記錄了 15 輪「量測 → 規劃 → 落地 → 再量測」的優化歷程。

### 0.2 三個必須知道的「修正提醒」（貫穿所有文件）

1. **路徑遷移**：01–08 與 README 撰寫時程式碼在 `cell_mask/hybrid/`；UI Phase 1 重構後改為 **`backend/algorithms/hybrid/`**。讀到舊路徑一律換算。09 之後的文件已用新路徑。
2. **架構遷移**：01/02/03/04 描述的是 **pre-precut 架構**（M0 逐 chunk 迭代器讀圖）；目前 HEAD 已改成 `precut_paired_tiles` / `PrecutStream`（先把 tile 落地或串流到磁碟再分析）。巢狀迴圈的敘述、03/04 的絕對秒數均已過時，只有「瓶頸排名的方向性」與模塊 I/O 大致仍成立。
3. **畫布/組成數字修正（round 15）**：round 8–14 的整片數字（27,565 tile、55.8% 背景、~61 GB peak RSS …）量的是 `scripts/full_wsi_validate.py --conform` 硬裁出的**未配準畫布**（`conform_to_intersection()` 把 IHC/DISH 裁到同尺寸但沒對齊，繞過了 `PrecutStream` 該擋下的檢查，根因見 doc 44）。**正確（配準後）畫布是 156,222×134,028、35,700 tile、20.938 GP、65.92% 背景**。round 8–14 的秒數/加速比作為「計時資料」仍有效（比例不受影響），但**不是有效的臨床輸出**，也不該再被當成「目前的畫布規模」引用。

### 0.3 資料夾結構總覽

| 類別 | 檔案 | 作用 |
| --- | --- | --- |
| 導覽 | `README.md` | 導覽表 + 各輪 TL;DR（本檔 §0、§1 吸收其內容） |
| 基礎文件（舊，pre-precut） | 01–08 | 架構、模塊、舊 benchmark、舊路線圖、dev 指南、依賴版本、踩坑、問題分析 |
| 量測／規劃／落地（輪次文件） | 09–46 | round 2–15 的 plan（規劃）/ implementation（落地）/ result（結果）配對文件 |
| 清單類 | `19-open-backlog.md`、`DISCOVERED-NOT-IMPLEMENTED.md`、`26-remaining-work-implementation-plan.md` | 待辦、已發現但未落地的優化 |
| 方法論 | `PERFORMANCE_BOTTLENECK_PLAYBOOK.md`（+ `.quickref.md`） | Discover→Analyze→Plan→Choose 的瓶頸分析方法 |
| 視覺化 | `pipeline-flow.html`、`measurement/perf_report.html` | 流程圖、舊 perf 報表 |
| 現況報告 | `measurement/bottleneck-list*.md`、`current-status-comparison*.md` | 現況清單（精簡版）與歷史版、中文版 |
| 專題結果 | `measurement/detect-all-dots-result.md`、`gil-contention-diag.md`、`pipeline-overlap-result.md` | 單項實驗結果 |
| 原始資料 | `measurement/_metrics_r6`、`_r7`、`_r8`、`_r9` | JSON/CSV/log/dmon 原始量測 |

### 0.4 輪次對照速查表（round ↔ 文件 ↔ 主題 ↔ 結論）

| Round | 日期 | 規劃/落地文件 | 主題 | 一句話結論 |
| --- | --- | --- | --- | --- |
| 1（baseline） | 2026-06/07 | 03、08、09 | 舊量測與重新規劃量測 | GPU 平均 29% → starvation 假說 |
| 2 | — | 10、pipeline-overlap-result | 單 process 兩段式 pipeline/overlap | −18.5%，idle 0.494→0.154 |
| 2–3 | — | 11、gil-contention-diag、12、detect-all-dots-result、13 | GIL 診斷、`detect_all_dots`、Cellpose 4.2.1.1 | doc 11 假設被推翻；②被 GPU 前段遮住 |
| 3 | — | 14/15/16 | `gc.collect` 頻率 | `gc.freeze()`，1.069–1.077x |
| 4 | — | 17/18 | GPU starvation 前置條件 | ⑧搬離 MAIN −8.0%/−5.0%；precut 串流化；batch_size 掃描負向 |
| 5 | — | 19、20/21 | 跨 tile 多行程（`workers`） | `workers=3` 3.09x；Candidate D 上線 |
| 6 | — | 22/23 | 下一輪優化 | `dot_detect_n_jobs=1` 1.60x；GIL 異常坐實 |
| 7 | — | 24/25 | GPU encode/decode 迴圈 | 玻片組成 55.8% 背景；B/D/E/F 關閉；Phase D 322.7 s |
| 8 | 2026-07-27 | 26/27 | 剩餘待辦 + 首次整片端到端 | `workers=1` 3.82h / `workers=4` 1.73h，2.216x；放行 gate 關閉 |
| 9 | 2026-07-27 | 28/29/30、31/32/33 | gc round2、Phase D GPU 化、tile-read I/O | re-freeze 上線；Phase D GPU 化 spike；prefetch 上線 |
| 10 | — | 34/35 | backlog：overlay 修復、整片 `workers=1` | Phase D GPU 化反轉為不值得；`B1_m3b_cellpose` +751 s 未歸因 |
| 11 | 2026-07-29 | 36/37 | cellpose 迴歸歸因、卡住缺陷、allocator | `expandable_segments:True`（4/12→0/12 OOM）；fail-fast 卡住修好 |
| 12 | 2026-07-29 | 38/39 | `workers` 擴展天花板 | 實為 1.745x；Phase D 佔 32.3% |
| 13 | 2026-07-29 | 40/41 | Phase D 管線化 | tifffile backend：Phase D 1.913x、e2e 1.200x |
| 14 | 2026-07-30 | 42/43 | GPU↔CPU 搬運 | 三項關閉負向；唯一發現 `workers=4` OOM 風險 |
| — | — | 44 | conform_to_intersection 位移調查 | 輪 8–13 餵的是未配準輸入 |
| 15 | 2026-07-31～08-01 | 45/46 | ETA 估計、正確畫布整片量測 | 35,700 tile；`workers=1` 3.023h、`workers=4` 1.478h（2.05x） |

---

## 1. README.md — 現況總結（2026-08-01，round 15 後）

### 1.1 TL;DR 的核心結論（由新到舊）

**Round 12–15（最新）**
- Round 12：`workers=4` 自 round 8 後首次重跑，**實際只有 1.745x、不是 2.216x**。tile 平行臂本身沒退化（排除 Phase D 後仍 2.279x），全部落差來自 **Phase D（縫合）成長到 wall 的 32.3%**。順帶關掉 batch-claiming（load-balancing 候選，端到端天花板僅 1.00006x）。
- Round 13：**Phase D 縫合管線化並上線為預設**（`config.stitch_backend = "tifffile"`：背景執行緒讀檔 + CPU box pyramid）：整片 **Phase D 1.913x、end-to-end 1.200x**，四道正確性 veto 全過，peak RSS 45.6 → 17.0 GB，Phase D 佔 wall 降到 20.3%；`pyvips` 保留為 fallback。
- Round 14：系統性排查 `workers>1` 的 GPU↔CPU 資料搬運——三個候選（讀檔並發、Cellpose `_from_device`、pinned memory）全部關閉負向、**沒動 pipeline code**：`_from_device` 24.2% 樣本裡 96.3% 是 GPU-wait（四行程搶同一張卡，2.63x 排隊），真正的 D2H 只佔 wall 0.18%。唯一新發現：`workers=4` 在裁切規模上 10 次 OOM 1 次（VRAM 餘裕是四個 worker 共用 ~2.5 GB）。
- Doc 44：找到更根本的問題——round 8–13 整片驗證餵的是 VALIS 配準前的原始輸出（`*_processed.tiff`）；正確輸入是 `module4_thumbnail.py` 產生的 `*_warped_lv0.tiff`（天生同尺寸）。`conform_to_intersection()` 與 `--conform` 已移除（69 行）。
- Round 15：在**正確畫布**首次量出全套數字：**35,700 tile、20.938 GP、65.92% 背景**（組織 tile 數幾乎沒變：12,167 vs 12,184，多出的 29.5% 全是背景）。組織/背景每 tile 係數 **0.7298 s / 0.0460 s（15.9x，R²=0.997）**；首次有誤差棒（n=3，CV<1%）；量到 Phase D page-cache 轉折（cache 內 11.26 s/GP vs 冷讀 60.90 s/GP，5.41x）；workers 的 serial 比例不是常數（576 tile 時 36.8%、4,096 tile 時 17.1%），故 doc 38/39 的單點 Amdahl 不能外推。**整片實測：`workers=1` 3.023 h、`workers=4` 1.478 h（2.05x）、peak RSS 13.66 GB**。與 round 1 同批 crop 對照：`workers=1` 快 2.85–2.88x、`workers=4` 快 5.92x（large crop）。交付 `scripts/eta_estimate.py`（`fit`/`estimate`），尚未接進 API/UI。

**Round 9–11（2026-07-27 → 07-29）**：第 8 輪留下的三大開放項目全部關閉。
- `gc.collect`：round 9 上線週期性 re-freeze；round 10、11 各自獨立整片確認 58.8 s → 19.5 s（逼近 doc 31 §7 預測天花板 1.19x）。
- tile read prefetch：round 9 單行程、round 10 多行程上線。
- Phase D GPU 化：round 9 在記憶體 slab 上量到 1.40x/2.15x，**round 10 補測讀檔/join 半邊後整個反轉**（兩候選都比現行 pyvips 慢）→ Phase 2 判定不值得做，正式關閉。
- **關鍵修正**：round 8 的「`B2r_tile_read` 佔 17.2% wall」被 `d6592c3`（移除每 tile 275 GB 中繼檔寫入）污染 5.3 倍；扣除後真實成本 **4.21% of wall**，Option L 已吃掉 99.8%。
- 可靠性：修好「fail-fast 之後行程卡住不退出」（`workers=4` 曾卡 49 分鐘；`task_q.cancel_join_thread()`，never → 0.32 s）。
- `expandable_segments:True`（`workers=4`）12 次交錯：對照組 4/12 OOM、開啟後 0/12；已在 `b3fa47d` 落地為 `config_example.py` 預設。

**Round 8（2026-07-27）**：本專案**第一次完整 WSI 端到端跑批**（`workers=1`、`workers=4` 各一次，皆通過正確性投票），`workers>1` 的生產放行 gate（19-open-backlog item 7）關閉——但帶 VRAM 但書：`workers=4` 在整片量到 **93.3% 卡用量（30,439 / 32,607 MB，餘裕 ~2.2 GB）**。同輪發現三個「裁切時以為關閉、整片不成立」的數字：`gc.collect`（16.1% wall）、tile read（17.2%）、Phase D（8.6%/19.3%）——後續全部關閉。

### 1.2 單 process（`workers=1`）瓶頸演進

- **#1 為 Cellpose GPU 前向**（UNet++ core mask + M2/M3b 兩次 Cellpose），歷經五輪優化：單 process 兩段式 pipeline/overlap、Cellpose 4.0.8→4.2.1.1（DINOv3 `cpdino` backbone + bfloat16）、`gc.freeze()`、CPU 前處理搬離 MAIN 臂、precut 串流化、**round 6 拔掉 `detect_all_dots` 的 joblib 平行派工（`dot_detect_n_jobs=1`）**。
- Round 6 的一行 config 改動實測 **1.60x**（large/441 錨點 484.7 → 302.7 s）——原因不是 CPU 段變快，而是背景執行緒少了 19 條搶 GIL 的 thread，MAIN 臂 Cellpose 前向自己快 43.4%。累計 large/441 錨點 **848.0 s → 302.7 s（−64.3%）**。
- 全 WSI 單 process 估算：~8.1h → ~5.3h（round 6）→ ~2.6h（round 7，用實測組成）→ 實測 3.82h（round 8，舊畫布）→ 3.023h（round 15，正確畫布）。

### 1.3 跨 tile 多行程（`workers`）建議值的演進

- Round 5：`workers=6`（round 5b 建議）；round 6 重掃後**下修為 `workers=4`（無人看管）/ `workers=5`（可接受重跑）**：`workers≥6` 新測 6 次跑 2 次因 CUDA allocator 碎片化 OOM。
- `workers=3` large/441 錨點 **3.09x**（482.8 → 156.1 s，效率 103%）；`workers=4` 疊加 round 6 改動 128.8 s。遠超原估（1.23–1.7x）——原估漏算了 MAIN/BG 兩臂的 GIL 競爭。
- Round 7（組成對齊真實玻片的 576 tile 錨點）：`workers=4` **2.14x**（188.8 → 88.3 s），兩臂模型 BG/MAIN = 0.47–0.53（BG 臂還有 47–53% 餘裕）——這是 round 7 把落在 BG 臂的 GPU 候選（doc 24 的 B/D/E/F）全以「wall 天花板 1.00x」關掉的依據。
- Round 8：整片 `workers=4` 加速 2.216x；Round 12：修正為 1.745x（Phase D 吃掉）；Round 13 後 e2e +1.200x；Round 15 正確畫布 2.05x。

### 1.4 導覽表（README 的「你想做的事 → 先讀」）

README 把全部文件分為以下主題線（本檔後續章節逐一展開）：
- 架構/資料流：01；模塊：02；舊 benchmark：03；舊路線圖：04；問題與量測計畫：08→09。
- 現況：`measurement/bottleneck-list.md`（現況）、`current-status-comparison.md`（baseline vs HEAD）、`DISCOVERED-NOT-IMPLEMENTED.md`。
- GPU 序列瓶頸：10 → pipeline-overlap-result；GIL：11 → gil-contention-diag；dots：12 → detect-all-dots-result；換模型後：13。
- gc：14→15→16；GPU starvation 前置：17→18；多行程：20→21；下輪：22→23；encode/decode：24→25；待辦壓縮：26→27；
- gc round2：28→31；Phase D GPU 化：29→32；tile read：30→33；round 10：34→35；round 11：36→37；round 12：38→39；round 13：40→41；round 14：42→43；畫布修正：44；ETA：45→46。
- dev：05；依賴：06；踩坑：07；視覺化：pipeline-flow.html；待辦：19、`../BACKLOG.md`。

**README 建議閱讀順序**：先 `measurement/bottleneck-list.md`（注意該檔原本停在 round 11，需補 round 12–15）→ 依接手項目挑 10–46 → 需要架構細節再看 01/02。03/04 只有歷史價值。**接手前先確認還開著的項目**：Phase D 冷讀子模型失準、跨玻片驗證只有一張玻片、`workers=4` VRAM OOM 風險未重測。

### 1.5 參考地圖（README 附）

- 模塊級權威規格：`backend/algorithms/hybrid/CLAUDE.md`
- 核配對 v3 說明：`docs/algo/elastic_matching_v3_explainer.html`（已更新為 v4，以 code 為準）
- sliding-window 縫合：`docs/algo/sliding-window-seam-stitch.html`
- 前後端傳圖 vs 傳路徑：`docs/algo/frontend_backend_split_architecture.html`（對應 UI 護欄「邊界一律檔案路徑 + JSON」）
- UI 交接：`../UI/README.md`（FastAPI + React + pywebview，Phase 1–3 已完工）
- 已移除死連結：`cell_mask/docs/6_30_report.md`

---

## 2. 基礎文件 01–08（pre-precut 時期，作為背景）

### 2.1 doc 01 — 架構與資料流

**Pipeline 全景**：輸入一對配對影像（IHC/Her2 tile、DISH tile，同尺寸；單 tile／ROI／整張 WSI）→ M0 讀 → M1 → M2 → M3 → M0 縫 → M4 匯出。

| 模塊 | 檔案 | 角色 |
| --- | --- | --- |
| M0 讀 | `m0_reader.py` | pyvips 隨機存取，把任意大小輸入切成 bounded-memory 的重疊 chunk，IHC/DISH 同 offset 對齊 |
| M1 | `m1_overlay.py` | UNet++ 產生 IHC 腫瘤 core mask，套到 IHC＋DISH，50/50 alpha blend 當 M2 輸入 |
| M2 | `m2_segmentation.py` | Cellpose 在疊合圖上切 instance mask（重疊視窗 + IoMin 去重） |
| M3 | `m3_module/*` | 逐細胞生成結果、DISH 核彈性配對、核內數 HER2/CEP17 點、Score 判擴增 |
| M0 縫 | `m0_stitch.py` | 每 chunk 局部結果去重、重編號、絕對化座標，增量貼回 slide-level 整圖 |
| M4 | `m4_export.py` + `m4_module/*` | CSV、醫師檢視 overlay、逐細胞 crop |

> M0 出現兩次：讀在最前、縫在 M3 之後；M0 是包在 M1–M3 外面的「分塊殼」。

**巢狀迴圈**（`hybrid_pipeline.py`）：
- 外圈 `run_batch`：掃目錄、`find_paired_tiles` 依檔名排序配對；**模型只載一次**（`_init_unet_inferencer` / `_init_cellpose_segmenter` / `_init_dish_cellpose_segmenter`，避免每 tile 重載 ~582 MB 權重）。
- 中圈 `process_single_tile`：無論輸入多大一律用 `default_tile_size`(1024) 分塊，記憶體恆定。
- 內圈 `_process_one_chunk`：單塊 M1→M2→M3，回傳帶絕對 offset 的 `ChunkResult`；`acc.add(cr)` 後 `cr` 出作用域立即可 GC。

**Chunk 記憶體生命週期**：任一時刻只有一個 chunk 的中間產物 + 一張正在被增量填的整圖畫布；記憶體不隨 chunk 數線性成長，只由整圖畫布大小決定（把 20k² ROI 從 ≈31 GB 壓下來的關鍵）。**但畫布仍是 full-H×full-W**（`StitchAccumulator.__init__` allocate 6 張整圖 numpy：instance/nucleus mask + core_mask + 3 張 RGB），對 156k×134k WSI 這是新的記憶體天花板。

**StitchAccumulator 與質心 core-ownership 去重**：
- 每個 chunk 沿 `overlap/2` 切出互不重疊鋪滿全圖的「核心區」（`_cut_lines`）；一顆細胞只算在其質心落在的那塊核心區（`bisect_right(cuts_x, gxc) != col`）。
- 視窗**內部**（`segment_windowed`）仍用 IoMin 去重；**跨 chunk** 改用質心落點，成本由 O(n²) 降到 O(n)。
- 同時做全域重編號（細胞 1..N、DISH 核 1..M、`assigned_dish_ids` 一併改寫）與座標絕對化。

**關鍵資源設定**：
- `pyvips.cache_set_max(0)`：預設會把每次 `crop()` 結果快取在 C heap，掃 WSI 數千次不釋放→RAM 單調成長；關掉後記憶體可預測。
- `run_batch` 每 tile 後 `torch.cuda.empty_cache()` + `gc.collect()`，作為 tile 邊界重置點。

**單塊退化 = 回歸基準**：輸入 ≤ 1024 只 yield 一 chunk → 核心區=整塊、無接縫 → 輸出必須 bit-identical 於 pre-M0 單影像路徑（GPU 推論非決定性，跨 run 比對用 noise floor）。

### 2.2 doc 02 — 逐模塊參考

（路徑相對 `backend/algorithms/hybrid/`。Invariant：影像 RGB `uint8 (H,W,3)`；core mask `uint8{0,1}`；instance mask `int32`，背景 0、細胞 1..N。⚠️ 描述 pre-precut 架構。）

- **M0 讀取 `m0_reader.py`**：`iter_paired_chunks(ihc_path, dish_path, tile_size=1024, overlap=256)`；pyvips `access="random"`；視窗格線沿用 `m2_segmentation._overlap_window_coords`（`stride=tile_size-overlap`）；越界用 `gravity(extend="white")` 補白；套件 `pyvips 2.2.3`、`numpy 1.26.4`；IO-bound。可優化：ROI-only 掃描、prefetch。
- **M0 縫合 `m0_stitch.py`**：`StitchAccumulator(positions, full_h, full_w, overlap).add()/.finalize()`、`clear_slide_edge_cells`；`_paint_unmatched_core_nuclei` 把未配對但質心在核心區的 DISH 核畫入（僅供輪廓）；瓶頸：6 張全尺寸畫布。
- **M1 `m1_overlay.py`**：`generate_ihc_core_mask`、`apply_mask_to_ihc_image`、`overlay_ihc_mask_on_dish`、`fuse_masked_ihc_with_dish`；非 ROI 填 `background_fill_value=255`；`dish×(1-alpha)+ihc×alpha`；空 core mask 短路成空 CSV。關鍵參數 `overlay_alpha` config=0.65/example=0.5、`core_close_kernel=7`、`mask_blur_sigma=0.0`。
- **M2 `m2_segmentation.py`**：`CellposeSegmenter`、`segment_windowed()`、`_dedup_instances()`（面積由大到小貪婪 IoMin；256 粗格空間雜湊）；舊文件寫 ViT-SAM backbone：`run_net` 32.3%、`get_rel_pos` 25.1%、`compute_masks` 8.1%、`flow_error` 5.3%；`cellpose_batch_size` 硬編 16；`cellpose_flow_threshold` 0.6/0.4、`cellpose_cellprob_threshold` -0.8/0.0（config/example）；`window_overlap_px=256`、`window_dedup_iomin=0.5`。
- **M3**：
  - `m3_cells_generator.py`：`build_all_positive_results`（一次 `center_of_mass`）、`enlarge_cell_instances`（`expand_labels`，面積放大 `cell_enlarge_area_factor=1.5`，**放大版只供配對/點偵測**）。
  - `m3_elastic_matching.py`：**以細胞為中心 + 重疊優先 + reach 候選**；候選 (a) 與綠框重疊的核、(b) `cKDTree.query_ball_point` 在 `reach=max(sqrt(factor*area/π), min_reach_px)` 內的核；排序鍵 `(is_reach_only, dist)`；貪婪一對一 + lock；0 核細胞分 drop-out（有過候選）與 0/0（從無候選）。參數 `dish_elastic_expand_factor=1.5`、`dish_elastic_min_reach_px` 20/0。
  - `m3_dot_detection.py`：`detect_all_dots(...)`；先 `_filter_dish_nucleus_by_core_mask`、`_build_nucleus_owner_mask`、逐細胞在自己核區域內偵測紅/黑點、`_finalize_per_cell` 算 Score；舊時 `n_jobs=None`→joblib 全核（後被 round 6 改為 1）；累計 17.1%，joblib `delete_folder` 4.8%。dot 參數在 config.py/example 有多處差異（`dot_red_h` 5/12、`dot_red_a_min` 17/25、`dot_red_min_area` 5/7、`dot_red_min_contrast` 7/10、`dot_merge_distance` 2/3 等）。
  - `m3_dot_kernels.py`：RGB→LAB，紅點看 a* H-maxima、黑點看 L* H-minima；多準則閘控；ring 對比；union-find 合併近距離點。
- **M4**：`m4_export.py` facade + `m4_module/{csv,overlay,cell_crops}.py`；CSV 欄位 `cell_id/centroid_x/centroid_y/reddot/blackdot/score`（excluded→NaN）；真正的 IO 大戶是 M1 debug PNG（`_write_m1_artifacts` 每 tile 7 張）。`cell_crop_size` 100/256。
- **UNet++ `unet_inference.py`**：`UNetPPInference`；≤1024² 直接推論、>1024² 無重疊滑動視窗；`smp.UnetPlusPlus`、FP32 + `cudnn.benchmark`；後處理形態閉合 + 移除 <550 px 碎片；`unet_encoder_name` timm-efficientnet-b4 / efficientnet-b4；batch_size 4/8。
- **資料型別 `hybrid_data_types.py`**：`DetectedDot`、`CellAnalysisResult`、`CellDotResult`（避免 m4 依賴 m3）。
- **設定**：`config.py` gitignored、`config_example.py` 範本；`compute_config_hash` = SHA-256 前 8 碼寫入每張 CSV。

### 2.3 doc 03 — 舊 benchmark（2026-06-29，M0 優化前 17 分鐘）

- 條件：`--batch --test`，3 個 4096² tile，RTX 5090。
- 每 tile：48.7 s（首 tile 含暖機 +1.4 s）/ 37.8 s / 31.5 s，合計 120.4 s → 39.3 s/tile；M2 Cellpose ~48%、M3 ~26%、M4 ~7%、PNG 寫出恆定 ~5.4 s。
- cProfile Top：`run_net` 32.3%、`get_rel_pos` 25.1%（每 tile 呼叫 12,672 次）、`detect_all_dots` 17.1%、`_write_m1_artifacts` 13.4%（PIL encode 13.0% 與之重疊）、`compute_masks` 8.1%、`flow_error` 5.3%、joblib `delete_folder` 4.8%。
- 三大瓶頸群：Cellpose ViT-SAM 前向（~70%）、debug PNG（~13%）、M3 + joblib overhead（~22%）。
- 資源：CPU 平均 15.2%（峰 76.8%）、**GPU 平均 29.3%（峰 99%）**、VRAM 4.7/32 GB、RAM 17.2 GB。解讀為 **GPU starvation**（供給不足，非算力不足）。
- WSI 估算：156,222×134,028 = 20,938 Mpx → 1,287 個 4096² tile；最佳 12h23m、推薦（70% 組織）8h43m、並行 4 → 2h10m、+ 關 PNG → 1h42m。（皆為外推）

### 2.4 doc 04 — 舊優化路線圖（pre-refactor）

- 短期：**S1 關 debug PNG**（省 ~16 s/tile、WSI 快 ~22%，風險極低）；**S2 joblib backend 調整**（省 ~4.8%）；**S3 Cellpose batch 16→32/64**（需先補 `Config.cellpose_batch_size`）。
- 中期：**M1 多 tile ProcessPoolExecutor 平行**（頂部註明當時「不安全：fork-under-CUDA」；後 round 5 以 `spawn` worker pool 實現）；**M2 換非 SAM Cellpose 模型**（省 ~60%，需重訓與驗證；後 round 3 換成 4.2.1.1 `cpdino`）。
- 長期：L1 GPU daemon 常駐（避免重載 582 MB）；L2 縫合畫布只縫 ROI／串流輸出。
- 設計決策：core-ownership 去重 O(n) vs IoMin O(n²)；分塊讀避免 20k² ROI 峰值 ≈31 GB。
- 棄用嘗試：cuCIM 換 reader（400 GB 壓力來自輸出端縫合畫布，換輸入 reader 無效）；舊「細胞膨脹搶核」彈性配對（已重寫）；`dish_elastic_expand_factor` 標棄用但 code 仍用。

### 2.5 doc 05 — 開發與測試指南

- hybrid 目錄**沒有自動化 test 檔**（codegraph 的 `test_*.py` 是幻影）；唯一自動檢查 `scripts/verify_gc_freeze.py`（驗證 `gc.freeze()` cadence/pairing invariant）。round 8 後新增 46 個測試（見 doc 27）。
- CLI：`python hybrid_pipeline.py --test` / `--ihc A --dish B --output out/`；**`--batch` flag 不存在**（只有 `--test/--ihc/--dish/--output`）。
- 跑法：`source .venv/bin/activate` → `cp config_example.py config.py` → 填模型與目錄路徑 → 跑；量測用 `scripts/perf_measure.py --ihc ... --gpu-dmon --metrics-dir ...`（monkeypatch 計時 + `nvidia-smi dmon` + `psutil`），配套 `aggregate_report.py`、`resource_analyze.py`、`arm_report.py`、`gc_ablation_report.py`。
- 模型位置：`models/unet/b4/best_model_unet_b4.pth`、`models/unet/b6/...`、`models/cellpose/cellpose_ihc_dish_best`（M2）、`cellpose_dish_best`（M3b）。
- 回歸基準：單塊輸入 bit-identical；GPU 非決定性→noise floor。`m0_stitch` 可用合成 numpy 單測。

### 2.6 doc 06 — 版本與依賴

- **權威優先序**：venv 實測 ≥ requirements.txt > pyproject.toml。Python 3.11.15。
- 實測：torch 2.11.0+cu130、torchvision 0.26.0+cu130、numpy 1.26.4、cellpose 4.0.8（舊；後升 4.2.1.1）、pyvips 2.2.3、scipy 1.17.1、scikit-image 0.24.0、opencv 4.8.1、Pillow 12.2.0、smp 0.5.0、joblib 1.5.3。
- GPU：RTX 5090 sm_120 `(12,0)`，只能 cu130；不可降版。
- 三方衝突：numpy 1 vs 2（requirements 2.2.6）、skimage 0.24 vs 0.25.2、pyvips 2.2.3 vs 3.1.1（別照 requirements 升）、opencv 在 requirements 內列三個互相矛盾的發行版、torch 小版本漂移。
- 原則：不要用 requirements 重建環境（從 venv `pip freeze` 反推）；升 GPU 相關套件前先驗證 cu130/sm_120；「缺依賴直接報錯、不做 fallback」。

### 2.7 doc 07 — 踩坑附錄

- **G1 codegraph 索引過期**：舊記錄 30 個檔 vs 實際 20 個 `.py`，11 個幻影檔；**round 8 重新核對現行路徑零幻影**（indexed 20 / git 22 / disk 37）。教訓：存在與否以 `git ls-files` 為準。
- **G2 `config.py` gitignored**：需 `cp config_example.py config.py`；**檔尾缺 `compute_config_hash()` 與 `config = Config()` 的問題已修復**（226 行）。
- **G3 `cellpose_batch_size` 沒接線**：Config 無該欄位、`getattr(...,16)` 永遠 16（round 4 已接線並掃描，結果負向）。
- **G4 失聯 spec docs**（`docs/sdd-elastic-dish-matching.md`、`docs/dish_dot_detection_spec.md`）與 HTML 漂移：**round 8 已全部關閉**（含額外找到 `m3_dot_detection.py` 的第二個死引用；explainer 更新為 v4）。
- **G5 `generate_ihc_core_mask` 參數名誤導**（`ihc_tile_path` 實為 ndarray）：**round 8 已更名為 `ihc_image: Union[np.ndarray, Path, str]`**。
- 共通根源：文件/索引/範本與 code 漂移，**以 code 為準**。

### 2.8 doc 08 — 問題與瓶頸分析（只定位問題＋調查方法）

- 問題：舊瓶頸排名（Cellpose ~70%、PNG ~13%、M3+joblib ~22%、GPU 29% starvation、輸出端整片畫布）可信度存疑——量測早於 M0 優化；HEAD `46e9c8d`（2026-07-05）大改 `m0_reader/m0_stitch/m2_segmentation/hybrid_pipeline`（chunked → pre-cut tiling、`ThreadPoolExecutor` 平行寫檔），舊文件與 HEAD 可能不一致；`perf_report.html` 僅手動 3 tile；WSI 估算全為外推；`cellpose_batch_size` 死 config；文件漂移。
- 調查方法：(2.1) `git show 46e9c8d` 對照重建心智模型；(2.2) 重量測 cProfile + dmon；(2.3) `torch.profiler`/Nsight 驗 starvation 假說、核對 `get_rel_pos` 12,672 次；(2.4) 確認 config 接線、先 `git ls-files` 再信 codegraph；(2.5) 完整 WSI 規模跑批。
- 這些調查後來都在 doc 09 以後逐一執行（見後續章節）。

---

## 3. Round 1–2：重新量測規劃、GPU 序列瓶頸（doc 09、10、11、12 與對應結果）

### 3.1 doc 09 — 深度瓶頸量測與分析計畫（純規劃，不含解法）

**為何重新規劃**（對照 HEAD `46e9c8d`）：

| 舊文件描述 | HEAD 實況 |
| --- | --- |
| M0 是逐 tile 迴圈內的 chunk 迭代器 | 已改為 `precut_paired_tiles()`：先把整張 ROI/WSI **預切成磁碟 tile 檔**（`ThreadPoolExecutor` 平行寫檔），分析階段再逐檔讀 → 讀取與分析成為兩個分開階段 |
| `StitchAccumulator` 配 6 張整圖畫布（WSI 下 ≈400 GB 天花板） | 整圖畫布**已移除**：每塊獨立落地到磁碟資料夾，只有「細胞表格」全域合併（`compute_tile_geometry` + `filter_and_absolutize`，純函式 O(n)），最終 overlay 用 `pyvips` 逐列/逐欄惰性 join 成 pyramid TIFF |
| 04 的 M1 提案 `ProcessPoolExecutor` | `run_batch()` 寫死序列迴圈，註解說明：三個 GPU 模型共用同一 CUDA context、fork-under-CUDA 不安全 |
| perf_report cProfile Top | 量的是舊架構；M2 dedup 已被重構；需重量 |
| `cellpose_batch_size` 未接線 | 依然未接線 |

**新增的兩個階段**（舊 perf_report 完全沒量過）：Phase A 預切、Phase D `_stitch_overlay_slide`。

**量測紀律**（核心方法論）：
1. 先量一個**端到端 wall-clock 總數**作為唯一錨點，在此之前不拆解子階段。
2. 這次端到端跑同時是「笨版本控制組」，用來偵測未來的負向優化，必須**原樣保存**（含完整 log/trace、commit hash、環境快照）。
3. 拆解一律**先算佔比再看秒數**；候選瓶頸以「佔比最大」認定。

**分階段量測項**：
- Phase A（預切）：wall、thread 數掃描、磁碟吞吐、來源檔大小與解碼時間；至少中型 ROI 與完整 WSI（1287 tile 等級）。
- Phase B（逐 tile 序列迴圈）：B1 GPU 本體（cProfile + `torch.profiler`/Nsight 時間軸）、B2 每 tile 落地檔案 I/O（core_mask/masked_ihc/dish_mask_overlay PNG、instance_mask/dish_nucleus_mask int32 TIFF、overlay_annotated TIFF、per-cell crop、可選 merge_overlay——與舊「7 張 PNG」不同，需重量）、B3 M3 分析、B4 tile 間 GC/CUDA cache 清理。
- Phase C 全域合併（`compute_tile_geometry` 等，檢查隱藏 O(n²)）。
- Phase D overlay 縫合（讀檔/join/lzw 壓縮三段拆解；是否為尾端單點）。
- Phase E API/Job 層（`BackgroundTasks` 排程延遲）。
- Regime 驗證：小規模 vs 真實 WSI。

**橫切維度**：GPU 利用率時間軸（非平均值）、CPU/thread 拓撲、RAM/VRAM 曲線、磁碟 I/O、config 死變數稽核、版本/環境戳記、**相鄰 Phase 邊界的供給 vs 消耗吞吐比對**（playbook anti-pattern #10：mistaking "looks parallel" for "is parallel"）。

**結果記錄格式**：必先產出 % 排名表；Amdahl 天花板 `1/(1-p)`；個位數 %（<~10%）者標記「已達 Amdahl 下限，本輪不深入」（呼應 CuPy 案例：優化了 9 版卻只佔 1.1%）；「快」與「是瓶頸」分開記。逐條瓶頸記錄欄位：現象、佔比、Amdahl 天花板、Phase、量測規模、新現象、分類方向、信心等級。

**解法方向七類**：演算法/模型複雜度、硬體限制規避、平行/併發、記憶體生命週期、I/O 與儲存佈局、軟體架構開銷、設定/死程式碼正確性。

### 3.2 doc 10 — GPU 序列瓶頸方案設計（只談 ①）

- **問題**：GPU 閒置 ~46–49% wall；B1（三個模型前向）只佔 45.5%。B2 PNG/TIFF 寫檔 9.05% + B3 `detect_all_dots` 30.7% + B4 gc 4.28% ≈ **44.0%**，幾乎精確對上 GPU 閒置——**①的閒置與②③的 CPU 時間是同一塊時間的兩種讀法**。錨點：large（441 tiles）= **848.0 s**（不可覆蓋）。
- **約束**：跨 tile `ProcessPoolExecutor` 不安全（fork-under-CUDA）；必須保留 fail-fast 與「記憶體有界」。
- **候選**：
  - (a) 跨 tile GPU batching（改動面大，且需先接線 `cellpose_batch_size`；列長期）。
  - **(b) 單 process 內兩段式 pipeline/overlap（推薦，本輪做）**：GPU 階段（B1）留在主執行緒、CPU/IO 階段（B2+B3+B4）丟背景執行緒；理論上限 `max(45.5%, 44.0%)` vs 目前 ~89.5%。
  - (c) 架構/硬體級（換非 SAM backbone、GPU daemon），長期。
- **驗收標準**：large 總時間明顯低於 848.0 s（負向優化偵測）；idle_frac 從 0.46–0.49 實質下降；正確性在雜訊地板內 + fail-fast 保留；記憶體宣稱重驗；ablation。
- **不處理**：②、③、④、⑤⑥⑦、`cellpose_batch_size` 死 config。
- **實作前開放問題**：Cellpose 前向在多執行緒餵送下是否 thread-safe；`gc.collect`/`empty_cache` 呼叫點重設計；fail-fast timing；重疊粒度先用最簡單兩階段。

### 3.3 pipeline-overlap-result — 方案 (b) 落地結果

- 實作（只動 `hybrid_pipeline.py`）：把每個 tile 切成 **GPU 前段**（主執行緒：讀檔→M1 UNet→M2 Cellpose→M3b DISH Cellpose，`_process_one_chunk_gpu`/`_process_precut_tile_gpu`）與 **CPU 後段**（背景執行緒：`detect_all_dots` + merge + 核心去重 + 所有落地寫檔，`_finish_chunk_cpu`/`_process_precut_tile_cpu`；完全不碰 torch）。`run_batch`「先收前一塊、再提交本塊」，同時最多兩塊在飛。`process_precut_tile`/`_process_one_chunk` 保留為同步 wrapper。（對 doc 10 §4「只動 run_batch」的修正：三個 GPU 前向與 `detect_all_dots` 交錯在同一函式，必須拆。）
- 驗收（121 tiles, 8192² medium 錨點）：
  - wall：255.1/255.0/254.7 s → **207.9 s（−18.5%）**。
  - GPU idle_frac：0.494 → **0.154**（mean SM 26.3% → 34.5%）。
  - 正確性：3557 vs 3557 cells，0 筆不符（優於雜訊地板：兩次原碼互比 2 顆質心漂移 >3px，reddot/blackdot 各 2、score 1 筆不符）。
  - peak RSS 2.95 → 3.06 GB（+4%）。
- 為何不到理論 ~50%：當時歸因於 `detect_all_dots` 用 joblib `prefer='threads'` 與主執行緒搶 GIL（**此假設後被 gil-contention-diag 推翻**）。

### 3.4 doc 11 — stage 2：GIL 競爭

- 新錨點：medium 121 tiles baseline 255.0 s → (b) 後 **207.9 s**。剩餘 idle 15.4% → Amdahl 天花板 ≈ 1.18（卡在停損線邊緣）。
- **歷史兩難**：`prefer='threads'` 來自 commit `b51ce6a`（Jun 30），目的是省掉 loky 後端的 memmap `delete_folder` 清理開銷（舊 4.8%），與 GPU 重疊無關。換回 process 後端要重新引入已被判不划算的開銷，且 fork-under-CUDA 需顯式 `spawn`。
- 候選：(a) 診斷優先（py-spy，推薦）；(b) 換 process 後端（需 spawn-safety + memmap 重量）；(c) 加深 pipeline depth 2。
- **✅ (a) 已執行（2026-07-08）**：原假設被推翻——拖 GIL 的 **81% 是主執行緒自己**。結論：(b) 不做；浮現新槓桿 tile 邊界 `gc.collect()`，但 (d) gc 重定位 ablation 為**負向並已還原**；整體**停損**，保持方案 (b) 現狀。

### 3.5 gil-contention-diag（py-spy 診斷詳細結果）

- 方法：`py-spy 0.4.2` rate 200 Hz；GIL run（28,807 樣本）與 wall run（537,986 樣本）；py-spy 膨脹 wall（513 s / 866 s vs 原生 207.9 s），故只看比例。
- **GIL 歸屬**：主執行緒 81.4%（`gc.collect()` 33.6%、Cellpose Python 後處理 26.1%、UNet++ Python 14.9%）；背景 18.6%（`detect.red_black` 12.5%、`regionprops` 4.0%）。
- **Wall 分佈**：主執行緒 native/CUDA（GIL 釋放）61.0%；主執行緒 Python（持 GIL）29.0%（含 `gc.collect` 3.7%、Cellpose/UNet mask 重建 ~19%）；背景 10.0%。
- 殘餘 GPU idle ≈ 主執行緒 29% GIL-held Python 中沒與 GPU kernel 重疊的部分。
- **(d) gc→背景 ablation（負向、已還原）**：wall 218.03 → 217.09 s（−0.4%，雜訊內）、idle 0.183 → 0.221（反升）；VRAM/RSS 不變。原因：depth-1 管線的重疊是**背景 CPU 綁定**，gc 搬到背景只讓背景那一極更長。
- 唯一未試方向：降 gc 頻率（每 N tile 一次）→ 後來成為 doc 14–16（`gc.freeze()`）。
- **追加深挖（2026-07-08）**：
  - 修正 1：UNet++ 14.9% GIL 中 **93.3% 是一次性 import/模型載入**（`_init_unet_inferencer`），不是逐 tile；wall 只佔 2.6%（對上 bottleneck-list ⑥）。
  - 修正 2：Cellpose 那 ~19% wall 全在第三方（cellpose 4.0.8 + segment_anything）：`_extend_centers_gpu`、`get_masks_torch`、`steps_interp`、`get_rel_pos` **已在 GPU 上**，卡的是 Python for-loop 逐輪 launch 小 kernel（槓桿是 CUDA graph/`torch.compile`/向量化）；只有 `fill_holes_and_remove_small_masks`（`fill_voids` CPU C 擴充）是真正 CPU-only。即使歸零天花板只有 ~1.23x，且需 patch 釘死版本的第三方套件 → 維持停損、列 backlog（理由更正為「天花板太小、改動面是第三方套件」而非「model-inherent」）。

### 3.6 doc 12 — `detect_all_dots` 優化方案（②）

- 前提（後被推翻）：①落地後 CPU 後段比 GPU 前段更長，`detect_all_dots` 成為穩態節拍。
- 第 0 步：在 ① 落地、無 py-spy 的乾淨狀態重量 `detect_all_dots` 的 Amdahl 天花板。
- 候選：
  - **(a) 消除迴圈內重複的 `disk()` 結構元素配置**（`_detect_red_dots`/`_detect_black_dots` 的 `disk(seed_dilate)`、`_compute_ring_stats` 的 `disk(ring_gap)`/`disk(ring_gap+ring_width)`），零風險、bit-exact。
  - (b) joblib 換 process 後端（需 `spawn` 驗證 + memmap 重量 + worker pool 跨 tile 重用）。
  - (c) `regionprops_table` 取代逐 blob 物件化。
  - (d) 整塊 tile 向量化（最高上限、最高風險，列 backlog）。
- `elastic_dish_nucleus_matching` 已向量化，無需優化。

### 3.7 detect-all-dots-result — ② 落地結果（2026-07-11，git `00f2c91`）

- **doc 12 前提被推翻**：`detect_all_dots` 在背景執行緒上被 GPU 前段**完全遮住**。
  - medium 121：e2e 184.9 s；detect 38.1 s（20.6%，名目天花板 1.26）；GPU 前段 143.0 s（77.3%）；每 tile detect 0.320 s vs GPU 前段 1.329 s。
  - large 441：e2e 724.7 s；detect 238.6 s（32.9%，名目 1.49）；GPU 前段 586.2 s（80.9%）；每 tile detect 0.579 s vs 1.329 s。
  - 有效 Amdahl 天花板 ≈ **1.0**。
- **量測衛生**：本機 CPU 頻率隨熱狀態變動 ~2×（同碼同輸入 detect 38.1 s boost vs 79.9 s throttle）；跨 stage 比較只在同一次跑內做。
- **(a) 已落地**：`m3_dot_kernels.py` 新增 `@lru_cache` 的 `_disk_footprint(radius)`（唯讀共享 footprint），替換 4 處 `disk(...)`；`r∈{0..8}` 與 `disk(r)` `np.array_equal`；report.csv 差異落在 GPU 雜訊地板（reddot/blackdot 各 2、score 1）；微基準 `disk(3)` 6.6 µs vs cache hit 0.04 µs。**端到端收益 = 0**（被遮住），保留為零風險元件級去重。
- **(b)(c)(d) 不執行**（依 doc 12 §2 停損）。下一槓桿在 GPU 前段（兩次 Cellpose 前向 = 80.9% wall）。
- 附帶修復：`scripts/perf_measure.py` 的 `wrap()` 對缺失符號改為 skip（`segment_masked_dish` 已不存在，M2 也走 `segment_windowed`，故 `B1_m3b_cellpose` bucket 現在合計 M2+M3b 兩次 Cellpose）。

---

## 4. Round 3：Cellpose 換模型之後，重新排序（doc 13–17）

### 4.1 doc 13 — Next optimization plan（post-round-3，2026-07-22）

- **Round 3 事實**：Cellpose 4.0.8 → **4.2.1.1（DINOv3 `cpdino` backbone + bfloat16）**，由同事換模型帶來，large wall **707.4 → 573.7 s（−18.9%）**，累計 −32.3% vs 原始 control；peak VRAM 5159 → 2787 MB。**非 bit-exact**：cells 12,922→13,150（+1.8% large）、3,558→3,647（+2.5% medium），一個 tile success→skipped；屬 model-quality 變更，需臨床驗證（本文件範圍外）。後續 ablation 必須以 round 3 自己的 cell count 為正確性基準。
- **營運須知**：共用多人伺服器，量測前必須 `nvidia-smi` 確認 GPU 空閒；每輪量測資料夾要附 `pip freeze`（round 2 缺這個，導致 ⑨ 無法歸因）。
- **已關閉清單（不要 re-litigate）**：①序列 pipeline（兩段 overlap）、② `detect_all_dots`（被 GPU 前段遮住）、③ PNG encode（被遮住）、process backend（負向）、gc 重定位（負向）、CUDA graph/向量化 Cellpose 迴圈（停損，天花板 1.18–1.23x）、Cellpose swap、`gc.collect` 頻率（以 `gc.freeze()` 落地）。
- **排序方法改變**：pipeline 已是兩條並行臂——**MAIN 臂（主執行緒）**：3 次 GPU 前向、`_read_rgb`、M1 overlay、`clear_slide_edge_cells`、`build_all_positive_results`、`enlarge_cell_instances`、gc+empty_cache；**BG 臂（背景執行緒）**：`detect_all_dots`、merge、PNG/TIFF encode、overlay render、per-cell crops、`filter_and_absolutize`。`wall ≈ max(MAIN, BG) + outside`（outside = precut A + stitch D + model init；兩個規模內驗證到 1.3%）。**位於 slack 臂（BG）的項目，self-time % 無意義，天花板 ≈ 1.00**。
  - 排名表（large/441，錨點 573.7 s）：GPU forwards 444.6 s/77.5% MAIN → ceiling 1.382x；`gc.collect` 36.4 s/6.3% MAIN → 1.083x（後完成）；CPU prep 28.4 s/5.0% MAIN → ~1.05x；precut A + stitch D 25.6 s/4.5% outside → ~1.05x；`detect_all_dots` 292.9 s/51.1% BG → 1.013x；PNG encode 78.7 s/13.7% BG → 1.013x。
  - **關鍵數字**：BG 臂 (387.3 s) 是 MAIN 臂 (538.3 s) 的 **72%**；MAIN 必須削減 34.0%（large）/36.8%（medium）GPU 前向才會讓 BG 臂變成新的關鍵路徑。
  - 「<10% 就放棄」的停損線對 **關鍵臂上、可趨近 0 的項目暫停適用**（全片 ~12.6 h 時 6% ≈ 44 分鐘）。
- **優先序**：
  1. P1 `gc.collect` 頻率 → **DONE**（`gc.freeze()`，1.069x attributable / 1.077x e2e，與預測 1.083x 幾乎吻合；Option A batching 建好、量過、無增益、已刪除）。
  2. P2 把閒置 CPU prep（`enlarge_cell_instances` 19.60 s + `build_all_positive_results` 8.81 s）從 MAIN 臂搬到 BG 臂；投影 wall 573.7 → ~501 s（~1.14x，與 P1 合計）；二階效應：MAIN→473.5 s、BG→415.7 s，margin 從 34% 縮到 ~12.2%。
  3. P3 precut A（~20.5 s/3.6%）與 stitch D（~5.1 s/0.9%）與 B 迴圈重疊（天花板 ~1.05x；全片 35,700 tile 時占比更高）。
  4. P4 把 `cellpose_batch_size` 接進 `Config`（目前 `hybrid_pipeline.py:206/218` 仍是 `getattr(...,16)`）並掃 16→32→64；先做 no-op 驗證；VRAM 餘裕 ~29.8 GB；需對照 post-P1/P2 margin。
  5. P5 隔離 `detect_all_dots` +22.3% 迴歸（239.4 → 292.9 s，僅 +1.8% cells；懷疑 scikit-image 0.25.2→0.24.0、numpy 2.2.6→1.26.4、opencv→4.8.1.78 降版 vs 新 checkpoint 細胞幾何，未隔離；ceiling 1.013x）。
  6. P6 每輪記錄 `pip freeze`（流程修正）。

### 4.2 doc 14 — `gc.collect` 頻率優化設計

- 背景：`gc.collect()` 每 tile 在 MAIN 執行緒跑一次（`hybrid_pipeline.py:798`），36.3 s、6.3% wall、MAIN 臂 → ceiling 1.083x；是 GIL 最大持有者（33.6% 樣本、3.7% wall）；重定位已失敗，**不要再提**；硬約束：memory-bounded invariant（RSS 隨累積細胞數成長，非 tile 數）。
- 選項：
  - **A** 固定 N 批次（N∈{4,8,16}）；
  - **B** `gc.collect()` 與 `empty_cache()` 解耦（`empty_cache` <0.3% wall，保持每 tile）；
  - **C** `gc.freeze()` 於 model init 後（掃描範圍最佳化，stdlib ≥3.7，gunicorn/uwsgi preload 同招；近零風險）；
  - **D** gen0/1 每 tile + 每 N tile full collect（A 的 RSS 驗證失敗時才用）；
  - **E** RSS/細胞數觸發的 adaptive 掃描（不是初始矩陣，僅記錄）；
  - **F** `gc.disable()` + 週期性 full（最高風險、直擊 memory-bounded invariant，最後手段，需延長壓力測試）；
  - **G** 讓 `per_tile_owned` 串流寫出（架構改動，範圍外）。
- RSS 兩項檢查：絕對上限（441 tile）+ 成長形狀（sawtooth 而非單調爬升）。
- 實驗矩陣 Exp 0→1(C alone)→2(A sweep, medium)→3(A confirm, large)→4(A+C)→5(D)→6(F stress)；順序 0→1→2→3→4。

### 4.3 doc 15 — `gc.collect` 實作紀錄

- **最終只上線 Option C**：`_frozen_gc_generation()` context manager（`gc.freeze()`；`finally: gc.unfreeze()`），套在 `run_batch` 的 tile loop `with` 上。`gc.collect()`/`empty_cache()` 仍每 tile 一次，無 config 欄位、無 CLI flag。
- 兩個計畫假設錯誤：
  1. **單獨 `gc.freeze()` 會在 API server 洩漏**：`backend/api/hybrid.py:39` 在長壽 FastAPI 行程的 background job 內呼叫 `run_batch`；未配對的 freeze 會把當下追蹤的垃圾永久釘住，RSS 隨「請求數」無界成長（單批 benchmark 看不出來）。故 `finally: gc.unfreeze()`（已查證專案內無其他 `gc.freeze()` 呼叫；已知 caveat：兩個並行 `run_batch` 會讓先結束者全域 unfreeze，只損失最佳化，不影響正確性；且並行本就因單一 CUDA context 而不安全）。
  2. 凍結點放在 `with`（比「`_init_*` 之後」晚 ~20 行），連 `stats`/`per_tile_owned`/`_collect` closure 一起凍，更有效。
- 檔案：`hybrid_pipeline.py`（+32/−1）；`config*.py` 不變（config hash `db2b7e6a`）；新增 `scripts/verify_gc_freeze.py`（驗證 cadence 不變、freeze/unfreeze 成對含 fail-fast 路徑，無需 GPU，約 1 分鐘）與 `scripts/gc_ablation_report.py`。

### 4.4 doc 16 — `gc.collect` 結果

- **Large 441 tile（n=3）**：基準 571.5 s → **530.4 s（−41.1 s，1.077x）**；attributable 109.7 → 72.7 s（**−37.0 ± 1.5 s，1.069x**）；medium：166.4 → 157.3 s（1.058x）。`gc.collect` 本身 36.71 → 0.52 s（83.2 → 1.2 ms/call，69x），呼叫次數不變。
- **噪音地板教訓**：medium 兩次相同跑差 0.2 s，但 large 的 Cellpose `B1` 本身 ±3%（434.4–446.7 s），故看 **residual（e2e − B1）** 才是乾淨訊號：109.0–110.5 → 71.9–73.4 s。早期稿曾宣稱 1.118x「超越天花板」是單次運氣（red flag #3），被兩次重複推翻——**沒有二階效應**。
- **成本驅動因子是 per-call 掃描量，不是呼叫次數**：freeze 關閉時 81.9/82.0/82.0/83.0 ms（N=1/4/8/16）；Option A 瞄準錯誤變數，疊在 C 上無增益（RSS 3.991 GB 反為最高）→ 刪除。
- 正確性：13148 vs 13149 cells，13139 配對，reddot/blackdot/score max|Δ| = 0；excluded-flag flip 18 vs 地板 15。記憶體：RSS 3.881 → 3.925 GB（+1.1%），sawtooth 完好。
- 基準漂移：Round 3 的 707.4/208.2 s 無法重現（今日基準 571.5/166.4 s）；`9e618d3`（disk footprint cache）、`5ee788c`（移除舊碼）之後 round-3 `report.csv` 已不是有效 reference（差 89 cells、462 excluded flips）。
- **⚠️ Round 8 更新**：整片規模 `gc.collect` 回到 **80.5 ms/call、2,218.4 s、16.1% wall**——`gc.freeze()` 只豁免「凍結時已存在」的物件，而 `per_tile_owned`（整片累積 356,255 個 `CellAnalysisResult` dataclass）在凍結之後建立、被完整追蹤、每次收集都被重掃。→ 由 round 9 週期性 re-freeze 解決（見 §10）。
- Follow-ups：large anchor 需 n≥3；Cellpose `B1` 占 large `run_batch` 84%（437/518 s）；round-3 參考過期；`B4_gc_collect` 只計明確呼叫，不含自動 gen 收集。

### 4.5 doc 17 — GPU starvation 前置條件（為何大槓桿要等小槓桿）

- **觸發觀察**：GPU 使用仍斷續。Round 3 後 GPU **idle_frac 反而上升**（large：overlap 輪 0.190 → round 3 **0.370**；mean SM 32.9 → 16.6%），因為 Cellpose 前向變快但 CPU 縫隙沒縮短 → 更 bursty。
- **確認 HEAD 未落地**：⑧ 仍在 MAIN 臂（`_process_one_chunk_gpu` 第 597/601 行）；`cellpose_batch_size` 仍未接線（207/219 行）；`gc.freeze()` 已上線；無多行程；無 GPU 端 transform/decode 路徑（`_read_rgb` 讀磁碟 PNG）。
- **新發現：tile 內部 GPU bubbles**：單 tile 的 GPU 前段不是連續 GPU-busy，而是 3 次前向之間夾著 CPU-only 段（UNet → M1 overlay glue → Cellpose #1 → `clear_slide_edge_cells` → `build_all_positive_results` → `enlarge_cell_instances` → Cellpose #2）。背景執行緒只處理「前一個 tile」的後段，填不了這些洞。⑧ 只是「臂平衡」修復，不是「bubble 消除」。真正消除需 (a) 第二個背景執行緒預取下一 tile 的讀取+glue，或 (b) pipeline depth ≥2 —— 皆需先以 `torch.cuda.Event` 量化。
- **為何三個大槓桿（跨 tile 多行程 / 加大 Cellpose batch / GPU 端 transform）要等**：
  - 多行程：fork-under-CUDA、單一 CUDA context、需 per-process 模型重載（VRAM×N）或 MPS；
  - batch：需對照重新量測後的 margin；batch 對 intra-tile bubbles 沒用，甚至更糟；
  - GPU 端 transform：`B2r_tile_read` 只佔 1.16% wall（medium），ceiling ~1.01x，且無現成路徑可「搬」。
- **優先序**：(1) 重量 MAIN/BG margin（`gc.freeze()` 後）→ (2) 落地 doc 13 P2（⑧）→ (3) 量化並視情況原型化 bubble overlap → (4) doc 13 P3（precut+stitch overlap）→ (5) 最後才用殘餘 idle_frac 衡量多行程/batch/GPU transform。
- **第一次 margin 重量（medium，暫定）**：MAIN 146.2 s（GPU 前向 130.35 s/82.7%；`_read_rgb` 1.83 s；M1 glue 3.88 s；`clear_slide_edge_cells` 1.62 s；⑧ 8.01 s；empty_cache+gc 0.50 s），BG 105.2 s（detect+merge+filter 77.49 s；PNG/TIFF/render/crop 27.72 s），outside 9.97 s；模型預測 156.2 s vs 實測 157.7 s（1%）。**BG/MAIN = 0.719 → MAIN 需削減 28.0%**（原 36.8%）。idle_frac 0.227、mean SM 22.75%。該 run 缺 `nvidia-smi`/`pip freeze`，不計入正式紀錄。

---

## 5. Round 4：GPU starvation 前置條件落地（doc 18）與開放待辦總表（doc 19）

### 5.1 doc 18 — round 4 實作與量測紀錄

- **協議**：GPU 空閒確認（89 MiB / 0%）、`--gpu-dmon --workers 8`、`pip freeze` + env stamp（含 checkpoint SHA-256，與 round 3 相同）、large n=3、medium n=2；基準價差 large 1.0%、medium 1.6%。配置標籤：`p0`（HEAD `2f89fea`，`gc.freeze()` 已在）、`p2`（⑧ 移出 MAIN）、`p3`（precut 串流化）、`p0ev/p2ev`（`--cuda-events`）、`p4_bs{16,32,64}`。
- **Item 1 — margin 重量**（`scripts/arm_report.py` 新增；先用它重現 round 3 紀錄：MAIN 538.3 / BG 387.3 / outside 28.0 / 預測 566.3 vs 實測 573.7，BG/MAIN 0.719）：
  - `p0`：large 538.5 s（536.4–541.9）、MAIN 503.4、BG 374.6、outside 28.0、預測偏差 −1.3%、**BG/MAIN 0.744 → MAIN 需削減 25.6%**（medium 154.0 s、0.736、26.4%）。bottleneck-list 上印的 34.0%/36.8% 已失效。獨立確認 doc 16：573.7 − 36.4 ≈ 537 s vs 實測 538.5 s。
  - 正確性參考：large 13152–13153 cells / 378 success / 63 skipped；medium 3647–3649 / 103 / 18。
- **Item 2 — ⑧ 移出 MAIN 臂（已採用）**：`build_all_positive_results` + `enlarge_cell_instances` 由 `_process_one_chunk_gpu` 移到 `_finish_chunk_cpu`（純函式、無共享可變狀態；`matching_mask`/`results_pre` 從 `_ChunkGpuState` 刪除）。
  - large 495.5 s（−8.0%，1.087x）、medium 146.3 s（−5.0%，1.053x）；MAIN 503.4 → 458.5 s、BG 374.6 → 382.6 s；margin 25.6% → 16.6%。doc 13 預測 ~501 s，實測 495.5 s。
  - **贏過自己的 arm model 預測**：模型預測 MAIN −28.4 s，實測 −44.9 s，因為兩個未改動 bucket 也變快（B1 GPU 前向 444.4 → 431.6 s；`detect_all_dots` 279.9 → 257.6 s）——GIL 競爭（與 gil-contention-diag 相反方向）：以前 ⑧ 與 `detect_all_dots` 跨執行緒搶 GIL，現在序列化在同一執行緒。**arm model 把兩臂視為獨立，故對「位置變更」只是下界估計**。
  - 正確性：per-cell veto；blackdot Δ=8 出現在**同一份 p2 程式碼兩次跑之間**，跨配置差距不超過配置內差距。VRAM（`dmon fb`）峰值 2787 MB 於六次 large 完全相同。
- **Item 3 — 以 `torch.cuda.Event` 量化 intra-tile bubble**（`--cuda-events`）：
  - 基準（medium 157.1 s）：M2→M3b gap 10.06 s（6.40%）、UNet→M2 gap 4.53 s（2.88%）、tile 邊界 3.07 s（1.96%），**合計 17.66 s（11.24%）**；Event 與 Python timer 吻合到 0.01 s（CPU glue 100% GPU-idle）；但這僅是總 device idle（~40 s）的 ~44%，其餘 ~56% 是 forward 內部 kernel-launch-bound（已停損）。
  - item 2 後：M2→M3b gap 10.06 → 1.66 s（只剩 `clear_slide_edge_cells`），總 17.66 → 8.69 s（−8.97 s）。
  - **決定不做 CUDA-stream/depth-2 redesign**：剩餘 8.69 s/6.07% wall（≈1.065x，即使歸零），分散在三個 segment，風險大。記錄為「sized and stopped-out at ~1.065x」。
- **Item 4 — precut A 與 stitch D**：
  - **Stitch D 不可重疊**：pyvips join 是 lazy，成本全在單次 `tiffsave` C 呼叫；且 D 在 `run_batch` 末尾無可重疊對象；pyramid TIFF 無法逐列增量編碼 → overlappable ≈ 0 s，5.10 s/1.0% 不值得。結構性負向，關閉。
  - **Precut A 串流化（已採用）**：tile grid 可由 header 推得（`m0_reader.read_size` 不解碼像素，`chunk_offsets` 純算術）。新增 `PrecutStream`：立即交出 `positions`，每對 tile 落地就 yield（同樣 8 執行緒 pool、`_crop_to_tile` + deflate）；處理順序可變因為 `run_batch` 會依 `(abs_y, abs_x, cell_id)` 排序再全域重編號、`_stitch_overlay_slide` 依座標讀 tile。242 個 medium tile 檔 sha256 逐位元相同。
  - 結果：`phaseA_precut_s` → 0.004 s；`outside` 27.9 → 7.64 s；large 480.3 s（−3.1% vs p2）、medium 140.8 s（−3.8%）；precut 回收 75%（large）/ ~98%（medium）；large spread 0.25%。
- **整合結果**：round-3 573.7 → p0 538.5（−6.1%）→ p2 495.5（−8.0%）→ **p3 480.3 s（−3.1%）**；累計本輪 −10.8%（1.121x），相對 round 3 含 `gc.freeze()` −16.3%。medium 154.0 → 140.8 s。全片線性外推 `wall ≈ 12.5 s + 1.0608 s/tile` → **~10.5 h**（35,700 tile；round 3 時 ~12.6 h）。p3 arm 狀態：MAIN 453.9、BG 381.7、outside 7.6 → BG/MAIN 0.841，MAIN 只剩 15.9% 可削減；單 process 下限 `BG+outside = 389.3 s`（1.23x）。
- **Item 5 — 三個大槓桿對照殘餘**：
  - **6.1 `cellpose_batch_size`：已接線但掃描為負向**。`Config.cellpose_batch_size: int = 16` 新增（config hash `db2b7e6a` → `ad41c42f`）；step 1 驗證 bit-exact no-op（142.5 vs 140.8 s、3648 cells）；掃描 16/32/64 → 142.5/144.7/144.0 s，每次 Cellpose 呼叫 ~518 ms 不變，VRAM 2787 MB 不變。**原因**：cpdino backbone 的 `bsize=384`，1024² tile 切成 4×4 = **16 個 patch**，恰好等於現有 batch size（1536 px → 25；2048 px → 49）；硬編碼 16 剛好是正確值。只有 `default_tile_size` ≥1536 才會有效。原本預測（25–36 patch）有誤，未先用 `cellpose.transforms.make_tiles` 驗證。
  - **6.2 跨 tile 多行程**：當時唯一尚有天花板的大槓桿；VRAM 2.79 GB × N、init 2.4 s × N；不受 389.3 s BG 下限限制（每個 process 各有 BG 執行緒）；合理天花板 1.23x–~1.7x，風險最大。
  - **6.3 GPU 端 tile 載入**：`B2r_tile_read` 5.87 s = 1.22%（ceiling 1.012x）；停損。**⚠️ round 8 更新：整片量到 17.2%（precut scratch ~49 GB 不再放得進 page cache）→ round 11 又證實 17.2% 被污染，真實 4.21%**。
- **方法論更正**：
  1. `idle_frac`（SM==0 的 1 Hz 樣本比例）是 knife-edge 指標，p0→p2 假性上升 0.32→0.43；改用 `SM<=3`（0.50→0.45）或 cuda-Event gap。
  2. `peak_cuda_reserved_gb` 不可靠（p2_large_r2 報 25.967 GB 而 `dmon fb` 為 2787 MB）；VRAM 讀 `dmon fb`。
  3. Arm 歸屬是「程式碼版本」的屬性而非 bucket 名稱（⑧ 移到 BG 後靜態 bucket→arm map 會誤判，誤差從 −1.5% 變 +4.2%）；`arm_report.py` 需顯式 `--moved/--moved-labels`。
- **改動檔案**：`hybrid_pipeline.py`（⑧ 搬移、`run_batch(..., tile_stream=None)`、`_run_single_tile_cli` 用 `PrecutStream`）、`m0_reader.py`（`PrecutStream`）、`backend/api/hybrid.py`（`/api/hybrid/tile` 用 `PrecutStream`）、`config*.py`（`cellpose_batch_size`）、`scripts/perf_measure.py`（`--cuda-events`、`--stream-precut`、`--cellpose-batch-size`）、`scripts/arm_report.py`（新）。
- **Follow-ups**：`perf_measure.py` 記錄每 bucket 的執行緒名稱；bottleneck-list 的 ①/⑧ 與 34.0%/36.8% 已過期；雙臂模型需第三項（precut 切片執行緒），否則 p3 後低估 2–4%；⑨ 部分自癒（279.9 → 257.6 s）；`clear_slide_edge_cells` 成為 M2/M3b 間唯一剩餘 CPU 工作（1.66 s）。

### 5.2 doc 19 — Open backlog（最後更新 2026-08-01，round 15）

> 一份「還欠什麼」的單一總表，取代需要把 01–46 全讀一遍。頂部提醒：**round 8–14 的整片數字量在錯誤畫布上**（見 §0.2）。

**效能待辦／狀態表**（編號沿用原檔）：

| # | 項目 | 狀態 | 天花板/結論 |
| --- | --- | --- | --- |
| 0 | `detect_all_dots` joblib fan-out 移除（`dot_detect_n_jobs=1`） | ✅ 已採用（round 6） | `workers=1` **1.60x**（large 484.7→302.7 s）；`workers=6` 約 0%；原本的 threads 後端在每個 process 數下都比序列慢（20 threads 慢 2.77x），多餘執行緒偷 GIL，移除後 GPU 前向反而快 43.4%；同時坐實 +192.9 s B1 異常 |
| 1 | 跨 tile 多行程 | ✅ 已建置、量測、採用（round 5）；**round 6 把建議 worker 數從 6 下修為 4（無人看管）/5**；round 8 放行 gate 關閉；round 11 allocator 修復；round 12 發現實際 1.745x；round 15 正確畫布 2.05x | `workers=3` 3.09x、`workers=4` 3.51x（large crop）；預估 1.23–1.7x 偏低因漏算 GIL 競爭；MPS（Candidate C）、加深 CPU 後端（Candidate A）停損；`workers≥5` 因 VRAM 餘裕（92.2% of card）仍不開放；**round 15 發現 serial fraction 非常數（576 tile 36.8%、4,096 tile 17.1%）** |
| 1b | Phase D 縫合（`_stitch_overlay_slide`） | ✅ **round 13 以 `tifffile` 管線化上線為預設，關閉為正向** | 歷程：doc 24 估 ~3 分鐘 → round 7 實測 322.7 s（超線性：14.11/14.16/19.90 s/GP @ 1/4/16.22 GP）→ round 8 整片 1,185 s（`workers=1` 8.6% wall、`workers=4` 19.3%）；`tiffsave` knob 消融全負向（tile 256/512/1024 = 0.948x/0.860x/0.686x；deflate 0.785x；`zstd` 1.2331x 且小 13.8% 但 **QuPath/BioFormats 無法開啟 → 因正確性被 veto**）；round 9 spike：nvImageCodec 因不支援 tiling 判死、`tifffile` 接受預壓縮 tile bytes，shape-matched 1.522x / +GPU pyramid 2.246x；意外修好既有缺陷（pyramid 層 predictor tag 沒標，縮圖 ~90% 黑，`predictor="none"` +9.4% 檔案大小、時間中性，**round 9 前的 slide 需重縫**）；round 10 補測讀檔後反轉（candidate B 0.788x / C 0.884x，**Phase 2 關閉負向**）；round 12 重開（`workers=4` Phase D 1,889.8 s = 32.3% wall）；pipelined read 1.365x/1.581x；round 13 上線：Phase D 1,889.8 → 987.8 s（**1.913x**）、e2e 5,854.9 → 4,877.4 s（**1.200x**，投影 1.135x、完美重疊下限 1.184x）；四道 veto 全過（CSV 356,225 vs 356,220 +0.0014%；pyramid audit PASS，`Predictor=2` 12 IFD 一致；各層與上層 2×2 box shrink bit-exact）；peak RSS 45.6 → 17.0 GB；檔案 7.50 → 5.85 GB；Candidate C 建議關閉。尚欠：整片 error-bar（n=1 vs n=1）、`overlay_pyramid_audit.py` 只測亮度（抓不到 tile 擺放錯誤）。round 15：在正確畫布 Phase D 佔 `workers=4` wall **24.1%**；page-cache 懸崖 11.26 vs 60.90 s/GP（5.41x） |
| 1c | `run_batch` 斷點續跑 | ✅ 已建置（round 8，opt-in） | `run_batch(checkpoint=True)` 將每個完成 tile 的 `owned` pickle 到 `output_dir/_resume/`（tmp+rename、config_hash 守衛）；fail-fast 不變；`--resume`；cold vs resumed `report.csv`/`summary.txt` byte-identical |
| 2 | `clear_slide_edge_cells` | Watch | ~1.2% wall；M2→M3b 間隙唯一剩餘 CPU 工作 |
| 3 | 隔離 `detect_all_dots` +22.3% 迴歸（⑨） | 大致被 round 6 超越 | 1.013x，沒有 wall 回報 |
| 4 | CUDA-stream / depth-2 bubble redesign | 關閉 | ≤1.065x |
| 5 | CUDA graph / 向量化 Cellpose 內部迴圈 | 停損 backlog（round 6 用 4.2.1.1 重新追：`get_rel_pos` 消失、其餘四個行號位移） | ~1.118x（原 1.23x）；MAIN 臂最大 Python leaf 改為 `_from_device` 24.2% 與 `_quantile` 10.1%（皆不在清單內） |
| 6 | GPU 端 tile 載入（`B2r_tile_read`） | ✅ 關閉（round 11） | Option L 兩個路徑都上線；真實成本 449.3 s = **4.21%** wall（非 17.2%，被 `d6592c3` 移除 275 GB/slide 寫入污染 5.3 倍），Option L 吃掉 99.8%；Option K 上界 1.208x 結算為 **1.044x** |
| 6b | `gc.collect` 整片規模 | ✅ 關閉（round 9，round 10+11 確認） | Option H 週期性 re-freeze：整片 2,218.4 → 58.8 s（−97.3%）、round 11 獨立 19.5 s；doc 31 §7 預測 ~1.19x，實測 1.186x；`gc.freeze()` 為 O(1)（0.00004–0.0006 ms/call），Option I 不需做；cadence 以累積細胞數（5,000）計；peak RSS 反而下降（70 次 freeze）；`workers>1` 未受影響 |
| 7 | 完整 WSI 規模驗證 | ✅ 完成（round 8）；round 15 在正確畫布重做 | round 8（舊畫布）：`workers=1` 3.82 h、`workers=4` 1.73 h（推估 2.6/1.25 h → +47%/+38%）、2.216x、report.csv −0.01%。sub-item：`_stitch_overlay_slide` 同時開 27,565 個 tile 無 `RLIMIT_NOFILE` 檢查 → `_ensure_nofile_limit()`；registration 對各 modality 輸出不同畫布（HER2 141818×114366、DISH 141658×114415、HE 141717×116400）使 `PrecutStream` fail-fast → 當時用 `--conform`（99.86% 保留），**後被 doc 44 證實錯誤並移除** |
| 7b | allocator 氣球（單 worker 吃 24.76 GiB 導致 OOM） | ✅ 根因與緩解（round 11）、預設已翻開（`b3fa47d`） | `expandable_segments:True` 在 `workers=4` 12 次交錯：對照 **4/12 OOM（33%）**、開啟 **0/12**（Fisher p=0.047），代價 +0.67% wall；`workers=4` 餘裕變為 92.2% of card；`workers≥5` 仍不開放；round 7 在 `workers=4` 觀察到 dmon 樣本 26,687/32,607 MB |
| 8 | API/Job 層並發行為（Phase E） | 從未量測 | `workers>1` 對 API 路徑需要常駐 worker pool，會重開 `gc.freeze()` per-call 設計 |
| 9 | 每輪記錄 `pip freeze` | 部分採用 | round 3、4 開始；round 2 缺 |
| 10 | fail-fast / 完成後行程卡住 | ✅ `workers=4` fail-fast 修好（round 11）；`workers=1` 2 小時卡住未重現 | `scripts/exit_latency_probe.py`（新）；`_kill_all()` 後無人 drain `task_q` → feeder thread 在 `send_bytes` 阻塞 → `multiprocessing` atexit join 永遠等；`task_q.cancel_join_thread()`：never → 0.32 s；第二缺陷：`PrecutStream` 在批次中止後仍切完整張玻片（576/576 tile，實際在 ~8 tile 中止）→ 有界 in-flight 視窗（`workers×8`），`tests/test_precut_stream_bounded.py` |
| 11 | `B1_m3b_cellpose` +751 s 迴歸 | 🟡 大致解釋；殘餘 530 s | round 11 排除 `e806938`/`d6592c3`；真正原因是 Option L prefetch 讀檔執行緒與主執行緒搶 GIL 的記帳膨脹（`B1_unet_coremask` +16.4%、`B2r_tile_read` +26.4%，總 wall 反降 1.21%）；量到的競爭係數 0.513 s/s 跨 576 tile → 整片偏差 4% 以內，解釋 221.2 s（29.5%）；殘餘 530 s 記為 n=1 對 n=1 的整片跑批間漂移 |
| 12 | 批次領取動態 tile 佇列（batch-claiming） | ✅ 關閉負向（round 12） | `scripts/mp_queue_claim_probe.py`：派送全部 27,565 tile 零工作僅 0.356 s = 0.006% wall → e2e 天花板 **1.00006x**；batch 越大工作者不平衡越嚴重（tile spread 160–493 → 1,011–1,344） |
| 13 | `workers=4` 在 crop 規模 10 次 OOM 1 次（即使有 `expandable_segments:True`） | 🟡 新（round 14），未進一步調查 | 三個同伴共 30.15 GiB，一個 worker 膨脹到 11.55 GiB（peers 9.30 GiB）；VRAM 餘裕 ~2.5 GB 是四者共用 |

**已關閉、不要重提**：跨 tile Cellpose 批次（round 6：G=2 ≤5.7%、G=4/8 負向、VRAM 1.17 → 8.01 GB；G=16 +6.6%/+5.9% 慢、15.8 GB = 48.6% 卡）；跨 tile UNet++ 批次（更差；`predict_batch` 其實是序列 `for`）；`detect_all_dots` process backend；gc 重定位；固定 N gc 批次；`cellpose_batch_size` 掃描（1024 px 下平坦）；**CUDA MPS**（在 RTX 5090 上實測有效、launch-bound 微基準 +44%，但端到端平坦）；加深 CPU 後段 pipeline（Candidate A，+2.8% 慢）。
**Round 7 關閉**：`mkdir` hoist（Candidate G，0.056% wall，+0.63%/+0.80% 雜訊內，已 revert；若輸出改到網路檔案系統需重新量）；背景 tile 空白檔寫入（Candidate F，24 ms/tile、7.5% wall、~6 分鐘全片，但在 BG 臂 47–53% 餘裕內 → wall 0；`os.link` 便宜 272–407x，真正理由是儲存：~157 GB/slide 的 10.2 MB 常數 TIFF）；`detect_all_dots`/`enlarge_cell_instances`/debug encode → GPU（Candidates B/D/E，BG 臂 → 1.00x）；G=16 跨 tile Cellpose 批次。

**正確性／臨床（阻擋性，非效能任務）**：**Round-3 Cellpose checkpoint 重訓需病理醫師/臨床簽核——至 round 7 仍未解**。細胞數 +1.8–2.5%、一個 tile success→skipped，屬 segmentation-quality 變更，round 3 之後所有效能收益（包含多行程）都建立在這個尚未驗證的模型上。

**文件↔code 落差（尚未修）**：`generate_ihc_core_mask` 參數名（round 8 已修）、失聯 spec docs（round 8 已修）、explainer HTML（round 8 已更新為 v4）；**仍開著**：無自動測試守護 `config.py`/`config_example.py` 同步；hybrid 目錄仍無通用 pipeline 正確性測試（`m0_stitch` 是最佳合成 numpy 測試候選）；codegraph 索引重新核對（round 8 已完成）。

---

## 6. Round 5：跨 tile 多行程（`workers`）——規劃（doc 20）與落地（doc 21）

### 6.1 doc 20 — 設計空間與實驗計畫（純規劃）

- **背景**：doc 19 判定它是當時唯一還有可量測天花板（~1.23x–1.7x）的槓桿，也是正確性風險最高者。單 process 的所有槓桿都受 `BG + outside = 389.3 s`（1.23x）下限約束；多行程每個 process 帶自己的 BG 執行緒，兩臂都平行化；當時 mean SM 僅 ~20%。
- **機器現況（2026-07-23 實查）**：RTX 5090 32607 MiB、Compute Mode `Default`、CUDA MPS 未啟動、torch 2.11.0+cu130、driver 580.159.03。
- **7 條不可協商的正確性不變量**：
  1. 全域細胞 ID 重編號保持**單一確定性後處理**（依 `(abs_y, abs_x, cell_id)` 排序後在 parent 中央重編號 1..N，不得在 worker 內重編）；
  2. fail-fast 是**整批**語意（worker 例外必須傳回 parent，parent 先終止所有兄弟 worker 再回傳）；
  3. 每個 tile **剛好一個 worker** 處理（動態佇列保證）；
  4. `gc.freeze()` 語意不得在常駐 pool 外洩（需 per-call freeze/unfreeze）；
  5. VRAM/RSS 有界不變量在 N-process 規模需重驗（`2.79 GB × N` 加每 process CUDA context 開銷）；
  6. 輸出必須通過 per-cell 正確性 veto（參考 round 4：large 13152–13153 cells/378 success/63 skipped；medium 3647–3649/103/18；以同碼雜訊地板為準）；
  7. 小/API server 請求不得回歸（`/api/hybrid/tile` 不傳 `workers`）。
- **候選架構**：
  - **A**：只把 CPU 後段（`_finish_chunk_cpu`）改為 `ProcessPoolExecutor`，GPU 維持單 process（不碰 CUDA、風險低；注意**與已關閉的 `detect_all_dots` joblib process backend 不同**：粒度是跨 tile 而非 per-cell；單獨效益可能小因為 BG/MAIN=0.841）；
  - **B**：N 個 `spawn` GPU worker，各自重載三個模型，從共享佇列領 tile、跑完整既有 per-tile pipeline（靜態 round-robin vs **動態 work queue（推薦）**）；最大未知是無 MPS 時 GPU 硬體排程器只做時間切片，不真正並行；
  - **C**：B + CUDA MPS（RTX 5090 為消費卡、不在 NVIDIA 官方支援矩陣，需實測；與已關閉的 CUDA-stream/depth-2 不同：那是單 process 多 stream，只觸及 inter-forward bubble）；
  - **D**：「B done right」——小 N（2–3）、逐字重用現有 `_process_precut_tile_gpu`/`_cpu`/`_frozen_gc_generation`，只新增外層 pool、work-stealing queue、結果收集、fail-fast（**建議首個建造目標**）；
  - **E**：`os.fork()` 於模型載入後（CLAUDE.md 已判不安全，記錄以關門）。
- **實驗順序**：(1) 玩具 CUDA 多 context 並行探測（零 pipeline code，決定整條線值不值得）→ (2) 找並行 knee → (3) Candidate D 原型（先小 crop：per-cell veto + fail-fast 注入）→ (4) medium/large 錨點（把雙臂模型擴為 N-arm）→ (5) 視 serialization-limited 與否決定 MPS → (6) A 可獨立平行原型。
- **停損**：step 1 負向→全線停、退回 A；N≥2–3 放不進 32 GB→下修；任何 correctness veto 失敗即停；**real-WSI 驗證未完成前不得放行生產**。
- **開放問題**：`spawn` vs `forkserver`、模型物件可否在 spawned child 乾淨匯入、每 process CUDA context 記憶體、與 `detect_all_dots` 內部 joblib 的嵌套（需 `n_jobs=1`）、常駐 pool 的 `gc.freeze()` 契約。

### 6.2 doc 21 — round 5 實作與量測紀錄

- **環境/協議**：git `9f02be1` + 變更；RTX 5090 / driver 580.159.03 / Compute Mode Default；config hash `ad41c42f` 不變；新增 25 tile 的 `small` crop；每次啟動前確認 GPU 空閒（`memory.used < 200 MiB` 迴圈等待）；n=2，推薦設定 large n=3。**損失的量測能力**：`perf_measure.py` 的 monkeypatch 只在 parent，worker 內的 bucket 全空，雙臂模型不適用（改看 e2e wall + `dmon`）。
- **Step 1 gate — 獨立 CUDA context 在這張卡上是否真重疊**（`scripts/mp_concurrency_probe.py`，barrier 同步、`spawn`，`speedup(N)=N·T(1)/T(N)`）：
  - 合成控制：`sm`（SM 飽和）N=2/4 為 0.94（context-switch 開銷）；`launch`（launch-bound）N=2/3/4/6 全為 **1.45x**，吞吐 236.65 vs 163.22 units/s（四位有效數字完全相同 → driver/context 排程器硬上限，SM 仍只 55–64%）；**1.45x 是無 MPS 時多行程能從 launch-bound 部分回收的上限**。
  - 真模型（`models` workload，三模型/process、跑原樣 `_process_precut_tile_gpu`）：N=1/2/3/4 → speedup 1.00/1.58/1.92/2.19，效率 100/79/64/55%，mean SM 35.2/45.7/73.6/74.0。**Step 1 決定性正向**。
  - 每 process 成本（N=1 才可信，`mem_get_info` 是 device-wide）：CUDA context 88.1 MB、三模型權重 828.4 MB、總計 916.5 MB、init 3.14 s；forward 時峰值 ~2787 MB（`dmon fb`）。
- **Step 2 knee**：launch-bound 合成 N=2；真 GPU 前段到 N=4 仍升；整條 pipeline knee 在 N=3（效率 93%，邊際 +0.70 → +0.27）。
- **Step 3 Candidate D 最小規模**：`run_batch(..., workers=1)` 為預設且與原路徑相同（單 process 路徑 diff 僅三行被搬移的 `_init_*`）；`_mp_tile_worker`（spawn、各自 init 三模型、逐行轉寫 run_batch 的 depth-1 GPU/CPU 重疊迴圈）、`_run_tiles_multiprocess`（動態佇列 + feeder thread，使串流 `PrecutStream` 仍與分析重疊）、`_finish_batch`（全域 merge/renumber/export/stitch 尾段逐字抽出，兩路徑共用）。
  - Correctness（small，w1 為 reference）：w1 37.0 s/1015 cells；w2 19.8 s、w3 16.5 s，cells 1015，**所有 per-cell 欄位 delta 皆為 0**，24 success/1 skipped。
  - **fail-fast 注入**（`scripts/verify_mp_failfast.py`，破壞 grid 中間的 tile）：`workers=3` 與 `workers=1` 皆 raise `RuntimeError`、提早中止（9/25、12/25）、無 worker 殘留（首版誤報因為把 `resource_tracker`/loky semaphore tracker 當成 survivor；改為只比對 `spawn_main` 子行程）。
  - **寫 fail-fast 測試時找到兩個真缺陷並修好**：(1) feeder thread 在批次中止後仍繼續切整片 WSI（加 `stop_feeding` event）；(2) 所有 worker 乾淨退出但結果缺漏時會卡住且看似成功（只在非零 exit code 才 raise）→ 現在 worker 全數退出且收集數 < total 即 fail-fast。
- **Step 4 medium / large 結果**：

| workers | medium wall / speedup / eff | large wall / speedup / eff | large FB peak | large RSS |
| --- | --- | --- | --- | --- |
| 1 | 138.1 s / 1.00 / 100% | 482.8 s / 1.00 / 100% | 2787 MB | 4.04 GB |
| 2 | 65.8 s / 2.10 / 105% | 208.9 s / 2.31 / 116% | 6233 MB | 6.52 GB |
| 3 | 49.3 s / 2.80 / 93% | **156.1 s / 3.09 / 103%** | 12354 MB | 9.26 GB |
| 4 | 45.0 s / 3.07 / 77% | **137.4 s / 3.51 / 88%** | 20667 MB | 11.97 GB |

  - tile outcome 在所有 worker 數相同（large 378/63、medium 103/18）；`w1` 對照 482.8 s vs doc 18 的 480.3 s（+0.5%）。
  - **結果遠高於 doc 20 預測（1.23–1.7x）**：原估只算裝置閒置，漏掉**兩臂間 GIL 競爭**（獨立 process 無共享 GIL）與 20 核真平行 CPU 工作（`detect_all_dots`、PNG、per-cell crops、`_read_rgb`、M1 glue）。`workers=2` 的超線性效率（116%）是 GIL 回收的簽名；near-idle（SM≤3）0.46 → 0.06。
  - Correctness（large，vs `w1_large_r1`）：`reddot 2 / blackdot 8 / score 4` 為 doc 18 §2 記錄的同碼 GPU 非決定性簽名；X-flips 3–18 < 同碼 21；僅 1–2/~13,140 cells 不同。
  - 記憶體：VRAM 2787 → 6233 → 12354 → 20667 MB（**超線性**，per-process 2787/3117/4118/5167 MB，疑 allocator）；RSS 近線性，sawtooth 完好（85–131 次 >50 MB 下降、ramp fraction 0.48–0.57 vs 對照 0.54）。
  - 全片線性外推（35,700 tile）：`workers=1` 10.68 h（round 4 的 10.52 h ✔）、`workers=2` 4.44 h（2.41x）、`workers=3` 3.31 h（3.23x）、`workers=4` 2.87 h（3.73x）；上限性質（~85% 組織密度）。
- **§4.7 round 5b 找極限**（medium 與 large 掃 N）：
  - medium：N=5 3/3、N=6 3/3、N=7 2/3（首次 OOM）、N=8 1/1、N=9 0/1、N=10 2/3、N=11 0/3、N=12/14/16/20 皆在**載入模型時** OOM。
  - large：N=5 3/3（~127–130 s）、N=6 3/3（~123.3 s）、N=7 1/1（121.2 s）、N=8 OOM。
  - 發現：(1) 單次成功不代表安全（`workers=7` 之後有 worker 吃 7.7/7.7/9.4 GB，3–4x 於 2.8 GB 穩態；`workers=10` 成敗交錯）——allocator 行為，PyTorch OOM 訊息自己點名 `expandable_segments:True`；(2) 風險隨 tile 總數上升（全片 ≈ 80x large crop）；(3) **`workers=12` 是確定性的牆**（`2.79 GB × 12 ≈ 33.5 GB > 32.6 GB`，載入模型即失敗）。
  - 當時建議：`workers=6`（large 123.3 s，比 `workers=4` 快 10.3%，0/6 失敗；`workers=7` ~25% 失敗率）；但完整無人看管 WSI 建議 `workers=4`（除非 `expandable_segments:True` 驗證有效或 `run_batch` 有斷點續跑）。**此建議在 round 6 被下修為 `workers=4`/5**。
- **Step 5 Candidate C（MPS）：在這張卡上實測可用，但端到端平坦 → 停損**：launch-bound 控制 N=2 1.45x → **1.94x**（97% eff）、N=4 2.09x（吞吐 236.65 → 316.77 units/s）；medium 真 pipeline `workers=2/3/4`：65.8/49.3/45.0 s vs MPS 64.9/49.4/44.4 s（−1.3%/+0.2%/−1.4%，皆在雜訊內）。原因：真 pipeline 到 knee 時已不是 serialization-limited（mean SM 58–78%、near-idle 0.06–0.16）。playbook anti-pattern #5（微基準勝利 ≠ 端到端勝利）。避免新增「必須運行的 daemon」運維依賴（GeForce 不受 NVIDIA 支援）。
- **Step 6 Candidate A：前提直接檢驗被推翻，不建造**：用暫時的 `HYBRID_BG_DEPTH` knob 把每 worker 的 CPU 後段 thread pool 深度從 1 提到 2——`workers=1` +2.8% 更慢（GIL 競爭）、`workers=3` −1.2%（雜訊內）。knob 已刪除。
- **7 條不變量如何被滿足**（表）：`_finish_batch` 是唯一重編號實作；first error→停 feeder、`terminate()` 所有兄弟、再 raise；動態佇列每 tile `put` 一次 `get` 一次；worker 每次 `run_batch` 重新 spawn 故 `gc.freeze()` 契約自動成立（**常駐 pool 設計是避開、未解決**）；VRAM/RSS 全 N 驗證；per-cell veto；`workers=1` 預設、API 不傳 `workers`。
- **程式碼改動**：`hybrid_pipeline.py`（`run_batch(..., workers=1)`、`_mp_tile_worker`、`_run_tiles_multiprocess`、`_finish_batch`）、`perf_measure.py`（`--mp-workers`）、新增 `mp_concurrency_probe.py`、`mp_scaling_report.py`、`verify_mp_failfast.py`、`small_{ihc,dish}.tiff`（gitignored）。`workers` 刻意是 `run_batch` 參數而非 `Config` 欄位（避免改 hash 使舊 CSV 無法比較）。
- **Follow-ups**：worker 內 per-bucket 計時不存在；persistent pool 是避開而非解決（API 若啟用 `workers>1` 每次請求要付 `3.14 s × N`）；VRAM per process 超線性未解；**real-WSI 驗證成為放行的綁定約束**；probe 的 N>1 VRAM 欄位無效；`hybrid_pipeline.py` 無 `--workers` CLI；無斷點續跑；`expandable_segments:True` 未試。

---

## 7. Round 6：下一輪優化週期——規劃（doc 22）與落地（doc 23）

### 7.1 doc 22 — Next optimization cycle 研究計畫（2026-07-25，純規劃）

- **觸發**：團隊希望從兩個角度繼續縮短全片 wall-clock：(A) 更多多行程/多執行緒平行；(B) 把 CPU 迴圈工作搬上 GPU 做批次/平行運算。**明確約束**：不得改 `default_tile_size`、`window_overlap_px`、`window_dedup_iomin` 等定義分割視窗/tile 邊界的幾何（已驗證且縫合正確性敏感）；此約束**不涵蓋** `cellpose_batch_size`/`batch_size`（GPU 派工參數，非幾何）。新事實：隊友已完整跑過一片真實 WSI 並確認程式可用。
- **累計進度鏈（large/441）**：control 848.0 s → ① 兩段 overlap 707.4 s（−16.6%）→ Cellpose 4.2.1.1 573.7 s（−18.9%）→ ⑧ off MAIN + precut 串流 480.3 s（−16.3%）→ **`workers=3` 156.1 s（−67.5%）**；全片（35,700 tile）`workers=3` 線性投影 ~3.3 h（上限）。
- **已關閉清單**：CUDA MPS、更深 CPU pipeline、`detect_all_dots` process backend、gc 重定位、固定 N gc 批次、CUDA-stream/depth-2、GPU 端 tile 載入。**需附 caveat 而非重新關閉**：round 4 的 `cellpose_batch_size` 掃描範圍比看起來窄——只測了 config 值本身，而該呼叫點在結構上不可能觸發該值控制的行為。
- **Step 0**：把隊友的整片結果以既有量測紀律正式化（`workers=1` vs `workers=3`、`--gpu-dmon`、per-worker tile 數與 idle、N=3 下 35,700 tile 的 VRAM/RSS、correctness veto），決定規則：通過則 item 7 關閉、item 1 放行生產。
- **Track A**：A1 以真實整片組織密度重調 worker 數；A2 CPU 核心競爭稽核（`detect_all_dots` 的 `Parallel(n_jobs=-1, prefer='threads')` 在 `workers=3` 下是否 3 倍超額訂閱）；A3 多請求並發負載測試（item 8，僅在確認部署會並發時才做）。
- **Track B**：
  - **B1 跨 tile Cellpose 批次——根因追到 wrapper 而非 GPU**：`segment_windowed` 每視窗呼叫一次 `segmenter.predict`，1024 px tile 只有一個視窗；`CellposeModel.eval` 對 Python list 是逐張序列迴圈；但對真正堆疊的 `(Lz,H,W,C)` ndarray 會走 `core.run_net`，其 `nimgs = max(1, batch_size // ntiles)` 本來就會把多張影像的 patch 塞進同一個 `IMGa` 緩衝一起 `_forward`——機制存在，只是我們的呼叫點永遠給 `Lz=1`。設計：累積 N 個 tile 的 RGB 後堆疊呼叫 `model.eval(stacked, batch_size=N*16)`；輸出 `yf`/`styles` 已按輸入影像分開，下游 flow→mask 仍 per-tile。風險：pipeline 結構改變（overlap 粒度從 1 tile 變 N tile）、與 `workers=N` 非加性、浮點順序、尾組處理、第三方介面釘版本。**Method 第一步：用微基準（不改 pipeline）**。
  - B1b UNet++ 同樣問題；研究線索 `cell_mask/unet_mask/inference.py` 的 `predict_batch`（**後被證實只是序列迴圈**）；ceiling 受 UNet 前向只占 ~0.4% wall 限制。
  - B2 `detect_all_dots` GPU 化（條件式，依 A2；slack 臂 ceiling 1.013x）；B3 `enlarge_cell_instances`/`build_all_positive_results`（只能與 B2 打包）；B4 對 cellpose 4.2.1.1 重新確認 kernel-launch 追蹤（行政性）。
- **Back-pocket 清單**（未排程）：`torch.compile`（UNet++ 0.4% 上限）、UNet CUDA graph、pinned memory（已結案）、DALI/TensorRT（需重新驗證模型）、bare `cellpose_batch_size` sweep（`Lz=1` 下平坦，只有與 B1 重構配對才有意義）。
- **建議順序**：Step 0 → B1 微基準 → A2 → A1 → B1/B1b 完整設計 → B4 → B2/B3（僅若 A2 發現競爭）→ A3。

### 7.2 doc 23 — round 6 落地與量測

- **環境**：git `a4a6254`；RTX 5090 / driver 580.173.02、20 CPU 核、cellpose 4.2.1.1（`cpdino`/`dino_vitb`）、torch 2.11.0+cu130、numpy 1.26.4、scikit-image 0.24.0、joblib 1.5.3；config hash `ad41c42f` → **`3d1087f2`**（加一個欄位）；large n=2、控制與候選交錯跑；微基準取 3 reps 的 min 並記錄 spread。
- **標題結論**：B1、B1b 兩個批次線**皆因量測而停損**；A2 稽核找到「比原本要找的更大的缺陷」：`detect_all_dots` 的 joblib fan-out 在**每個 process 數下都是淨損失**，20 threads 比 plain serial 慢 **2.77x**；修復只需一行 config。
- **範圍決策**：step 0（整片驗證）由專案負責人**本輪不執行**（理由：tile 獨立處理、crop 足以量吞吐/記憶體/正確性；~11 h 跑批 + ~270 GB 暫存輸出代價高；隊友已確認整片可跑），A1 縮小到 crop，A3 只記錄。**順帶發現兩個與先前假設矛盾的事實**（後被 round 7 更正）：整片是 27,565 tile（141,818 × 114,366，stride 768 → 185 × 149），非 35,700；本片背景亮度 ~213 且僅 ~39% 的格子高於它——「幾乎全是白背景」不成立（**round 7 又證實這個 ~61% 組織是錯的**）。
- **B1 追根後停損**（`scripts/cellpose_batch_probe.py`，8 個真實組織 tile）：
  - 機制已驗證（`models.py:231` list 逐張迴圈；4-D ndarray 走 `core.run_net`；`bsize=384` → 16 patch = 現有 batch size → round 4 平坦由原始碼重現）。
  - 結果（ms/tile，G=1/2/4/8）：M2：253.1/238.6/254.6/259.4（−5.7%/+0.6%/+2.5%，G=2 的 −5.7% 在 7.1% rep spread 內）；M3b：245.2/242.1/253.9/265.2；`run_net`-only 193.6/184.3/198.5/204.1 → 前向占 76% 且**不攤提**（成本與 patch 數成正比，而非每次呼叫固定開銷）；peak alloc 1.17 → 8.01 GB 線性成長；masks 在所有 G **bit-identical**。結論：不建 N-tile gather；VRAM 拿去多開 worker（~10–20%/個）更划算。
- **B1b 停損**：`predict_batch` 是 `for img_path in tqdm(...)` 呼叫 `predict_single` 的序列檔案層包裝，無可移植；實測 G=1/2/4/8 → 23.78/25.11/27.93/27.81 ms/tile（+5.6%/+17.5%/+16.9%），非 bit-identical（0.0024–0.0058% 像素）；1024 px 時走 `_predict_direct`，`_predict_sliding_window` 從不被觸發。
- **A2 稽核**（`scripts/cpu_contention_probe.py`，`--prepare` 跑真 GPU 前段並 pickle `_ChunkGpuState`；`--run` 以 P 個同步 process 重放整個 BG 臂）：`n_jobs=-1` 下 `detect_all_dots` 中位數 P=1 694.8 ms、P=3 329.8、P=6 364.2（**越多 process 反而越快**，與超額訂閱相反）；`n_jobs` 掃描：1 → 252.4/274.2/287.9 ms（P=1/3/6），2、4、8、−1 全部更慢；單 process ms/tile：1 → 260.3、2 → 296.7、…、12 → 527.0、−1 (20) → **721.5**；dot counts 全部相同。機制：每個 `_detect_one_cell` 任務是 ~4 ms 的小 Python 級工作，joblib 派工 + GIL 交接超過工作量（**anti-pattern #10：「looks parallel」≠「is parallel」**），也修正 bottleneck-list ② 一直寫的「已 CPU 平行」。
  - **改動**：`config*.py` 新增 `dot_detect_n_jobs: int = 1`；`hybrid_pipeline.py:636`（`_finish_chunk_cpu`）傳 `n_jobs=config.dot_detect_n_jobs`。
  - **端到端**（large n=2、交錯）：`workers=1`：471.9/497.5 s（均 484.7）→ 302.7/302.8 s（302.7）= **−37.5%、1.60x**；`workers=6`：120.8 → 120.7 s（~0%）。**增益來源不在改動處**：`B3_detect_dots` 267.1 → 100.2 s（−62.5%）；**`B1_m3b_cellpose`（MAIN 臂兩次 Cellpose 前向）410.2 → 232.0 s（−43.4%，−178.2 s）**——19 條多餘 BG 執行緒搶 GIL，把驅動 Cellpose 的主執行緒餓死（呼應 gil-contention-diag 的 81.4%）。**關閉測量紀錄中一個懸案**：bottleneck-list ① 把 B1 +192.9 s 記為「與 GIL 競爭一致、尚未隔離」的假設，這次移除多餘執行緒返還 −178.2 s，同量級同方向同機制。
  - **為何 `workers=6` 不增益**：round 5 的 3.09x 本來就是在回收同一筆 GIL 損失，兩者是同一回收的兩條路、**不疊加**；`workers=6` 此時已是 GPU-bound（~120 s）。
  - **實務意義**：`workers=1` 仍是生產預設（API 不傳 `workers`）；large 單 process 484.7 → **302.7 s**；per-tile 1.0608 → **0.686 s**；全片單 process 估算（27,565 tile）~8.1 h → ~5.3 h；與多行程差距由 4.0x 縮到 2.5x。
  - Correctness：最壞 `reddot 2 / blackdot 8 / score 4`，與 doc 21 同碼雜訊指紋相同；X-flips 21 = 同碼上限；僅 1/~13,140 cell 不同；`n_jobs` 掃描證實 dot counts 完全相同。
  - **`workers=6` 本輪 OOM 兩次**（`w6_nj1_r1`、`a1_w6_nj1_r2`）：某 worker 恰好吃 **24.76 GiB**，兄弟只有 1.1–1.7 GiB；victim tile 不同（`tile_x1536_y0`、`tile_x5376_y0`）。與本改動無關（不分配 CUDA 記憶體），但是 `workers≥6` 的主導風險（2/6，對照 round 5b 的 0/6）。
- **B2/B3 不建**：A2 沒有發現超額訂閱；CPU 壓力以一個整數解決（每 worker 用 1 核而非 20 核，CPU 時間少 2.77x）；BG 臂的 `detect_all_dots` leaf（`_binary_erosion`、`xyz2lab`、`rgb2xyz`）僅 1.2–2.2% 取樣。
- **A1（crop 範圍）worker 重調**（large，`dot_detect_n_jobs=1`）：`workers` 1/3/4/5/6/8 → 302.7/142.9/128.8/122.7/119.9(n=1)/119.2 s；5→6 只買 2.3%、6→8 0.6%（GPU-bound ~119 s）。**建議從 `workers=6` 下修為 `workers=4`（無人看管）/`workers=5`（可接受重跑）**，`workers=4` 距地板 +8.0%。caveat：這是 crop 曲線，整片是否同樣成立仍是 item 7 的問題。
- **B4 重新追蹤 cellpose 4.2.1.1**（py-spy `--gil` 與全執行緒、100 Hz、medium、`workers=1`；`scripts/gil_trace_report.py`）：五個舊函式在 MAIN 臂 self%：`_extend_centers_gpu` 5.46%、`steps_interp` 2.09%、`fill_holes_and_remove_small_masks` 1.68%、`get_masks_torch` 1.35%、`get_rel_pos` **0.00%（已不存在，backbone 是 CPDINO/dino_vitb）**，合計 **10.58% → ceiling 1.118x**（原 1.23x）；新的最大兩個 leaf 為 `_from_device`（`cellpose/core.py:141`）24.2% 與 `_quantile` 10.1%（皆不在 item 5 清單）；item 5 維持停損。
- **程式碼改動**：僅 `config.py`/`config_example.py`（`dot_detect_n_jobs`）與 `hybrid_pipeline.py:636`；新增量測工具 `cellpose_batch_probe.py`、`unet_batch_probe.py`、`cpu_contention_probe.py`、`gil_trace_report.py`；`run_batch` 預設仍 `workers=1`。
- **仍開著**：整片驗證（item 7）、`workers≥6` allocator 氣球、A3、Cellpose 內部（若重開目標應為 `_from_device`/`_quantile`）、B1/B1b 已關閉、B2/B3 未建、⑨ 被超越、**round-3 checkpoint 臨床簽核仍阻擋中**。
- **重現注意**：spawn workers 從磁碟重新讀 config，故掃描 `dot_detect_n_jobs` 要在 run 之間改 `config.py`，`perf_measure.py` 的 in-process override 到不了 worker。

---

## 8. Round 7：GPU encode/decode 與逐項迴圈——調查（doc 24）與落地（doc 25）

### 8.1 doc 24 — 調查計畫（純規劃，無程式改動；三次修訂）

- **範圍**：找「仍 CPU-bound、可能搬到 CUDA / `cu*` 系列」的項目，特別是 encode/decode 與大量重複的獨立逐項迴圈。**修訂 1**：基準由 441 tile crop 改為全片 27,565 tile；crop 背景只佔 12–15%，而當時（錯誤）認為真實玻片有 39% 背景；**修訂 2**：上游 commit `a64b92e`（2026-07-25）刪除 `m0_reader.py` 死碼並把 `run_batch` 預設 `workers=1` 改成 `workers=4`（僅記錄）；**修訂 3**：目標是**最短總 wall-clock**，沒有偏好單/多 process，因此每個候選都要同時問 (a) 有沒有幫助、(b) 在 `workers>1` 下是否淨損失。
- **§0 全片重新基準**：grid 185 × 149 = 27,565 tile（62.5× 441 tile crop；注意 35× 是像素比）；**背景 tile 的短路是現有程式**：`_process_one_chunk_gpu` 先跑 UNet core mask，`core_mask.sum()==0` 就 `return None`，不跑兩次 Cellpose；`_process_precut_tile_cpu` 對 `tg.chunk is None` 呼叫 `_write_blank_tile`，連 `detect_all_dots` 都不跑。**但 `_write_blank_tile` 仍寫與真 tile 相同的六個檔**（`core_mask`/`masked_ihc`/`dish_mask_overlay` PNG；`instance_mask`/`dish_nucleus_mask`/`overlay_annotated` TIFF），~10,750 個背景 tile × 6 = ~64,500 次 encode，單價從未量過。
  - 組成不匹配 vs 固定每次呼叫開銷攤提兩種機制；§0.4 把 round 6 的 per-bucket 數字依「該 bucket 實際跑在哪個族群」重新乘以全片族群：MAIN ≈ 171.6 + 14.2 ≈ 186 min（3.1 h）、BG ≈ 148 min（2.5 h）；單 process wall 約 3.1–3.4 h（官方 blended 5.3 h 偏高）。
  - **§0.5 多行程下什麼會變、什麼不變**：`_mp_tile_worker` 每個 worker 都是同一個兩段式 arm 結構，所以候選的 MAIN/BG ceiling 是**per-worker 不變**；真正新增的是**共享 VRAM 餘裕**——`workers=4` 時 VRAM ≈ 20.7/32 GB，加上 `workers≥6` 的 24.76 GiB 氣球失敗模式；任何新 GPU 函式庫（nvTIFF/nvImageCodec/CuPy/cuCIM）在每個 import 它的 process 都加自己的 CUDA context；`workers=3→4` 就買 ~11%，丟掉一個 worker 比單一 tile encode 優化代價高。Candidate A（parent 中跑一次）免疫。
- **§1 已關閉清單的「全片重新框架」**：GPU 前向（1.118x，20 分鐘上限，風險阻擋）、跨 tile Cellpose/UNet 批次（負向；G=16 可選驗證）、`detect_all_dots`/`enlarge` GPU 化（更確信關閉）、precut A（~1.8 分鐘）、CUDA MPS、GPU 端 tile 載入（~2.5 分鐘）。
- **§2 候選**：
  - **A — 最終 overlay 縫合（Phase D）TIFF encode**（`_stitch_overlay_slide`，pyvips row-then-column `join()` + 單次 `tiffsave(lzw, tile, pyramid)`；parent 進程、與 workers 無關）：crop 5.05–5.11 s（0.46 GP），全片 16.2 GP 約 35× → 外推 ~3 分鐘；在多行程下相對佔比**變大**；候選：nvTIFF（無確認 Python binding、無一流 pyramid 支援）、nvCOMP（無 LZW）、cuCIM `cuslide2`（write 路徑待查）。
  - B — 每 tile debug 陣列 PNG/TIFF + per-cell crop 編碼（~52.2 min，BG 臂 slack，不建）；C — int32 label TIFF（併入 B）；D/E — `detect_all_dots`/`enlarge`/`build_all_positive_results` GPU 批次（不建）；
  - **F — 背景 tile placeholder 寫入**（`_write_blank_tile`；真正新發現：此前無任何 bucket 單獨量過）；若固定開銷主導則解法是「寫一組空白 tile 一次再 hardlink」，非 GPU；
  - **G — 每次寫入前冗餘的 `mkdir(exist_ok=True)`**（`_save_tile_array` 每個 tile 6 次，全片 ≈165,390 次 syscall；多行程下 4 個 process 同時對同一 inode 呼叫是穩定性疑慮）：doc 24 認為「最高信心、最低風險」。
- **§3 環境/資源閘**：RTX 5090 sm_120 對 nvImageCodec/nvTIFF/nvCOMP/cuCIM 的驗證不明；全無依賴已裝；round 3 的 `uv sync` 綑綁升降版導致 ⑨ +22.3% 無法歸因 → 新 GPU codec 依賴必須獨立 `uv sync`、獨立 `pip freeze`、獨立 benchmark；共享 VRAM 預算（`workers=4` 餘裕 ~11.3 GB nominal）。
- **§4 優先序**：(1) 組成匹配 crop 在 `workers=1`/`4` 各量一次（直接量 F 與 G）→ (2) 量 Candidate A 的實際規模 → (3) 原型/量 G → (4) 環境+VRAM spike → (5) B/D/E 不建 → (6) 可選 G=16 Cellpose 批次。

### 8.2 doc 25 — round 7 落地與量測（git `025f9a5`；config hash `3d1087f2` 不變；**無 pipeline 程式改動**）

- **標題**：doc 24 §4 第 1 項原本要「微調」組成估計，結果**推翻了它**：以 pipeline 自己的規則（UNet++ core mask 全空 = 背景）量到，玻片是 **55.8% 背景**（非 39%）。**Candidate G 停損**；**Candidate F 真實但 wall 為 0**（24 ms/背景 tile，在有 47–53% slack 的 BG 臂上）；**Candidate A 是唯一倖存者**——Phase D 在真實 16.2 GP 實測 **322.7 s**，是 doc 24 外推的 1.8 倍，完全序列、parent 進程；GPU 編碼器（nvImageCodec，無損 LZW，encode 步驟 19.2x）是 doc 24 清單中唯一既存在於 CUDA 13 又能在本機跑的。
- **§1 組成前提檢驗**：
  - brightness 熱圖重現但閾值敏感：cell mean ≥213 → 2.4% 背景、≥200 → 39.0%（doc 23 的數字其實是 ≥200，非它宣稱的 213）、≥188 → 55.1%；p50 = 191、p90 = 211、max = 226，±13 灰階使答案從 2% 擺盪到 39%（16×）。
  - **真實規則**（`scripts/tissue_calibrate.py`，抽樣格子、完全模擬 `PrecutStream` 切 tile、跑真 M1 forward）：n=250（seed 0）58.0%（CI 51.9–64.1%）；n=800（seed 1）55.3%（CI 51.8–58.7%）；brightness 最佳閾值（≥188–191）仍錯分 ~12% tile。
  - **全格掃描**（`scripts/core_mask_map.py`：27,565 次真實 M1 forward，1,043.7 s）：**背景 15,386（55.82%）、組織 12,179（44.18%）**；n=800 抽樣誤差只有 0.6 pp。
  - 影響：背景 ~10,750 → **15,386**；組織 16,815 → **12,179**（−28%）；Candidate F 母群 +43%（92,316 encode）。**所有 MAIN 臂成本縮 ~28%，唯一隨背景 tile 成長的 BG 臂成本增 ~43%**。
- **§2 錨點與 arm 天花板**：
  - **comp24**（`composition_crop.py` 以 brightness proxy 選窗 (96768, 1536)，18,688² = 576 tile）**實際 73.4% 背景**（proxy 預測 55.2%；proxy tile 級準確 ~88% 但錯誤空間相關，window 級誤差大）；`workers=1` 134.91 s、`workers=4` 65.54 s（**2.06x**，大 crop 為 2.35x——背景重的負載多行程收益較少）；MAIN 127.2 s（Cellpose M2+M3b 85.5 s/62.6% wall、UNet 16.0、`enlarge_cell_instances` 7.1、`_read_rgb` 5.1、M1 glue 4.7、`build_all_positive_results` 4.1），BG 67.1 s（`detect_all_dots` 26.0、PNG 25.6、**blank-tile 寫入 10.1**、render 3.2、crops 1.7、TIFF 0.4），**BG/MAIN ≈ 0.52**（round 6 緻密 crop 為 0.74）。
  - **match24**（以精確 map 選窗 (46080, 768)，576 tile）：實際 **322/576 = 55.9%** 背景；`workers=1` 188.8 s、BG/MAIN **0.470**、MAIN 需削減 53.0%；`workers=4` 88.3 s（2.14x）。blank-tile 寫入率 0.0237 vs 0.0241 s/tile（−1.7%，兩個獨立 crop 重現）。
  - **結論**：任何在 BG 臂的候選（B/D/E/F）wall 天花板 = **1.00x**，直到 MAIN 先縮 ~50%。
- **§3 Candidate G（mkdir hoist）已原型、量測、停損**：mkdir 呼叫 3,462（= 6 × 576 + 6）、0.075 s、**0.056% wall**（ceiling 1.0006x）；全片 165,390 次 ≈ **0.83 s**。並發：`fs_write_probe.py` P=1/2/4/6 mkdir µs/call 2.60/2.58/5.11/5.33（P≥4 時一兩個 process 看到 8–12 µs），全片最壞 ≈ 2 s；e2e A/B（n=2 交錯）：`workers=1` 134.91 → 135.76 s（+0.63%）、`workers=4` 65.54 → 66.06 s（+0.80%），在 spread 內；correctness 所有欄位 bit-identical。**不上線（已 revert）**：會引入新不變量（`_save_tile_array` 不再自給自足、需先跑 `_ensure_output_dirs`）。NFS 注意：本機 NVMe ~2.6 µs/call，網路檔案系統可能貴三個數量級，若輸出改到 NFS/SMB 要重量。
- **§4 Candidate F 首次量測**：`F_write_blank_tile` 423 次 10.18 s（24.07 ms/tile、7.5% wall）；其中 PNG 8.91 s（7.02 ms/次，6.6%）、TIFF 0.89 s（0.70 ms/次）；對照真 tile PNG 55.82 ms/次。**磁碟面**：三個 TIFF 未壓縮 → 423 個背景 tile 寫 4.3 GB（**10.2 MB/空白 tile**），全片 **~157 GB 常數資料**（估整片輸出 ~270 GB 的一半以上）。`os.link` 替代方案：407×/272×/283×（P=1/4/6），可去掉 ~99.6% 空白 tile 寫入成本與 ~157 GB；**但 BG 臂有 47–53% slack → 零 wall**（BG 70.3 → 64.2 min vs MAIN 149.7 min），**不建**（儲存面理由另議）。
- **§5 Candidate A（Phase D）真實規模**（`scripts/stitch_probe.py`：重建 `chunk_offsets` grid、用 `core_crop_bounds` 算每個 tile core crop 尺寸，邊緣 896/768/250 px 三類；以真實 overlay tile 當模板、其餘位置 hardlink；annotated 比例 44.75%）：35,840×28,928（1.04 GP）14.63 s（14.11 s/GP）；71,168×57,344（4.08 GP）57.80 s（14.16 s/GP）；**141,818×114,366（16.22 GP，27,565 tile）322.7 s（19.90 s/GP，輸出 5.76 GB）**。**實際 5.4 分鐘，非 ~3；全片規模超線性（+40%/像素）**，任何 crop 線性外推都會系統性低估。穩健性：stitch 一次開 27,565 個 pyvips 影像，`RLIMIT_NOFILE` 1,048,576 才通過，常見 1,024 soft limit 會失敗（→ round 8 修）。價值：`workers=1` 占 2.59 h 的 **3.5%**（ceiling 1.036x）；`workers=4`（2.17x 平行部分）約 74 min 中占 **7.3%**（ceiling 1.078x）。
- **§6 全片投影重建**（`scripts/wsi_projection.py`；以 `match_w1_r1` 速率 × 實測族群：12,178 組織 + 15,387 背景）：`B1_m3b_cellpose` 0.5259 s/tile → 106.7 min；`B2_png_encode` 29.7；`B3_detect_dots` 28.4；`B1_unet_coremask`（全 tile）13.3；`B3_enlarge_cells` 8.7；`F_write_blank_tile` 6.1；`B3_build_results` 5.6；其他 19.2；**MAIN 149.7 min、BG 70.3 min（0.470）、Phase D 5.4 min → `workers=1` ≈ 2.59 h、`workers=4` ≈ 1.24 h**；comp24 同算法得 2.70/1.33 h（4% 價差）。歷史對照：round 5 ~10.5 h（錯誤 tile 數）→ round 6 ~5.3 h（85% 組織 crop blended）→ doc 24 3.1–3.4 h（方法對、組成輸入錯）→ **round 7 ~2.6 h / ~1.25 h**。仍是單一 crop 速率外推。
- **§7 跨 tile Cellpose G=16**：緻密 crop（8×8、100% 組織）、26 個真實輸入：M2 238.5 → 254.3 ms/tile（**+6.6%**）、`run_net` +1.6%、peak 15,840 MB；M3b 268.5 → 284.3（**+5.9%**）、`run_net` +9.6%、15,838 MB；`workers=4` 時 4 × 15.8 GB = 63 GB 算術上不可能；G=16 時 cell count 變動 ≤0.15%（G ≤ 8 為 bit-identical）。關閉。
- **§8 環境 + VRAM gate**（throwaway venv，一次一個套件）：
  - `cupy-cuda13x` 14.1.1：可裝但 import 失敗（numpy ≥2 ABI，專案 pin `numpy<2`）；13.6.0：可 import（報 cc 120）但首個 kernel JIT 失敗（無 CUDA toolkit，只有 driver + torch bundled libs）→ **不可用**；`nvidia-nvimgcodec-cu13` 0.9.0/0.8.0 **可用**；`nvidia-nvtiff-cu13` 0.8.0.82 **無 Python module**（只有 shared libs）；`nvidia-nvcomp-cu13` 5.3.0 可用但**無 LZW**；`cucim-cu13` 26.06.00 **唯讀、無 write**。
  - 編碼吞吐（4096²）：pyvips lzw 129.3 MP/s；+tile 136.3；**+tile+pyramid+bigtiff（`_stitch_overlay_slide` 實際呼叫）93.0 MP/s**；**nvImageCodec TIFF 1,789.6 MP/s**（compression tag 5 = LZW、photometric RGB、bit-identical round-trip）→ **13.8x（對 strip LZW）/ 19.2x（對 pipeline 配置）**；但**無 pyramid、無 BigTIFF、單頁**。結論：entropy coding 不是 floor，pyramid/container 工作是 322.7 s 的大部分。
  - VRAM：idle 41 MB → 建構 `nvimgcodec.Encoder()` 557 MB → 一次 4096² encode 1,261 MB（~0.5 GB context、~1.2 GB 含緩衝）；Candidate A 免疫；`workers=4` 內 ×4 ≈ 5 GB，本輪已觸及 **26.7/32.6 GB** 瞬時峰值。
- **doc 24 §4 各項處置**：1 做完且推翻前提；2 做完（322.7 s）；3 做完後停損；4 做完（nvImageCodec 只對 Candidate A 通過）；5 確認不建；6 做完關閉；新 Candidate F 量測後不建。
- **新增量測工具**：`tissue_calibrate.py`、`core_mask_map.py`、`composition_crop.py`、`fs_write_probe.py`、`stitch_probe.py`、`wsi_projection.py`、`gpu_codec_spike.py`；`perf_measure.py` 新 bucket（`F_write_blank_tile`、`B2_{png,tiff}_encode_blank`、`G_mkdir_*`）；`arm_report.py` 把 `F_write_blank_tile` 加入 BG 臂。
- **停損的 Candidate G patch 保留**（`_TILE_OUTPUT_SUBDIRS`、`_ensure_output_dirs(output_dir, merge_dir)`，`run_batch` 中 `compute_tile_geometry` 之後、`workers>1` 分支前呼叫一次）。
- **仍開著**：整片驗證（但兩個輸入已實測：背景 55.8%、Phase D 322.7 s；估算降至 ~2.6/~1.25 h）；doc 19/23/24 的 39%/61% 主張已被取代；Candidate A（先試 `tiffsave` 旋鈕與跳過空白區，再考慮 GPU）；`workers=4` 瞬時餘裕只有 ~6 GB；round-3 checkpoint 臨床簽核。

---

## 9. Round 8：剩餘待辦計畫（doc 26）與落地（doc 27）——第一次完整 WSI 端到端

### 9.1 doc 26 — 剩餘工作實施計畫（2026-07-27）

- **目的**：把 `19-open-backlog.md`、`DISCOVERED-NOT-IMPLEMENTED.md`、`bottleneck-list.md`、`current-status-comparison.md` 四份重疊清單，套用 quickref 的 Discover→Analyze→Plan→Choose **在待辦清單本身**，得到單一排序計畫。排序原則：ceiling <~5% 的項目列出但不排程；可靠性/正確性 gate 排在任何 ceiling 的效能項目之上（不可放行的 3.5x 不如可放行的 1x）；純文件漂移最後批次處理。
- **Discover（去重清單）**：
  - 效能：整片 WSI 驗證、多行程放行決策（2.06–3.51x）、`workers≥6` OOM、無斷點續跑、`_stitch_overlay_slide` 缺 `RLIMIT_NOFILE` 檢查、Phase D `tiffsave` 旋鈕、Phase D GPU 化（nvImageCodec）、`clear_slide_edge_cells`、`detect_all_dots` +22.3% 迴歸、整塊向量化、Cellpose 內部、API 並發負載測試、常駐 worker pool、A1 worker 重調、worker 內 per-bucket 計時、`pip freeze`。
  - 正確性：Cellpose 4.0.8→4.2.1.1 checkpoint 需病理醫師簽核。
  - 文件漂移：七項（#32–#38）。
  - 不帶入：所有 🔴 stop-lossed（#7, #9, #11–13, #17–25）與三個無 sizing 的 ⚪ 背口袋點子（#27 `torch.compile`、#28 UNet CUDA graph、#29 DALI/TensorRT）。
- **Analyze 兩條依賴鏈**：Chain A（整片驗證 → 放行 → A1 調校；`workers≥6` OOM 與斷點續跑應先於或與驗證同步處理，否則失敗的驗證跑批根因含糊）；Chain B（Phase D 便宜旋鈕 → 之後才考慮 GPU 化）。其餘獨立。
- **Plan 分層**：
  - **Tier 0**（阻擋性）：0.1 整片驗證（先 `workers=1`，後 `workers=4`；先硬化再驗）、0.2 `expandable_segments:True` 重跑 `workers=6` 掃描、0.3 per-tile 完成記錄的 resume、0.4 `RLIMIT_NOFILE` 檢查。
  - **Tier 1**：1.1 Phase D 旋鈕、1.2 GPU 化（視 1.1）、1.3 worker 內計時（先於 0.1 的 `workers=4`）。
  - **Tier 2**：`clear_slide_edge_cells`、`detect_all_dots` 迴歸（機會性）。
  - **Tier 3**：3.1 checkpoint 臨床簽核（治理，向使用者/團隊提出）。
  - **Tier 4**：七項文件漂移（批次一次做）。
  - **Tier 5**：多行程放行決策、A1（需 Tier 0 完成）。**Tier 6**：Phase E 並發測試、常駐 pool（需產品決策）。**Tier 7**：`detect_all_dots` 整塊向量化、Cellpose 內部（不建議排程）。
- **建議順序**：0.2–0.4 → 1.3 → 0.1 → 1.1（可平行）→ Tier 5 → Tier 4 → 3.1（立即提出）→ Tier 2/6/7。
- **明確排除表**：跨 tile Cellpose/UNet 批次、CUDA MPS、CPU 後端 process pool（Candidate A）、fork 重用模型（Candidate E）、CUDA-stream/depth-2、CUDA graph/向量化 Cellpose、GPU 端 tile 載入、Candidates B/D/E（BG 臂 slack + CuPy 在本機跑不起來）、Candidate F、Candidate G、`torch.compile`/DALI/TensorRT。

### 9.2 doc 27 — round 8 落地（2026-07-27）

- **post-refactor 位置註記**：M0 拆分後函式搬到 `m0_module/`（`_ensure_nofile_limit`/`_join_overlay_tiles`/`_stitch_overlay_slide` → `m0_module/m0_stitch.py`；多行程 → `m0_module/m0_multiprocess.py`；checkpoint → `m0_module/m0_checkpoint.py`；`PrecutStream` → `m0_module/m0_reader.py` 經 `m0_slide` facade）。重構以純搬移驗證（除一個相對 import 深度外逐位元相同），但**以三種方式弄壞了 `perf_measure.py`**（harness 靠 patch 父行程 namespace，被 patch 的名字搬走了）：(1) `install_wrappers()` 因直接屬性存取 `HP._save_tile_array`/`HP._write_blank_tile` 啟動即 `AttributeError`；(2) **22 個 per-stage bucket 中 15 個悄悄變空**（`wrap()` 用 `getattr(..., None)` 並印 skip，run 看似完成但是空洞的量測，三者中最危險）→ 13 個 wrap 目標改指 `m0_tile_runner`；(3) `LAST_MP_WORKER_TIMINGS` 未 re-export，最後寫入前 `AttributeError`（整個量測做完才丟失結果）→ 透過 `m0_slide` re-export。仍有三個 wrap 在重構前就已死（`segment_masked_dish`、`export_per_cell_images`、`process_precut_tile`）；`HP.precut_paired_tiles` 仍未 re-export。
- **總表**：0.2 allocator knob + `workers≥6` OOM sweep（12 次）→ **候選修復未被證實，預設維持 off、`workers≤4–5` 維持**；0.3 斷點續跑 → 已建、輸出與 cold run byte-identical；0.4 RLIMIT_NOFILE → 已建 + 7 tests；1.1 `tiffsave` 旋鈕 13 config → **關閉負向**；1.3 worker 內計時 → 4 個 parent-only bucket → **26 個 worker-side bucket**；0.1 整片驗證 → **完成，3.82 h / 1.73 h（+47%/+38% vs 投影）、2.216x**；Tier 4 七項全關（外加發現一個清單外的死引用）；3.1 臨床簽核 → 升級；Tier 2/5/6/7 依 doc 26 不做。**淨效果**：關閉兩個可靠性缺口、一個量測盲點，**零效能改動被採用**；自動化測試從無 → 48 個；`verify_mp_failfast.py` 在 `workers=1`/`3` 仍通過。**整輪不是負向結果——整片驗證重置了地圖**：發現三個來自 crop 的數字在整片規模不成立（`gc.collect` 16.1% wall、tile read 17.2% vs 1.22%、Phase D 19.3% vs 7.3%）。
- **兩個無先前紀錄的發現**：(1) **registration 階段對每個 modality 輸出不同畫布**（HER2 141818×114366、DISH 141658×114415、HE 141717×116400），`PrecutStream.__init__` 對尺寸不等 fail-fast，整片無法啟動（前七輪都用同座標 crop 故從未碰到；**後被 doc 44 證明此處理有誤**）；(2) `perf_measure.py` 非串流路徑呼叫 `HP.precut_paired_tiles`（未 re-export）→ `AttributeError`。
- **§1 Tier 0.4**：新 `_ensure_nofile_limit(needed)`——soft 不足時自行升到 hard（無須特權）；hard 也不足時報錯並說明需要設多少、且 per-tile 分析輸出已在磁碟無須重跑；`_stitch_overlay_slide` 拆為 `_join_overlay_tiles`（lazy join，guard 在這，**以 tile 數**而非第一次 `EMFILE` 觸發）+ `_stitch_overlay_slide`（join + `tiffsave`）；`tests/test_stitch_nofile_guard.py` 7 cases（用 `monkeypatch` 假造 limit，因真正降低 hard limit 在行程內不可逆）。
- **§2 Tier 0.3 斷點續跑**：`run_batch(..., checkpoint: bool = False)`；每個完成 tile 的 `owned` list（唯一無法由磁碟重建者）以 tmp + `os.replace` pickle 到 `output_dir/_resume/tile_x{ax}_y{ay}.pkl`；啟動時載入、過濾到目前 grid、`_skip_completed` 跳過；`config_hash.txt` 守衛（不符則全丟並大聲警告）；fail-fast 不變；單/多行程共用 parent 端 `_record(ax, ay, owned)`，多行程新增 `on_tile` callback，worker 不碰 checkpoint；**預設關**（API 單 tile 請求只多 I/O），`--resume`（`hybrid_pipeline.py`/`perf_measure.py`）開，`full_wsi_validate.py` 恆開。25 tile smoke ROI `workers=2`：cold 16.69 s → resumed 4.56 s（仍付 precut + 全域 merge/CSV + Phase D），`report.csv`（738 行/737 cells）與 `summary.txt` byte-identical；`tests/test_run_batch_resume.py` 12 cases（空 owned list 往返視為 completed、損毀 checkpoint 只多重算一 tile、異 grid 的 tile 被忽略）。
- **§3 Tier 1.1 `tiffsave` 旋鈕消融**（`stitch_probe.py --ablate`，4.055 GP = 92×75 = 6,900 tile、annotated 0.4474；每 variant 與出貨呼叫只差一個參數）：

| config | encode s | total s | speedup | out GB |
| --- | --: | --: | --: | --: |
| baseline（128 px tile、完整 pyramid、LZW、BigTIFF） | 57.02 | 59.45 | 1.0000 | 1.84 |
| tile_256 / 512 / 1024 | 59.67 / 66.10 / 83.71 | — | 0.9484 / 0.8602 / 0.6861 | 1.79 / 1.78 / 1.81 |
| predictor_horizontal / none | 56.21 / 55.59 | — | 1.0023 / 1.0143 | 1.84 / **2.38（+29.7%）** |
| depth_onetile / shrink_nearest / subifd | — | — | 1.0062 / 1.0228 / 0.9960 | — |
| deflate | 72.53 | 75.73 | 0.7850 | 1.62 |
| **zstd_1** | **44.91** | **48.21** | **1.2331** | **1.58（−13.8%）** |
| BOUND_no_pyramid（QuPath 需要，否決） | — | — | 1.2583 | 1.20 |
| BOUND_no_compression（8.9x 大小，否決） | — | — | 1.2628 | 16.25 |

  - **LZW encoding 占 Phase D 的 20.8%**（baseline → no-compression bound 12.37 s）；tile 越大越慢（128 px 已是對的）；pyramid depth、predictor 已是有效預設（byte-identical）；zstd_1 距無壓縮 bound 僅 2.4%。
  - **zstd 被 correctness veto 殺掉**：zstd 為無損（tile 級與 ~1 GP 規模抽樣 2048² patch byte-for-byte 相同，兩者皆 `bigtiff=True` 與 10 級 pyramid），**但 QuPath（經 BioFormats）打不開 zstd TIFF**（對照組：`overlay_slide_lzw.tiff` 446.7 MB 可開、`overlay_slide_zstd.tiff` 385.2 MB 報 "cannot open"）→ 「病理醫師打不開的輸出是壞掉，不是更快」。`_stitch_overlay_slide` 不變。**對 GPU 化的新硬約束**：必須 BioFormats 可讀 → 排除 GPU encoder 最擅長的現代 codec；19.2x 是 LZW 數字故仍成立，但工程量比 doc 24 估的更大。caveat：4.055 GP 數字不外推（Phase D 超線性）；贏家需在 16.22 GP 重確認。
- **§4 Tier 1.3 worker 內計時**：`_install_worker_probe()` 讀 `HYBRID_MP_WORKER_PROBE`（`module:callable`），在三個 `_init_*` 之前呼叫（故 init 也被量測），失敗只 log；worker 在結束時以既有 result queue 發送 `("timings", {...})`，parent 在 `join` 前 drain，結果存 `LAST_MP_WORKER_TIMINGS`；`scripts/mp_worker_probe.py` 重用 `perf_measure.install_wrappers()` 使 parent/worker 的 bucket 同源；`perf_measure.py --worker-timings` 寫 `worker_timings_total`。25 tile `workers=2`：4 → **26** 個 worker-side bucket（`B1_m3b_cellpose` n=48 Σ17.307 s；`B3_detect_dots` 5.867 s；`B2_png_encode` 4.016 s；`B1_unet_coremask` n=25 3.967 s；`init_unet` 2.781 s；`init_cellpose_m2` 1.339 s；`init_cellpose_m3b` 1.038 s…）；per-worker model init **~2.58 s**（doc 20 假設 3.14 s）；`B1_unet_coremask` 觸發 25 次（含背景）、M2/M3b 與 M3 僅 24 次（組織 tile）證實背景快速路徑。注意：Σt 是跨 worker 的 CPU 時間總和非 wall；shim 增加 hot path 開銷，`--worker-timings` 只能用來看階段拆解、不可比 wall。
- **§5 Tier 0.2 allocator 組態 vs `workers≥6` 氣球**：新增 `config.cuda_alloc_conf`（預設 `""`），由 `_run_tiles_multiprocess` 在 worker spawn **之前**寫入 `os.environ`（唯一能到達 worker allocator 的時點）；`scripts/alloc_conf_probe.py` 交錯（非分塊）跑兩條件，各 6 次、`workers=6`、在 round 7 的 comp24 crop（x=96768, y=1536, size=18688，576 tile、137 success/439 skipped、bg=0.762）；dmon `fb` 欄位按 header 解析（先以 round 7 儲存檔驗證能重現 26,687 MB）。結果：control 1/6 OOM、median 65.46 s、median peak 22,968 MB、max 27,836 MB；`expandable_segments:True` 0/6、66.80 s、24,040 MB、28,700 MB。**缺陷重現**（`control_w6_r1` 一個 worker 24.62 GiB vs 五個兄弟 1.11–1.32 GiB）。**候選修復未被證實且機制反駁**：(1) `expandable_segments` 沒降低 peak VRAM（反略高）= 正面證據反對碎片化假說，支持 doc 19 #7b「byte-identical 24.76 GiB 是可重現的病態配置」；(2) 0/6 vs 1/6 統計不可區分（Fisher p≈1.0）；(3) wall +2.0%。**另得**：`workers=6` median 65.46 s ≈ round 7 `workers=4` 的 65.55 s → **`workers=6` 不比 `workers=4` 快，只增加 OOM 風險**。**決定**：`cuda_alloc_conf` 預設 off、`workers≤4–5` 不變；下一步改為根因分析 24.76 GiB 氣球；`workers=6` 真實瞬時餘裕 ~4 GB。**（round 11 後續：在 `workers=4` 正式掃描後 12 次 4/12 → 0/12，預設已翻開。）**
- **§6 Tier 0.1 整片驗證**：`scripts/full_wsi_validate.py`（preflight：grid、以**實測 12.3 MB/tile** 估輸出空間 + precut scratch、`RLIMIT_NOFILE`、CUDA、三模型檔；resume 恆開；先 `workers=1` 後 `workers=4` 皆 `--gpu-dmon`；兩份 `report.csv` 的 correctness 比對對 noise floor；每次跑完寫部分結果）。
  - **preflight 找到的 blocker**（見上）；`--conform` 把 pair 裁到交集 **141658×114366（保留 99.86%）**，存放於 `<output-root>/_conformed`；理由「intersection 而非 union 白填」；**conform 放在 runner 而非 pipeline**（避免放寬 `PrecutStream` 的守衛）。（**這個處理在 doc 44 被證實是根本錯誤**。）
  - 預估成本：27,565 tile、16.219 GP、輸出 ~513 GB（12.3 MB analysis + 6.3 MB precut/tile）、`/home` 9.7 TB 可用（`/data/nvmessd` 僅 319 GB 且不可寫）。
  - **`workers=1` 結果**（2026-07-27 04:29 → 08:19，**27,565 tile 無任何 fail-fast abort**）：wall **13,762 s = 3.82 h**（投影 ~2.6 h → +47%）；success/skipped 10,801 / 16,764；cells 356,255 rows、有效 44,535；peak RSS **61.13 GB**；peak CUDA reserved（parent）2.19 GB；分析輸出 347.0 GB；config hash `3d1087f2`。**組成預測在一個 tile 內命中**：背景 15,385（vs 預測 15,386）；`skipped=16,764` 還含 1,379 個 tissue-but-no-owned-cells。→ **+47% 全在 per-tile rates**。
  - **三個只有整片才暴露的階段**（parent-side TIMINGS 占 wall 比例；兩臂重疊故不加總為 100%）：`B1_m3b_cellpose` 24,360 次 6,358.9 s（46.2%）；**`B2r_tile_read` 55,130 次 2,368.5 s（17.2%，crop 時 1.22%）**；`B3_detect_dots` 2,333.2 s（17.0%）；**`B4_gc_collect` 2,218.4 s（16.1%，crop 時整批 0.52 s）**；`B2_png_encode` 2,040.6 s（14.8%）；**`D_stitch_overlay` 1,185.4 s（8.6%，probe 322.7 s → 3.7x 樂觀）**；`B1_unet_coremask` 734.8 s（5.3%）；`F_write_blank_tile` 457.1 s（3.3%）。
    1. `gc.collect` 80.5 ms/call（2,218.4/27,565），回到 pre-freeze 水準：`gc.freeze()` 只凍結「凍結時已存在」的物件，而 `per_tile_owned` 累積 **356,255 個 `CellAnalysisResult`** 在凍結後建立、被追蹤、每次收集重掃；→ 最有吸引力的剩餘目標（plausible 便宜修法：週期性重新 freeze 或讓累積結果離開 GC 視野）。
    2. `B2r_tile_read` 17.2%：precut scratch ~49 GB 不再放得進 page cache（stop-loss 數字不成立，需重新推導，位於仍有 slack 的臂）。**（round 11 證實 17.2% 本身被污染。）**
    3. Phase D 為 probe 的 3.7x（probe 以 hardlink 複製少量真 tile → 壓縮與快取比真實 tile 好）；`workers=1` 占 8.6%（doc 19 記 3.5%）。
    4. **peak RSS 61.13 GB**（先前最高 ~4 GB），由 stitch 持有 27,565 個 lazy pyvips 影像驅動（mid-stitch 觀察到 12,027 個 open fd）；主機需求 ~64 GB RAM。`RLIMIT_NOFILE` guard 實戰通過：在常見 1,024 soft limit 主機上這次跑批會在 3.5 小時分析後的最終步驟死掉。
  - **`workers=4` 結果**（08:19 → 10:03）：wall **6,211 s = 1.73 h**（+38% vs 1.25 h）、**speedup 2.216x**（落在 round 7 預測的 2.06–2.17x 帶內）；success/skipped 10,800/16,765；peak RSS 61.67 GB；**peak GPU 30,439 MB（vs `workers=1` 2,739 MB）**；Phase D 1,200.8 s = **19.3%** of wall（doc 19 記 7.3%）；`overlay_slide.tiff` 5.86 GB；**correctness veto 通過**：`report.csv` 356,255 vs 356,221 rows（−0.01%，一個 tile success→skipped）、amplified cell count 兩者皆 0。
  - **§6.6 意義**：
    1. **19 #7 關閉，放行 gate 滿足——但帶 VRAM 但書**（30,439/32,607 MB = **93.3%、餘裕 ~2.2 GB**；結論：**放行 `workers=4`、32 GB 視為硬底線、不在 worker pool 內加任何 GPU 函式庫、`workers≥5` 在氣球根因前不開放**）；
    2. **Phase D ceiling 近 3x**（`workers=4` 時 19.3% → 1/(1−0.193) = **1.239x** vs doc 19 的 1.078x）→ 推翻 §3.2「GPU 化維持關閉」的建議，BioFormats 約束仍在且不致命（19.2x 是 LZW 數字）；Phase D 成為管線中最大的單一槓桿；
    3. 兩個 crop 導出的 stop-loss（gc、tile read）不應在舊數字上重提、也不該繼續在舊數字上關閉；
    4. **主機需求首次量測**：~64 GB RAM、~32 GB VRAM（`workers=4`）、~350 GB 磁碟/slide、`RLIMIT_NOFILE` ≥ ~28,000。
- **§7 Tier 4 文件漂移全關**：#32 `generate_ihc_core_mask` 參數改 `ihc_image: Union[np.ndarray, Path, str]`，移除 `# pyright: ignore`；#33 `m3_elastic_matching.py` 死引用移除（並在 `m3_module/m3_dot_detection.py` 找到並移除第二個）；#34 `config.py`/`config_example.py` 死引用移除；#35 explainer HTML 更新為 v4（Part B counting 仍正確；Part A matching 重寫含兩階段鎖定圖；參數表錯三處——記載了不存在的 `dish_elastic_max_dist_px` 與 `dot_blue_exclude_threshold`、把仍在用的 `dish_elastic_expand_factor` 劃為棄用；四種 outcome → 三種（「核數 ≥2 多核排除」在一對一配對下不可能）；加上「以 code 為準」的頁首/頁尾）；#36 `tests/test_config_parity.py` 6 cases（parity = config surface：欄位名、順序、型別、非站台特定預設值、模組尾 `Config`/`compute_config_hash`/`config`；站台特定欄位白名單；第六個測試保證白名單不會悄悄藏住改名的欄位——首跑抓到猜錯的 `merge_tile_dir`）；#37 `tests/test_m0_stitch.py` 21 cases（`compute_tile_geometry`：cut line 落在後一 tile 起點半個 overlap、五種格式錯誤 grid 都 raise、`overlap ≥ tile` raise；`core_crop_bounds`：把所有 core crop 累加進全片覆蓋陣列斷言 `min==max==1`；`filter_and_absolutize`：細胞掃過 overlap 帶每個位置恰被認領一次、`cell_id` 不重編號；`clear_slide_edge_cells`：只清指定邊、無邊仍 relabel、全四邊只留內部、輸入不被修改）；#38 codegraph 重核：indexed=20、git-tracked=22、on-disk=37、**幻影 0**，唯一未索引為 gitignored 的 `config.py`。
- **§8 Tier 3.1**：不是工程工作，升級；round-3 `cpdino` 換模型使細胞數 +1.8–2.5%、一個 tile success→skipped；**round 3 以來所有效能成果都建立其上**；至 round 7 未解。
- **§9 刻意不做**：Tier 2.1/2.2、Tier 5、Tier 6、Tier 7、doc 26 §4.2 排除表。
- **§10 Follow-ups**：(1) `perf_measure.py` 非串流路徑已死（一行 import 可修，或刪 flag）；(2) **加入 `cuda_alloc_conf` 進 `Config` 時，最初被納入 `compute_config_hash`，allocator sweep 首跑即 `worker config_hash b147f01b != parent 656ac4c3`**——`_mp_tile_worker` 的 hash 守衛正確運作（parent 修改 config singleton，spawn 的 worker 重新 import 看不到）；修法：`_HASH_EXCLUDE = {"cuda_alloc_conf"}`（只改變「如何執行」而不改「產出什麼」的 runtime 旋鈕不計入 hash），修後 hash 回到 `3d1087f2`（與 round 7 所有 metrics 檔一致），`tests/test_config_parity.py` 直接斷言此性質；(3) registration 階段的 per-modality 畫布尺寸應在上游明文；(4) `pip freeze` 維持（本輪只新增 `pytest` + `pluggy` + `iniconfig`，以 `--no-deps` 安裝）；(5) zstd 決策「只差一次 QuPath session」——（實際已在本輪完成否決）；(6) 24.76 GiB 氣球根因才是真正的活問題。
- **§11 Artifacts**：`measurement/_metrics_r8/`：`stitch_ablate_4gp.json`、`alloc_conf_w6.json`、`worker_timings_probe.json`、`pip_freeze.txt`（並有 `fullwsi_*` 整片資料）；含 reproduce 指令。

---

## 10. Round 9：三條開放線並進——gc round 2（28→31）、Phase D GPU 化（29→32）、tile read I/O（30→33）

> Round 9 最重要的變更不在這三份計畫之中：追查 QuPath 當機時發現**每個在 round 9 之前寫出的 `overlay_slide.tiff`，縮圖金字塔層全部解成雜訊**（見 §10.5 第 (b) 項）。

### 10.1 doc 28 — `gc.collect` round 2 設計

- **回顧**：`gc.freeze()` 只凍結「凍結時已存在」的物件；`per_tile_owned`（每個 tile 追加 `(abs_x, abs_y, owned)` tuple 與其中每個 `CellAnalysisResult`）在凍結後才建立，整片累積 **356,255 個**，被 27,565 次 `gc.collect()` 重掃 → doc 27 量到 80.5 ms/call、2,218.4 s、16.1% wall。
- **根因（讀碼驗證）**：`_record(abs_x, abs_y, owned)` 是單/多 process 共同的 parent 端匯合點，會 append 新物件；doc 15 §2.2 其實已寫「appended to a frozen list are not themselves frozen」。`CellAnalysisResult`（`hybrid_data_types.py`）是純標量 dataclass，無反向引用/巢狀/循環 → **永久凍結無正確性風險**。
- **為何 doc 14/16 沒抓到**：doc 14 §2 對 Option C 不要求 RSS 重驗；沒料到 freeze 的好處有隱藏範圍限制，只在「一次長批次內 per-call 成本趨勢線」才現形。
- **選項**：
  - **H 週期性 re-freeze（首選）**：在既有 `_frozen_gc_generation()` 範圍內再次 `gc.freeze()`（累加式，單一 `finally: gc.unfreeze()` 不變）；cadence 變數應為**累積細胞數而非 tile 數**（避免重蹈 doc 14 Option A 的 crop-tuned-N 脆弱性）；必須先量 `gc.freeze()` 本身的成本（唯一未驗證假設）。
  - **I（後備）**：`_record` 立即把 `owned` pickle 成 `bytes`（不被 GC 追蹤），`_finish_batch` 再 unpickle；僅當 H 的 Exp 1 顯示 freeze 反覆成本不攤提才做。
  - **J（機會式，不排程）**：`checkpoint=True` 時 `per_tile_owned` 在 RAM 與 `_resume/*.pkl` 已重複，可由磁碟串流回來。
- **正確性/記憶體**：freeze 只改掃描範圍不改可達性；`verify_gc_freeze.py` 需從「freeze exactly once」改為「freeze N≥1 次、unfreeze exactly once」。
- **實驗矩陣**：Exp 0 合成累積 harness（不需 GPU）重現 1.2 → 80.5 ms 爬升；Exp 1 量 freeze 本身重複成本；Exp 2 cadence sweep；Exp 3 以 crop 串接數次取得真實物件；Exp 4 整片 `workers=1` 重跑。
- **與 doc 30 的關係**：此修法降低「被追蹤物件數」而非 RSS（凍結不釋放記憶體），不應假設它能把 RAM 還給 page cache；win 在 `gc.collect()` 內的 CPU 時間。

### 10.2 doc 31 — gc round 2 落地（Option H 上線）

- **標題**：Option H 以 doc 28 建議的形狀上線。唯一自陳的未驗證假設**先被量測，結果決定性有利**：`gc.freeze()` 是 **O(1) linked-list splice**，每次 0.00004–0.0006 ms，與已凍結/新追蹤量無關；27,565 次 freeze 共 **1.2 毫秒**。Option I 因計畫自己的停止規則而不建。量測也發現 doc 28 §4 沒指出的風險，它決定了 cadence 不是「每 tile」。**未宣稱整片勝利**（Exp 4 沒跑，只有標明為投影的 §7）。
- **改動**（3 檔、~60 行）：`m0_module/m0_tile_runner.py` 新增 `_PeriodicFreezer`（`note(n_cells)`：累積 `pending`、超過 `every_cells` 就 `gc.freeze()`）、`_REFREEZE_EVERY_CELLS = 5_000`、`_frozen_gc_generation()` 現在 yield 一個 freezer；`hybrid_pipeline.py` 單 process 迴圈在 `_collect` 內呼叫 `refreezer.note(len(owned))`；`scripts/verify_gc_freeze.py` 擴充。
- **為何 hook 放在 `_collect` 而非 `_record`**（對計畫的範圍更正）：`workers>1` 時 `_record` 由 `_run_tiles_multiprocess(on_tile=_record)` 在 parent 呼叫，**在 `_frozen_gc_generation()` 範圍之外**，那裡 `gc.freeze()` 會成為沒有對應 unfreeze 的 freeze（API server 無界 RSS）。這同時收窄修法範圍：**回歸是 `workers=1` 現象**——multiprocess workers 每 tile `gc.collect()` 但不累積；parent 累積但不 per-tile collect。
- **Exp 0**（`scripts/gc_refreeze_probe.py`，真實 `CellAnalysisResult`、27,565 tile、55.8% 背景、29.24 cells/組織 tile → 356,255 cells）：frozen once 369.4 s，per-call 0.001 → **31.2 ms**；never frozen（上限 5,000 tile）81.2 s；絕對值比實測 80.5 ms 低 ~2.6x（harness 不產生 numpy/torch 暫態垃圾），僅形狀可信。
- **Exp 1**（freeze 本身成本）：cadence 1/10/100/1,000 tile → 27,565/2,757/276/28 次、per call 0.000043/0.000055/0.000088/0.000596 ms、全片總計 0.0012/0.0002/0.00002/0.00002 s——**flat**；**freeze is free**，doc 28 預測的「too-frequent freezing 反覆付 O(n)」被證偽。
- **Exp 2**（cadence sweep，全 27,565 tile 的 collect + freeze 總成本）：每 1/5/25/100/500/2,000 tile → 0.004/0.020/0.105/0.490/2.942/12.072 s；每 100/500/2,000/10,000/50,000 cells → 0.031/0.164/0.757/4.292/27.833 s（freezes 3,046/676/176/35/7）；none（今日）416.7 s。**doc 28 預測的 U 形沒有出現，只有單調斜坡**（記錄為計畫預測失敗）。cadence 因此由另一個 doc 28 §4 沒指出的風險決定：**`gc.freeze()` 凍結「當下所有被追蹤物件」，不只想釘住的結果**；任何暫態物件（背景臂的 in-flight `ChunkResult`、torch/cellpose 內部）若之後成為不可達循環就不會被回收直到 `run_batch` 返回。每次 freeze 是一次曝險：每 tile 一次 = 27,565 次、每 5,000 累積細胞 ≈ 71 次。故 `_REFREEZE_EVERY_CELLS = 5_000`（collect 成本仍可忽略 ~2.1 s）；以 cells 而非 tile 計：背景 tile 貢獻 0 cell，安靜區域不觸發 freeze（有測試 2b）。
- **invariant guard** 擴充（freeze 次數 = entry 1 + 每次 cadence crossing；unfreeze 恰一次；全背景批次只在 entry freeze 一次；mid-loop fail-fast 後仍恰好 unfreeze 一次；期望值讀 `TR._REFREEZE_EVERY_CELLS` 不重複寫死；harness 之前 stub 回傳 `[]` 現在偽造 29 個 `CellAnalysisResult`/tile；第 3 節改在中途失敗）；**修兩個既有問題**：`run_batch` 預設 `workers=4` 使腳本默默跑到 multiprocess 路徑，改顯式 `workers=1`；`--control-ref` 改為可選（沒有早於 freeze 變更的 ref 存活於 squash 後的歷史；直接對 tile 數斷言「每 tile 一次 collect」更強）。
- **Exp 3**（match24，576 tile、55.9% 背景，`perf_measure.py --no-refreeze` 只拿掉 cadence、保留 entry freeze）：control per-call decile means 0.47 → 1.37 ms（末/首 **2.94 爬升**）；Option H 0.47 → 0.33（**0.71 持平後下降**，約 5,000 cells 處 cadence 觸發）；`B4_gc_collect` 0.447 → 0.363 s（−18.8%）；RSS 4.338 → 4.309 GB；stats 224/352 相同；**e2e wall 193.6 vs 200.2 s 是雜訊**（機制僅佔 0.084 s；doc 33 後來在同 anchor 量到同配置 188.2 ± 0.22 s，兩者皆是冷 page cache 的前兩次）。需要新增 per-call `gc.collect()` 序列：bucket 總計無法區分「flat 且便宜」與「爬升但短」——這正是回歸能活過 round 1–7 的原因。
- **Exp 4 未跑**（3.82 h）。**投影**：harness no-refreeze 416.7 s vs 實測 2,218.4 s（5.3x）；5,000-cell cadence 下降至 ~2.1 s；剩餘為 per-call 底 ~1.2 ms × 27,565 ≈ 33 s → 約 2,185 s of 13,762 s = 15.9% → ceiling **~1.19x**（**之後 round 10、11 在整片實測 1.186x，幾乎完全命中**）。
- **處置表**：H 上線；I 不建；J 仍 flagged；Exp 0/1/2/3 完成；Exp 4 未跑；correctness veto 通過；`verify_gc_freeze.py` 更新。Follow-ups：Exp 4 應與 doc 33 Option K 與 L ablation 同一次整片跑；觀察該次 peak RSS（~71 次 freeze 是真實曝險）；`_REFREEZE_EVERY_CELLS` 是模組常數非 config（刻意）。

### 10.3 doc 29 — Phase D GPU 化設計

- **背景**：Phase D 真實成本 1,185.4 s（`workers=1` 8.6%）/ 1,200.8 s（`workers=4` 19.3%）；synthetic probe 322.7 s 為 3.7x 樂觀；便宜旋鈕全死，`zstd` 被 QuPath 否決；naive ceiling 1.094x/1.239x；nvImageCodec 1,789.6 vs 93.0 MP/s = 19.2x，但無 pyramid/BigTIFF；VRAM 免疫（parent 內跑一次）。
- **§1 核心分析：19.2x 不轉移為 19.2x（甚至 1.239x）的 Phase D 勝利**：19.2x 是微基準（包含「無 pyramid vs 有 pyramid」差異）；1.239x 是「Phase D 歸零」的 Amdahl。拆 doc 27 §3 消融表：baseline encode 57.02 s；`BOUND_no_pyramid` −12.79 s（pyramid ≈22.4%）；`BOUND_no_compression` −13.06 s（LZW ≈22.9%）；兩者近似獨立 → 壓縮+pyramid ≈45%，**剩餘 ≈55%（container/tile-buffer/TIFF 結構組裝）兩個 bound 都碰不到**。校正後 ceiling：只換壓縮 ≈1.052x；壓縮+GPU pyramid ≈**1.093x**（`workers=4`）；非 1.239x。
- **硬約束**：必須 BioFormats 可讀（LZW、tiled、pyramidal、BigTIFF；非 zstd）。
- **未解技術問題（不憑斷言回答）**：是否有維護中的 Python TIFF 寫入庫接受外部預壓縮 tile bytes（`tifffile` 最可能）；否則退回「GPU 只做 pyramid（~22%），壓縮+container 仍在 CPU」。
- **Phase 1 spike**（獨立腳本，重用 `stitch_probe.py`/`gpu_codec_spike.py`）：4.055 GP/6,900 tile 合成輸入 → nvImageCodec 編 LZW → 以 §2.2 的答案組成 pyramidal BigTIFF → **計時整體（encode + assembly）**→ 以與 zstd 相同的 QuPath 協議驗證無損與可讀性。**決策點**：Phase 1 是否「顯著高於」1.05x/1.09x；Phase 2 只有 Phase 1 通過才建。**correctness 是 veto 不是平手判準**（BigTIFF 組裝錯誤 — 壞 offset、畸形 IFD — 也會表現為「打不開」）。

### 10.4 doc 32 — Phase D Phase 1 spike 結果（round 9）

- **標題**：doc 29 的兩個問題答案相反——container assembly 其實**已被專案現有依賴解決**（`tifffile` 接受預壓縮 tile bytes），但 **nvImageCodec 產不出那些 bytes**（TIFF encoder 拒絕 tiling、strip layout 內部固定）。19.2x 路線**因輸出形狀而死**（非速度）。doc 29 §1.2 的「≈55% 不可降的 container 組裝」是 **pyvips 的假象**：`tifffile` container 寫入只占 ~1%。最終勝者：**CPU codec + `tifffile` container + GPU 生成的 pyramid 層**，**略高於** doc 29 §1.3 校正 ceiling（1.109x vs 1.093x @`workers=4`）。
- **Q1 預壓縮 passthrough → YES**：`tifffile.TiffWriter.write` 接受 `Iterator[bytes]` 作為 `data`（需與 `compression/predictor/...` 相容）；已驗證 LZW + horizontal differencing 的 tile 餵入後產出 tiled BigTIFF 讀回 pixel-identical；`tifffile` 2026.3.3 已是專案依賴。**無聲陷阱**：`tifffile` 對預壓縮輸入只「宣告」`predictor` tag、不實際套用轉換——宣告 `predictor=True` 卻傳未差分的 bytes，會得到打得開但解碼為垃圾的檔案。
- **Q2 nvImageCodec 能否產出 tile bytes → NO**：(a) `EncodeParams(tile_width, tile_height)` → "Tiling is not supported with TIFF encoder."，encode 回 `None`；(b) 內部 strip 約 8 KB/strip（256² → RowsPerStrip 10、26 segments；512² → 5、103；1024² → 2、512；4096² → 1、**4,096**）；(c) 獨立 LZW stream 不可串接（各自以 EOI 結束；實測串接兩個 stream 只解出 6,144 位元組中的 3,072）。19.2x 數字仍真，但無法用於本 pipeline 需要的輸出形狀。**對 doc 25 §8.1 的更正**：`nvidia-nvtiff-cu13`「無 Python module」準確但誤導——它正是 nvImageCodec 的 TIFF codec 來源，應改寫為「無 Python binding，但 nvImageCodec TIFF 支援所必需」。
- **spike**（`scripts/phase_d_container_spike.py`；0.268 GP/16,384² slab，best of 3；輸出形狀檢查：8 pages / 128×128 tiles / LZW / BigTIFF / 解析度 tag 與 baseline 一致 / **每個金字塔層 pixel-identical**——第一次跑因 256 px tile 且 pyramid 停在 512 px 而「做得比較少所以較快」，故改為 shape-matched）：

| 候選 | wall | vs pyvips | pyramid | encode | **container** | 輸出 |
| --- | --: | --: | --: | --: | --: | --: |
| A `pyvips_tiffsave`（今日） | 2.84 s | 1.000x | — | — | — | 42.38 MB |
| B `tifffile_cpu`（無 GPU） | 2.028 s | **1.4004x** | 1.01 s | 0.996 s | 0.022 s | 42.10 MB |
| C `tifffile_gpu_pyramid` | **1.323 s** | **2.1466x** | **0.081 s** | 1.222 s | 0.021 s | 42.15 MB |

  - container 組裝 0.021 s ≈ C 總時間的 **1.6%**；GPU pyramid 比 CPU 快 ~12.5x（1.010 → 0.081 s），這一個替換就是 B 與 C 的全部差異。
- **ceiling 重算**（套到 Phase D 真實成本，join ~5% 不動）：B：Phase D → ~863 s、省 ~322 s、`workers=1` 1.024x、`workers=4` **1.055x**；C：→ ~583 s、省 ~602 s、**1.046x / 1.109x**；doc 29 估 1.052x / 1.093x。**C 贏過較高估計 1.6 個百分點、B 約等於較低估計**——只夠證明 Phase 1.5，非 Phase 2 綠燈。
- **QuPath 檢查跑了兩次**：第一次 A、B 開得起來、**C 讓應用程式當機**，且 A 與 B/C 顯示尺度不同；修正後三者渲染一致、C 的 correctness veto 清除。**兩個真缺陷**：
  - **(a) spike 自己的 predictor 編碼錯誤**：`np.diff(t, axis=1, prepend=t[:, :1])` 讓每列第一個輸出值為 0 而非像素絕對值；正確：`out[0] = in[0]; out[k] = in[k] - in[k-1]`。所有自動檢查都漏掉，因為合成影像每 64 px 畫黑線、容器 tile 128 px → 每個 tile 第一欄剛好是 0，使 buggy 與正確編碼在 level 0 相同；`verify()` 只查 level 0；縮小層錯誤，C 的 pyramid 退化到 level 7 純黑（min=max=mean=0）導致 QuPath 當機。**錯誤檔案也「更小」**（退化層幾乎不佔空間）——候選同時「更快又更小」應該引起更仔細檢視。已修（encoder prepend 0；`verify()` 檢查**每一層**）。
  - **(b) 既有、已出貨的 `_stitch_overlay_slide` 金字塔層壞掉（更嚴重）**：libvips 8.15.1 `tiffsave` 預設 `predictor="horizontal"`，只在**全解析度 IFD** 寫 tag 317 Predictor=2，卻對縮小層像素資料仍做水平差分 → 任何依各 IFD 自己 tag 解碼的 reader 無法還原層 1..N。在**真實 pipeline 產出的 `overlay_slide.tiff`** 上量：L0 predictor 2、decoded mean 242.3、0.5% 零像素；L1 predictor **absent**、mean 12.8、**90.8%** 零像素；L2–L8 absent、mean 16.3 → 83.0、88% → 37% 零像素。這**不是 reader 嚴格度**（`tifffile`、libtiff via Pillow 都解成雜訊）；手動逐 tile 還原 L1 精確重現源的 2×2 box shrink（max|Δ|=0）。**影響**（QuPath 中確認）：level 0 正確，故檔案開得起來也看起來對，這就是它活過八輪與所有測試的原因；**病理醫師縮小（zoom out）時錯誤**。**修法**：`predictor="none"`；真實 overlay（18,688² level 0）：horizontal 93.35 MB/4.14 s/9 層中 **8 層解成雜訊** vs none 102.08 MB/4.18 s/**9 層全對**（**+9.4% 磁碟、時間中性**；doc 27 §3.2 已量到速度中性，但當時沒檢查是否「正確性中性」）。較便宜替代（只補 tag 317）需要手寫 BigTIFF IFD，因風險被否決；若 Phase 2 上線，`tifffile` 對每層寫對 tag 問題自然消失。`tests/test_stitch_pyramid_levels.py` 2 tests（真 `_stitch_overlay_slide` 輸出每層需亮且非退化；Predictor tag 跨 IFD 一致；無修法時 level 1 mean 6.9 → 失敗）。
  - **5.2 仍待**：**所有 round 9 之前產出的 `overlay_slide.tiff` 金字塔層都是壞的，需重新縫合**（`_stitch_scratch/` 仍在時 stitch 可單獨重跑；`run_batch(checkpoint=True)` 的 checkpoint 覆蓋所有 tile 時直接跳到 `_finish_batch`；scratch 已清則需重跑批次）；整片輸出尚未重生並重開。
- **§6 scale gap**：spike 在連續記憶體 slab 上跑，doc 29 要求的是 4.055 GP/6,900 tile 規模；真實 Phase D 從不把輸入放進記憶體（27,565 tile 檔經 pyvips 惰性 join 避免 ≈400 GB canvas）；`tifffile` 版需要等價的 lazy source（依 container 寫入順序讀 `_stitch_scratch/`）——**未 scoped 的真工作**；spike 不量測 read/join 這一半；`stitch_probe.py` 在 M0 split 後壞掉，本輪修復（doc 33 §5）。
- **處置**：§2.2 Q1 YES；Q2 NO；Candidate A（nvImageCodec GPU encode）**關閉**（因輸出形狀）；§2.3 Phase 1 spike 完成（`measurement/_metrics_r9/phase_d_container_spike.json`）；QuPath 步驟「failed → root-caused → fixed → re-checked」；Phase 2 未開始（閘門變成 §6 的 read/join 與 §4 的薄邊際）；`zstd` veto 不變；**`_stitch_overlay_slide` 金字塔層修復**。**建議順序**：重生舊 `overlay_slide.tiff` → 以修好的 `stitch_probe.py` 在 4.055 GP 從磁碟取真 tile 重跑 spike → 才 scope Phase 2；B 無需 GPU，若在規模與 QuPath 下成立，比任何 GPU 版小且低風險，doc 29 §3 的「prefer simplest」會偏向 B（除非 C 邊際值得 torch 依賴）。

### 10.5 doc 30 — `B2r_tile_read` / precut scratch I/O 設計

- **Discover（先前紀錄）**：doc 18 §6.3（stop-loss 1.22%/1.012x）→ doc 19 #6 重開 → DISCOVERED #13/#44 → bottleneck-list 17.2% → doc 27 §6.4；**尚無人修**。
- **Analyze 1（讀碼驗證）**：`_read_rgb`（`skimage.io.imread` + 通道正規化 + `astype(uint8)`）在 `_process_precut_tile_gpu` 頂端於**MAIN 執行緒同步**呼叫兩次（IHC、DISH），**先於**該 tile 的 GPU 前向；`PrecutStream._cut` 的 8 執行緒 pool 寫檔那一側已由 Precut A 解決，**讀回那一側從未處理**。**更正紀錄**：bottleneck-list 說它「在有 slack 的臂」——依程式碼它其實在 **MAIN/critical 臂**；該說法應讀成「若搬到背景執行緒，就能與有 slack 的臂並行」（計畫性而非描述性）。
- **Analyze 2（為何整片變差 14x 與 doc 28 的關聯）**：crop 規模整份 scratch 能放在 OS page cache，讀取等於讀 cache（所以 1.22%）；整片 scratch ~49 GB，而 process RSS 同時峰值 61.13 GB（主要來自 `per_tile_owned` 累積與 stitch 持有的 27,565 個 pyvips handle）→ **行程 RSS 與 page cache 競爭**，剛寫的 tile 在幾個 tile 後讀回之前就被驅逐。doc 28 降低追蹤物件數而非 RSS，**不該假設它順便修好這個**。
- **選項**：**K**（免費）doc 28 上線後重量；**L**（推薦首建）一個 tile 超前預讀（與 Precut A 同構）：doc 27 §6.4 數字：`B1_m3b_cellpose` 6,358.9 s/24,360 次 ≈261 ms/tissue-tile；`B2r_tile_read` 2,368.5 s/55,130 次 ≈43 ms/tile；但**平均混合了組織與背景 tile**，背景 tile（55.8%）只剩 UNet（`B1_unet_coremask` 734.8 s/27,565 ≈27 ms/tile）可躲，**組織/背景分拆未量**；設計：背景 helper 為 N+1 tile 呼叫 `_read_rgb`，重用或另設單執行緒 pool；順序無風險；**fail-fast 必須保留**（預取讀取失敗要在該 tile 被消費時浮現）；無新依賴、無 VRAM 成本、`workers=1` 與多行程同等受益。**M**（`workers=1`；larger）：`PrecutStream._cut` 手上已有 in-memory `pyvips.Image`，分析迴圈卻從磁碟讀回（另一個庫 `skimage`）——若直接交出像素資料可省整個往返，ceiling 到 17.2%，但需解兩個開放問題：多行程相容性（`spawn` 跨進程傳 `pyvips.Image`/大 numpy 需 IPC 設計）與 bit-exact 重驗（`m0_reader` 既有「逐位元一致」只涵蓋「讀檔」）；**N**（記錄不排程）：N1 更便宜的 scratch 壓縮（deflate → 更快 codec 或不壓縮；**無 BioFormats 約束**因為 scratch 是 pipeline 內部）；N2 以 pyvips 讀回；N3 `workers=1` 完全移除 precut scratch（與 resume/多行程耦合；應與 Candidate F 的 ~157 GB 與 doc 27 §6.2 的 ~513 GB 磁碟足跡一併討論）。
- **Choose 順序**：K → 量組織/背景拆分（延伸 doc 27 §4 的 worker-timing）→ L → 僅 L 令人失望才 M → N 不排程；成功準則為在實際（或接起來的）多千 tile 跑批上 `B2r_tile_read` 占 wall 的比例，永遠不是 `_read_rgb` 的微基準；ablate K → L → M 個別歸因。

### 10.6 doc 33 — tile read I/O 落地（Option L 上線）

- **標題**：Option K（整片重量）因需 3.82 h 跑批無法做；先做 §4 step 2——結果：**讀取成本集中在組織 tile（62.7%），而每個組織 tile 自身有 ~41x 的 GPU 工作可躲**。Option L 建成、測試、ablate；M 與 N 未建。
- **Step 2 量測**（`perf_measure.py` 把每次讀取按檔案路徑暫存，`_process_precut_tile_gpu` 返回時（`chunk is None` ⇒ 背景）回溯歸屬；match24，576 tile、55.9% 背景，pre-L 程式碼）：組織 254 tile/508 reads/3.561 s/**62.7%**/14.02 ms per tile/GPU 工作 ~574 ms（UNet 29.4 + 2× Cellpose 544.5）；背景 322 tile/644 reads/2.195 s/37.3%/6.82 ms/~29.4 ms（UNet only）；全 576/1,152/5.756 s/9.99 ms。有利結果：組織 tile 的讀取只是其 GPU 工作的 **2.4%**；背景 tile 較難（23%），但只占 37.3% 的成本。背景 tile 近乎全白，deflate 檔小且解碼快——此拆分是**內容屬性**，應可轉移到整片，只是量級不行。
- **Option L 實作**：`m0_tile_runner.py` 新 `prefetch_tile_reads(tiles, pool, depth=1)`（deque 維持 depth+1 個 `pool.submit(_read_tile_pair, ...)`）；`_process_precut_tile_gpu` 新增可選 `preread` 參數（其他 5 個呼叫點已檢查、保持原內嵌讀取）；`run_batch` 單 process 迴圈用**第二個專用單執行緒 pool**（`tile-read`），而非共用 `tile-cpu`（共用會讓讀取排在上一 tile 的 CPU 尾巴之後，把要重疊的東西序列化）。
  - **fail-fast 完整保留**：讀取例外在 `read_fut.result()`、**該特定 tile 被消費的那一刻**重新拋出，轉成同一條 `preread = None` → `tg = None` 路徑，批次同樣中止；排除兩種失敗模式（吞掉例外使批次越過缺洞、或對錯 tile 浮現）。`tests/test_tile_read_prefetch.py` 6 tests（每 tile 恰好依序 yield 一次；空 stream；單 tile stream；預取 submission 順序（斷言 submission 而非 execution，因為真執行緒池非確定）；失敗讀取為該 tile 重新拋出且不影響其他；`preread` 確實繞過內嵌讀取，否則預取會讓每次讀取加倍而非搬移）。記憶體有界（depth=1 → 最多兩個 tile 的像素資料 ~12 MB）；處理順序不變。
- **Ablation**（`perf_measure.py --no-prefetch` 還原內嵌讀取；3 次交錯 A,B,A,B,A,B）：control 188.06/188.08/188.55（均值 188.23 s，sd 0.22）vs prefetch 184.90/186.25/183.63（184.93 s，sd 1.07）→ **−3.30 s、−1.75%、1.018x**，兩組範圍不重疊；`stats` 六次全相同（224/352）；RSS 4.26–4.33 vs 4.31–4.39 GB。機制自我檢查：從關鍵路徑移除的讀取成本 5.605 s = 2.98% wall，實測節省 1.75% → 回收 ~59%（符合預期，背景 tile 無法完全藏住讀取）。bucket：`B2r_tile_read` 5.605 → 7.111 s（組織 3.503 → 4.454、背景 2.102 → 2.657）；**bucket 總量上升而 wall 下降是工作被成功搬到平行臂的簽名**（讀取與主執行緒 GPU 前向競爭 GIL/CPU，每次牆鐘變長但不在關鍵路徑）。
- **全片投影**（標明為投影）：doc 27 的整片彙總 2,368.5 s over 12,184 tissue / 15,381 background，以實測 per-read 比例 2.06 套用：組織 120.5 ms/tile → 1,468 s（**全部可藏**，120.5 ms vs ~574 ms）；背景 58.6 ms/tile → 901 s（~29.4 ms of 58.6 可藏 → 452 s）；總 2,369 s，可藏 ~1,920 s = **81.1%** → 13.9% of 13,762 s → **~1.16x**。局限：背景 tile 空間聚集成片，一個 depth=1 預取會讀取受限；`depth` 已是參數但未量測不改。
- **對紀錄的更正**：bottleneck-list「on the arm with slack」現在作為計畫才成真（讀取已跑在背景執行緒）；**`scripts/stitch_probe.py` 在 M0 split 後壞掉卻無人發現**（`from m0_reader import ...`）→ 修為直接從 `m0_module.m0_stitch` 匯入（doc 29/32 依賴它）。
- **處置**：K 未做；step 2 完成；L 上線（單 process 路徑、depth=1、專用 `tile-read` pool、6 tests）；M 未建；N1/N2/N3 未建；correctness veto 通過。**Follow-ups**：整片 `workers=1` 一次跑同時解 gc Exp 4 + Option K + Phase 2 決策；背景 tile 連續時 `depth=2–3`；**多行程 worker 仍內嵌讀取——`prefetch_tile_reads` 套進 `_mp_tile_worker` 是剩下最高價值的一塊**（`workers=4` 是出貨建議）。

---

## 11. Round 10：關閉 round 9 的待辦（doc 34 規劃、doc 35 落地）

### 11.1 doc 34 — round 10 計畫（純規劃）

- **Discover（round 9 三份落地文件留下的 7 項）**：(1) 任何早於 `predictor="none"` 修復的 `overlay_slide.tiff` 金字塔層解成雜訊（正確性缺陷，非效能）；(2) `_mp_tile_worker` 仍內嵌讀取，`prefetch_tile_reads` 只接在單 process（doc 33 稱為「最高價值的剩餘部分」）；(3) `stitch_probe.py` 尚未在 4.055 GP 從磁碟取真 tile 重跑（「read/join 這一半」，doc 32 §4 的 1.109x/1.055x ceiling 取決於此）；(4) 整片 `workers=1`（3.82 h）一次跑同時解 doc 31 Exp 4 與 doc 33 Option K；(5) 背景 tile 成片時 `depth=2–3`；(6) Option J；(7) Option M/N。6、7 無新動作。
- **Analyze**：(1) 不在 Amdahl 序列內（任何人縮小就會看到雜訊）；(2) 與 (4) 透過「production 是什麼」互相作用：`workers=4` 才是出貨預設，Option L 全部量測證據都屬於 production 不跑的單 process 路徑；先出貨 (2) 再花 3.82 h 做 (4)，才不是只驗證沒人部署的組態；(3) 比 (4) 便宜且獨立（4.055 GP/6,900 tile，只動 stitch）；(5) 在 (4) 下游。
- **順序**：1 重生壞 overlay（先做，correctness）→ 2 `_mp_tile_worker` 接 prefetch（先做 crop `workers=4` 交錯 ablation）→ 3 `stitch_probe.py` 4.055 GP（可與 1、2 並行）→ 4 整片 `workers=1`（doc 31 Exp 4 + doc 33 Option K 一次解決）→ 5 `depth=2/3`（僅 4 顯示背景成片時）。2.2：多行程 worker 的 fail-fast 路徑（轉成 `("error", ...)` 訊息而非 raise）必須用 `codegraph_explore` 重新追蹤，不可假設與 `run_batch` 同形。2.4：預算視為單一 3.82 h 跑批，不要在 2 出貨前啟動（否則生產路徑改變後還要第二次 3.82 h 重驗，犯「一次改好幾個」的反模式 #7）。

### 11.2 doc 35 — round 10 落地

- **標題**：五步全部解決。Step 1：兩個出貨的整片 overlay 以直接量測確認壞掉並重生驗證乾淨；Step 2：`_mp_tile_worker` 現在預取，正確性以測試鎖定，但六次交錯 `workers=4` 重複量到 −0.83%（**95% 下無法與雜訊區分**，雖然效應大小剛好符合機制預測）；**Step 3 推翻 doc 32 §4**：加入 read/join 後 candidate B、C 比出貨的 pyvips baseline **更慢**（0.788x、0.884x；記憶體 spike 曾量到 1.400x/2.147x），Phase 2 as specified 已死；**Step 4（三份文件在等的整片跑批）1.347x（13,762.5 → 10,217.7 s）**並如預測結清 doc 31 Exp 4（`gc.collect` 2,218.4 → 58.8 s、70 次 freeze、peak RSS 反而下降 1.09 GB）；doc 33 Option K 只結清一部分（無關 commit 污染讀取 bucket 絕對大小）；**Step 5 閘門達成但獎金僅 1.04% wall，量測後刻意不建**。意外發現三個缺陷、外加一個未解釋的迴歸。
- **Step 1 壞 overlay**（`scripts/overlay_pyramid_audit.py` 新增，套用 `tests/test_stitch_pyramid_levels.py` 兩個檢查到已部署檔案）：`fullwsi_w1`/`w4` 修前 Predictor tags `2,1,1,…,1`、L3 decoded mean 25.03/**81.1% 零像素**、L5 35.34/73.3%、L8 81.82/36.8%、L11 69.94/47.5% → **FAIL**；`fullwsi_w1` re-stitch 後 `1,1,…,1`、L3 **240.71/0.0%**、L5 240.76、L8 240.88、L11 241.14 → **PASS**。修復不需重算批次：`_stitch_scratch/` 成功即刪，但兩個 run 都在鄰近資料夾保留每 tile annotated overlays（27,565 tile、46 GB）；audit 腳本 hard-link 成新 scratch，使 `rmtree` 只刪 link；**re-stitch 1,352.9 s，檔案 5.86 → 7.50 GB（+28.0%）**（doc 32 在 18,688² crop 量到 +9.4%，整片大三倍——值得記錄，因為大家會外推 doc 32 的數字）；舊檔保留為 `overlay_slide.tiff.predictor-broken`；`fullwsi_w4` re-stitch 1,315.0 s，也 5.86 → 7.50 GB，在 step 4 的 run 離機後才做（22 分鐘 pyvips stitch 峰值 ~45 GB RSS，會污染 step 4 的 peak RSS/page cache）；兩片重生後各層 decode 一致（means 240.71 → 241.14、零像素 0.0%）；**Step 4 自己的新輸出**由修好的程式碼直接寫出也 audit 乾淨（Predictor 在 12 個 IFD 一致）。
- **Step 2 多行程 prefetch**：`_mp_tile_worker` 在自己的專用 `tile-read` pool（每 worker 一個）驅動 `prefetch_tile_reads`；新增 `_drain_task_queue`（把跨 process 工作佇列加 poison sentinel 轉成 iterable；每 tile 仍恰被一個 worker `get` 一次）；一個行為變化：預取 worker 從共享佇列多拿一個任務（四個 worker 多持四個 tile 對 27,565 無關）。**fail-fast 路徑確實與 `run_batch` 不同**——worker 把 `None` chunk 轉成 result queue 上的 `("error", …)` 訊息而非 raise；預取例外在該 tile 被消費時於 `read_fut.result()` 重拋、記錄於該 tile、轉成同一條 `preread = None` → `tg = None` → worker abort 路徑。`tests/test_mp_tile_read_prefetch.py` 6 tests（佇列轉接形狀與 poison、control arm 與預取 generator 契約一致含例外浮現位置、worker 依序恰處理每 tile 一次且用預取像素、失敗讀取為自己 tile 中止批次、ablation env var 到達 worker），全套 60 → 66 tests。
  - **ablation 開關必須是 env var**：`perf_measure.py --no-prefetch` 只改 parent namespace 的 `prefetch_tile_reads`，`spawn` worker 重新 import 拿到原版（與 doc 25 §2.2 對 `_save_tile_array` 的陷阱相同；control arm 悄悄無效會使兩臂相同）→ worker 讀 `HYBRID_MP_NO_PREFETCH`（沿用 `HYBRID_MP_WORKER_PROBE` 先例）；`--no-prefetch` 現在兩者都設，有測試鎖定。同時修兩個量測缺口：`wrap_tile_read_split` 之前只 patch `hybrid_pipeline` 的 `_process_precut_tile_gpu` binding → 組織/背景拆分 bucket 在 `workers>1`（production）全空，現在也裝到 `m0_multiprocess` binding；Option H freeze 次數從 `run_batch` 外讀不到 → `perf_measure.py` 現在回報 `gc_refreeze`。
  - **結果（match24，`workers=4`，6 對交錯）**：stats 12 次全相同（224/352）；RSS 11.70–11.81 GB；control 105.30 s（sd 1.16）vs prefetch 104.43 s（sd 0.84）→ **−0.83%、1.0084x，範圍重疊**；前 3 對 −0.42%（重疊）、後 3 對 −1.24%（不重疊，page cache 暖機：sd 1.74 → 0.52、0.89 → 0.08）——後半是事後挑選，只當一致性檢查；**|Δ|/SE = 1.50（需 ~2）**。**機制無歧義**：`B2r_tile_read` 5.769 → 6.626 s（**+14.9%，六對全部皆然**）而 wall 不升。效應大小正好符合算術：`workers=1`（doc 33）可藏讀取 3.06% wall → 實省 1.75%（回收 57%）；`workers=4`（此處）可藏 1.37% → 實省 0.83%（回收 61%）。**處置：保留，不宣稱勝利**（576 tile anchor 的 scratch 放得進 page cache、`B2r_tile_read` 僅 3% wall，是量測的錯誤尺度）。doc 34 §1 希望「step 2 先出貨可讓 step 4 的昂貴跑批旁邊有 production path 的證據」——**證據回來不確定**；排序仍對（不需第二次 3.8 h 重驗）。
- **Step 3 Phase D 在規模**（`stitch_probe.py --candidates`：B/C 取得 lazy source——由上而下的 band walk，一次 join 一列 source tile（4 GP 時 ~163 MB）、切成容器 128 px tile 列、同時推過 pyramid 縮減；另有 **read-only arm** 只 drain band source 而不做其他事，第一次直接量測所有候選共享的成本）；輸入 6,900 個真實 annotated overlay tile、4.055 GP、annotated 44.74%：read-only 30.44 s；**A `pyvips_tiffsave` 62.55 s（2.44 GB，1.000x）**；**B `tifffile_cpu` 79.33 s（read 36.87、pyramid 18.39、encode 18.22、container 0.55；1.89 GB；0.788x）**；**C `tifffile_gpu_pyramid` 70.71 s（read 36.04、pyramid 6.59、encode 22.23、container 0.54；0.884x）**；皆 shape-matched（11 pages、128×128 tile、level-0 同形）且通過每層 decode audit。**讀取占 A 整體 wall 的 48.7%**（30.44/62.55）→ A 的 encode+pyramid+container 只有 ~32 s；C 等價工作 29.4 s、B 37.2 s。doc 32 的「container 非不可降、是 pyvips 特有」作為 container 寫入的陳述仍成立（0.55 s、<1%），但**不是更快 Phase D 的路**：pyvips 把那段時間花在「讓讀取與編碼重疊」，串流候選卻序列付出。**假設性**：若 B、C 的讀取完全藏在自己的編碼之後（沒人寫過的執行緒實作）→ 42.5 s（**1.47x**）/ 34.7 s（**1.80x**）；套到 doc 27 §6.5 `workers=4`（Phase D 1,200.8 s of 6,211 s）→ 端到端 ~1.094x，低於 doc 32 §4 的 1.109x。**決定：Phase 1.5 完成並否定地結案，Phase 2 不被證成**；doc 32 §7「若 B 在規模下成立就偏好 B」——B 是三者中最差。red flags #1（最佳化版比最笨 baseline 慢）與 #5（spike 只是問題一半的微基準）。QuPath 開啟驗證仍未跑（需要人；輸出保留於 `/home/taro/r10_phase_d/`；因無候選被採用，不再決策相關）。
- **Step 4 整片 `workers=1`**（無 `--resume`：續跑做較少工作、wall 不可比；分析 2 h 50 m + stitch 19 m 40 s）：wall 13,762.5 → **10,217.7 s（1.347x、−25.8%）**；stats 10,801/16,764 → 10,800/16,765（一個 tile 在 `core_mask.sum()==0` 閾值上的 GPU 非決定性）；peak RSS 61.13 → **60.04 GB（−1.09 GB）**。
  - **歸因與必須先排除的混淆**：`d6592c3` 在兩次 run 之間移除所有 per-tile 中繼寫入（baseline 在磁碟留下 275 GB 的 `instance_mask`、`dish_nucleus_mask`、`cell_crops`、`masked_ihc`、`core_mask`）。依 `wall ≈ max(MAIN, BG) + outside` 拆 arm：**MAIN 13,084.7 → 9,493.9 s（−3,590.8）**；BG 5,292.9 → 2,826.9 s（−2,466.1）；`tile-read`（新臂）1,581.2 s；outside（Phase D）1,185.4 → 1,182.4 s；**wall −3,544.8 s ≈ MAIN −3,590.8 s**。`d6592c3` 刪掉的 2,466 s 寫入**全在 BG 臂**（對 MAIN 有 ~7,800 s slack）→ 對 wall 貢獻 ≈ 0（反直覺：刪 275 GB 寫入沒讓 pipeline 變快，因為它們從不在關鍵路徑）。MAIN 變化：`B4_gc_collect` **Option H −2,159.6 s**；`B2r_tile_read` 完全離開 MAIN（**Option L −2,368.5 s**）；`B1_m3b_cellpose` **+751.0 s**；`B1_unet_coremask`/`BM1_*`/其他 +144 s。
  - **doc 31 Exp 4 結清且投影準確**：`B4_gc_collect` 2,218.4 → **58.8 s（−97.3%、37.7x）**；per-call decile means 1.2 → … → 80.5（爬升）→ **16.19 → 0.94 → 0.41 → 0.42 → 0.40 → 0.39 → 0.41 → 0.39 → 0.75 → 1.04**（末/首 decile 0.06，首 decile 16.19 ms 是模型 init 垃圾在第一次 freeze 前被收集）；freeze 次數 1 → **70**；peak RSS 下降。doc 31 §7 預測「~2,185 s、15.9%、1.19x」，實測 **2,159.6 s = 15.7%、1.186x**；預測的 ~33 s 底線實為 58.8 s（~2.1 ms/call，同量級）；doc 31 §4 的 RSS-pinning 疑慮答案是 **否**。
  - **doc 33 Option K：關鍵路徑主張成立；孤立數字不成立**：`B2r_tile_read` 2,368.5 s 內嵌於 MAIN → 1,581.2 s 在 `tile-read` 執行緒並行；−2,368.5 s 對 MAIN = baseline wall 的 17.2% → **1.208x** 貢獻（略優於 doc 33 投影 1.16x）；**但 bucket 絕對大小被混淆**（doc 27 §6.4 把讀取成本追到 RSS 壓力下的 page cache 驅逐；移除 275 GB 並行寫入恰好是緩解 page cache 壓力的改變）→ Option K 正式仍未結清，乾淨實驗需第二次整片 `--no-prefetch`（~2.9 h，未跑）。首次整片組織/背景拆分：tissue 24,360 reads/1,037.8 s/42.6 ms per read/85.2 ms per tile；background 30,770/543.4 s/17.7 ms/35.3 ms；比例 **2.41**（crop 2.06，17% 內）。
  - **一個迴歸**：`B1_m3b_cellpose` 6,358.9 → 7,109.9 s（**+751.0 s、+11.8%**），MAIN 臂、同 tile 數、本輪沒動 M2/M3b；嫌疑 `e806938`（「更改判讀依據」）未量測；它回吐 Option H 贏得的 21%。
- **Step 5 `depth=2/3`：閘門達成、獎金太小、不建**：背景 tile 讀取（2 檔）**35.32 ms** vs GPU 工作（UNet only）**28.39 ms** → 每背景 tile 赤字 **+6.93 ms** × 15,385 = **106.6 s = 1.04% wall**；doc 33 §4 算術方向正確，但低於 playbook 的 ~10% 線一個量級、也低於任何 anchor 能解析的程度（§2.3 剛證明六對重複解析不出 0.83%）；驗證 ≤1.04% 需另一次 ~2.9 h 整片，且會與 §4.3 真正需要的 `--no-prefetch` arm 搶同一次。
- **三個途中發現的缺陷**：(6.1) `stitch_probe.py` 的 `build_inputs` 把合成 tile 網格寫到 `overlay_annotated/`，而 `_join_overlay_tiles` 讀 `_STITCH_SCRATCH`（`_stitch_scratch/`）→ 該 rename 之後每次跑都會 `FileNotFoundError`（round 9 只修了 stack trace 看得到的 import 就停；現改從 pipeline 常數取目錄名）；(6.2) **`workers=4` 一次跑 OOM GPU 然後 hang 49 分鐘**——control arm 一次 `torch.OutOfMemoryError`（"Process 2264989 has 24.76 GiB memory in use" of 31.36 GiB）餓死三個兄弟 = doc 19 #7b/DISCOVERED #2 的 allocator 氣球**在 `workers=4`（production 推薦）上被觀察到**（同命令其他十一次成功）；fail-fast 正確運作，**但退出不行**——raise 之後 parent 活著發呆 49 分鐘需手動殺，簽名（終止的 worker + `multiprocessing.Queue` feeder thread 無法 flush 到已關閉 pipe，於 interpreter exit join）指向 `_run_tiles_multiprocess` 的 `_kill_all()` 路徑（本輪未動）；(6.3) candidates 寫的 pyramid 比 baseline 淺（B、C 9 頁 vs A 11 頁，因 level-count 規則用 `min(h,w)` 而 pyvips 用 `max(h,w)`，橫向 slide 停早兩層；做得比 baseline 少卻仍輸）→ 修正後才引用。
- **處置與 follow-ups**：Step 1 DONE（+ `overlay_pyramid_audit.py`）；Step 2 SHIPPED（6 tests、env-var ablation 開關、不宣稱 wall 勝利）；Step 3 DONE（推翻 doc 32 §4）；Step 4 DONE（Exp 4 結清；Option K 部分）；Step 5 MEASURED NOT BUILT；correctness veto 通過（stats 12 次一致；整片 10,800/16,765 vs 10,801/16,764）。**Follow-ups（依優先）**：(1) `B1_m3b_cellpose` +751 s 是整輪最大未解釋數字且在 MAIN，先量；(2) **post-fail-fast hang 是最危險缺陷**（兩次：`workers=4` fail-fast 後 49 分鐘、`workers=1` 已印出最終 JSON 之後 2 小時——後者無多行程，原因不只 task queue；需有界 queue join + 稽核 `main()` 之後仍存活的 non-daemon 執行緒）；(3) `workers=4` 的 CUDA allocator 氣球（`expandable_segments:True` 在 `workers=4` 從未測，掃描便宜）；(4) 整片 `--no-prefetch` arm（~2.9 h）比 (6) 更值得；(5) 若重訪 Phase D，閘門是「讀取管線化」（假設上限 ~1.47x/1.80x 僅 Phase D、端到端 `workers=4` ~1.094x）；Phase D 占 wall 比例因周圍變快而成長（8.6% → 11.6%）；(6) `depth=2/3`：1.04%；(7) QuPath 對 `cand_{A,B,C}.tiff` 的檢查未跑（現為資訊性）。

---

## 12. Round 11：關閉 round 10 的待辦（doc 36 規劃、doc 37 落地，2026-07-29）

### 12.1 doc 36 — round 11 計畫（純規劃）

- **Discover**（doc 35 §8 的七項）：(1) `B1_m3b_cellpose` +751.0 s（+11.8%）未歸因；(2) 失敗後/完成後 hang（`workers=4` fail-fast 後 49 分鐘；`workers=1` 已印出最終 JSON 後 2 小時）；(3) `workers=4` allocator 氣球（1/12），`expandable_segments:True` 從未在 `workers=4` 量過；(4) Option K 只是上界（`d6592c3` 移除 275 GB 並行寫入的混淆）；(5) Phase D read pipelining（假設上限）；(6) `depth=2/3`（1.04%）；(7) QuPath 驗證（資訊性）。
- **Analyze**：`e806938` 的完整 diff 只動兩個程式檔——`hybrid_pipeline.py`（一行 docstring）與 `m4_module/csv.py`（`DotStatsSummary.from_results` 的有效細胞過濾與 `write_summary_csv` 標籤，移除 `her2_dot_count >= 1` 條件並重新命名輸出欄位）——**完全不碰 M2/M3b/Cellpose**，不可能移動 GPU 前向 bucket；`d6592c3`（M0 split）較合理（把 1,065 行搬出 `hybrid_pipeline.py`、新增 547 行到 `m0_tile_runner.py`，M2/M3b GPU 呼叫編排就在那）；另一候選是 n=1 的跑批間變異。項目 2 與 3 是 `_run_tiles_multiprocess`（`m0_multiprocess.py:250-398`）與 `_kill_all()` 中已出貨的可靠性缺陷，但**不一定是同一個 bug**：`workers=1` 的 hang 沒有多行程，需自己的診斷。項目 3 獨立且便宜。項目 4 是昂貴不可重複資源；其與 1–3 的依賴是保護操作者時間（2 的 hang 風險）。5、6 不需新動作；7 從主動追蹤移除。
- **順序**：1 歸因 → 2 診斷並修 hang → 3 掃描 `expandable_segments` → 4 整片 `--no-prefetch`（~2.9 h）→ 5 不動作。Step 1 不要從 `e806938` 開始（diff 已排除）：在 match24 anchor 上，對 `1c0b31e`（M0 split 前）與當前 HEAD 各跑一次 `workers=1`，比 `B1_m3b_cellpose`；若 crop 看不到偏移，就記為「crop 未重現，需第二次整片才能解」，不追（因第二次 2.9–3.8 h 跑批對 4 有更強主張）。Step 2 把兩種形狀視為兩個候選機制（`workers=4`：terminated worker + `multiprocessing.Queue` feeder thread 無法 flush；修法候選 `_kill_all()` 前先 drain queue 或有界 join；`workers=1`：稽核 `main()` 之後存活的 non-daemon 執行緒）。Step 3 用既有 `config.cuda_alloc_conf` 與 `scripts/alloc_conf_probe.py`（`m0_multiprocess.py:280-284` 已把 `PYTORCH_CUDA_ALLOC_CONF` 寫進 worker 環境）。Step 4：`workers=1`、`--no-prefetch`（`perf_measure.py` 已同時設 parent binding 與 `HYBRID_MP_NO_PREFETCH`）、在今日程式碼上，直接與 Step 4（round 10）的 1,581.2 s 比。

### 12.2 doc 37 — round 11 落地

- **標題**：**Step 1 沒有重現它被派去歸因的迴歸，卻找到了成因**：match24 anchor 上，`B1_m3b_cellpose` 跨整個 commit 範圍只動 **+1.24%**（非 +11.8%），在 Option L 預取兩側都拿掉時整段**持平**（`B1_unet_coremask` +0.02%）；**真正讓 bucket 移動的是 Option L 自己**：預取打開使 `B1_unet_coremask` **+16.4%**、`B2r_tile_read` **+26.4%**，**同時 wall 下降 1.21%**。round 10 的 +751 s 是此膨脹在整片規模的樣子——併行造成的記帳假象，而非新增 GPU 工作；這個預測 step 4 被排來檢驗，並**只確認一半**：競爭係數從 576 tile 轉移到 27,565 tile 誤差在 4% 內，但只解釋 751 s 中的 221 s。**Step 2 以 stack trace 而非假設找到 hang**：parent 在 `multiprocessing` 的 atexit handler 永遠阻塞，等待 `task_q` 的 feeder thread join，而 feeder thread 自己卡在寫入 reader 剛被終止的 pipe；修好 + 修好探針暴露的第二缺陷（`PrecutStream` 在批次放棄後仍切整張玻片）；`workers=4` fail-fast 退出延遲 **never → 0.32 s**；`workers=1` hang **在 crop 與整片規模都沒重現**。**Step 3**：control 12 次中 **4 次**以 byte-identical 24.76 GiB 簽名 OOM，`expandable_segments:True` **0/12**，代價 +0.67% wall。**Step 4 把 Option K 結清為 1.044x（非 1.208x）並解釋差距**：在讀取位置與 doc 27 baseline 相同的條件下，讀取 bucket **449.3 s vs baseline 的 2,368.5 s，−81%**——讀取從來不貴，貴的是與它並行的 275 GB per-tile 寫入。Option L 現在隱藏 449.3 s 中的 448.4 s（**99.8%**，天花板是 4.21% wall，非 doc 33 對照的 17.2%）。
- **途中發生的兩件事**：`main` 在 20:56 被 `git pull` 從 `6b0286c` fast-forward 到 `2b3ad68`（numpy-2 constraint bump、Windows 可攜性、含 ROI 支援的分析 UI）；前兩輪量測 sweep 因本輪自己造成的設計缺陷被丟棄。
- **§1 Step 1 歸因**：
  - **兩次 sweep 被丟棄**：A,B,A,B 交替使每個 pair 的第二臂永遠佔後面的 slot，而機器暖機過程使兩臂跑得越來越慢（PRE `B1_m3b_cellpose` 134.03 → 157.58 → 158.81；HEAD 141.37 → 169.60 → 174.04；每 slot 漂移 ~+11.8 s ≈ 報告的差異 +11.53 s）→ **量到的是 sweep 順序而非程式碼**（反模式 #7 從後門進來：被改的第二件事是「何時」跑）；第二個 sweep（`--no-prefetch` vs PRE）中途跳階（wall 221/214 → 191/190/190/188）。**最後報告的 sweep：三個 arm 的 3×3 Latin square、重複兩次，每臂在每個 slot 位置出現次數相等，從已暖機狀態開始**；arm 內 sd 0.3–0.9 s（~189 s wall 的 0.2–0.5%），slot 殘差 189.68/189.29/188.38 s。doc 35 §2.3 說「anchor 解析不出 0.83%」是針對**熱狀態未穩、未平衡的 sweep**，並非 anchor 本身；穩定且平衡後 1.2% 解析在 d/SE = 7。
  - **三個 arm**：PRE = `1c0b31e`（`d6592c3` 之前：無 M0 split、無 Option H/L，讀取內嵌於 MAIN）；HEAD = `2b3ad68`（今日，預取）；**HEAD_NP** = `2b3ad68` + `--no-prefetch`（Option L 被拿掉）——使 PRE→HEAD_NP 只改 refactor；n=6 per arm、**18 次 stats 全相同（224/352）**。
  - 結果（wall / `B1_m3b_cellpose` / `B1_unet_coremask` / `B2r_tile_read`）：PRE 188.63±0.42 / 135.93±0.32 / 16.65±0.05 / 5.42±0.05；HEAD_NP 190.52±0.87 / 137.48±0.37 / 16.66±0.05 / 5.58±0.06；HEAD 188.22±0.81 / 137.61±0.43 / 19.39±0.10 / 7.05±0.05。比較（d/SE）：**PRE→HEAD**：wall −0.22%（1.0）、m3b **+1.24%（6.97）**、unet +16.44%（55.7）、read +30.03%（49.6）；**PRE→HEAD_NP（refactor 單獨）**：wall +1.00%（4.4）、m3b +1.14%（7.1）、unet **+0.02%（0.08）**、read +2.85%（4.6）；**HEAD_NP→HEAD（Option L 單獨）**：wall **−1.21%（4.3）**、m3b +0.10%（0.5）、unet **+16.42%（54.3）**、read **+26.43%（42.4）**。
  - **結論**：+11.8% 在 crop 規模不重現（+1.24%，小 9.5 倍）；**`d6592c3` 由量測而非論證洗清**（讀取位置固定時 unet +0.02%）；`e806938` 已由 diff 排除；**真正移動 bucket 的是 Option L，且不增加任何工作**（bucket 是主執行緒呼叫的牆鐘計時，並行讀取執行緒與主執行緒競爭使所有主執行緒 bucket 讀起來變長，而裝置做的事相同）。crop 上帳目 0.4 s 內閉合：讀取離開 MAIN −5.58 s、B1 膨脹 +2.86 s（154.14 → 157.00 s）→ 預測 wall −2.72 s、實測 −2.30 s，即 **每搬走 1 秒讀取，B1 膨脹 0.513 秒**。
  - **對 round 10 與 step 4 的預測**：baseline 讀取總計 2,368.5 s × 0.513 ≈ **預測 1,214 s B1 膨脹**；round 10 觀察 `B1_m3b_cellpose` +751.0 s、`B1_unet_coremask` + `BM1_*` + misc +144 s；觀察/預測 0.62–0.74——方向與量級吻合（crop 讀取是 page-cache 命中、執行緒空轉 decode 時持 GIL 較多，整片是真磁碟 I/O、阻塞在 `read()` 的執行緒競爭較少）。**round 10 的「最大未解釋數字」與其頭條 Option L 勝利是同一現象的相反符號**：若成立，Option L 對 MAIN 的淨貢獻約 −1,617 s（非 −2,368.5 s）。**預測在 step 4 跑之前就先寫下**：若成立，`--no-prefetch` 的 `B1_m3b_cellpose` 應回落到 baseline 的 ~6,359 s；若仍在 ~7,109.9 s，歸因錯誤。
- **§2 Step 2 hang**：
  - **量測所有 benchmark 都看不到的缺陷**：`scripts/exit_latency_probe.py`（新）量「子行程退出 − 子行程最後一行輸出」；工作在子行程跑，parent 為每行輸出打時間戳，子行程超時時 parent 送 `SIGUSR1` 給子行程註冊的 `faulthandler`，所有執行緒 stack 進同一個 log（`py-spy` 在此主機無法 attach——`ptrace_scope=1`、無 root）；fail-fast 透過既有 `HYBRID_MP_WORKER_PROBE` hook 觸發。
  - 修前：clean `workers=1` 0.69 s、clean `workers=4` 0.29 s、fail-fast `workers=1` 0.77 s、**fail-fast `workers=4` never**（探針於 65 s SIGKILL；先前一次無界嘗試 14 分鐘後手動殺）。**`workers=1` 完成後 hang 不重現**（記為 not reproduced，不是 fixed）。
  - **stack dump**：main 在 `multiprocessing/util.py` `_exit_function` → `_run_finalizers` → `queues.py` `_finalize_join` → `threading.py` `join`；feeder thread 在 `queues.py` `_feed` → `connection.py` `send_bytes` → `_send`。`_kill_all()` 終止 worker，無人再讀 `task_q`，pipe 填滿，feeder thread 永遠阻塞在 `write()`，atexit handler join 它；**一個正確的 fail-fast 變成永不返回的行程**。修法一行（標準庫專為此情境的原語）：`task_q.cancel_join_thread()`（放棄批次後不該遞送的工作被丟掉是正確的，不是繞路）：`stop_feeding.set(); task_q.cancel_join_thread(); _kill_all(); raise`。
  - **探針暴露的第二缺陷：precut 從不停止**：`_run_tiles_multiprocess._feed` 的註解說中止後必須真的停止切整張玻片，實際沒有——`PrecutStream.__iter__` 在 yield 第一個 tile 前就把**所有**位置 submit 給 thread pool，消費者的 `break` 只停止「拉取」而不停止「切割」；hang 的那次 run：批次在 tile ~8/576 中止，scratch 目錄仍有 **576/576 個切好的 tile、842 MB**；整片規模 = 27,565 tile 對、數十 GB 與數分鐘 I/O。**修法**：`__iter__` 現在維持有界視窗（`workers × 8`）in-flight cuts，generator 關閉時取消未開始的；排乾的 stream 產出相同 tile、相同 completion-order 語意，只有被放棄的 stream 行為不同；視窗不會成為瓶頸（切一對幾十 ms vs 每 tile 分析數百 ms）；`tests/test_precut_stream_bounded.py` 3 tests（對修前程式碼第一個測試失敗：「abandoning the stream took 2.51s」且切完所有 500 位置）。
  - **結果**：clean `workers=1` 0.69 → 0.69 s；clean `workers=4` 0.29 → 0.28 s；fail-fast `workers=1` 0.77 → 0.67 s；**fail-fast `workers=4` never → 0.32 s**；有界 precut 視窗在 `workers=4` 的成本：step 3 的第一次 run 104.4 s vs round 10 六次 prefetch 均值 104.43 s，stats 與 peak RSS（11.79 vs 11.70–11.81 GB）相同。`workers=1` 兩小時 hang：未修（未重現）；doc 35 提的「稽核存活的 non-daemon 執行緒」已查：該路徑 joblib 用 `prefer='threads'` 與 `dot_detect_n_jobs=1`（無 loky 行程池），`run_batch` 每個 executor 都在 `with` 內；剩餘假說是規模相關的 teardown（61 GB RSS 在 62 GB 機器 + 27 GB swap），crop 無法測試——step 4 是工具，其退出行為見 §4。
- **§3 Step 3 `expandable_segments` @ `workers=4`**（`alloc_conf_probe.py` 未改動、match24、12 次交錯/條件、共 24 次、~45 分鐘）：control 12 次 **4 次 OOM（33%）**，三次為 byte-identical 24.76 GiB 氣球（第四次 20.49 GiB）；`expandable_segments:True` **0/12**（Fisher 單尾 **p = 0.047**）。median wall（成功 run）104.59 vs 105.23 s；mean 104.31±0.70（n=8）vs 105.01±0.91（n=12）；median peak framebuffer 15,303 vs 20,621 MB；**max peak 18,193 vs 30,061 MB（92.2% of card）**；mean RSS 11.90 vs 11.88 GB。**代價**：wall +0.70 s（+0.67%，d/SE 1.83）、RSS 不變、**peak framebuffer 上升**——expandable segments 不使用較少記憶體，是阻止 allocator 碎片化到「176 MB 請求無法被滿足」的狀態，並會長進任何可得餘裕；doc 27 §6 的 32 GB 硬底線（93.3%）不會因此放寬，反而收緊。**建議**：`workers>1` 開啟；本輪不改 `config_example.py`（一個 anchor、一張卡；預設自 doc 27 出貨為 `""`）——後來 §9 在明確指示下翻開。**對 step 2 的意外確認**：control run 8 OOM、fail-fast 正確、**返回**（sweep 在 26.4 s driver wall 後進到 run 9）——正是 round 10 花 49 分鐘手動殺的情境，在真實非注入的 allocator 失敗上重現四次，step 2 的修復有效；沒有它此 sweep 會在第三次 run 停住。
- **§4 Step 4 整片 `--no-prefetch`**（`workers=1`、無 `--resume`、今日程式碼；分析 2 h 37 m + stitch 20 m 39 s = **2 h 57 m 46 s**；metrics `prefetch_ablated: true`）：correctness veto 全軸通過：stats **10,800/16,765（與 round 10 相同）**；356,226 cell rows；peak RSS 60.17 GB（round 10 60.04）；70 次 `gc.freeze()`（與 round 10 相同）；7.50 GB overlay `overlay_pyramid_audit.py` 各層 PASS、Predictor=1 跨 12 IFD、各層 mean 240.71 → 241.14、零像素 0.0%。

| | doc 27 baseline | round 10 | **round 11** |
| --- | --: | --: | --: |
| prefetch | off | **on** | **off** |
| per-tile 中繼寫入 | **275 GB** | 無 | 無 |
| **wall** | 13,762.5 s | **10,217.7 s** | **10,666.1 s** |
| `B1_m3b_cellpose`（n=24,360） | 6,358.9 | 7,109.9 | **6,888.7** |
| `B1_unet_coremask`（n=27,565） | 734.8 | — | 738.7 |
| **`B2r_tile_read`**（n=55,130） | **2,368.5** | 1,581.2 | **449.3** |
| — tissue / background | — | 1,037.8 / 543.4 | 274.3 / 175.0 |
| `B4_gc_collect` | 2,218.4 | 58.8 | 19.5 |
| Phase D stitch | 1,185.4 | 1,182.4 | 1,239.2 |
| peak RSS | 61.13 GB | 60.04 GB | 60.17 GB |
| `gc.freeze()` 次數 | 1 | 70 | 70 |

  - **Option K 結清：1.044x**（wall prefetch off 10,666.1 − on 10,217.7 = **448.4 s**），**不是** doc 35 §4.3 記錄的 1.208x 上界。**差距原因可量測**：與 doc 27 baseline 相同的讀取位置（內嵌於 MAIN）下，讀取 bucket 449.3 s vs 2,368.5 s（**−81.0%**）——同程式路徑、同片、同樣 55,130 次讀取；唯一改變是其中一次同時在寫 275 GB per-tile 中繼檔。**doc 27 §6.4 對「page-cache 驅逐而非固有 I/O」的診斷被直接量測證實**；被混淆的數字不只是「略微」膨脹，是**真實成本的 5.3 倍**。Option L 的 Amdahl 天花板在它出貨前就塌縮：doc 33 前提 17.2% → 今日 **4.21%**，實測節省 **4.20%**，回收 **99.8%**——**Option L 近乎完美地最佳化了一個已經縮小 5 倍的問題**；doc 33 的「17.2%」與其下游（含 doc 35 §5 `depth=2/3` 算術）都是以屬於已被取代的寫入模式的數字估的。caveat：一次對一次、相隔三天；prefetch-on arm 是 round 10 的（其 metrics JSON 已不在磁碟，只能用 doc 35 公布數字）；兩臂也差本輪的有界 precut 變更，已驗證不限流（切割領先差距整個被取樣的 sweep 都釘在 64 tile 上限）。
  - **預先登記的預測：機制確認、量級錯**：`B1_m3b_cellpose` 降到 **6,888.7 s——只回收 221.2 s of 751.0 s（29.5%）**，非「大部分」，預測**未確認**；但機制與係數緊密確認：crop 量到搬 1 秒讀取 → B1 膨脹 0.513 秒；套用於此 run 實際搬走的讀取（449.3 s × 0.513 = **230.5 s 預測**）vs 實測膨脹（6,888.7 → 7,109.9）**221.2 s**，整片隱含比例 **0.492（與 crop 0.513 差 4.1%）**。§1.4 的錯是**輸入**而非機制：把比例乘以 2,368.5 s（baseline 讀取總計），而 §4.1 剛證明該數字被一個不復存在的寫入模式膨脹 5.3 倍。**round 10 的 +751 s = 221 s prefetch 競爭（crop 與整片雙重量測）+ 530 s 未歸因**；該殘差是對 doc 27 baseline +8.33%，**不是** M0 split（crop 量到範圍 +1.14%，整片 ≈72 s）、**不是** Option L（此處已量）——是不同日子兩次單次整片跑批的變異；§1.1 量到同一張 GPU 在一次暖機 sweep 內此 bucket **漂移 +17%**，故 8.3% 落在該帶內。解決需重複整片 run（~3 h 每次），且無事依賴答案。
  - **其他**：**`workers=1` 完成後 hang 在整片規模也沒重現**（正是 doc 35 看到卡兩小時的形狀：`workers=1`、27,565 tile、60.17 GB RSS、最終 JSON 已印出；最後 log 行 02:35:41、metrics 寫入 02:35:42、行程消失，**<2 秒**；規模相關 teardown 假說不被支持）；Option H 整片第二輪成立（`B4_gc_collect` 19.5 s，比 round 10 的 58.8 s 更低、70 freeze、RSS 持平）；Phase D 現為 wall 的 **11.6%**（1,239.2 s of 10,666.1 s）；讀取不再是任何定義下的瓶頸（449.3 s = 4.21%，低於 playbook ~10% 線）；tissue:background per-read 比例 1.97（274.3/24,360 vs 175.0/30,770；doc 35 為 2.41、doc 33 crop 2.06）。
- **§5 Step 5**：無動作，Phase D Phase 2、`depth=2/3`、QuPath 皆維持關閉/移除。
- **§6 途中發生的事**：(6.1) `main` 20:56 在第二與第三輪 sweep 之間 fast-forward 到 `2b3ad68`（三個 commit：numpy-2 constraint bump、Windows 可攜性、含 ROI 的分析 UI）；hybrid 側編輯為 ROI plumbing（`PrecutStream(region=...)`、`origin` 參數穿進 `compute_tile_geometry`/`_validate_axis`）與 Windows 對 `import resource` 的守衛，**per-tile 分析 hot path 不變**；venv 仍解到 `numpy 1.26.4`；sweep 2 跨越 pull 故也丟棄；報告的 sweep 全在 pull 之後。(6.2) `backend/tests/{test_chunked_upload,test_module1_strips,test_resume}.py` 收集失敗：`ModuleNotFoundError: No module named 'tuspyserver'`（pull 引入的 `backend/api/tus_compat.py` 新依賴，venv 沒有）——stash 本輪變更重跑確認為既有問題，不在量測輪中途安裝依賴；hybrid 套件 **79 passed**。(6.3) round 10 整片 metrics JSON 已不在磁碟（`/home/taro/r10_fullwsi_w1/` 只保留輸出與 45 GB precut scratch），所以 step 4 用 doc 35 公布數字比較，**不能**重現 doc 35 §4.1 的 MAIN/BG arm 拆解；本輪產出保留於 `/home/taro/r11_step1{,b,c,d}/_metrics/`（36 次 crop）、`r11_step2/*.json`、`r11_step3_alloc/alloc_conf_w4.json`（24 次）、`r11_fullwsi_w1_np/_metrics/`。
- **§7 處置**：Step 1 MEASURED（crop 未重現；221 s 歸為 Option L 競爭、530 s 未歸因落在漂移帶內）；Step 2 FIXED（`task_q.cancel_join_thread()`、bounded precut、3 tests、`workers=1` hang 未重現）；Step 3 SWEPT and SHIPPED（§9）；Step 4 DONE（Option K 1.044x）；Step 5 無動作；correctness veto 通過（step-1 18 次、step-3 20 次成功、整片 10,800/16,765）；**79 tests pass**。
- **§8 Follow-ups**：(2) `B2r_tile_read` 4.21% 且 Option L 捕獲 99.8%——所有讀取側待辦都以 17.2% 估，需以 449.3 s 重讀 doc 30/33/35；(3) 530 s 殘差是 critical arm 最後一個未歸因數字，需整片重複才能建立**這個專案從未有過的誤差棒**（docs 27/35/37 的整片比較都是 n=1 vs n=1），且誤差棒在 crop 規模以平衡、暖機的 sweep 便宜得多；(4) **量測衛生已被示範而非論述**：兩次 sweep 因順序/熱混淆被丟棄；未來 A/B 預設平衡；(5) 保留整片 `_metrics/*_timings.json`；(6) `tuspyserver` 缺失；(7) **`scripts/exit_latency_probe.py` 應納入 pre-release checks**（~90 s）。
- **§9 決定——`cuda_alloc_conf` 預設翻開**：follow-up 1 在本輪（明確指示）內關閉：`cuda_alloc_conf: str = ""` → `"expandable_segments:True"`，**同時**改 `config_example.py` 與本機 gitignored 的 `config.py`（後者因 `config.py` 早於本輪 `cp` 而存在，只改範本不會改變 `run_batch(workers>1)` 實際行為）；`tests/test_config_parity.py`（8 tests）仍通過（檢查結構一致而非值）；`cuda_alloc_conf` 已在 `_HASH_EXCLUDE` 外（不影響 hash、不使 `--resume` checkpoint 失效）；`m0_multiprocess.py` **未改**（knob 早已由 doc 27 §5 端到端接好：寫進 `os.environ["PYTORCH_CUDA_ALLOC_CONF"]`，僅在呼叫端未設時）。**硬體底線重述（非繼承）**：doc 27 §6.6 的 32 GB 底線是 knob **關閉**時的 30,439 MB（93.3%）；§3 量到的是**相反於「修 OOM」直覺**的結果——`expandable_segments` 不降低 peak VRAM，而是升高（max 18,193 → 30,061 MB），故 **32 GB 底線不是被放寬，而是被收緊**：`workers=4` production 應視為常態居於 ~92% of 32 GB card；「不在 worker pool 內加任何 GPU 函式庫而不重量」規則延續，但要跨的數字更靠近天花板；`workers≥5` 更堅定不開放。不重開 doc 27 §5 的 round-8 決定（以當時證據正確：`workers=6`、n=6、p≈1.0；round 11 是以出貨 worker 數、雙倍樣本重問而得到可解析的答案）。

---

## 13. Round 12：`workers=1→4` 只有 ~2x 的完整成因（doc 38 規劃、doc 39 落地，2026-07-29）

> 使用者提問：「`workers=1` vs `workers=4` 只有大約 2x，load balancing、pipelined stitching、開更多 worker 能否更接近完全利用硬體？」

### 13.1 doc 38 — 計畫（純規劃）

- **§0 兩個更正**：(a)「`expandable_segments:True` 尚未翻為預設」的陳述已過時——HEAD（`b3fa47d`）已把 `config_example.py` 的 `cuda_alloc_conf` 預設改為 `"expandable_segments:True"`；**VRAM 餘裕並未因此變寬**——peak framebuffer 從 18.2 GB 升到 30.1 GB（92.2% of 32,607 MB），因為旋鈕消除的是 allocator OOM 失敗模式，不節省記憶體。(b) 順手發現：`run_batch(..., workers: int = 4, ...)`（`hybrid_pipeline.py:128`）目前預設 `4`，而所有文件說預設是 1；`git log -S"workers: int = 4"` 追到 `a64b92e`（2026-07-25 "adjust worker count in run_batch"）；但兩個真實 call site 都不受影響（`backend/api/hybrid.py:89` 明確傳 `workers=1`；CLI argparse 預設 `1`，`hybrid_pipeline.py:435`）→ latent footgun，非 live bug，不在本輪範圍。
- **§1 Discover——把散落的數字排成一條線**：
  - 使用者觀察的 ~2x = round 8 唯一的整片 `workers=4`（13,762.5 s / 6,211 s = **2.216x**）；該數字自 round 8 後**從未重跑**，而 round 9–11 的 gc/prefetch 勝利只在 `workers=1` 確認 → 2.216x 的分子分母不是同一份程式碼上量的（round 11 `workers=1` 已降到 10,666.1 s，1.290x）。
  - **crop 加速比 3.09–3.51x vs 整片 2.2x 已被量測兩次**：tissue-dense large/441（14.1% 背景）`workers` 1/2/3/4 → 482.8/208.9/156.1/137.4 s（2.31/3.09/3.51，效率 116/103/88%）；composition-matched match24（55.9% 背景）188.8 → 88.3 s = **2.138x**（Phase D 在 576 tile crop 只有數秒，不影響此比）；整片 2.216x。**一旦組成對齊，加速比從 3.51x 掉到 2.14x，且發生在 Phase D 進入前**——所以使用者看到的「只有 2x」**主要不是 Phase D 造成，而是真實玻片的組成本身**（doc 21 §4.2 把 3.51x 歸因於「回收的 GIL 競爭」與「跨 process 重疊前向」，兩者的回報都與每個 tile 實際做多少工作成正比；背景 tile 在 UNet++ core mask 為空時直接短路，沒有 GIL 競爭可回收）。
  - **Phase D 是唯一完全序列的階段且不隨 worker 數縮放**：`_finish_batch()` 在整個 worker pool（或單 process 迴圈）完全耗盡後只呼叫一次（`hybrid_pipeline.py:245/:257/:342`），做全域重編號 + 表格匯出 + `_stitch_overlay_slide`（`:379`）——硬屏障。Round 8 ~1,185 s（`workers=1` 占 8.6%，`workers=4` 同絕對值占 **19.3%**）；round 11 `workers=1` 已升到 1,239.2 s（11.6%）——「Phase D 占 wall 比例持續成長純粹因為周圍都變快」。此線已被完整調查兩次且兩次都判「GPU 化不值得」：doc 29 校正後 ≈1.05–1.09x → doc 32 記憶體 slab 一度 1.40x/2.15x → doc 35 把讀取/join 半邊加回、真實磁碟規模下**完全反轉**（0.788x/0.884x，因讀取占 Phase D 自己 wall 的 48.7% 而 pyvips 已免費將讀取與編碼重疊）；**唯一已識別卻從未建的槓桿：串流讀取管線化**，doc 35 §3.3 已算出其天花板 ~1.094x——「這正是使用者說的 pipelined stitching，其天花板已被量過，不是把 2.2x 變 4x 的東西」。
  - **三個方向對照**：**load balancing**——早已是動態工作佇列（doc 20 §2 Candidate B/D；`_run_tiles_multiprocess`（`m0_multiprocess.py:250`）用 `ctx.Queue()` + feeder thread，每個 worker 空閒才拉下一個 tile）；沒有任何一輪把加速缺口歸因於此。**Pipelined stitching**——唯一真正開著的槓桿，ceiling ~1.094x。**更多 worker**——VRAM 硬體上限，非演算法：`workers=4` 整片 30,439/32,607 MB（93.3%，~2.2 GB 餘裕）→ allocator 修復預設化後 92.2%，餘裕**更緊**；crop 邊際曲線已收斂（103% → 88%，doc 21 §2「knee 是 N=3」）；每 worker 3 個 CUDA context 的模型組在 32 GB 卡上 `workers≥5` 不建議，直到每 worker VRAM 足跡真的縮小。
- **§2 Amdahl 帳**（新算術，使用 round 8 的配對）：
  - 計算 1（Phase D 完全隱藏且 tile 平行完美 4x 的理論天花板）：`13,762.5 − 1,185 = 12,577.5 s`；理想 `12,577.5/4 + 1,185 = 4,329.4 s`；**理想加速 3.179x——即使完美 4x，Phase D 單獨就把天花板釘在 ~3.18x**。
  - 計算 2（由實測 6,211 s 反推 tile 平行部分實際縮放）：`6,211 − 1,199 = 5,012 s`；實際平行加速 `12,577.5/5,012 = 2.509x`，與 match24 獨立量到的 **2.138x**（也是排除 Phase D 的純 tile 平行加速）互相印證——**真實玻片組成本身把可達 tile 平行度限制在 ~2.1x–2.5x**（remaining 2.51 vs 2.14 差距與整片 `workers=4` 佔 93.3% VRAM vs crop 63% 同量級）。
  - 計算 3（若疊上 Phase D 管線化 ~1.094x）：`2.216 × 1.094 ≈ 2.42x`。
  - **誠實結論**：在這張卡、這個模型組、這個真實玻片組成下，可達天花板約 **~2.5x–2.8x，不是 4x**；缺口是兩個已量到的獨立效應之和：(a) 55.8% 背景使 tile 平行本身限於 ~2.1–2.5x；(b) Phase D 完全序列、不隨 `workers` 縮放，「完全管線化」也只多 ~1.09x。**load balancing 沒有可歸因的缺口；`workers≥5` 卡在 VRAM**。
- **§3 計畫表**：0 修正 `bottleneck-list.md`/`19-open-backlog.md` 的過期句子；**1 重跑乾淨整片 `workers=4`（高成本、無人看管；整份 Amdahl 帳建立在 round 8 的數字上）**；2 Phase D 讀取管線化（先在 `scripts/stitch_probe.py` 規模做便宜 spike，天花板 ~1.094x）；3 批次領取動態佇列的合成微基準（低成本；若在雜訊地板則關閉不建）；4 縮小每 worker VRAM 以容納 `workers=5`（高成本、有 correctness 風險；**不建議**，記錄以關門，如 doc 20 Candidate E）。
- **§4 停損**：step 1 結果若明顯偏離 2.216x → 先找原因；step 2 spike 若不接近 ~1.09x → 依 doc 35 §3 的 ablation 紀律關閉；任何 correctness veto 不豁免。**不是對 4x 的承諾**。

### 13.2 doc 39 — 落地與結果

- **標題**：五項全部解決。**Item 0** doc 38 已修兩檔；同一句過時陳述還活在三個檔案，現已修（`README.md:265`、`docs/BACKLOG.md` 標頭/item 1/item 7b（🟡 → ✅）、`DISCOVERED-NOT-IMPLEMENTED.md:50`）。**Item 1 整片 `workers=4` 重跑是頭條，並觸發 doc 38 §4 的第一個停損**：今日程式碼 `workers=4` **5,854.9 s**，對當前 `workers=1` baseline 是 **1.745x，不是 2.216x**（比值下降 21%）；**原因不在 tile 平行臂**（doc 38 §2 預測正確，從第三個獨立方向再證實：**2.279x**，在 §2 推出的 2.1–2.5x 帶內），而是 **Phase D 從 1,239.2 s 成長到 1,889.8 s，現占 `workers=4` wall 的 32.3%**——該階段 Amdahl 天花板 **1.477x**（doc 27 §6.6 記 1.239x）。**Item 3 決定性負向**：派送全部 27,565 tile 零工作僅 **0.356 s = 0.006% wall** → 批次領取 e2e 天花板 **1.00006x**，不建。**Item 2 推翻 doc 35 §3.3**：未建的槓桿現已存在，band 讀取管線化後串流候選從「比出貨路徑慢」變成 **1.365x 與 1.581x 更快**；控制組幾乎精確重現 doc 35 §3.2；「來源在 page cache 中」這個 spike 可能說謊的明顯方式被**量測而非論證**：真實 46 GB 來源冷讀，讀取占 Phase D **整片 51.9% vs doc 35 crop 的 48.7%**；端到端投影 **1.135x**，完美重疊底線 1.184x。**Item 4 刻意不做**。correctness veto 全過；一個本輪造成的量測缺口被如實記錄。
- **§2 Item 1 整片 `workers=4`**：一次完整 27,565 tile、`--mp-workers 4 --stream-precut`、無 `--resume`、HEAD；**刻意不傳 `--worker-timings`**（會使 wall 不可比；代價是 §6.2）。

| | round 8 `w=1` | round 10 `w=1` | round 11 `w=1`（no-prefetch） | **round 12 `w=4`** |
| --- | --: | --: | --: | --: |
| **wall** | 13,762.5 s | **10,217.7 s** | 10,666.1 s | **5,854.9 s** |
| **Phase D stitch** | 1,185.4 | 1,182.4 | 1,239.2 | **1,889.8** |
| Phase D 占 wall | 8.6% | 11.6% | 11.6% | **32.3%** |
| peak RSS | 61.13 GB | 60.04 GB | 60.17 GB | **45.56 GB** |
| analysis 輸出 | 347 GB | — | 55.24 GB | 55.25 GB |
| `overlay_slide.tiff` | 5.86 GB（壞） | 7.50 GB | 7.50 GB | 7.50 GB |
| `config_hash` | `3d1087f2` | — | `d2ccc46b` | `d2ccc46b` |
| `cuda_alloc_conf` | `""` | `""` | `""` | `expandable_segments:True` |

  - vs round 10（prefetch ON = 出貨預設）：10,217.7 / 5,854.9 = **1.745x**（誠實數字，因為與今日出貨配置相同）；vs round 11（prefetch OFF）1.822x。round 8 `workers=4` 6,211 s → 絕對值 1.061x 更快：round 9–11 的勝利有轉移到 `workers=4`，只是遠少於在 `workers=1` 的轉移。
  - **correctness veto（vs round 11 `workers=1`，唯一同 `config_hash` 的整片 run）**：`report.csv` rows 356,226 → 356,220（−0.002%）；valid cells 45,736 → 45,732（−0.009%）；HER2/CEP17 dots 148,405/108,464 → 148,385/108,458（−0.013%/−0.006%）；HER2:CEP17 ratio、mean HER2/cell 1.37、3.24 相同；verdict Not amplified 相同；stats 10,800/16,765 → 10,801/16,764（一個 tile）；比 round 8 的 −0.01% 更緊；`overlay_pyramid_audit.py` 各層 PASS（Predictor=1 跨 12 IFD、各層 mean 240.71 → 241.14、零像素 0.0%）。
  - **doc 38 §2 Amdahl 帳重算**：parallel portion w=1 = 10,217.7 − 1,182.4 = 9,035.3 s；w=4 = 5,854.9 − 1,889.8 = 3,965.1 s；tile 平行加速 **2.279x**。「純 tile 平行加速（排除 Phase D）」的三條獨立路徑一致：match24 直接量 **2.138x**、round 8 整片反推 2.510x、**round 12 整片反推 2.279x**——**真實玻片 55.8% 背景組成本身把 tile 平行度限制在 ~2.1–2.5x，這不是調校失敗**（背景 tile 在空 core mask 時短路，幾乎沒有 GIL 競爭可回收，這正是 tissue-dense crop 量到 3.51x 而真實玻片永遠做不到的原因）。**doc 38 錯的是頭條**：其「天花板 ~2.5–2.8x」假設 Phase D 維持 1,185–1,199 s；實際 Phase D 占 `w=4` wall **32.3%**、Amdahl 1/(1 − 0.323) = **1.477x**——doc 38 §1.3 預測了方向（「最終成為主導項」），這次比趨勢暗示的更猛；項目 2 從「唯一已知小天花板的槓桿」變為「整條 pipeline 的主導剩餘項」。
  - **Phase D 成長 52%，本文件不知道為什麼**：與產生**相同 artifact** 的 run 比較——round 8 的 1,200.8 s 產出的是 doc 35 §1 證明金字塔層壞掉的 5.86 GB 檔案；同 artifact 的比較是 round 10 re-stitch（1,352.9 s 與 1,315.0 s，皆 7.50 GB）與 round 11 整片（1,239.2 s、7.50 GB）→ 本次對 byte-size 相同 7.50 GB 輸出需 1,889.8 s = **+40% 到 +52%**。已知：輸出不同嗎？否（7,500,478,515 B vs 7,500,468,791 B（r11）、7,500,511,297 B（r8 re-stitch）；各層 audit 相同）；整個 run peak RSS **45.56 GB** 低於 r11 的 60.17 GB，stitch 視窗（t=3,965 → 5,855 s）平均 35.0 GB/peak 45.6 GB；Phase D 是未變更程式碼、於 parent 單執行緒。**未知（刻意不擇一）**：(1) 寫回/page-cache 競爭（`workers=4` 在 1.2 h 而非 2.5 h 內寫完 55 GB 輸出與 27,565 個 `_stitch_scratch` tile，dirty-page 速率約兩倍，stitch 對更深的 writeback backlog 開始）；(2) 一般整片跑批間變異（doc 37 §5 已指出本專案**從未建立整片數字的誤差棒**，每個整片比較都是 n=1 vs n=1）。**這是 n=1 且必須當 n=1 讀**；不改變 tile 平行結論（三個來源印證），但 32.3%/1.477x 取決於一個已知會移動 ±50% 的量的單次量測。
- **§3 Item 3 批次領取：負向關閉**（doc 38 事先承諾「若在雜訊地板，關閉不建」）：`scripts/mp_queue_claim_probe.py`（新）重現 `_run_tiles_multiprocess` 的 IPC 形狀（spawn context、feeder thread、`ctx.Queue()`、poison sentinels、每 tile 一個攜帶真 `CellAnalysisResult` list 的 `("ok", …)` 結果訊息），在真實 185×149 grid 與真實 55.81% 背景組成上，組織聚集成中央橢圓（散佈會平均掉每個 batch，隱藏要找的唯一失敗模式）；只改變**領取**端。**null arm（零工作）決定性**：`per_tile`（出貨）0.356 s（18.84 µs/tile、1.000x）、`batch:8` 0.291 s（2.08 µs/tile、1.221x）、`batch:32` 0.288 s（0.93 µs/tile、1.233x）；全 27,565 tile 總領取成本 0.356 s ÷ `w=4` wall 5,854.9 s = **0.0061%** → e2e 天花板 **1.00006x**，比值得建造的低四個數量級（反模式 #2）。**modelled-work arm**（實測 per-tile 時長壓縮 50x——對本問題是**保守**的，因為壓縮工作而保留全額 IPC，使領取開銷的占比膨脹 50x）：`per_tile` 48.857 s（tile spread 160、493）；`batch:8` 48.530 s（1.0068x，304、512）；`batch:32` 48.459 s（1.0082x，**1,344、1,011**）。即使 IPC 占比膨脹 50x，批次領取只買 **0.7–0.8%**（真實時長下 ≈ **0.015%**）；而 **tile spread 隨 K 單調成長**——doc 20 §2 Candidate B 選動態佇列要避免的負載不平衡確實存在、且 batch 越大越糟。**不建**。
- **§4 Item 2 管線化 Phase D 讀取（推翻 doc 35 §3.3）**：`_prefetch_bands()`（新，在 `scripts/stitch_probe.py`）在背景執行緒取 band k+1 同時編碼 band k（與 pipeline 自己的 `prefetch_tile_reads` 同形、depth 1，最多兩個 band 常駐）；`--pipelined` 把每個管線化 arm 與**自己的序列對照**交錯跑，其他一概不變（反模式 #7）。
  - **控制組重現 doc 35 §3.2（4.055 GP/6,900 tile）**：read-only 30.44 → **30.53 s**；A 62.55 → **62.40 s**；B 79.33/0.788x → **79.52/0.785x**；C 70.71/0.884x → **72.90/0.856x**（四個獨立重現皆在 3% 內）。
  - **新 arm（wall/read/pyramid/encode/container/vs A）**：A 62.40 s（1.000x）；B 79.52 s/36.07/18.67/18.80/0.54/0.785x；**B + pipelined read 45.72 s**/**1.56**/19.11/19.09/0.57/**1.365x**；C 72.90 s/36.56/6.98/23.24/0.55/0.856x；**C + pipelined read 39.47 s**/**4.23**/6.23/22.87/0.56/**1.581x**；全部 shape-matched（11 pages、128×128 tile、level-0 同形）且 per-level decode audit PASS。讀取幾乎消失：B 36.07 → 1.56 s（95.7% 隱藏）、C 36.56 → 4.23 s（88.4%）；兩者落在 doc 35 理想上限（1.47x/1.80x）的 7–12% 內。doc 35 的處置在其自己的條件下被反轉，原因恰是 doc 35 所指：pyvips 靠重疊讀取與編碼獲勝，串流候選以前序列付出，現在不再。
  - **端到端**（把 1.581x 套到 §2.1 的 `workers=4`）：Phase D 1,889.8 s → 1,195.3 s（省 694.5 s）；新 wall 5,160.4 s → **1.135x**；`workers=4` speedup 1.745x → 1.980x（高於 doc 35 §3.3 的 ~1.094x，因為 Phase D 占比成長，不是槓桿變好）。
  - **spike 可能說謊的明顯方式被量測**：`stitch_probe.py` 以 hardlink 把 ~24 個相異 tile 的池子填成 6,900 個位置 → **來源在 page cache**；真實 Phase D 走 27,565 個**相異** tile、46 GB 從磁碟讀；管線化只能省 `min(read, other work)`——若真實未快取讀取遠大於其餘則省得更少。每像素絕對成本顯示風險真實：probe A 4.055 GP 62.4 s = 15.4 s/GP；真實 Phase D 16.201 GP 1,889.8 s = **116.6 s/GP（7.6x）**。
  - **§4.4 真實來源的未快取讀取**（`scripts/phase_d_real_read_probe.py`（新）把真實 46 GB `overlay_annotated` hard-link 進 scratch，在真實 141,658×114,366 幾何下 drain `stitch_probe.py` 的 band walk；`/proc/<pid>/io` 確認 **46.5 GB `read_bytes`**）：**REAL read-only：981.00 s / 149 bands / 16.20 GP（60.55 s/GP）**。**讀取占比幾乎精確轉移**：doc 35 §3.2（4.055 GP、cache-resident）30.44/62.55 = **48.7%**；本輪（16.20 GP、真實來源、冷）981.0/1,889.8 = **51.9%**。spike 的**絕對**數字不外推（每像素 7.6x 誤差，doc 27 §6.4 已發現 Phase D 外推錯 1.8x，這次更糟），但**驅動管線化勝利的那個比例會轉移**：讀取在兩種規模、cached synthetic 與 cold real 都約占 Phase D 一半，正是隱藏它有用的區間。殘餘非讀取工作 1,889.8 − 981.0 = 908.8 s，故完美重疊 Phase D 不會低於 **981.0 s（1.926x）**。三情境：今日 1,889.8 s/5,854.9 s/1.000x；實測 spike 比 1.581x → 1,195.3 s/5,160.4 s/**1.135x**；完美重疊 981.0 s/4,946.1 s/**1.184x**。doc 38 §4 閘門通過；兩個 caveat：981.0 s 是**冷讀**，而真實 Phase D 讀的是分析階段幾分鐘前自己寫的 tile，故實際 in-situ 讀取在此或以下 → 51.9% 是**上界**；spike 之下一切仍是 spike（doc 32 → doc 35 曾在同一步反轉）。**一個額外條件**：候選 B/C 寫 1.89 GB（vs A 2.44 GB）因套用 Predictor 2 而出貨路徑不套用；doc 35 把 QuPath 開啟/渲染一致檢查停放為「不再決策相關」，**現已再次決策相關且從未被執行——未經它不得採用**。
- **§5 Item 4 刻意不做**：`workers≥5` 是 doc 38 明確標「不要做」的項目；§2.3 加強反對（更多 worker 的邊際回報受同一組成天花板限制 ~2.28x，而主導 wall 的項目（Phase D 32.3%）完全不隨 worker 縮放——第五個 worker 無法觸及三分之一的 wall）；**本輪沒有任何新 VRAM 證據**（見 §6.2）。
- **§6 doc 38 §4 決策閘門評估**：step 1 偏離 2.216x → **觸發**（1.745x，21% miss；§2.3/§2.4 做了「找原因」）；step 2 spike 接近 ~1.09x → **通過**（Phase D 1.581x、e2e 1.135x；cache-resident 風險由直接量測消除；建議建造並重量，不以 spike 採用）；item 3 → 關閉、不建（0.006%）；correctness veto → 全過（§2.2、§4.2）。**§6.2 本輪造成的量測缺口**：沒傳 `--worker-timings` → **本 run 沒有 per-stage 拆解**（`workers>1` 下 parent 的 monkeypatch 計時只看到 parent，除 `D_stitch_overlay` 與兩個 `C_export_*` 外全空；`gc_refreeze` 因 freeze 發生在 worker 內讀 `n_freezes: 0`——量測假象，非迴歸）；**沒有 peak-VRAM**（沒傳 `--gpu-dmon`；`peak_cuda_reserved_gb` 為 `0.0` 因 parent 在多行程下從不在裝置上分配；兩次 spot `nvidia-smi` 讀 9,181 與 5,017 MiB 是單點非 peak）——任何 `workers≥5` 討論仍建立在 round 8 與 round 11 的 VRAM 數字上。
- **§7 仍待**：(1) 把管線化讀取建進 `_stitch_overlay_slide` 並在整片重量（→ round 13）；(2) 整片數字的誤差棒（Phase D +40–52% 是 n=1 對 n=1）；(3) 對 Predictor-2 候選的 QuPath 開啟/渲染檢查；(4) `run_batch` 的 `workers: int = 4` 預設（latent footgun）。
- **§8 重現**：整片 `perf_measure.py --mp-workers 4 --stream-precut --metrics-dir ...`（參考主機 1 h 38 m）；`overlay_pyramid_audit.py`；`mp_queue_claim_probe.py --work null/sleep`；`stitch_probe.py --candidates --pipelined --slide-w 70909 --slide-h 57183`；`phase_d_real_read_probe.py`（16 分鐘、冷讀 46 GB；需 drop caches 或用近期未讀的來源）。metrics 歸檔於 `measurement/_metrics_r12/`。

---

## 14. Round 13：Phase D 管線化縫合上線（doc 40 規劃、doc 41 落地，2026-07-29）

### 14.1 doc 40 — 計畫（純規劃）

- **doc 39 §7 留下的四項**：(1) 把管線化讀取建進 `_stitch_overlay_slide` 並在整片重量；(2) 整片數字的誤差棒；(3) Predictor-2 候選的 QuPath 檢查——**2026-07-29 使用者已在 QuPath 打開候選輸出並確認正常，閘門清除**；(4) `run_batch(..., workers: int = 4, ...)` 預設與文件的不符——**2026-07-29 決定 `workers=4` 就是預期預設**（非要回退的 bug）。`docs/BACKLOG.md` item 1b 已寫同樣的下一步。
- **§1.1 槓桿報酬已從兩個獨立方向量過**：B+pipelined read 45.72 s 1.365x；C+pipelined 39.47 s 1.581x；真實 46 GB 來源冷讀占 Phase D 51.9%（vs 48.7%）；e2e 投影 **1.135x**（完美重疊底線 1.184x）→ `workers=1→4` 加速由 1.745x → ~1.980x。
- **§1.2 「建進 `_stitch_overlay_slide`」不是「在出貨路徑加一個預取執行緒」**：出貨路徑（`m0_stitch.py:438` → `:396` → 一次 `pyvips` `tiffsave`）本來就免費把自己的讀取隱藏在自己的編碼之後（這正是它在 doc 35 贏過兩個 tifffile 候選 0.788x/0.884x 的原因）；`_prefetch_bands` 無法接到 pyvips 上，因為 pyvips 沒有可管線化的讀取步驟；勝利全在 tifffile 串流候選這一側。所以「build it」= **替換** pyvips join+encode 呼叫為 `scripts/stitch_probe.py` 的 `cand_tifffile_streamed()` 移植（`_band_source` 逐 band 讀、`_prefetch_bands` 包裝、`_encode_tile_row` 的 Predictor-2 LZW、CPU 或 GPU `_shrink2_*` 做 pyramid），而非增量補丁。
- **§1.3 B vs C**：B（`tifffile_cpu`）1.365x、CPU pyramid（`_shrink2_cpu` box filter）、不碰 CUDA；C（`tifffile_gpu_pyramid`）1.581x、GPU pyramid（`_shrink2_gpu`，torch/CUDA）、**在 `workers>1` 的 parent 首次碰 CUDA**（`hybrid_pipeline.py:250` 與 `m0_multiprocess.py:250` 註解：`workers>1` 時 parent 從不碰 CUDA、模型只在 spawn 的 worker 內載入；`_finish_batch` → `_stitch_overlay_slide` 在同一個 parent、所有 worker 結束之後執行）——不是不安全，但是 doc 39 未提的架構首次。**建議先出貨 B**（playbook「prefer the simplest solution that clears the bar」）。
- **§1.4 QuPath 檢查已通過**：候選 B/C 寫 **1.89 GB vs 出貨路徑 2.44 GB**，因套用 TIFF Predictor 2 而出貨 `pyvips` 呼叫為 `predictor="none"`（`m0_stitch.py:475` 的註解記載 round 9 缺陷：pyvips 預設 *horizontal* predictor；tifffile 候選依 `_encode_tile_row` 的 docstring 正確使用 Predictor 2）；使用者 2026-07-29 在 QuPath 直接打開 doc 39 §4.2 audit 通過的 `B + pipelined read`/`C + pipelined read` TIFF，確認渲染無差異——這是 programmatic `--verify` audit 無法取代的人眼渲染檢查（round 9 / doc 32 §5.1 即因只信 programmatic 檢查而吃虧）。
- **§1.5 誤差棒真實但不閘門建造**：報酬來自 spike，並由獨立的冷讀量測印證，兩者都不依賴 32.3% 恰好正確；建造出貨後再量整個 `workers=4` wall 幾乎免費多一個整片資料點；專用重複 run 活動（~3 h × 數次）不納入關鍵路徑。
- **§1.6 `run_batch` 預設不符——已決定並已對齊**：`run_batch` docstring（`hybrid_pipeline.py` `workers:` Args）從「`1`（預設）」改為「`4`（預設）」，理由改寫：單 tile 請求保持低 worker 數的理由不再是「預設保護你」，而是 **`backend/api/hybrid.py:89` 明確傳 `workers=1`** 覆蓋新預設（該覆蓋現在**更**承重；單一 API 請求為一個 tile 付 4 個 worker 的模型 init 成本會是真回歸）；CLI `--workers` argparse 預設（`build_arg_parser`）從 `1` 改 `4`，移除舊的「`>1` 尚未通過整片驗證」（round 8 已清除、round 12 再確認）；`backend/algorithms/hybrid/CLAUDE.md` 的 Running 與 M0 架構段與 `README.md` 的跨 tile 多行程段由「預設 `workers=1`、尚未放行生產」更正為 `workers=4` 預設並引用 round 8/12。
- **§3 計畫表**：0 ✅ 對齊 `workers=4`；1 ✅ QuPath 通過；**2 把候選 B（`tifffile_cpu` + `_prefetch_bands`）移植進 `_stitch_overlay_slide`，放在預設 pyvips 的 config 開關之後**（比照 round 11 `expandable_segments` 的上線模式：先在旗標後量測，之後以獨立部署決定翻預設），保持 `backend/tests/test_stitch_scratch_cleanup.py` 與 `tests/test_stitch_pyramid_levels.py` 綠燈並擴充；3 整片 `workers=4` 重量（`perf_measure.py --mp-workers 4 --stream-precut`，對 `d2ccc46b` baseline 做 `report.csv` diff + `overlay_pyramid_audit.py`；若遠低於 ~1.135x 先找原因）；4 整片誤差棒活動（2–3 次每個 `workers=1`/`4`）；5（stretch）以 C 換 B 取 1.365x → 1.581x。
- **§4 閘門**：item 1 已通過；item 3 若明顯低於 ~1.135x（或像 doc 32→35 那樣反轉）→ 停，不翻預設；任何 correctness veto 不豁免；`workers≥5` 維持關閉。

### 14.2 doc 41 — 落地：候選 B 已建、整片量測、成為預設

- **結果一張表**：

| | round 12（`pyvips`） | round 13（`tifffile`） | |
| --- | --: | --: | --: |
| **`workers=4` end-to-end wall** | 5,854.9 s | **4,877.4 s** | **1.200x** |
| Phase D（`D_stitch_overlay`） | 1,889.8 s | **987.8 s** | **1.913x** |
| Peak RSS | 45.56 GB | **16.95 GB** | −62.8% |
| `overlay_slide.tiff` | 7.50 GB | 5.85 GB | 0.78x |
| `report.csv` rows | 356,220 | 356,225 | **+0.0014%** |
| `config_hash` | `d2ccc46b` | `d2ccc46b` | 相同 |

  doc 40 §1.1 投影 **1.135x**（完美重疊底線 **1.184x**），實測 **1.200x——高於兩者**；doc 38 §4 停損不觸發；預設翻為 `"tifffile"` 是出現數字之後的獨立明確決定。
- **§1 建了什麼**：`config.stitch_backend` 選擇編碼器；`_stitch_overlay_slide` 成為 dispatcher；出貨主體**逐字**搬到 `_stitch_overlay_slide_pyvips`（舊路徑逐位元保留，作為 fallback 與量測控制組）；`_stitch_overlay_slide_tifffile` 是 `scripts/stitch_probe.py` `cand_tifffile_streamed()` 的移植：`_band_source`（一次一列 source tile，以 `_join_overlay_tiles` 同樣方式 join——`arrayjoin` 對非均勻 grid 會錯誤補邊）、`_prefetch_bands`（單執行緒、depth 1）把讀取藏在編碼之後、`_shrink2_cpu` 做 pyramid、`_encode_tile_row` 自己套 TIFF Predictor 2、`tifffile.TiffWriter` 做容器。**與 spike 程式碼兩個刻意差異**：(1) per-phase 計時器、`keep_levels`、GPU pyramid 分支被移除（C 是另一個決定；`perf_measure.py` 已從外部計時 Phase D）；(2) `_slide_dims` 取代 spike 的尺寸探測——`cand_tifffile_streamed` 為了得知完成高度/寬度而呼叫 `_join_overlay_tiles`，會同時打開 27,565 個 tile，且該呼叫在 spike 的 `t_all` 之前，所以其成本從未計入 1.365x；直接引入會對出貨路徑增加未量測成本；`_slide_dims` 每欄/每列只讀一個 header（整片 332 次 open），因同欄同寬、同列同高（`core_crop_bounds` 構造保證）而精確，tifffile 路徑因此只需 `_ensure_nofile_limit(cols + 256)` 而非 `cols × rows + 256`。**`stitch_backend` 放入 `config._HASH_EXCLUDE`**：它**確實**改變輸出 bytes，需要論證——兩個 hash 消費者都不讀 `overlay_slide.tiff`：worker 守衛比較 parent 與 child 的演算法設定、resume checkpoint 儲存每個 tile 的 `owned` 細胞清單，兩者都由 M1–M3 決定，而它們在 Phase D 之前就完成；納入 hash 會為了換編碼器而丟掉多小時整片 checkpoint；兩次 run 皆回報 `config_hash = d2ccc46b`。
  - **測試**：`backend/tests/test_stitch_scratch_cleanup.py` 與 `tests/test_stitch_pyramid_levels.py` **兩個 backend 都跑**（擴充而非取代，pyvips 覆蓋完整保留）：3 → 8 tests；新增 `test_backends_agree_on_layout_and_level0_pixels`（doc 32 §3 的不可比性守衛作為測試：頁數相同、容器 tile 大小相同、**每一層 pixel-identical**；這把 `_shrink2_cpu` 釘死到 pyvips 自己的 `region_shrink` kernel，不是假設——§3.2 在真實出貨的 141658×114366 overlay 上量到 L3..L11 皆為上一層的 bit-exact 2×2 box shrink）、`test_unknown_backend_fails_loudly`（拼錯 backend 要 raise，不可默默 fall through 到 pyvips，否則所有對此開關的量測都是謊言）。
- **§2 整片確認**：同 doc 39 §2 協議、同 conformed pair、同主機、`workers=4`、`--stream-precut`；`scripts/perf_measure.py` 新增 `--stitch-backend`（不必編輯 gitignored `config.py` 再讓它留在翻開狀態；選值記錄在 `_timings.json`）：`end_to_end_total_s` 5,854.9 → 4,877.4；`runbatch_BCD_s` 5,854.8 → 4,877.3；`D_stitch_overlay` 1,889.8 → 987.8；`peak_rss_gb` 45.558 → 16.953；tiles 10,801/16,764 → 10,800/16,765。**Phase D 的 1.913x 超過自己 spike 的 1.365x**：doc 39 §4.4 量到冷讀在整片占 Phase D **51.9% vs crop 48.7%**，槓桿隱藏讀取——更多讀取可隱藏 = 更多勝利；Phase D 現為 `workers=4` wall 的 **20.3%**（987.8/4,877.4），自 32.3% 下降，不再是 pipeline 最大單項。
  - **兩個沒人預測的效應**：(1) **peak RSS 45.6 → 17.0 GB**——band 串流取代了持有 27,565 個檔案的 pyvips lazy join；doc 27 §6 記為「新的、先前未陳述的主機需求」的 61–62 GB 大幅放寬（非刻意優化，值得明說因為它改變文件化的部署約束）；(2) **resolution tag 現在精確**：pyvips/libtiff 把 `XResolution` 存為 C `float`，25.4 變 `float32(25.4) = 25.399999618530273`，寫成有理數 `13316915/524288`；QuPath 讀到 1000.000015 µm/px，回報 slide 寬 141,658,002.13 µm；tifffile 寫 `127/5` = 精確 25.4 → 精確 1000 µm/px → 141,658,000.00 µm（相對差 1.5×10⁻⁸，無實際影響）。**兩者都是佔位校準**（1 px = 1 mm，約為真實 40× 像素的 ~4000x 偏差），故在 QuPath 中取得的任何 µm 測量在兩者都無意義；本輪未改變此事，但因差異在 QuPath 可見而記錄。
- **§3 correctness veto**：`report.csv` 356,225 vs 356,220（**+0.0014%**，比 round 12 的 −0.002% 與 round 8 的 −0.01% 更緊）；`summary.txt` mean black dots 3.24 → 3.25、black total 148,385 → 148,417、red total 108,458 → 108,462、valid cells 45,732 → 45,735；verdict split 到小數點一位相同（ratio <2 75.3%、≥2 24.7%）；Phase D 無法影響這些（在表格寫出之後才跑）。**artifact**：`overlay_pyramid_audit.py` 各層 PASS、`Predictor=2` 跨 12 IFD 一致；**金字塔自洽**：L3 → L11 每層都是上一層的 **bit-exact** 2×2 box shrink（`maxdelta=0`）——audit 腳本**不**做此檢查（只測各層亮度），這證明移植的 `_shrink2_cpu` 在真實規模建出數學上正確的 pyramid，而非只是看起來亮；**無 tile 位移**：27,565 個 level-0 容器 tile 與 r12 artifact 比較，最差 tile 4.28% 像素不同（annotation 非決定性）；被錯置的 tile 會差 ~100%。
- **§4 預設翻開**：比照 round 11 `expandable_segments`：先旗標後量測、再獨立決定翻預設；`config_example.py`/`config.py`：`stitch_backend: str = "tifffile"`；`pyvips` 保留為 fallback 與未來 Phase D 量測的控制組（絕不丟棄控制組）；翻開後 `compute_config_hash` 仍回 `d2ccc46b`，既有 resume checkpoint 跨變更仍有效。
- **§5 這輪沒做**：(5.1) **item 4 誤差棒活動沒跑**（2–3 次 × `workers=1`/`4`，11–16 h 獨佔 GPU），整片比較仍是 n=1 vs n=1；(5.2) **item 5 候選 C 應維持關閉**：Phase D 現占 20.3% wall，C 對 B 在 spike 規模的增量 1.581/1.365 = 1.158x 僅 Phase D，e2e 約 2–3%，代價是 `workers>1` parent 有史以來**第一次 CUDA allocation**——B 拿走大部分勝利後，報酬/架構風險比反而變差；(5.3) **QuPath 回報停放而非解決**：期中收到一份回報——某些 slide 有些 tile 渲染錯誤，歸因於 round 9 的 pyramid 修復；在 `r12_fullwsi_w4/overlay_slide.tiff` 無法重現，且一個發現反駁了該歸因：**修復後檔案的 level 0 與 `overlay_slide.tiff.predictor-broken` 逐位元相同**（27,565 個取樣容器 tile `maxdelta=0`）；修復前檔案 `L0 pred=2, L1..L11 pred=1`，其 level 0 一直解碼正確，修復只動過縮小層，無法解釋全倍率可見的現象；同時檢查乾淨：27,565 個 source tile 恰為預期 core-crop 尺寸（`join(expand=True)` 未悄悄錯誤補邊）、grid 加總精確 141658×114366、`workers=1`/`workers=4` stitch 吻合到 0.0002%；一個虛驚：每個 core seam 上 45× 像素不連續是**刻意的**藍色虛線 grid（`config.draw_window_grid = True`，畫在左側 tile 的最後一欄）；停放至有能重現的 slide。**暴露的缺口**：`overlay_pyramid_audit.py` 只測各層亮度，會放行有 tile 擺放缺陷的檔案；§3.2 的「層對降採樣」檢查與 source-tile-size 檢查便宜且更強——本輪沒做。
- **§6 重現**：`perf_measure.py --mp-workers 4 --stream-precut --stitch-backend tifffile`（參考主機 1 h 22 m）；`overlay_pyramid_audit.py`；`stitch_probe.py --stitch-backend pyvips,tifffile --slide-w 70909 --slide-h 57183`（每個 backend 計時一次出貨的 `_stitch_overlay_slide`，arm 之間重建輸入因成功 stitch 會刪除輸入；`--candidates` 量 spike 程式碼，此量實際出貨的）。**4.055 GP 消融**：pyvips 63.30 s/2.44 GB/PASS；tifffile 46.52 s/1.89 GB/PASS → **1.361x（vs doc 39 spike 的 1.365x，重現在 0.3% 內）**，Predictor-2 大小比 2.44 → 1.89 GB 與 doc 35 §3.4 記錄相同——在投入 1.6 h GPU 於 §2 之前就已證明移植保留了 spike 的勝利。
- **§7 仍欠**：(1) item 4 誤差棒；(2) 強化 `overlay_pyramid_audit.py`（金字塔自洽與 source-tile-size 檢查）；(3) QuPath 渲染回報需要重現 slide；(4) 真實解析度校準（兩個 backend 都寫佔位 1 px/mm）。

---

## 15. Round 14：`workers>1` 下的 GPU↔CPU 資料搬運（doc 42 規劃、doc 43 落地，2026-07-30）與畫布位移調查（doc 44）

### 15.1 doc 42 — 計畫（純規劃；範圍明確限定 `workers>1`）

- **§0 為何「聚焦 `workers>1`」改變能引用什麼**：本專案歷史上**所有**與 GPU↔CPU 資料搬運相關的既有量測都在 `workers=1`（或早於跨 tile 多行程，或多行程重跑被跳過/wall 結論被保留）。逐項檢查：Option L 的 `B2r_tile_read` 預取機制（doc 35）在 `workers=4` 機制確認、crop wall 主張被保留（−0.83% 在雜訊內）；「4.21% wall、99.8% 隱藏」僅 `workers=1`（doc 37 §4.1）；CUDA-stream/depth-2 bubble 重設計（doc 17 §3 → doc 18 §3，round 4）**早於多行程**；Cellpose `_from_device` 24.2% MAIN 臂樣本（doc 23 §7，`workers=1` medium crop）僅 `workers=1`；worker 內 depth-2 管線已在多行程下測過並拒絕（doc 21 §6，+2.8% 慢、W=3 持平；`m0_multiprocess.py:225-226` 註解）；CUDA MPS 是真正與 `workers>1` 相關的測試（合成 launch-bound +44%、端到端平坦）。四個 process 共用一張實體 GPU 與一個實體磁碟是質上不同的制度：PCIe 頻寬與 GPU copy engine 是共享的、每個 process 有自己的 CUDA context 與 default stream 且無內建協調、OS page cache 面對四個併發讀者。**每個先前數字視為「`workers=1` 已知，`workers=4` 未知」。**
- **§0.1 直接回答：`workers=4` 對這些方向零收益嗎？否**：UNet++ `.to(device)` 小是因為資料量微不足道（~12.6 MB/tile vs 組織 tile ~600 ms GPU 計算）；已試過的唯一 CUDA-stream 設計之所以被封頂是因為那個特定 bubble 很窄（四個 process 已讓 GPU 忙碌後更窄）；MPS 平坦是因為本 pipeline 並非受 launch/context-switch 開銷限制。真正未被證明為小的兩列：**Cellpose `_from_device`**（單 process GPU 執行緒時間 24.2%，從未在 `workers=4` production 制度量過）與**磁碟 scratch 往返**（若 `_from_device` 是真實可修的搬運/同步成本，四個 worker 各自付，修一次四個一起受益）。
- **§1 Discover（以 `codegraph_explore` 追蹤 `m0_multiprocess.py`）**：`_run_tiles_multiprocess`（`:250`）用 `mp.get_context("spawn")`；**parent 在這條路徑上完全不碰 CUDA**；`PYTORCH_CUDA_ALLOC_CONF` 於 spawn 之前寫入 `os.environ`；`_mp_tile_worker`（`:138-247`）每個 worker 獨立呼叫三個 `_init_*`（自己的權重、自己的 CUDA context、自己的 default stream），然後跑自己的兩執行緒 pool 迴圈（專用 `tile-read` 單執行緒 pool 做 `prefetch_tile_reads` depth 1 + 專用 `tile-cpu` 單執行緒 pool）；**四個 worker = 四個獨立讀取 pool、四個獨立 CPU pool、四個獨立 CUDA context 同時競爭同一張 GPU 與同一個磁碟**；depth 刻意封頂 1；**`workers=4` VRAM 餘裕緊**：92.2% of 32,607 MB（round 11，`expandable_segments:True`），約 **2.5 GB 是四個 worker 合計的餘裕**，非每個 worker；任何增加每 worker 裝置端狀態的設計都要對共享 2.5 GB 天花板評估（host 端 pinned buffer 不計）；磁碟 I/O 現在是四路併發（doc 27 §6.4 的 page-cache 競爭診斷是以**一個** process 的 RSS 推論，四個 process 各持模型權重與 per-tile 狀態、四個 `tile-read` 執行緒對同一 scratch 目錄發出重疊隨機讀）。
- **§2 Amdahl 重新推導（`workers=4` 制度）**：零磁碟 scratch——機制確認但整片 `workers=4` 的 wall 占比從未量（需先量）；CUDA-stream bubble——ceiling ≤1.065x 在多行程之前量，多行程下更小；worker 內 depth-2——已在多行程測過，平坦；UNet++ `.to(device)` pinned memory——×4 併發仍在 ≤~50 MB in flight，仍遠低於 PCIe 頻寬；Cellpose `_from_device`——24.2% 是單 process 數字，四 process 併發呼叫時搬運/同步可能膨脹或占比縮小（分母變小）→ **未知，且多了一個維度**（不只是「真搬運 vs GPU-wait」，還有「4 路 GPU 競爭是否改變兩者」）；MPS——已測平坦。
- **§3 計畫（皆在 `--mp-workers 4` 下，cheapest-first）**：**Item 1** 在 crop 規模 `--mp-workers 4` 量 `B2r_tile_read` 在四路併發讀取下的真實占比（`perf_measure.py --worker-timings`，與 `workers=1` 參考 14.02 ms/tile 組織、6.82 ms/tile 背景（doc 33 §1）比）——決定 Option M 的跨 process IPC 是否值得；**Item 2** 在真實 4 路 GPU 競爭下追蹤 `_from_device`（**用 Nsight Systems 而非 py-spy**——py-spy 無法分辨 `cudaMemcpyAsync` 與 `cudaStreamSynchronize` 且在此主機 attach 不穩（doc 37 §2）；Nsight 原生追蹤整個多 process 樹）；**Item 3** pinned memory/`non_blocking` 微基準（在一個 spawned worker 內、在其他三個 worker 的併發負載下，不是隔離環境）；**Item 4** 僅在 1 或 2 顯示 `workers=4` 特有 ceiling ≥10% wall 時：設計 Option M 的跨 process IPC（`multiprocessing.shared_memory`）；或 per-worker 修復（pinned staging buffers、`non_blocking=True`、每 worker 專屬 copy stream，需對共享 2.5 GB 檢查）；或若為 GPU-wait → 不是資料搬運修復，而是 MPS 家族問題，doc 21 §5 已測平坦，重申而非新提；**Item 5** 不排程：CUDA-stream bubble、worker 內 depth-2、MPS、fork-based 重用（Candidate E）。
- **§4 閘門**：VRAM 硬約束（共享 2.5 GB）；item 1 若顯示磁碟競爭未實質惡化 → 停；item 2 若 GPU-wait 主導 → 停；correctness veto 現明確 per-worker（只在真正四路併發下才出現的 bug 不會在 `workers=1` 檢查中暴露）；一切以真實 `--mp-workers 4` wall 的比例判斷。

### 15.2 doc 43 — 落地：三項全部關閉負向，沒有動 pipeline code

- **結果一張表**：

| doc 42 項目 | 開啟時的問題 | `--mp-workers 4` 實測 | 閘門 |
| --- | --- | --: | --- |
| **1** `B2r_tile_read` 在 4 路併發讀取 | 四個讀者是否在磁碟/page cache 競爭，使讀取超過 `workers=1` 的 4.21%？ | 每個 worker wall 的 **1.34–1.39%**，*完全暴露*（prefetch OFF）；Amdahl **1.014x** | **停** — Option M IPC 重設計維持停放 |
| **2** Cellpose `_from_device` 在 4 路 GPU 競爭 | doc 23 §7 的 24.2% 是真 D2H 搬運嗎？競爭會惡化它嗎？ | 24.2% **重現**（worker wall 的 25.6%）但 **96.3% 是 GPU-wait**；真正 D2H 拷貝僅占它的 **0.7%**、wall 的 **0.18%** | **停** — GPU-wait 分支 |
| **3** pinned/`non_blocking` 對 ~12.6 MB/tile | 4 process 的 PCIe 競爭是否改變「可忽略」預測？ | pinned H2D 隔離下快 **14.1x**，但端到端價值 **~0.4% wall**；競爭只多 **+4–8%** median | **停** — 預測被證實非假設 |

  唯一在四路競爭下**改變**的，是 doc 42 預測不會是搬運問題的那個：`_from_device` 內每次呼叫 GPU-wait 從 `workers=1` 的 **38.7 ms → 101.8 ms（2.63x）**——四個 process 的 kernel 在一張裝置上排隊，也就是 doc 21 §5 已用 MPS 量過、端到端平坦的失敗模式。
- **§1 實際跑了什麼**（全部用 match24 組成匹配 crop、576 tile、55.9% 背景，`--stream-precut`，出貨的 `cuda_alloc_conf=expandable_segments:True`，`config_hash` 皆為 **`d2ccc46b`**——與 round 9 相同，故 `workers=1` 參考數字是同配置產生）：`r14_m24_w1_r{1,2}`（`workers=1` control）、`r14_m24_w4_r{1,2}` + `r14_m24_w4_pf_p{1,2}`（`workers=4` prefetch ON）、`r14_m24_w4_noprefetch_r{1,2}` + `r14_m24_w4_npf_p{1,2}`（prefetch OFF，item 1 的 ablation）、`r14_m24_w{1,4}_fromdev`（item 2 split probe）、`transfer_probe.json`（item 3，0 vs 3 sibling）。`workers=4` 在 **104.7 s** 跑完 crop vs `workers=1` 的 **188.3 s** = **1.80x**（落在 rounds 12–13 記的 1.745x/2.138x 帶內，故為常態制度）。
  - **工具替換**：doc 42 指定 Nsight Systems，但此機**沒有 `nsys` 也沒有 `py-spy`**，安裝剖析器不在範圍；改以算術產生同樣的拆分：`torch.cuda.synchronize(X.device)`（耗盡 stream 欠的 → GPU-wait）→ `host = X.detach().cpu()`（已閒置的裝置只能搬位元組 → D2H copy）→ `out = host.to(torch.float32).numpy()`（host 端計算 → cast）。位於 `scripts/perf_measure.py:wrap_from_device()`，由 `HYBRID_PROBE_FROM_DEVICE=1` 閘門，**預設關**（額外 synchronize 會序列化 pipeline 賴以建立的 GPU/CPU 重疊，所以任何報告 wall-clock 的 arm 都沒裝）；`spawn` 子行程繼承環境，經 `mp_worker_probe:install` 重新匯入 `perf_measure`，一個 env var 到達四個 worker。synchronize 是**搬移**一個 `.cpu()` 本來就要付的等待而非新增——wall 確認（probed 189.5 s（w1）、105.1 s（w4）vs unprobed 186.8–189.8 與 103.3–107.2 s）。
- **§2 Item 1：讀取不競爭，ceiling 1.4%**：每次讀取成本（ms/read：tissue / bg / all）：r9 `workers=1` inline 7.010/3.409/4.997（doc 33 的 14.02/6.82 ms/**tile** = 這些 ×2）；r9 `workers=1` prefetch 8.852/4.075/6.182；r14 `workers=1` 9.276–9.377/4.588–4.683/6.655–6.753；**r14 `workers=4` 四 worker 彙總 6.864–7.065/4.488–4.720/5.536–5.754**——四個併發讀者**比一個便宜 ~25%**，doc 42 §1 提的競爭假設在此主機**不成立**；可能機制（提供為吻合的解釋，未獨立驗證）：不是磁碟效應而是 GIL/CPU——每個 `workers=4` process 花更多時間阻塞在競爭的 GPU 上，使自己單一 `tile-read` 執行緒得到更多無競爭 CPU；r14 `workers=1` prefetch 比 r9 更差（9.28 vs 8.85 ms/read）、同 `config_hash` → host/kernel 在 round 10–13 的漂移，故決策依據 round 內 `w1` vs `w4` 配對。**Ablation**（prefetch OFF 使每次讀取同步且完全暴露在 worker 主執行緒上 = 消除磁碟 Option M 所能買到的上界）：`workers=4` prefetch ON 時間配對 103.25、104.67 → **103.96 s**；OFF 105.66、105.39 → **105.53 s**；差 **+1.57 s（+1.51%）**；全部 4+4 arm 彙總 ON 104.68 / OFF 104.99 → **+0.30%**（前兩個 ON arm 與 OFF arm 相隔 ~7 小時，故背靠背重跑；兩種框架不一致的幅度超過各自大小，本身就是發現——噪音底線領域）；暴露讀取成本直接讀自 prefetch-OFF arm 的 worker timings：**5.68 s 與 5.86 s**（4 worker 合計）= **每 worker 1.42/1.46 s** vs ~105 s wall = **1.34%/1.39%**，Amdahl **1.0136x/1.0141x**。**決策（doc 42 §4 bullet 2）：停**——磁碟競爭未使 `B2r_tile_read` 惡化而是改善；`workers>1` 的「零磁碟 scratch」ceiling 為 **1.4% wall**，低於 `workers=1` 的 4.21%；Option M 的跨 process IPC 重設計維持停放——現在是對 `workers>1` 被**確認**而非假設；順帶：出貨的 prefetch 在 `workers=4` crop 買到 ~1.5%（與 doc 35 保留的「−0.83% 雜訊內」一致），沒有害處；doc 27 §6.4 整片 17.2% 是不同制度（RSS 驅逐 page cache，49 GB scratch），沒理由移除。
- **§3 Item 2：標竿候選 96% 是 GPU-wait**（`cellpose.core._from_device` 三向拆分）：

| | `workers=1` | `workers=4`（4 worker 彙總） |
| --- | --: | --: |
| 呼叫次數 | 1,016 | 1,016 |
| `_from_device` 總計 | 42.171 s（41.5 ms/call） | 107.449 s（105.8 ms/call） |
| ├ **GPU-wait**（sync） | 39.302 s（**93.2%**） | 103.437 s（**96.3%**） |
| ├ **D2H copy** | 0.708 s（**1.7%**） | 0.761 s（**0.7%**） |
| └ host cast+numpy | 2.161 s（5.1%） | 3.250 s（3.0%） |
| 搬運 D2H 位元組 | 7.195 GB → 10.16 GB/s | 7.195 GB → 9.45 GB/s |
| **D2H 占 worker wall** | 0.374% | **0.181%** |

  - **doc 23 §7 的標題數字重現，詮釋不成立**：`_from_device` = 107.4 s / 4 workers = 26.9 s per worker vs 105.1 s wall = **25.6%**，與 py-spy 在 `workers=1` 報的 24.2% MAIN 樣本近乎精確吻合；但 **96.3% 是 worker 阻塞在自己的 Cellpose 前向上**——`.cpu()` 對 CUDA tensor 會等 stream 排空才搬位元組，取樣剖析器分不出兩者（quickref red flag「剖析器的熱函式最後只是總時間的一小片」，由量測而非直覺抓到）；**實際搬運幾乎不理會另三個 process**：per-call D2H 0.697 → 0.749 ms（+7.5%）、有效頻寬 10.16 → 9.45 GB/s（−7.0%）——doc 42 §2「搬運可能排在 sibling kernel 之後而膨脹」被量測且**不發生**；**真正膨脹的是 GPU-wait：38.7 → 101.8 ms/call（2.63x）**——這是本輪找到的真實、大型、`workers=4` 特有的效應，但不是資料搬運成本，而是四個 process 的 kernel 在一張裝置上序列化，且已被計價（`workers=4` 1.80x 端到端加速是**扣除**這個排隊之後的淨值）。**決策（doc 42 §4 bullet 3）：停，不設計搬運修復**——pinned staging、`non_blocking=True`、per-worker copy stream 都只是在最佳化 **0.7%** 那一片，**96.3%** 那片不動——D2H 若變成零成本 Amdahl **1.0018x**；MPS 是能處理 96.3% 的修復家族，doc 21 §5 已測端到端平坦。
- **§4 Item 3：pinned memory 14x 更快，仍不值得**（`scripts/gpu_transfer_probe.py` 新；1024×1024×3 float32 = 12.58 MB = 一個 tile 的 UNet++ 輸入；測量行程本身是 `spawn` 子行程而非 parent，因為 `_run_tiles_multiprocess` 讓 parent 離開 CUDA）：

| 傳輸 | 隔離（median ms / GB/s） | 3 siblings | 競爭成本 |
| --- | --: | --: | --: |
| H2D pageable | 3.343 / 3.76 | 3.526 / 3.57 | +5.5% |
| **H2D pinned** | **0.237 / 53.16** | 0.256 / 49.13 | +8.2% |
| H2D pinned + `non_blocking` | 0.234 / 53.73 | 0.239 / 52.64 | +2.0% |
| D2H `.cpu()`（pageable） | 0.718 / 17.54 | 0.746 / 16.86 | +4.0% |
| D2H pinned | 0.323 / 38.94 | 0.341 / 36.95 | +5.4% |

  pinned 在隔離下 H2D **14.1x 勝利**且在競爭下存活；仍不值得建造——3.343 − 0.237 = **3.11 ms/tile 省下**；每 worker 144 tile = **0.45 s** vs ~105 s wall = **0.42%**，Amdahl **1.004x**（反模式 #5 最純粹形式：14x 微基準勝利價值四分之一個百分點）。次要觀察：median 在競爭下穩，tail 不穩（D2H `.cpu()` p95 0.735 → 1.420 ms（+93%）而 median 僅動 4% → 四路競爭增加 jitter 而非持續頻寬損失）；`.cpu()`（0.718 ms）比 `copy_` 進預分配 pageable tensor（1.099 ms）更快，故 pipeline 已在較好的 pageable 路徑；pipeline 內 Cellpose D2H（0.697–0.749 ms/call，平均 ~7 MB）與此微基準 `.cpu()` 列在大小差異內吻合。
- **§5 附帶發現：`workers=4` VRAM 如 doc 42 §1 所言一樣緊**：八個 `workers=4` ablation arm 中有一個（`r14_m24_w4_pf_p2` 第一次嘗試）以 `torch.OutOfMemoryError` 死於分配 **48 MiB**，且 `expandable_segments:True` 已啟用：「GPU 0 ... 31.36 GiB total, of which 8.44 MiB is free. Process 1369473 has 9.30 GiB Process 1369471 has 11.55 GiB Process 1369474 has 9.30 GiB」——三個 sibling 合計持 30.15 GiB，其中一個膨脹到 11.55 GiB（peers 為 9.30 GiB）；重跑成功，故本輪 `workers=4` crop 跑批 **10 次嘗試 1 次失敗**（8 ablation arm + split-probe arm + 1 重試）。非 doc 42 所問、非搬運問題，但直接證明 doc 42 §1「~2.5 GB 是四個 worker 合計，非每個」，且是出貨預設的**production fail-fast 風險**（在比整片小 48 倍的 crop 上）；記錄、未修，歸類為 `workers=4` 穩定性待辦。
- **§6 本輪改了什麼**：**無 pipeline code**。兩個僅量測的新增（皆在 `backend/` 之外）：`scripts/perf_measure.py` 的 `wrap_from_device()`（由 `HYBRID_PROBE_FROM_DEVICE=1` 閘門；新增 bucket `X_fromdev_gpuwait`、`X_fromdev_copy_d2h`（含位元組數）、`X_fromdev_cast_numpy`、`X_fromdev_TOTAL`）與 `scripts/gpu_transfer_probe.py`。correctness veto 不被觸發（沒有任何 worker 的搬運模式被改）。
- **§7 backlog**：**對 `workers>1` 以量測關閉**：「零磁碟 scratch」/Option M IPC（ceiling 1.4%，且讀取在四路併發下更便宜而非更貴）；Cellpose `_from_device` 作為搬運目標（96.3% GPU-wait；doc 23 §7 的線索跨四輪被當成「標竿候選」，現在在正確制度下以直接量測結案）；pinned memory/`non_blocking`（0.42%，在真實競爭下確認而非預測）。**重申而非重開**：CUDA-stream bubble、worker 內 depth-2、CUDA MPS（`workers=4` 每次呼叫 GPU-wait 2.63x 膨脹就是 MPS 形狀的問題，MPS 已測端到端平坦）。**誠實總結**：doc 42 中每個被點名的方向——pinned memory、shared memory、零磁碟 scratch、zero copy、非同步傳輸、CUDA streams——現在在 `workers=4` 都是被量測而非被假設，且沒有一個是時間所在；`workers=4` 的時間在 GPU kernel 執行與四個 process kernel 對一張裝置的排隊；**這條 pipeline 在 `workers>1` 沒有 GPU↔CPU 資料搬運瓶頸**；round 15 在沒有新證據前不該再找；doc 42 §0 的紀律（把每個數字標上其被量測的 process 制度）讓這在一輪內可回答，而非五輪。
- **§8 重現**：`perf_measure.py --mp-workers 4 --stream-precut --worker-timings`（`--mp-workers 1` control、`--no-prefetch` ablation）；`HYBRID_PROBE_FROM_DEVICE=1 perf_measure.py ... --label r14_m24_w4_fromdev`（絕不在要引用 wall-clock 的 arm 設定）；`gpu_transfer_probe.py --siblings 0,3 --iters 200`。metrics 於 `measurement/_metrics_r14/`。

### 15.3 doc 44 — 畫布位移：錯誤的輸入檔，以及隱藏它的 crop

- **發生了什麼**：round 8–13 的整片跑批餵給 hybrid pipeline 的是 `HER2_processed.tiff`/`DISH_processed.tiff`——它們是 **Module 1 輸出 = VALIS 的輸入，不是配準後影像**（`thriple_image_layer/config_example.py:26`「Module 1 輸出 / Module 2 輸入」；`:48` 以 `HER2_processed.tiff` 作 registrar 的 `reference_img_f`）。它們是原始 CZI→BigTIFF 轉換，各自在自己的 mosaic bounding box 上，不共享座標系；doc 27 §6.1 自己的 preflight 就顯示三個 modality 三個不同尺寸：IHC 141818×114366、DISH 141658×114415、HE 141717×116400。`PrecutStream.__init__`（`m0_module/m0_reader.py:130-138`）對不等 IHC/DISH 尺寸拒絕——本來能抓到；但 round 8 為效能驗證加入的 `scripts/full_wsi_validate.py` 中的 `conform_to_intersection()` 把兩者裁成 `min(width), min(height)`，**使它們大小相等卻未對齊**，守衛因此通過、pipeline 分析了未配準輸入。**這就是整個缺陷**：該 crop 本身的裁切只是從共同 `(0,0)` 原點起 160 px/49 px 的小片（0.14%），本身不會位移任何東西——它只是停用了檢查；回報的差距（141658×114366 vs 正確的 156222×134028）是原始與配準畫布的差，不是 crop 造成的。**橘色圓圈**（由 `cr.dish_nucleus_mask` 繪製，`m0_tile_runner.py:501-507`）佐證：未 warp 的 DISH 使每個 tile 的內容落在錯誤的絕對座標，所以每個圓都位移，而 tile 內的細胞仍分割同一塊局部組織、局部看起來正常——正好是回報的症狀。
- **修法——使用原流程已產生的檔案**：`module4_thumbnail.py` 本來就在全解析度 warp 兩個 modality 並輸出為 merge 的中間檔：`level = config.thumbnail.level`（預設 0 = 全解析度）；`dish_temp = temp_dir / f"dish_warped_lv{level}.tiff"`、`her2_temp = temp_dir / f"her2_warped_lv{level}.tiff"`；`dish_warped = dish_obj.warp_slide(level=level, non_rigid=True, crop="overlap")`、`her2_warped = her2_obj.warp_slide(level=level, non_rigid=True, crop="overlap")`。兩者都用 `crop="overlap"`，故**落在同一畫布、尺寸相同**，`PrecutStream` 守衛不需任何 conform 就通過。所以：讓 hybrid pipeline 指向 `<temp_dir>/her2_warped_lv0.tiff` 與 `<temp_dir>/dish_warped_lv0.tiff`，取代 `*_processed.tiff` 一對：`python hybrid_pipeline.py --ihc <temp_dir>/her2_warped_lv0.tiff --dish <temp_dir>/dish_warped_lv0.tiff --workers 4`。無演算法變更、無新程式碼、不改原流程。
- **變更**：**移除** `scripts/full_wsi_validate.py` 的 `conform_to_intersection()` 與 `--conform`（69 行；該函式為 round 8 效能驗證所加，不屬於原 pipeline）；`preflight()` 既有的「IHC size != DISH size」檢查不動，現在是唯一守衛——不匹配的一對會大聲失敗而非被默默拉平；其餘不變，特別是 `core_crop_bounds`/`_join_overlay_tiles`/`m0_module/m0_stitch.py`（tile seam-trim）不動且無關（round 13 doc 41 §3.2 已驗證它們不造成 tile 位移）；並刪除過期的 `_conformed/{ihc,dish}_conformed_141658x114366.tiff`。
- **對效能歷史的備註**：round 8–13 的 wall-clock 數字仍作為**計時資料**有效（秒數、tile 數、GB、RSS 與組織是否對齊無關），只是從未是有效的臨床輸出；在配準後的一對重新量測會改變 tile grid（更大畫布 → 更多 tile）與絕對 wall，但不改變那些輪次比較的比例（→ round 15 實際重量）。

---

## 16. Round 15：UI 預估時間（ETA）估算基準（doc 45 規劃、doc 46 落地，2026-07-30～08-01）

### 16.1 doc 45 — 量測與分析計畫（純規劃；比照 doc 09 的規格）

- **動機**：`../UI/13-numpy2-migration-and-analysis-ui.md` §7 與 `../BACKLOG.md`（"Analysis (hybrid) UI progress + cancellation"）記錄一個卡在演算法層的缺口——`run_batch` 執行中完全不回報進度，UI 只有 `pending/running/done/error`；`frontend/src/components/HybridPanel.tsx`（第 10–13 行）的設計原則：「backend 只回報一個整體 job status…**偽造 stage cursor 就是捏造進度**」。本計畫不是繞過它，而是把它延伸到「預估時間」：使用者想在送出分析（尤其是以小時計的整片）之前或期間，看到**有實測依據**的預估秒數/區間。模型設計成兩段式：送出前的靜態估計（`estimateTiles()` + 模型係數）；執行中的動態校正（需要逐 tile callback，不在範圍）。也是給醫師/教授報告的資料基礎（第 6 節誠實揭露）。
- **§1 現況盤點**：
  - 手臂模型（`wall ≈ max(MAIN, BG) + outside`，`outside = Phase_A + Phase_D + model_init`，誤差 <1.3%）；Phase A/D/init **不隨 `workers` 縮放**（序列、在 parent）。
  - 既有工具全部可重用：`perf_measure.py`、`composition_crop.py`（`--grid` 控制尺寸、`--target-bg` 控制比例，兩者正交）、`core_mask_map.py`、`wsi_projection.py`（要擴充成 `workers>1`、非線性 Phase D、多點迴歸）、`full_wsi_validate.py`（已移除 `conform_to_intersection()`）、`stitch_probe.py`、`generate_report.py`。
  - **doc 44 的畫布修正對舊數字的影響**：正確配準畫布 **156222×134028**（比舊畫布寬 +10.2%、高 +17.2%）；round 8–13 的 wall-clock 作為時間量測仍有效但不能作為本模型最終校準錨點；粗估 35,500–36,000 tile 僅為面積比例推算（巧合接近 round 6 前被推翻的「35,700 tile」），需先用 `core_mask_map.py` 在校正畫布上實測排除巧合。
  - **`estimateTiles()`**（`frontend/src/components/HybridPanel.tsx:22-34`）：`TILE_PX=1024`、`OVERLAP_PX=256`、`along(extent) = extent <= TILE_PX ? 1 : ceil((extent − TILE_PX)/(TILE_PX − OVERLAP_PX)) + 1`、`along(w) * along(h)`——尺寸輸入端已有；缺：(a) 背景/組織比例輸入（目前每個 tile 一樣貴）、(b) 有實測依據的每 tile 秒數係數。
  - **tile 尺寸恆定**（`config_example.py:207` `default_tile_size=1024`、`window_overlap_px=256`）：跨玻片尺寸可遷移性的前提——玻片越大只是 tile 數變多，單 tile 成本剖面不變；只需重驗背景比例（目前只有一張真實玻片 55.8%，跨玻片變異未知）。
  - 現有 anchor 清單（large 441 tile 14.1% 302.7/128.8 s；match24 576 tile 55.9% 188.8/88.3 s；comp24 576 tile 73.4% 134.9/65.5 s；舊畫布整片 27,565 tile 55.8% 10,666.1/4,877.4 s）——**沒有任何一列改變玻片本身的大小**；專案至今從未量過第二張不同物理尺寸的真實玻片；使用者已確認以「同一片裁切 + 純幾何合成大畫布」處理缺口。
  - `DISCOVERED-NOT-IMPLEMENTED.md` 全文核對：**沒有任何「預估時間」/「ETA」/「進度條」想法**——全新範疇。
- **§2 目標**：(1) 依 arm、依 tissue/background 分群的每 tile 秒數係數，以迴歸（非單一 anchor 外推）估計信賴區間；(2) Phase A 與 Phase D 的 scaling law（Phase D 已知真實冷讀下超線性：cached 15.4 s/GP vs 冷讀 116.6 s/GP，含讀檔 51.9%；Phase A 從未被系統性量過）；(3) `workers=1..4` 的 scaling 曲線（目前只有 1、4 兩點，doc 38/39 的 Amdahl 帳是兩點外推）；(4) 重複跑批取真正的變異數（所有整片數字都是 n=1）；(5) 可重複執行的量測腳本組合與報告骨架。**非目標**：不寫 UI/API；不設計逐 tile 進度 callback（`BackgroundTasks` 框架限制）；不驗證醫學正確性；不新增真實病人案例做尺寸母體抽樣（尺寸方向資料完全來自同片裁切 Set A 與純幾何合成 Set D——本計畫最大已知限制）。
- **§3 Discover 待確認**：(1) doc 44 的修正是否已被任何量測驗證（README 只到 round 12，doc 44 沒有 round 編號）；(2) Phase A 從未有 scaling 曲線；(3) MAIN 臂上 GPU 前向從未被拆成顯式 tissue tile 的 s/tile（唯一做過分群的是 `B2r_tile_read`：tissue 7.01 ms/read vs background 3.41–4.72 ms/read）；(4) 背景比例是否隨玻片而變——無法回答，列為限制。
- **§4 量測資料集（每組只變一個自變量）**：
  - **Set A 尺寸 sweep**（`composition_crop.py --target-bg 0.558`、`--grid` 11/21/24/~32/64/96，全部 `workers=1`）；
  - **Set B 組成 sweep**（固定 `--grid 24` = 576 tile、`--target-bg` 0.10/0.25/0.50/0.558/0.734/0.90，需先 preflight 確認這張玻片上存在對應窗口；對每個 bucket 以 `(n_tissue, n_background)` 對 `bucket_wall` 做線性迴歸、報告 R²）；
  - **Set C workers sweep**（沿用 2–3 個中型 anchor，`--workers 1,2,3,4`；`wall(w) ≈ serial + parallel/w`，`serial_portion` 應約等於 Phase A + Phase D + init，不吻合就查原因，不採信擬合值）；
  - **Set D 合成放大畫布**（純幾何，不跑完整 pipeline：只測 `stitch_probe.py`（含 `--pipelined`）與 Phase A 的獨立量測——重複貼上的組織對 MAIN/BG 的統計沒有意義、會污染 Set B 的係數）；
  - **Set E 重複跑批**（不要選整片級 anchor——選分鐘級中型 anchor，`workers=1`/`4` 各 n=3；整片 n=3 要 `(3×2.96h)+(3×1.35h)≈12.9h` GPU）；
  - **Set F 校正畫布整片驗證**（gated；最貴排最後；先確認 doc 44 現況並以 `core_mask_map.py` 重算 tile grid；`workers=1`/`4` 各一次起跳；作為 held-out 驗證點，不用來 fit）。
- **§5 建模**：Set B 迴歸 `bucket_wall ≈ a + b_tissue·n_tissue + b_bg·n_bg`（`wsi_projection.py` 的核心改進）；Set A/D 曲線先試線性、系統性彎曲則 power law（`s = k·GP^p`）或分段（page-cache 轉折）；Set C Amdahl；合成模型 `estimate = max(MAIN, BG)/parallel_speedup(workers) + Phase_A(GP) + Phase_D(GP) + init`（Phase A/D 不除以 workers）；Set F 驗證：**不吻合就指認失準子模型，不調整模型去遷就一個驗證點**（避免重蹈 CuPy 案例）；信賴區間 = 模型點估計 ± (Set E 變異 ⊕ 迴歸殘差) 平方和開根號，超出已驗證範圍的預測標「外推，誤差界未知」。
- **§6 產出**：原始資料（含 git/config hash/硬體戳記）、係數表（json/yaml）、`scripts/eta_estimate.py` 參考實作（純函式、不碰 `backend/api`/`backend/schemas`/`frontend`）、給醫師/教授的報告骨架。**§7 風險**：只有一張真實玻片（最大限制）；Set D 是幾何外推；doc 44 修正若未實跑過則 Set F 是第一次；不改 pipeline/API/UI；單一硬體型號。**§8 順序**：Discover 核對 → Set B → Set C → Set A → Set D → Set E → Set F。**§9** 與 `wsi_projection.py` 的關係：擴充而非另起爐灶（多點迴歸、非線性 Phase D、新增 `workers` 參數與信賴區間）。

### 16.2 doc 46 — round 15 量測執行與效能報告（2026-07-31～08-01）

- **環境**：RTX 5090（32,607 MiB）/ driver 580.173.02 / CUDA 13.0 / torch 2.11.0+cu130 / Python 3.11.15 / 20 cores / 62 GB RAM；git HEAD `8c3e8a3`，`config_hash d2ccc46b`（與 round 13/14 相同，未改任何 pipeline 程式碼）；總 GPU 時數約 7.95 小時（成功步驟加總）。
- **§0 摘要**：(1) **doc 44 的正確畫布首次被實測**：156222×134028、**35,700 tile**（204×175 grid、20.938 GP）、背景 **65.92%**（非沿用多輪的 55.8%）；(2) **首次解出顯式 tissue/background 每 tile 係數**：組織 **0.7298 s**、背景 **0.0460 s**，比值 **15.9x**、R² **0.99694**；(3) **首次有誤差棒**：n=3，CV **0.90%（`workers=1`）/ 0.49%（`workers=4`）**；(4) **Phase D page-cache 轉折首次被量到**：同一幾何 cached **11.26 s/GP** vs 冷讀 **60.90 s/GP**（**5.41x**）；(5) **模型在 held-out 的 Set F 上 core 準確（+4.3%/−6.7%）、Phase D 失準（−56%）**——依 doc 45 §5.5 不回頭遷就驗證點。與最舊資料比較：同一批 crop 檔案 round 1 對今日程式碼 `workers=1` 快 **2.85–2.88x**、`workers=4` 快 **5.92x**（large crop）；整片 round 1 當年投影 **18.87 h** vs 今日實測 `workers=4` **1.478 h**（**12.77x**）。
- **§1 Discover**：
  - **doc 44 修正從未被實跑過**（確認）：`docs/hybrid-pipeline/` 無比 45 更新的文件；唯一存在的正確畫布產物是 `backend/algorithms/hybrid/output/full_slide_run/`（2026-07-30，`_resume` 有 35,701 筆）但**沒有任何 timings JSON**（那是產品跑批非量測）→ Set F 是正確畫布上第一次有計時的整片跑批。
  - **正確畫布 tile grid（實測，取代 doc 45 §1.3 推算）**：`her2_warped_lv0.ome.tiff`/`dish_warped_lv0.ome.tiff` 都是 156222×134028（尺寸相等，`PrecutStream` 自然通過）；對比舊畫布（round 8–13）：141658×114366 → **156222×134028**（+10.3%/+17.2%）；gigapixels 16.20 → **20.938**（+29.3%）；tile 27,565 → **35,700**（204×175，+29.5%）；背景 55.8% → **65.92%**；**組織 tile 12,184 → 12,167（−0.14%）**；背景 tile 15,381 → 23,533（+53.0%）。**關鍵：組織 tile 數幾乎沒變，正確畫布多出的 +29.5% tile 全部是背景**（配準 `crop="overlap"` 把同一塊組織包在更大的畫布裡）；背景塊只有組織塊 1/15.9 成本，故畫布變大的代價遠小於 tile 數增幅。**「35,700」不是巧合**：當年（round 6 前）的假設是對著真正配準後的畫布算的，被推翻是因為它被拿去和錯誤 conformed 畫布（27,565）比較；**舊假設一直是對的，錯的是比較對象**。
  - **Phase A 從未有 scaling 曲線**（確認並已補）；在出貨組態下 Phase A 根本無法從一般跑批讀出——`--stream-precut` 把預切重疊進分析迴圈，`phaseA_precut_s` 只涵蓋讀檔頭與算格線（各尺度 0.03–0.05 s）。
  - **程式碼落差**：(1) `scripts/full_wsi_validate.py:81` 的 `from m0_reader import chunk_offsets`（`m0_reader` 已移入 `m0_module/`）→ preflight `ModuleNotFoundError` → **已修**；(2) `scripts/core_mask_map.py:36` 同上 → **已修**；(3) `core_mask_map.py:79` 的 `HP.generate_ihc_core_mask` 已搬到 `m1_overlay.py:53` → **已修**；(4) `scripts/perf_measure.py:769` 非 `--stream-precut` 分支呼叫 `HP.precut_paired_tiles`，**該函式已完全不存在**（被 `PrecutStream` 取代，只剩 `hybrid_pipeline.py:133` docstring 提及）→ **未修**（本輪無量測走這條路徑，未經驗證改動量測工具）；(5) **`run_batch` 的 `stats["skipped"]` 不等於背景 tile 數**。
  - **`stats.skipped` ≠ 背景比例**（解釋了 round 13 記錄的矛盾：doc 45 §1.6 引用 round 13 記錄「10,800 success / 16,765 skipped」= 60.8% 卻說背景 55.8%）：b10 直接拆開：576 tile = 58 背景（core mask 全空 → `F_write_blank_tile`）+ 518 進入完整處理（499 產出可用細胞、19 沒有）；`stats = {success: 499, skipped: 77}`，77 = 58 + 19。**建模必須用 `F_write_blank_tile` 計數，不能用 `skipped`**；Set F 實測 `F_write_blank_tile` = 23,533 = 65.92%，與 `core_mask_map.py` 獨立測量完全一致。
- **§2 實際花費**：core_mask_map（35,700 tile 全片 UNet 掃描）1,170.3 s；crops（11 個）2,168 s；legacy（round-1 原始 crop 重測，3 尺寸 × 2 worker）636 s；Set B 950 + 367 s；Set C 3,290 s；Set A 3,177 s；Set E 517 s；Set D 2,254 s；**Set F 16,212 s**；Phase A probe 570 s；合計 ≈ **8.11 h**。驅動腳本 `/home/taro/r15/{run_all,make_crops,run_sets,run_setD,run_setF,run_legacy}.sh`。兩次重跑源自 driver script 自身缺陷（`case "${1:?...{B|C|A|E}}"` 會靜默破壞 pattern matching；crop 命名 `b_real` vs `breal` 不一致），已修並加 fail-fast。
- **§3.1 Set B 組成 sweep（本計畫核心產出）**：固定 576 tile（18688×18688、0.349 GP）、`workers=1`，用 `core_mask_map` 的 `--map` 標記（pipeline 自己的 empty-core-mask 判準，**非** `composition_crop.py` 的亮度 proxy）：

| run | 背景 | 組織 | 背景比例 | wall | s/tile |
| --- | --: | --: | --: | --: | --: |
| b10 | 58 | 518 | 10.07% | 381.3 s | 0.6619 |
| b25 | 144 | 432 | 25.00% | 322.3 s | 0.5596 |
| b50 | 288 | 288 | 50.00% | 240.2 s | 0.4170 |
| b_real | 380 | 196 | 65.97% | 164.7 s | 0.2859 |
| b73 | 423 | 153 | 73.44% | 131.9 s | 0.2290 |
| b90 | 518 | 58 | 89.93% | 65.7 s | 0.1140 |

  對 core 時間（wall − Phase D）最小平方：**`core = 0.7298 s × n_tissue + 0.0460 s × n_bg`（截距 0.001 s），R² = 0.99694，殘差 RMS 6.03 s，最大相對殘差 6.47%**。**一個組織 tile 成本是背景 tile 的 15.9 倍**；doc 45 §3.3 指出 round 7 只「假設」背景塊會被短路成本趨近 0 而從未量過顯式係數：實測**不是 0，是組織塊的 6.3%**。`--map` 準確度：預測 58 個背景 tile，`F_write_blank_tile` 實測 **58**——零誤差（亮度 proxy 的落差因此不存在，doc 24 的提醒在此路徑可關閉）。**逐 bucket 係數表**（s/組織 tile、s/背景 tile、R²）：`B1_m3b_cellpose`（MAIN，tissue）0.60168/−0.01165/0.9995；`B3_detect_dots`（BG，tissue）0.29984/−0.01924/0.9938；`B3_enlarge_cells`（MAIN，tissue）0.05058/−0.00297/0.9921；`B1_unet_coremask`（MAIN，all）0.04091/0.02887/0.9726；`B3_build_results`（MAIN，tissue）0.03144/−0.00030/0.9998；`B2_render_overlay`（BG，tissue）0.02346/−0.00067/0.9994；`B2r_tile_read`（MAIN，all）0.02169/0.00755/0.9791；`Bs_clear_edge`（MAIN，tissue）0.01947/−0.00161/0.9795；`F_write_blank_tile`（BG，background）0.00007/0.00310/0.9940（負的 s/背景 tile 是最小平方雜訊，絕對值 <0.02 s）。**臂模型**：MAIN 0.8016 s/組織 tile、BG 0.3244 s/組織 tile，**BG/MAIN = 0.405**（`bottleneck-list.md` 在 match24 量到 0.470 同量級；BG 臂仍有 ~59% 餘裕）。
- **§3.2 Set A 尺寸 sweep（固定組成 65.9%）**：a11（121 tile、0.076 GP、80/41、wall 39.8 s、Phase D 0.7 s、Set B 預測 34.4 s、誤差 +15.9%）；a21（441、0.268 GP、291/150、124.6 s、2.8 s、125.6 s、−0.8%）；b_real（576、0.349、380/196、164.7 s、3.6 s、164.1 s、+0.3%）；a32（1,024、0.617、675/349、280.8 s、6.4 s、292.2 s、−3.9%）；a64（4,096、2.441、2,700/1,396、1,230.6 s、25.1 s、1,168.7 s、+5.3%）；a96（9,216、5.474、6,074/3,142、2,723.5 s、186.5 s、2,630.2 s、+3.5%）。**每 tile 速率跨尺寸穩定（doc 45 §1.5 前提成立）**：扣掉 Phase D 後 core/組織 tile 0.953（121）、0.812、0.822、0.786、0.864、0.808 s；排除 121 tile 那點（2.5 s model init 占 39.8 s 的 6%），**21x 尺寸範圍內只有 ±5% 變動**；在 576 tile 擬合的 Set B 係數外推到其他尺寸誤差 −3.9%~+5.3%。
- **§3.3 Set D + Set A — Phase D 的 page-cache 轉折**：Set D 用 `stitch_probe.py` 合成 scratch（hard link 重複真實 overlay tile；doc 45 原提議 4x/9x 實體大 TIFF，同曲線但不用寫 ~400 GB）：GP 1.04/4.08/16.22/20.94/41.88/83.75 → Phase D 11.2/44.7/182.6/241.8/556.7/1,240.9 s → s/GP 10.80/10.94/11.26/11.55/13.29/14.82；cached regime 擬合 **Phase D = 10.135 × GP^1.0665、R² = 0.98981**（到 84 GP 幾乎線性）。**但 hard link 使來源常駐 page cache**：同一幾何（141818×114366、27,565 tile、16.22 GP）Set D 探針 182.6 s（11.26 s/GP）vs round 13 實跑（真實相異 tile、冷讀）**987.8 s（60.90 s/GP）→ 5.41x 差距**（比 doc 39 §4.4 以不同尺度 spike 估的 15.4 vs 116.6 s/GP 更乾淨）。這解釋了 Set A 的 a96 異常：5.474 GP、9,216 個真實相異 tile、~18 GB scratch 實測 34.07 s/GP = cached 曲線的 **3.0x**——不是面積造成的超線性，而是同一個 cache 轉折在較小尺寸就到達。**結論：Phase D 正確模型是分段的——scratch 裝得進 page cache 時 ~10.1 s/GP × GP^1.07，裝不進時約 5x；轉折點由主機可用 RAM 決定、不由玻片大小決定**（本機 62 GB，轉折在 scratch 8 GB 與 18 GB 之間）。
- **§3.4 Phase A 獨立曲線**（首次量測）：a11 5.74 s（0.04746 s/tile）、a21 19.58 s（0.04439）、b_real 22.01 s（0.03821）、a32 40.86 s（0.03990）、a64 156.75 s（0.03827）、a96 322.02 s（0.03494）；**`Phase A = 0.06268 × n_tiles^0.9364`，R² = 0.9995**——指數 0.936 略次線性（與 Phase D 超線性相反）；外推 35,700 tile ≈ 1,149 s ≈ 19.1 分鐘**如果它是獨立序列階段**。**但在出貨組態下不是**：Set F `phaseA_precut_s` 實測 0.03 s；**doc 18 §4.2 的 Phase A 串流化在整片尺度的實測價值約 19 分鐘，占 `workers=1` wall 10.6%、占 `workers=4` wall 21.6%**（後者尤其可觀，因 Phase A 序列不隨 workers 縮放）；也修正 doc 45 §5.4 的模型公式（`+ Phase_A(total_GP) + Phase_D(total_GP)` 在串流組態下會重複計算 Phase A，`eta_estimate.py` 沒有獨立 Phase A 項）。
- **§3.5 Set C workers scaling（四點擬合取代兩點外推）**：b_real（576 tile）w1–w4 164.7/110.2/94.8/91.0 s、e2e 加速 1.81x、core 加速 1.84x、Amdahl serial 比例 36.8%、R² 0.9938；a64（4,096 tile）1,230.6/691.1/559.6/499.7 s、2.46x/2.54x、17.1%、0.9947；**full slide（35,700 tile）10,883.4 → 5,321.4 s、2.05x、core 2.36x、(23.0%)、n=2**。**serial 比例不是常數**（576 tile 36.8%、4,096 tile 17.1%）；doc 38/39 的 Amdahl 帳是在單一尺度用兩點外推並把 parallel portion 當 pipeline 固定性質——**不是**；ETA 模型若用單一 `speedup(workers)` 常數，兩端必有一端錯。**doc 45 §4 Set C 要求的交叉印證失敗（並查明原因）**：Phase A + Phase D + init：b_real 0.04 + 3.60 + 2.52 = 6.16 s vs Amdahl 擬合的 serial 62.6 s（**10.2x**）；a64 0.03 + 25.10 + 3.52 = 28.64 s vs 229.8 s（**8.0x**）——擬合 serial 項比架構上所有序列階段加總還大 8–10x；依 doc 45 指示不直接採信擬合值：原因是 **`workers>1` 不讓 GPU 吞吐變 4 倍**（只有一張 RTX 5090，四個 worker 行程仍把 kernel 排到同一裝置），Amdahl 歸給「serial」的絕大部分其實是 **GPU 裝置競爭**，與 round 14 從另一端獨立量到的（`_from_device` 每次呼叫 38.7 → 101.8 ms = 2.63x）一致。**因此 doc 38/39 推估的「~2.5x–2.8x 組成上限」不是 `workers=4` 的實際天花板**；實測是隨規模變動的 1.81x（576）→ 2.46x（4,096）→ 2.05x（全片，被 Phase D 拖累，core 本身 2.36x）。
- **§3.6 Set E 誤差棒（本專案第一次）**：固定 b_real（576 tile、65.97% 背景）n=3：`workers=1` 164.7/162.6/165.5 s、均值 164.2 s、sd 1.48 s、**CV 0.90%**；`workers=4` 91.0/91.5/91.9 s、均值 91.5 s、sd 0.45 s、**CV 0.49%**。**run-to-run 變異 <1%**。兩個推論：(1) ETA 信賴區間由**迴歸殘差**（6.5%）主導，不是執行雜訊；(2) **回頭看本專案很多 n=1 歷史結論其實是可信的**：round 12 的 Phase D「1,239.2 → 1,889.8 s」（+52%）與 round 11 未解釋的 530 s 殘差曾被標註「可能只是變異」，在中型錨點 0.9% CV 下，run-to-run 雜訊**不可能**解釋 52% 的變動，那些是真實效應。限制：此 CV 量在 576 tile，變異是否隨規模放大仍未知（Set F 每個 worker 數只跑一次）。
- **§3.7 Set F 正確畫布整片驗證（held-out）**：`workers=1` 與 `workers=4` 各一次、35,700 tile/20.938 GP、出貨組態（`--stream-precut`、`tifffile` stitch backend、prefetch 開）：`workers=1` **10,883.4 s = 3.023 h**、Phase D 1,332.1 s、core 9,551.3 s、Phase D 占 12.2%、peak RSS 13.66 GB、352,881 cells；`workers=4` **5,321.4 s = 1.478 h**、Phase D 1,282.0 s、core 4,039.4 s、Phase D 占 **24.1%**、peak RSS 13.24 GB、352,874 cells（**−0.002%**）。背景 tile 實測 23,533/35,700 = **65.92%**（與 `core_mask_map.py` 完全一致）；正確性 veto 通過；**peak RSS 13.66 GB**（round 8 的 61.13 GB −77.6%、round 13 的 16.95 GB −19.4%）——**README.md:174–182 仍寫「full-slide 需 64 GB RAM/實測約 60 GB peak RSS」已過時三輪**；Phase D 在 `workers=4` 占 24.1% wall，重現 round 12 的結構（tile 平行臂縮放良好，端到端被序列的 Phase D 拖住）。
- **§4 模型與 held-out 驗證**（`scripts/eta_estimate.py`）：`wall = core(n_tissue, n_bg)/speedup(workers, scale) + phase_d(n_tiles) + init`；`core = 0.7298·n_tissue + 0.0460·n_bg`；`phase_d = 0.001783 · n_tiles^1.209`（擬合於 Set A，R² = 0.778）；`speedup = 1/(f + (1−f)/w)`，`f` 依 log(tile 數) 在量測錨點間內插、範圍外夾住不外推；`init = 2.52 s`；**沒有獨立 Phase A 項**。Set F 完全沒有參與任何係數擬合。驗證：`workers=1` core 預測 9,962.8 vs 實測 9,551.3（**+4.3%**）、Phase D 569.6 vs 1,332.1（**−57.2%**）、wall 10,534.9 vs 10,883.4（−3.2%）；`workers=4` core 3,770.7 vs 4,039.4（−6.7%）、Phase D 569.6 vs 1,282.0（**−55.6%**）、wall 4,342.8 vs 5,321.4（**−18.4%**）。**診斷（不調整模型去遷就驗證點，而指認失準子模型）**：(1) core 子模型健全（外推 3.9 倍於擬合尺度、擬合時沒看過的組成，誤差 +4.3%/−6.7%）；(2) **Phase D 子模型失準 ~56%**：它擬合於 Set A，而 Set A 最大的一點（9,216 tile）正好落在 page-cache 轉折上，轉折之後沒有任何資料；用一條穿過轉折的冪律再外推 3.9 倍必然低估——R² 0.778 就是當場可見的警訊；(3) **`workers=1` 的 −3.2% 是假象**：+4.3% 的 core 誤差與 −57% 的 Phase D 誤差恰好相消，`workers=4` 時 core 縮小、Phase D 不變，相消消失，真實 −18.4% 顯現；**把 −3.2% 當成模型準確度會是錯的**。修好 Phase D 需要 9,216–35,700 tile 之間、冷讀 regime 的量測，本輪沒有編列；在那之前玻片尺度 ETA 在 Phase D 這項帶有已知 −56% 偏誤。
- **§5 與最舊資料的比較（使用者要求的重點）**：
  - **5.1 同一批 crop 檔案（round 1 vs 今日程式碼，like-for-like）**——round 1 control（git `96a28ba`、`config_hash db2b7e6a`、完全序列 `run_batch`）的 `small`/`med`/`large` crop 檔案至今仍在磁碟上：small（25 tile）54.3 → **22.5 s（2.41x）**、`w4` 26.3 s（對 round 1 2.06x）；med（121）243.3 → **85.2 s（2.85x）**、`w4` 54.6 s（**4.46x**）；large（441）848.0 → **294.6 s（2.88x）**、`w4` **143.2 s（5.92x）**。附註：(1) **`bottleneck-list.md` 的招牌數字 302.7 s 在今日程式碼被確認**（294.6 s，快 2.7%，雜訊內；自 round 6 後從未重驗）；(2) **小於約 100 tile 時多行程反效果**：25 tile `workers=4`（26.3 s）比 `workers=1`（22.5 s）**慢**（4 份 model init 攤不掉）——UI 送出小 ROI 時不應用 `workers=4` 加速比報 ETA；(3) **加速比隨規模成長**（2.06x → 4.46x → 5.92x，固定成本被攤薄）——任何單一「我們快了 N 倍」不講規模就沒有意義。
  - **5.2 整片：從第一次投影到今天**：round 1 投影（`9.4 + 1.903 s/tile`，從未實跑；假設 35,700）67,946 s = 18.87 h；round 8 首次實跑（舊錯畫布 27,565 tile）`w1` 13,762.5 s = 3.82 h、`w4` 6,211 s = 1.73 h；round 11 乾淨對照 `w1` 10,666.1 s = 2.96 h；round 13 最佳舊數字 `w4` 4,877.4 s = 1.355 h；**round 15（正確畫布、35,700 tile）`w1` 10,883.4 s = 3.023 h（6.24x）、`w4` 5,321.4 s = 1.478 h（12.77x）**。**round 8–13 那幾列的 tile 數比 round 15 少 29.5%**（錯誤畫布），直接把 4,877.4 與 5,321.4 s 相比會**低估**今日的效能。**12.77x 拆成三個來源**：**程式碼改進 2.85x**（round 1 與 round 15 用同一種投影法：441 crop 的 blended 速率 × 35,700：67,946 → 23,849 s）、**組成修正 2.19x**（真實玻片是 65.9% 背景，非 441 crop 的 14.1%：實測 23,849 → 10,883 s）、**多行程 2.05x**（`workers=1` → `workers=4`）；合計（w4）67,946 → 5,321.4 s = **12.77x**——「約 2.85x 來自真正工程優化，約 2.19x 來自『當年用組織密集 crop 外推整片本來就高估』，另 2.05x 來自多行程；只報 12.77x 而不拆解會誇大工程貢獻」。
  - **5.3 每組織 tile 成本（跨畫布唯一可比口徑）**：兩個畫布組織 tile 數幾乎相同（12,184 vs 12,167）：round 8 `w1` 13,762.5 s / 12,184 = 1.1296 s/組織 tile；round 11 `w1` 0.8754；**round 15 `w1` 0.8945**；round 8 `w4` 0.5098；round 13 `w4` 0.4003；**round 15 `w4` 0.4374**。以此口徑 round 15 比 round 11/13 **略慢**（+2.2%/+9.3%）——**非回歸**，是正確畫布多出的 8,152 個背景 tile 的真實成本（8,152 × 0.046 s ≈ 375 s）加上更大畫布的 Phase D（20.94 vs 16.22 GP）；在正確輸入上這就是真實成本。
  - **5.4 記憶體**：round 8 61.13 GB → round 13 16.95 GB → **round 15 13.66 GB（−77.6%）**；README 的 64 GB RAM 需求過時三輪，應更新為約 16 GB。
- **§6 產出**：係數檔 `/home/taro/r15/coefficients.json`；參考實作 `scripts/eta_estimate.py`（`fit` / `estimate` 兩個子命令，**不碰** `backend/api/`、`backend/schemas/`、`frontend/`——接進 UI 是下一份落地文件的工作且需先過 frontend-backend-boundary 審查）；`scripts/phase_a_probe.py`（Set D 授權的薄 wrapper）；原始資料 `measurement/_metrics_r15/`；campaign driver `/home/taro/r15/*.sh`；Set F 完整輸出 `/home/taro/r15_fullwsi/`；用法：`eta_estimate.py fit --metrics-dir /home/taro/r15/_metrics --out /home/taro/r15/coefficients.json`；`eta_estimate.py estimate --coefficients ... --roi-px 156222x134028 --background-share 0.6592 --workers 4`。
- **§7 限制（doc 45 §7 延續並更新）**：(1) 只有一張真實玻片（**比 doc 45 寫的更嚴重**：同一張玻片換一個正確畫布定義，背景比例差 10 個百分點；跨病人/染色批次變異完全未知，而模型對此輸入很敏感（組織塊貴 15.9x），上線後必須持續回測）；(2) **Phase D 子模型已知失準 −56%**；(3) Set D 大尺寸曲線是 cached regime（形狀指數 1.07 可遷移，每 GP 絕對成本不行——同幾何差 5.41x；已驗證真實規模上限 **20.94 GP**，41.88 與 83.75 GP 只有 cached 數字）；(4) 變異只在 576 tile 量過（CV <1%），是否隨規模放大仍未知；(5) 單一硬體型號（§3.5 顯示 `workers>1` 行為由 GPU 裝置競爭主導，換卡後 workers 曲線整條改變）；(6) 本輪未改動任何 pipeline/API/UI 程式碼（§1.4 修的三處都在 `scripts/` 的量測工具內且都是 bitrot；`config_hash` 全程 `d2ccc46b`）；(7) `workers=4` 的 VRAM 風險未在本輪重測（round 14 記錄過 `expandable_segments:True` 下 1/10 次 crop 跑批 OOM；Set F `workers=4` 單次成功但 n=1 不足以說風險消失）。
- **§8 仍開放**：(1) **Phase D 冷讀 regime 的量測**（最高優先，把 ETA 用在整片前的前提）；(2) `perf_measure.py:769` 死路徑；(3) README.md 硬體需求過時三輪；(4) 逐 tile 進度 callback（框架層阻擋）；(5) 跨玻片驗證。

---

## 17. `DISCOVERED-NOT-IMPLEMENTED.md` — 「已發現但從未落地」全稽核（2026-07-26 編纂，後續輪次以註記更新）

> 這是一次性、完整讀過 01–25 與 `measurement/*.md` 後編纂的帳本：每一個「Discover/Analyze/Plan 工作浮現、但從未造成 pipeline code 改動」的想法、候選或發現——不論仍開著、量測後被明確拒絕（stop-lossed）、或被其他事項阻擋。**不是常態維護 backlog**（日常追蹤以 `19-open-backlog.md` 為準）。閱讀符號：🟢 開著/可行動；🟡 被阻擋/待決定；🔴 已停損（量測或推理後明確拒絕）；⚪ 從未執行（量測/調查步驟本身未跑，答案未知）；✅ 稽核編纂後已在後來輪次關閉（保留原位使帳本完整，並標出關閉它的文件）。
> 稽核之後的重要更正（標頭註記）：**Round 8** 以 doc 26/27 關閉一批；**Round 9–11** 三項改變處置：#43（`gc.freeze()` round 2 已上線並在整片確認兩次）、#44（tile-read round 2 已上線，「17.2%」被 5.3x 污染，真實 4.21%）、#6（Phase D GPU 化被 spike、在真實規模反轉、負向結案）；**Round 12–15**：#6 **重開並上線**（round 12 建了缺少的管線化讀取槓桿、round 13 以 `config.stitch_backend = "tifffile"` 預設上線，整片 Phase D 1.913x/e2e 1.200x）；#1 的 round-8「2.216x」**修正為 1.745x**（round 12）再修正為隨規模變動的 **2.05x–2.36x**（round 15，正確畫布）；新項目：round 14 系統性關閉 `workers>1` 的 GPU↔CPU 資料搬運調查（三個候選皆負向，無 code 變更）；**最重要**：doc 44 發現本稽核數字所依賴的 round 8–14 整片跑批全在**未配準畫布**（27,565 tile/55.8% 背景）而非正確的**配準畫布**（35,700 tile/20.94 GP/65.92% 背景）；round 15 在正確畫布重測整片（`workers=1` 3.023 h、`workers=4` 1.478 h、peak RSS 13.66 GB）並交付 `scripts/eta_estimate.py`。

### 17.1 效能候選——GPU / pipeline 架構（#1–#31、#43–#46）

| # | 候選 | 狀態 | 天花板/證據 | 來源 |
| --- | --- | --- | --- | --- |
| 1 | 跨 tile 多行程（`workers>1`） | 🟡 → **gate 滿足（round 8）**：整片 2.216x、correctness veto 通過；放行決策取決於 **VRAM** 但書（`workers=4` 30,439/32,607 MB、93.3%）而非速度；後續 round 12 修正為 1.745x、round 15 為 2.05x–2.36x | tissue-dense crop 3.09–3.51x；組成匹配 crop 2.06–2.17x | doc 21、19 #1 |
| 2 | allocator 碎片化 OOM（單一 worker 恰好膨脹到 24.76 GiB；先見 `workers≥6`、後在出貨 `workers=4` 確認） | 🟡 → **round 11 根因與緩解**：`expandable_segments:True` 在 `workers=4` 12 次交錯 control 4/12 OOM、knob 0/12（Fisher p=0.047）、+0.67% wall、peak framebuffer 升到 92.2%；commit `b3fa47d` 把預設翻開 | 可靠性缺陷，非速度 | doc 21 §4.7、23 §4.6/§6、27 §5、37 §3、38 §0 |
| 3 | 整片規模端到端驗證 | ✅ **round 8 完成**（自 doc 09 §3.6 起開著）：3.82 h/1.73 h vs 投影 2.6/1.25 h、2.216x；preflight 浮出 crop 輪次碰不到的 blocker（每 modality 畫布尺寸）；發現**三個來自 crop 的數字在規模不成立** | 放行 item 1 的綁定 gate | 19 #7、22 §1、27 §6.4/§6.6 |
| 4 | `_stitch_overlay_slide` 一次開 27,565 個 tile 且無 `RLIMIT_NOFILE` 檢查 | ✅ **round 8**：`_ensure_nofile_limit()`（soft 自動升到 hard；hard 不足時開啟前大聲失敗）、7 tests | 常見 1,024 限制主機會失敗 | 25 §5.2、19 #7、27 §1 |
| 5 | Phase D 便宜非 GPU 旋鈕（`tiffsave` tile-size/pyramid-depth、跳過常數背景區） | ✅ **round 8 負向關閉**：13 config 單旋鈕消融，全死（tile size 單調更差、pyramid depth no-op）；`zstd` 贏 1.2331x 但 QuPath/BioFormats 打不開 → correctness veto | stitch 322.7 s/16.2 GP 超線性；ceiling 1.036–1.078x | 25 §5.3/§11、27 §3 |
| 6 | Phase D GPU 化（nvImageCodec / `tifffile` 容器） | 🔴 **round 10 負向關閉**（round 9 記憶體 spike 1.400x/2.147x；round 10 從磁碟取真 tile 後反轉為 0.788x/0.884x；讀取占 baseline wall 48.7% 而 pyvips 免費重疊）→ **round 12 重開、round 13 上線**（管線化讀取）；round 9 順帶修好既有缺陷：每個 `overlay_slide.tiff` 金字塔層解成雜訊（`predictor="none"`、兩個整片 overlay 重生） | round 11 時整片 stitch 為 `workers=1` wall 的 11.6%（1,239.2 s）；其占比持續成長純粹因周圍變快 | 24 §2.1、25 §8.2、32、35 §3 |
| 7 | Candidate F——背景 tile placeholder 去重（`os.link` 取代重新編碼） | 🔴 速度面停損（BG 臂 47–53% slack，任何 worker 數 wall 皆 0），但標為**獨立的儲存決策**（~157 GB/slide 相同未壓縮 TIFF），從未決定 | 比現行寫入路徑便宜 407x/272x | 25 §4.3/§11 |
| 8 | Candidate G——把冗餘 per-call `mkdir()` 提出 `_save_tile_array` | 🔴 已建、量測、**revert**（0.056% wall；端到端 ablation 讀起來略負） | patch 逐字保存於 doc 25 §10，可在 arm 平衡或儲存後端改變（如 NFS，同呼叫貴 ~1000x）時一個 edit 復活 | 24 §2.6、25 §3 |
| 9 | Candidates B/D/E——`detect_all_dots`、`enlarge_cell_instances`/`build_all_positive_results`、per-tile debug encode → GPU（CuPy/cuCIM） | 🔴 兩次停損（doc 22 B2/B3、doc 24/25）——真實組成下 wall ceiling 1.00x（BG 臂 slack 47–53%）；CuPy 在本機**跑不起來**（numpy≥2 ABI 與專案 `numpy<2` pin 衝突；numpy-1 線無 CUDA toolkit 不能 JIT） | MAIN 先縮 ~50% 才重開 | 22 §3、24 §2.2–2.4、25 §2.3/§8.1 |
| 10 | `clear_slide_edge_cells`——M2→M3b 裝置閒置縫隙最後的 CPU glue | 🟢 watch item、未行動（單獨太小，~1.2% wall） | 僅在 bubble-closing 重啟時 | 19 #2、18 §5 |
| 11 | CUDA-stream/depth-2 bubble 重設計 | 🔴 sized 並停損（⑧ 落地後 ≤1.065x） | 僅在 intra-forward launch-bound idle 被 upstream 修好時重開 | 17 §3、18 §3、19 #4 |
| 12 | CUDA graph / 向量化 Cellpose 內部 kernel-launch Python 迴圈（`_extend_centers_gpu`、`steps_interp`、`get_masks_torch`、`fill_holes_and_remove_small_masks`） | 🔴 兩次停損（4.0.8 ~1.23x；4.2.1.1/`cpdino` 重新確認 ~1.118x，`get_rel_pos` 已不適用） | 需 patch 釘版本的第三方 `cellpose`/`segment_anything` | gil-contention-diag「追加深挖」、19 #5、23 §7 |
| 13 | GPU 端 tile/transform 載入（把 `_read_rgb` decode 搬到 GPU） | 🟢 **round 8 重開**（round 4 在 441 tile crop 停損於 1.22%/1.012x，但整片量到 17.2% wall）→ round 9–11 以 Option L 預取上線；**round 11 證實 17.2% 本身被污染，真實 4.21%、Option L 已吃掉 99.8%** | — | 17 §4、18 §6.3、19 #6、27 §6.4 |
| 14 | 隔離 `detect_all_dots` +22.3% 迴歸（round 3 ⑨）——numpy/scikit-image/opencv 降版 vs 重訓 checkpoint 的細胞幾何 | 🟢 開著、無 wall 回報（ceiling 1.013x；大致被 round 6 的 `dot_detect_n_jobs=1` 超越）；便宜可解（在儲存的 instance mask 上以兩組依賴重跑） | — | bottleneck-list ⑨、19 #3 |
| 15 | API/job 層多請求/並發行為（Phase E）——從未量測 | ⚪ 從未執行；自 doc 09 §3.5 起旗標，仍未做負載測試；取決於「並發玻片服務是否是部署計畫的一部分」 | — | 19 #8、22 §2 A3、23 §1 |
| 16 | 每輪量測資料夾附 `pip freeze` | 🟡 部分採用（round 2 沒有、無法追補，這是 #14 無法歸因的原因） | 流程紀律 | 19 #9 |
| 17 | N 路多行程下的 CPU 核心競爭稽核（Track A2） | 🔴 已執行——超額訂閱假設**錯誤**；找到更大不同缺陷：`detect_all_dots` joblib fan-out 在**每個** process 數下都是淨損失（比序列慢 2.77x） | 以 `dot_detect_n_jobs=1` 一行 config 修正 | 22 §2 A2、23 §4 |
| 18 | 跨 tile Cellpose 批次（把 N 個 tile 堆成 `(N,H,W,3)` 一次 `eval()`，走 `cellpose.core.run_net` 的 `Lz>1` 真批次路徑） | 🔴 根因追到且確認為真，之後量測**負向**——所有 group size 持平到更差（≤+8.2%）、VRAM 線性成長（1.17 → 8.01 GB）；G=16 重測仍更差（+5.9%/+6.6%）且 15.8 GB（卡的 48.6%） | 不建；`cellpose_batch_size` 維持 16 | 22 §3 B1、23 §2、25 §7 |
| 19 | 跨 tile UNet++ 批次（含移植 `cell_mask/unet_mask/inference.py` 的 `predict_batch`） | 🔴 研究線索追完發現是死路（`predict_batch` 是逐影像序列迴圈）；量測所有 group size 嚴格更差 | 開始前就被 ~0.4% wall 封頂 | 22 §3 B1b、23 §3 |
| 20 | CUDA MPS | 🔴 已測——在此消費級 RTX 5090 上確實可用（launch-bound 微基準 +44%）但**端到端平坦** | — | 20 Candidate C、21 §5 |
| 21 | Candidate A——只用 CPU 後端的 process pool | 🔴 以便宜的 depth-2 thread knob 檢驗前提並被推翻——單 process +2.8% 更慢、多行程下持平 | knob 建、量、刪 | 20 Candidate A、21 §6 |
| 22 | Candidate E——跨 worker 的 fork 模型重用 | 🔴 明確「不建」——架構不安全（CUDA context 非 fork-safe）；僅為關門而記錄 | — | 20 Candidate E |
| 23 | `gc.collect` 批次（固定 N cadence） | 🔴 已建、量測、除 `gc.freeze()` 外無增益、**刪除** | `gc.freeze()` 上線 | 16 |
| 24 | `detect_all_dots` → process-safe joblib backend（loky/spawn） | 🔴 以 py-spy 診斷後判為不值得——GIL 競爭**不是** GPU idle 的驅動；所需 spawn-safety/memmap 驗證腳本**從未寫**，因決策樹沒走到那支 | 被 round 6 無關的 `dot_detect_n_jobs=1` 完全取代 | 11 §4(b)、gil-contention-diag、12 §3(b) |
| 25 | `detect_all_dots` 的 `regionprops_table` 向量化改寫（doc 12 §3(c)） | ⚪ 從未執行——是 (b) 不值得時的後備；整條分支在有人回去跑 doc 12 的 Step 0 之前就被 round 6 的 `dot_detect_n_jobs=1` 變得無意義 | — | 12 §2/§3(c) |
| 26 | `detect_all_dots` 整塊 tile 向量化（完全沒有 per-cell Python 迴圈，doc 12 §3(d)） | 🟢 原則上開著（明列 backlog：最高 ceiling、最高 correctness 風險），從未拾起 | — | 12 §3(d) |
| 27 | UNet++ inferencer 的 `torch.compile` | ⚪ 從未執行——背口袋點子，開始前就被 ~0.4% wall 封頂 | — | 22 §4 |
| 28 | UNet++ 呼叫的 CUDA graph capture | ⚪ 從未執行——同 #27 的 ~0.4% 上限 | — | 22 §4 |
| 29 | NVIDIA DALI/TensorRT 用於 UNet++ 或 Cellpose 前向 | ⚪ 從未執行——未 sized，僅旗標，會增加 fp16/int8 correctness 風險 | — | 22 §4 |
| 30 | A1——以**真實**整片組織密度重調 `workers`（相對於 crop） | 🟡 部分完成（round 6/7 在 crop 上重掃，建議由 6 → 4–5，再在組成匹配規模確認 4），但「真實整片密度」那半仍開著，繫於 #3 | — | 22 §2 A1、23 §6、19 #7 |
| 31 | Cellpose 4.0.8 SAM-backbone kernel 目標已被取代——任何未來對 cellpose 內部的嘗試應改鎖 `_from_device`（`cellpose/core.py:141`，MAIN 臂 Python 樣本 24.2%）與 `_quantile`（10.1%） | 🟢 開著的觀察、尚未嘗試（正確——ceiling 小且是第三方內部）；**round 14 之後 `_from_device` 被證實 96.3% 是 GPU-wait，關閉** | — | 23 §7、43 |
| 43 | `gc.freeze()` 的 crop 規模勝利（~0 成本）在整片規模不成立——退回 16.1% wall | ✅ **round 9 關閉，round 10/11 在整片確認兩次**：週期性 re-freeze 每 5,000 累積細胞（`gc.freeze()` 量到 O(1)）；整片 `workers=1` `B4_gc_collect` 2,218.4 → 58.8 s（round 10）→ 19.5 s（round 11）；peak RSS 反而下降 | — | 16、19 #6b、27 §6.4、35 §4.2、37 §4.3 |
| 44 | `B2r_tile_read` 的 crop 停損（1.22%）在整片不成立——17.2% wall | ✅ **round 9 單 process、round 10 多行程上線；round 11 結清**：整片 `--no-prefetch` 乾淨重跑證實「17.2%」被 `d6592c3`（每 slide 寫 275 GB per-tile 中繼檔）污染 5.3x——真實無混淆成本 **449.3 s = 4.21% wall**，Option L 已隱藏 99.8% | — | 18 §6.3、19 #6、27 §6.4、37 §4.1 |
| 45 | 失敗後/完成後行程 hang | ✅ **`workers=4` fail-fast 案例 round 11 關閉**（`scripts/exit_latency_probe.py` + `task_q.cancel_join_thread()`：never → 0.32 s；同時修好 `PrecutStream` 在批次放棄後仍切整張玻片）；`workers=1` 兩小時 hang **未重現**（記為未重現而非已修） | — | 35 §6.2、37 §2 |
| 46 | `B1_m3b_cellpose` +751 s/+11.8% 迴歸（round 8 baseline vs round 10 整片） | 🟡 **round 11 大致解釋，530 s 殘差留為未歸因雜訊**：Option L 預取經 GIL 競爭膨脹主執行緒 wall-clock bucket（係數 0.513 s/s，整片偏差 4% 內，解釋 221.2 s/29.5%）；其餘 530 s 落在直接量測的跑批間漂移帶內（同 GPU 在一次暖機 sweep 內此 bucket 移動 +17%） | — | 35 §4.4、37 §1 |

**（另：round 14 新增項目——`workers>1` 的 GPU↔CPU 資料搬運調查三候選全關閉負向，無 code 變更；見 doc 43。）**

### 17.2 文件↔code 漂移（#32–#38）——全部 ✅ 於 round 8 關閉

| # | 項目 | 結果 |
| --- | --- | --- |
| 32 | `generate_ihc_core_mask` 參數名 `ihc_tile_path: Path` 但實收 ndarray | 改名 `ihc_image: Union[np.ndarray, Path, str]`，移除 call site 的 `# pyright: ignore` |
| 33 | `docs/sdd-elastic-dish-matching.md` 在 `m3_elastic_matching.py` docstring 被引用但不存在 | 移除；並在 `m3_dot_detection.py` 找到並移除第二個未列出的同款引用 |
| 34 | `docs/dish_dot_detection_spec.md` 在 `config.py`/`config_example.py` 註解被引用但不存在 | 兩個檔案都移除 |
| 35 | `docs/algo/elastic_matching_v3_explainer.html` 描述錯誤的（以核為中心）配對演算法 | 更新為 v4：Part A 重寫為以細胞為中心、移除過時「多核排除」outcome、更正參數表（列了兩個不存在的參數並把仍在用的 `dish_elastic_expand_factor` 劃掉） |
| 36 | 沒有自動測試守護 `config.py`/`config_example.py` parity | `tests/test_config_parity.py`，6 cases |
| 37 | `backend/algorithms/hybrid/` 零自動 pipeline 正確性測試 | `tests/test_m0_stitch.py`，21 cases（含整片覆蓋斷言：core crop 恰好分割整片一次） |
| 38 | codegraph 幻影檔案清單未在現行路徑重核 | 重核：**0 幻影**（20 indexed、0 indexed-but-absent；唯一未索引 `.py` 是 gitignored 的 `config.py`） |

### 17.3 正確性/臨床驗證（#39）

- **#39 Round-3 Cellpose checkpoint 重訓（4.0.8 → 4.2.1.1 `cpdino` 換模型）需病理醫師/臨床簽核**：🟢 開著，自 round 3 起 pending，至 round 7 未解——細胞數 +1.8–2.5%、一個 tile success→skipped，每個 round 3 之後的效能勝利都建立在這個未驗證模型上（來源：doc 13 §3、bottleneck-list-history round-3 correctness note、19 §2、23 §9）。

### 17.4 被提出但從未回答的量測問題（#40–#42）

| # | 項目 | 狀態 |
| --- | --- | --- |
| 40 | 多行程 worker 內的 per-bucket 計時不存在 | ✅ round 8 關閉：env-gated probe hook；4 個 parent-only bucket → 26 個 worker-side bucket |
| 41 | API server 路徑的常駐 worker pool 設計（目前每次 `run_batch` 重 spawn，若 API 啟用 `workers>1` 要付 `~3.14 s × N` init；常駐 pool 需重設計每次呼叫的 `gc.freeze()`/`unfreeze` 契約） | 🟡 被 #15（多請求行為是否為部署需求）阻擋 |
| 42 | `run_batch` 無斷點續跑 | ✅ round 8 關閉：`run_batch(checkpoint=True)`、opt-in、config_hash 守衛、fail-fast 不變、cold vs resumed 輸出 byte-identical |

### 17.5 本稽核**不包含**

已上線並採用的項目（`gc.freeze()`、precut 串流化、⑧ 搬離 MAIN、`cellpose_batch_size` 接線、`dot_detect_n_jobs=1`、跨 tile 多行程核心機制——它們**確實**改了 code，見 `measurement/bottleneck-list.md` 與 `current-status-comparison.md`）；沒有產生自己候選的純 Discover/Analyze/規劃文件（08、09）；playbook 文件（方法論，非任務清單）。

---

## 18. `PERFORMANCE_BOTTLENECK_PLAYBOOK.md`（與 `.quickref.md`）——整個資料夾共同使用的方法論

> 這份 playbook 是**所有 10–46 號文件引用的紀律來源**（Discover → Analyze → Plan → Choose；反模式編號 #1–#10 被反覆引用）。它本身來自另一個專案（CuPy/GPU WSI patch 篩選研究 v0–v32），被提煉成通用方法論。

### 18.1 背景（給外部讀者）

- 那個專案是一個**平行計算方法論研究**：固定任務（對 gigapixel WSI 切 1024×1024 滑動視窗 patch → 灰階 → 算白/黑像素比例 → 丟掉過白/過黑，留下座標），用 **32 個版本（v0–v32）** 比較不同平行化策略。**正確性鐵律**：所有版本留下的座標集合必須與單核 CPU 基準逐位元一致（Jaccard = 1.0）；灰階用整數公式 `(R·19595+G·38470+B·7471+32768)>>16` 精確複製 PIL `convert('L')`。
- 三條主線：**線 A（v0–v10）CuPy 計算線**（把像素統計搬到 GPU：批次、pinned memory、async streams、fp16、fused kernel、全堆疊 v9）——「最佳化非瓶頸」的災難現場，是主要教材；**線 B（v11–v22）GPU 解碼線**（發現真瓶頸是 CPU 解 JPEG 後，繞過 OpenSlide，把 JPEG 解碼搬上 GPU：nvJPEG/手寫 CUDA kernel；最佳 v16 端到端 62x）；**線 C（v23–v32）** 在 GPU 解碼管線上重演線 A 的每個概念以交叉驗證。
- 硬體：小切片 GTX 1060 3GB（Pascal，fp16 只有 fp32 的 1/64）+ 8 核；大切片 RTX 5090 32GB + ~20 核。來源：`RESEARCH_RESULTS.md`、`FINAL_REPORT.md`、`V1_V32_COMPARISON.md`、`hardware_probe.py`。

### 18.2 核心教訓與案例數字

- **TL;DR**：先量測再最佳化。**Amdahl 定律**：某部分佔總時間比例 `p`，就算加速到無限快整體上限也只有 `1/(1−p)`。案例：GPU kernel `p = 1.1%` → 上限 ≈ 1.011x，卻在其上做了 v1–v9 九個版本、七層最佳化。
- **發現**：計畫書假設 v9 全最佳化堆疊比多核 CPU 快 8–15x；實測：v0a 單核 CPU 3.96 s（1.00x）、**v0b 多核 CPU 1.32 s（3.01x，贏家）**、v1 逐 patch GPU 4.84 s（0.82x）、v2 批次 0.74x、v3 hybrid 0.72x、v6 fp16 0.75x、**v9 全最佳化堆疊 4.73 s（0.84x）**——**負最佳化鐵證：加了 GPU 與七層最佳化，成品比一個 `for` 迴圈單核 CPU 還慢**。關鍵動作：在 GPU 工作兩側掛 CUDA event 計時器：v9 總 4.732 s、kernel **0.052 s（1.1%）**、其餘（I/O + Python）4.680 s。對照組是照妖鏡（v0a、v0b 同時保留）。
- **分析**：`hardware_probe.py` 量 roofline 兩端——GPU luma kernel 吞吐 ~17,500 patch/s（資料已在 GPU）vs OpenSlide 單執行緒解碼 ~27 patch/s（比值 ~640x，**GPU 被餵料餓死**）；OpenSlide 解碼是否釋放 GIL：單執行緒 27 patch/s（1.0x）、8 執行緒 93 patch/s（3.4x，會 scale → 多執行緒就夠）、8 行程相近。三個思考工具：**Amdahl 先算天花板；roofline 找「誰餓死誰」；分離「單元成本」與「管線成本」**（GPU compute-only 單獨量 1024 px 快 6.98x、2048 px 快 8.37x，但在完整管線被 I/O 完全蓋住——微觀優勢 ≠ 端到端優勢）。
- **規劃**：讓瓶頸定義解法空間而非「你會的技巧」；第一層「並行化供給端」（v0b 多行程讀取 3.0x；v10 單行程多執行緒讀取 + GPU 過濾 6.77x）；第二層「拆掉地板」（v11–v22 把 JPEG 解碼搬上 GPU）；每個新版本只針對當下量出的瓶頸（v19 共用 fd+lock → 量出序列 → v20 `os.pread` 真並行 → v21 producer thread 預取 → v22 destuff 也並行）。**節奏：修一個瓶頸 → 重量 → 新瓶頸浮現 → 再修**。解法優先序：(a) 並行化瓶頸、(b) 移出關鍵路徑（預取/重疊/pipeline）、(c) 消除瓶頸本身。
- **選擇**：不換硬體路徑時 v10 6.77x；允許拆地板時 v16（手寫 optimized CUDA 解碼 × 單執行緒餵料）大切片端到端 **62.0x**、解碼階段 47x、甚至贏過 nvJPEG 固定功能硬體的 29x。反直覺（只靠量測才發現）：async streams 幾乎沒用（瓶頸是 compute 本身時傳輸重疊沒東西可藏）；fp16 在 Pascal 上反而更慢（1/64）；「終極堆疊」v18–v22 輸給更簡單的 v16。**正確性是一票否決**：fp16/nvJPEG/手寫 float-IDCT 有 ±1 LSB 漂移，門檻邊緣 patch 可能翻轉決策，每個 benchmark 明確報出 kept-set 差異。
- **反模式速查（#1–#10）**：1 沒量測就最佳化；2 最佳化非瓶頸（佔比 <10%）；3 沒有對照組/基準線；4 信直覺熱點不信量測；5 把微觀勝利當端到端勝利；6 堆技巧當作前進（每層需 ablation 證明）；7 一次改很多無法歸因；8 忽略硬體/平台特性；9 為了速度犧牲正確性而不自知；10 「假並行」當「真並行」（共用 fd + lock 其實序列）。
- **SOP checklist（發現/分析/規劃/選擇四段）**：發現：誠實基準 + 最笨對照組；端到端量總時間；分解成 % 佔比表；最大佔比 = 候選瓶頸（不是最慢絕對值）。分析：算 Amdahl 天花板；roofline 兩端量吞吐；確認瓶頸性質（I/O、序列化、鎖、同步、配置、頻寬）；量硬體真實能力。規劃：讓瓶頸定義解法；依序考慮 (a)(b)(c)；先試最便宜的。選擇：一次一個最佳化；每層 ablation（零貢獻就砍）；正確性 gate 一票否決；以端到端 wall-clock 選、偏好簡單解；瓶頸移動就回到「發現」。
- **給其他專案 AI 的三句話**：(1) 一行 Amdahl 算術勝過一個月的最佳化直覺；(2) 對照組是照妖鏡；(3) 快 ≠ 是瓶頸、慢 ≠ 值得先修——瓶頸是量出來的、而且會移動。

### 18.3 `PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`

上述的**一頁濃縮版**（約 4 KB）：TL;DR（Amdahl）、四步流程（Discover → Analyze → Plan → Choose）、red flags（最佳化版比最笨基準慢/相等；堆疊多個最佳化總時間幾乎不動；實測加速遠低於理論；profiler 熱函式其實只占總時間一小片；追「看起來計算量大」而非 I/O/序列化/鎖/配置/同步點）、反模式 checklist #1–#10、「三件要記住的事」。**本資料夾 10–46 號文件統一用 `quickref` 當作操作 checklist**。

---

## 19. `measurement/` 現況報告

### 19.1 `measurement/bottleneck-list.md` — 現況瓶頸清單（精簡版，逐 round 敘事搬到 `-history`）

> 這份是「每個瓶頸一行：現況、最新量測結果、證據文件連結」；**其逐項列仍停在 round 11**，round 12–15 以頂部兩段摘要補充。

- **標頭註記**：HEAD = round 15（2026-08-01）；整份檔案逐項列仍是 round 11 時寫的；doc 44/46 確認所有 round 8–14 的整片列（27,565 tile、55.8% 背景）是在**未配準畫布**量的，正確畫布 **35,700 tile、20.94 GP、65.92% 背景**（組織 tile 12,167 vs 12,184 幾乎不變）。**Rounds 12–15 摘要**：round 12 `workers=4` 1.745x（非 2.216x）、Phase D 占 32.3%、batch-claiming 負向（1.00006x）；round 13 Phase D 管線化上線（Phase D 1.913x、e2e 1.200x、占 wall 20.3%、peak RSS 45.6 → 17.0 GB）；round 14 排除 `workers>1` GPU↔CPU 搬運瓶頸；round 15 畫布更正 + 首次誤差棒 + Phase D page-cache 懸崖；正確畫布 `workers=1` 3.023 h、`workers=4` **1.478 h**（2.05x）；`config_hash d2ccc46b` 自 round 9 起未變。機器：RTX 5090 / CUDA 13.0 / torch 2.11.0+cu130。Round 9–11 摘要：round 9 上線 Option H（gc 週期 re-freeze）與 Option L（tile-read prefetch，僅單 process）並關閉 nvImageCodec 路線；round 10 把 Option L 接進多行程、在真實規模跑 Phase D 讀取/join（**反轉** round 9 的 in-memory spike）、跑第一次整片 `workers=1`（被 `d6592c3` 混淆）；round 11 以乾淨 `--no-prefetch` 整片跑批解混淆、修好 fail-fast hang、在 `workers=4` 掃 allocator knob；**`B2r_tile_read` 的「17.2% wall」屬於已移除的寫入模式，真實無混淆成本 4.21%，Option L 隱藏 99.8%**。
- **目前 anchors 表**：baseline control（441 tile、14.1% 背景）848.0 s；large/441 crop 現碼 `workers=1` 302.7 s／`workers=4` 128.8 s；**match24（576 tile、55.9% 背景，實測 55.8%）188.8 s／88.3 s**；整片（27,565 tile、round 8 baseline）13,762.5 s = 3.82 h／6,211 s = 1.73 h；round 10（`workers=1`、prefetch on、被 `d6592c3` 混淆）10,217.7 s = 2.84 h（1.347x）；round 11（`workers=1`、`--no-prefetch`、乾淨）10,666.1 s = 2.96 h（1.290x）；round 12（`workers=4`、今日程式碼）5,854.9 s = 1.63 h。累積單 process 在 tissue-dense crop：**848.0 → 302.7 s（−64.3%）**；`workers=4` 已放行生產（帶 VRAM 但書）。round 12 重跑 `workers=4`：對 round 10 的 `workers=1` 是 **1.745x、非 2.216x**；tile 平行臂本身 **2.279x**（match24 2.138x、round 8 2.510x 獨立印證 2.1–2.5x 組成帶）；Phase D 1,239.2 → 1,889.8 s、占 `workers=4` wall **32.3%**（Amdahl 1.477x；n=1，見 doc 39 §2.4）。
- **手臂模型（現況，真實玻片組成）**：`wall ≈ max(MAIN, BG) + outside`；match24（55.9% 背景，`workers=1`）：**MAIN 187.4 s、BG 88.0 s、outside 7.5 s → BG/MAIN = 0.470 → MAIN 必須削減 53.0% 才讓 BG 臂（`detect_all_dots`、PNG）變關鍵**——比 round 7 前所有 tissue-dense crop 暗示的（低至 15.9%–28%）更寬；更多組織 tile 讓 MAIN 負載增加比 BG 快，與「大部分是背景的玻片」的直覺相反。
- **全部瓶頸現況表（摘錄每列結論）**：
  - ① GPU 前向（序列 pipeline / 3 次 per-tile 前向）：**DONE（多階段）**，仍是主要；overlap −16.6%、Cellpose 換版 −18.9%、`dot_detect_n_jobs` 使 MAIN 本身快 43.4%、累積 848.0 → 302.7 s（−64.3%）；剩餘內部 ceiling ~1.118x（launch-bound Python 迴圈、第三方）。
  - ② `detect_all_dots`：**「免費」解決**（BG 臂被遮住，ceiling 1.00–1.013x；真實組成 BG 臂 slack 47–53%）。③ PNG/TIFF 編碼：**HIDDEN**（同臂）。
  - ④ per-tile `gc.collect()`：✅ **上線並在整片確認（round 9；round 10+11 確認）**：`_PeriodicFreezer` 每 5,000 累積細胞 `gc.freeze()`（O(1)）；整片 2,218.4 → 58.8 s（−97.3%，37.7x）→ round 11 19.5 s；70 freeze；RSS 反降 1.09 GB；doc 31 §7 投影 1.19x、實測 1.186x；`workers>1` 不受影響。
  - ⑤a Precut A：**DONE**（串流進分析迴圈；20.3 s → 0.004 s；crop −3.1%/−3.8%）。
  - ⑤b Phase D（`_stitch_overlay_slide`）：🟢 **round 12 重開為正向 → ✅ round 13 上線為預設、正向關閉**（詳見上文 §14）；歷程記錄 round 10 record（B 0.788x、C 0.884x、「Phase 2 不被證成」）、round 12 管線化讀取（B 0.785x → 1.365x、C 0.856x → 1.581x；讀取 36.07 → 1.56 s（95.7% 隱藏）、36.56 → 4.23 s（88.4%））、真實來源冷讀 981.0 s = 51.9%（vs 48.7%；絕對成本 15.4 vs 116.6 s/GP 不轉移）、投影 1.135x（完美重疊 1.184x）；Phase D 占 wall 比例（8.6% → 19.3% → 11.6% → **32.3%**）；**round 9 找到並修好的既有缺陷**：libvips 只在 level 0 標 Predictor=2 卻也對縮小層做差分，所有 round 9 前的 `overlay_slide.tiff` 縮小層 ~90% 解成黑（QuPath 確認）；`predictor="none"` 修復（整片 +28.0% 檔案大小，非 spike 暗示的 +9.4%；時間中性）；round 10 兩個整片 overlay 皆確認壞並重縫（`overlay_pyramid_audit.py`）；**round 13**：`config.stitch_backend = "tifffile"`、Phase D 1,889.8 → 987.8 s（1.913x）、e2e 5,854.9 → 4,877.4 s（1.200x）、四道 correctness gate 過、peak RSS 45.6 → 17.0 GB、artifact 7.50 → 5.85 GB、Phase D 降到 20.3% wall；round 14 確認 `workers>1` 無伴隨的 GPU↔CPU 搬運瓶頸；round 15 在正確畫布 Phase D 占 `workers=4` wall **24.1%**。
  - `B2r_tile_read`：✅ **兩條 process 路徑皆上線（round 9 單 process、round 10 多行程）；整片結清（round 11）**——Option L 在專用 `tile-read` 執行緒一個 tile 超前預取；round 9 crop −1.75%（1.018x）；round 10 接進 `_mp_tile_worker`，機制確認（`B2r_tile_read` +14.9%/+26.4% 而 wall 下降）但 crop wall 主張（−0.83%）在雜訊內；**round 11 乾淨整片 `workers=1 --no-prefetch`**：讀取位置與 doc 27 baseline 相同時 **449.3 s = 4.21% wall，較 baseline 的 2,368.5 s/17.2% 下降 81.0%**（同樣 55,130 次讀取），差距是 `d6592c3` 已移除的 275 GB/slide 並行寫入與 OS page cache 競爭；**Option K = 1.044x，非 round 10 記的 1.208x 上界**；Option L 隱藏 448.4/449.3 s = 99.8% 一個本身只有 4.21% wall 的天花板；也削弱（不重開）doc 35 §5 `depth=2/3` 的 1.04% 天花板；整片 tissue:background per-read 比例 1.97–2.41（crop 2.06）。
  - ⑥ model init：資訊性（441 tile 時 0.37% wall，整片規模進一步攤薄）；⑦ API/job 層（Phase E）：資訊性、可忽略（~10⁻⁷）。
  - ⑧ 閒置 CPU prep（`enlarge_cell_instances` + `build_all_positive_results`）：**DONE**（搬到 BG 臂；−8.0%/−5.0%；也移除兩臂間 GIL 競爭使未改動 bucket 變快）。
  - ⑨ `detect_all_dots` +22.3% 迴歸：🟢 開著、成因從未隔離——但無意義（ceiling 1.013x；被 `dot_detect_n_jobs` 取代）。`cellpose_batch_size` 死 config：**DONE**（接進 `Config`；掃描 16/32/64 持平，僅在 tile ≥1536 才有效）。`detect_all_dots` joblib fan-out（`n_jobs=-1` → `1`）：**DONE**（獨立量 2.77x；e2e `workers=1` 1.60x（484.7 → 302.7 s）；隔離了 ① 遺留的 +192.9 s GIL 異常）。
  - 跨 tile 多行程（`workers>1`）：✅ **上線**；round 12 首次重量：**1.745x、非 2.216x**；correctness veto 通過且比 round 8 更緊（356,220 vs 356,226 rows、−0.002%、verdict 相同、pyramid audit 12 層乾淨）；tile 平行臂不退化（2.279x）；損失全在 Phase D；**出貨 `workers=4` 但 peak 30,439/32,607 MB（93.3%，~2.2 GB 餘裕）：32 GB VRAM 視為硬底線、不在 worker pool 內加任何 GPU 函式庫、`workers≥5` 在 allocator 氣球根因前維持關閉**；doc 38 Amdahl 帳：缺口是兩個獨立已量到的效應——真實 55.8% 背景限制 tile 平行 ~2.1–2.5x（match24 2.138x，排除 stitch）、Phase D 完全序列的 Amdahl 稅（完全管線化 ~1.09x，從未建）；load balancing 由量測關閉（派送全部 27,565 tile 零工作 0.356 s、0.006% wall、1.00006x；batch-claiming 使 worker 不平衡單調更糟）；**round 13 出貨了這列缺口所繫的 Phase D 修復**（`workers=4` e2e 4,877.4 s）；round 14 沒找到 `workers>1` 搬運瓶頸（磁碟 1.4%、`_from_device`/pinned 0.18%/0.42%）但發現 `workers=4` crop 規模 1/10 OOM；**round 15 在正確畫布 `workers=1→4` 2.05x（排除 Phase D 為 2.36x）**，且組成導出的「serial fraction」不是常數（576 tile 36.8%、4,096 tile 17.1%）。
  - allocator 碎片化 OOM：✅ **根因與緩解、預設已翻開（round 11 量；`b3fa47d`）**：`workers=6` 的 round 6–8 掃描與 round 10 `workers=4` ablation 都間歇命中 byte-identical **24.76 GiB** 氣球；先前 `workers=6` 規模掃描發現 `expandable_segments:True` 沒降低 peak VRAM 且 +2.0% wall；**round 11 在 `workers=4` 重掃 12 次交錯**：control **4/12 OOM（33%）**、`expandable_segments:True` **0/12**（Fisher p=0.047）；代價 +0.67% wall、RSS 不變、**peak framebuffer 升到 92.2% of card（max 18.2 → 30.1 GB）**——knob 移除失敗模式、不創造餘裕。
  - 整片 WSI 驗證：✅ **DONE 且已重跑兩次**（round 8 首個完整 slide；round 10 `workers=1`（混淆）；round 11 乾淨）；`stats` 三次相同（10,800–10,801/16,764–16,765）；peak RSS 60.04–61.13 GB；registration 對每 modality 輸出不同畫布（**後被 doc 44 證實 `--conform` 的處理是錯的**）。
  - `_stitch_overlay_slide` `RLIMIT_NOFILE` guard：✅ DONE；`run_batch` 斷點續跑：✅ DONE（opt-in；12 tests）；多行程 worker 內 per-bucket 計時：✅ DONE（26 個 worker-side bucket）；`B2r_tile_read` prefetch 多行程路徑：✅ 上線（round 10）；post-fail-fast hang：✅ 修好（round 11）；`B1_m3b_cellpose` +751 s 迴歸：🟡 大致解釋（221 s 來自 Option L 競爭），530 s 殘差未歸因；CUDA allocator 組態 knob：（見 allocator 列）；玻片組織/背景組成前提：**DONE**（55.8% 背景，非 39%——**後被 doc 44/46 更正為 65.92%（正確畫布）**）；Candidate F：🔴 量過、不建（24 ms/tile、7.5% wall、全在 slack BG 臂；`os.link` 便宜 272–407x，是儲存論點非速度）；Candidate G：🔴 建、量、revert（0.056% wall）；跨 tile Cellpose/UNet++ 批次：🔴 停損（G=16 +5.9–6.6%、15.8 GB）；CUDA MPS：🔴 停損（+44% 合成、端到端 0%）；CPU 後端 process pool/更深 BG pipeline：🔴 停損（+2.8% 更慢）；fork 重用：🔴 不建；CUDA-stream bubble：🔴 停損（≤1.065x）；CUDA graph/向量化 Cellpose 內部：🔴 停損（~1.118x）；多請求並發（Phase E）：🟢 開著、從未量。
- **記憶體**：crop 規模 VRAM peak（單 process）2787 MB（與 tile 數無關）；多行程下每 process VRAM 超線性（2787 → 3117 → 4118 → 5167 MB @ N=1–4 crop）——這才是限制安全 worker 數的因素。**Round 8 整片實測**：peak RSS 61.13 GB（`workers=1`）/ 61.67 GB（`workers=4`）；peak GPU 2,739 MB / 30,439 MB（93.3%）；隱含主機需求 ~64 GB RAM、~32 GB VRAM（`workers=4`）、~350 GB 磁碟/slide、`RLIMIT_NOFILE` ≥ ~28,000；round 10/11 `workers=1`：RSS 60.04–60.17 GB；round 11 `expandable_segments` 下 `workers=4` peak framebuffer 可達 92.2% of card。**以上（rounds 8–11）都量在錯誤畫布上**；**round 15 在正確畫布重量 RSS：13.66 GB（`workers=1`）/ 13.24 GB（`workers=4`）**，較 round 8 下降 77.6%（部分來自 round 13 的 stitch backend 改變——band 串流取代 lazy 整片 join——部分來自 round 8 baseline 本身量在錯誤輸入）；VRAM 未在正確畫布重量。
- **分類總表**：1 演算法/模型複雜度——② `detect_all_dots`、① 內的 Cellpose launch-bound 內部；3 平行/併發——① 序列 → overlap pipeline、⑧ CPU prep 位置、跨 tile 多行程；4 記憶體生命週期——④ per-tile gc、RSS 細胞結果累積；5 I/O 與儲存——③ PNG、⑤a/⑤b precut 與 stitch、背景 tile placeholder 寫入；6 架構/框架——① 無跨 tile GPU 批次、⑥ init、⑦ API；7 config/死碼——`cellpose_batch_size`（已修）、`detect_all_dots` joblib fan-out（已修）、⑨ 迴歸成因（未解）。

### 19.2 `measurement/current-status-comparison.md` — baseline vs 現況（精簡）

> ⚠️ **頂部註記：round 15 時已過期、未重新製表**——下面各表（特別是 full-WSI 段）皆在 round 8–11 畫布（27,565 tile、55.8% 背景）= 未配準輸入；正確畫布 35,700 tile/20.94 GP/65.92% 背景；round 15 整片重跑：`workers=1` **3.023 h**、`workers=4` **1.478 h**（2.05x）、peak RSS **13.66 GB**（較本檔 60+ GB 低，部分來自 round 13 stitch backend、部分來自 round 8 baseline 量在錯誤畫布）；也尚未反映 round 13 Phase D 修復（預設 `stitch_backend = "tifffile"`，Phase D 在舊畫布占 20.3% / 正確畫布 24.1% `workers=4` wall）與 round 14 的 GPU↔CPU 搬運排除。Baseline：git `96a28ba`、完全序列 `run_batch`；current：round 11（2026-07-29），config hash `3d1087f2`（round 7 起未變；`cuda_alloc_conf` 當時預設 `""`——後被翻開）。
- **§1 端到端錨點**：small 25 tile baseline 54.3 s；medium 121 tile 243.3 s；large 441 tile 848.0 s → **302.7 s（`workers=1`，−64.3%）**／128.8 s（`workers=4`）；**match24（55.9% 背景，匹配真實玻片 55.8%）576 tile 188.8 s／88.3 s**；各規模無負最佳化。**GPU 利用率**：idle_frac（large/441，sm==0）baseline 0.459 → 現況 0.06–0.19（依 `workers`）；mean SM 28.3 → 16.6–78.1（單 process SM% 下降是 Cellpose 本身加速的副作用，非回歸）。
- **§3 記憶體**（baseline → crop 規模現況 → 整片現況）：VRAM peak 5159 MB（large/441）→ **2787 MB**（單 process；隨 `workers` 超線性）→ **2,739 MB（`workers=1`）/ 30,439 MB（`workers=4`、93.3%）**；peak RSS 4.04 GB（large/441）→ ~3.9 GB（隨累積細胞數而非 tile 數）→ **61.13 GB（`workers=1`）/ 61.67 GB（`workers=4`）round 8；60.04–60.17 GB（`workers=1`）rounds 10–11**。
- **§4 整片實測（27,565 tile、16.2 GP）**：baseline（外推）~18.9 h；round 8 `workers=1` 13,762.5 s = 3.82 h；`workers=4` 6,211 s = 1.73 h（2.216x）；round 10（`workers=1`、prefetch on）10,217.7 s = 2.84 h（1.347x）；round 11（`workers=1`、prefetch off）10,666.1 s = 2.96 h（1.290x）；success/skipped 10,801/16,764 → 10,800/16,765；`report.csv` 356,255 → 356,221（−0.01%）→ 356,226；`B4_gc_collect` 2,218.4 s（16.1%）→ 58.8 s → 19.5 s；`B2r_tile_read` 2,368.5 s（17.2%）→ 1,581.2 s（混淆）→ **449.3 s（4.21%，結清數字）**；Phase D 1,185.4 s（8.6%）/ 1,200.8 s（19.3%）/ 1,239.2 s（11.6%）；peak RSS 61.13/61.67/60.04/60.17 GB。baseline 是 3-tile crop 外推的上界，從未在整片跑；round 8 是第一次實測。
- **§5 仍值得最佳化的**：截至 round 11，round 8 留下的三個最大項目全已關閉（`gc.collect`、Phase D GPU 化（負向）、tile read）；剩餘：allocator 氣球（已緩解，當時預設未翻——後翻）、`B1_m3b_cellpose` 530 s 殘差。
- **§6 重現**：`core_mask_map.py`（針對 pipeline 自己的 core-mask map）→ `composition_crop.py --grid 24 --map ...` 產生 match24 → `perf_measure.py` `workers=1`/`4`（`--gpu-dmon --stream-precut --mp-workers N`）→ `stitch_probe.py` 在整片尺寸 + `wsi_projection.py`；原始 artifacts：baseline `_metrics/`、現況 `_metrics_r7/`（含 `env_stamp_r7.txt`、`pip_freeze_r7.txt`、`core_mask_map.npz`）。

---

## 20. `measurement/` 其餘檔案——歷史版、翻譯版、HTML、原始資料

### 20.1 `bottleneck-list-history.md`（77 KB）——逐輪完整演進史（round 1–8）

- 它是 `bottleneck-list.md` 的**舊版完整本體**（控制組時期的 ①–⑦ 原始紀錄 + 各輪 `Update:` 註記），內容與前面各章（doc 10–27）重疊，這裡只列**這份文件獨有或最適合當索引的資訊**：
  - **閱讀順序**：①–⑦ 項目本體是 round 1 控制組時期的原始紀錄，各附日期註記；現況請看「Round-3 anchors」「Current ranking」「Round-4 anchors」「Round-5 anchors」與文末重新排序的優先序。每個 % 都是相對**該輪自己的錨點**（依 plan §5.1）。
  - **Anchors（控制組 / 樸素版本基線，保留不覆寫）**：small（25 tile，5×5）54.3 s、2.17 s/tile、precut A 1.45 s、22 cells、3 背景 tile、RSS 2.82 GB、VRAM 4.68 GB；medium（121，11×11）243.3 s、2.01、5.57 s、103 cells、18 背景、3.07 GB；large（441，21×21）848.0 s、1.92、20.58 s、379 cells、62 背景、4.04 GB。整片（35,700 tile @1024px）線性投影 `total ≈ 9.4 s + 1.903 s/tile` ⇒ **~18.9 h**（上限）。
  - **Round-3 anchors（Cellpose 4.2.1.1 / cpdino）**：medium 166.6 s（−20.0% vs overlap、−31.6% vs control）、large 573.7 s（−18.9%、−32.3%）；整片線性 `12.6 s + 1.2722 s/tile` ⇒ **~12.6 h**；**這不是單變量換版**：cellpose 4.0.8 → 4.2.1.1（DINOv3 `cpdino` + `use_bfloat16=True`）、兩個 checkpoint 重訓、venv 由 `uv.lock` 重建（torch 2.10→2.11、numpy 2.2.6→1.26.4、scikit-image 0.25.2→0.24.0、pyvips 3.1.1→2.2.3、opencv→4.8.1.78）。
  - **Amdahl 停損（修訂版）**：原則「<~10% 列出但不深入」；2026-07-22 修訂：overlap 落地後 self-time ÷ wall 不再是關鍵路徑份額——兩條臂（MAIN：3 GPU 前向、`_read_rgb`、M1 overlay、`clear_slide_edge_cells`、`build_all_positive_results`、`enlarge_cell_instances`、`gc.collect`+`empty_cache`；BG：`detect_all_dots`、merge、PNG/TIFF、`render_overlay_image`、per-cell crops、`filter_and_absolutize`）；`wall ≈ max(MAIN, BG) + outside`（large：max(538.3, 387.3) + 28.0 = 566.3 s vs 實測 573.7 s；medium 164.7 vs 166.6，1.3% 內）；第二次修訂：在關鍵臂且可趨近 0 的項目暫停「<10% 放棄」（全片 ~12.6 h 時 6% ≈ 45 min），標為「sub-floor but actionable」。
  - **round-3 排名（large/441）**：① GPU forwards 444.6 s/77.5%/MAIN/1.382x；④ `gc.collect` 36.4 s/6.3%/MAIN/1.083x（已解）；⑧ 閒置 CPU prep 28.4 s/5.0%/MAIN/~1.05x（新）；⑤ precut A + stitch D 25.6 s/4.5%/outside/~1.05x；② `detect_all_dots` 292.9 s/51.1%/BG/1.013x；③ PNG 78.7 s/13.7%/BG/1.013x；⑥/⑦ 2.5 s/µs/0.4%/~1.00。**Arm slack**：BG 387.3 s vs MAIN 538.3 s，BG/MAIN 0.719（overlap 輪 0.496），MAIN 必須削減 **151.0 s = 34.0%**（medium 36.8%）——後被 round 4 重量為 25.6%/26.4%，⑧ 搬走後 15.9%。
  - **Round-4 anchors**：round-3 紀錄 573.7/166.6 s；`p0`（`gc.freeze()` 已在）538.5/154.0；`p2`（⑧ 搬到 BG）495.5/146.3；`p3`（precut 串流化）**480.3/140.8**；累積 −16.3%（本輪兩個 code 改動單獨 −10.8%）；整片 `12.5 s + 1.0608 s/tile` ⇒ ~10.5 h；p3 後 MAIN 453.9、BG 381.7、outside 7.6 → BG/MAIN 0.841、MAIN 還剩 15.9%、單 process 下限 `BG + outside = 389.3 s`（1.23x）。兩個量測警告：`idle_frac`（SM==0）是 knife-edge、不可跨配置比較（p0→p2 假性 0.32→0.43 而 wall −8%；near-idle SM≤3 0.50→0.45）；`peak_cuda_reserved_gb` 不可靠（一次報 25.97 GB vs `dmon fb` 2787 MB）。
  - **Round-5 anchors**：`workers=1`（= p3）482.8/138.1 s、2.31x（`workers=2`）、**3.09x（`workers=3`，推薦）**、3.51x（`workers=4`）；FB peak 2787/6233/12354/20667 MB；RSS 4.04/6.52/9.26/11.97 GB；整片 `8.9 s + 0.3337 s/tile`（`workers=3`）⇒ ~3.3 h（`workers=1` 10.68 h，與 round 4 的 10.52 h 一致）；遠高於 doc 20 的 1.23–1.7x，原因是漏算兩臂間 GIL 競爭；`workers=2` 的 116% 超線性效率是簽名；近閒置 0.46 → 0.06；MPS（+44% 合成、端到端平坦）與 Candidate A（+2.8% 慢）關閉。
  - **各項 deep-record**：① 現象 GPU idle ~46–49%、mean SM ~28%、B1 45.5%、VRAM 5.16/32 GB；overlap 後 idle 0.494 → 0.154（121 tile wall −18.5%）；2026-07-11 再確認 idle 0.459 → 0.190（large）/0.477 → 0.163（medium）、wall −16.6%/−14.5%；**B1 絕對秒數也成長**（386.0 → 578.9 s，+192.9 s）：成因 1 為 harness 重新標記（`feedfbd` 刪除 `segment_masked_dish`、M2 與 M3b 都走 `segment_windowed`，故 `B1_m3b_cellpose` n=824 = 2 × 412；控制組等價成本 187.0 + 185.5 = 372.55 s vs 現 564.48 s，仍有 +51% per-tile）、成因 2「與 GIL 競爭一致但尚未隔離」（後由 round 6 `dot_detect_n_jobs=1` 的 −178.2 s 隔離）；round 3 後 B1 → 444.6 s。Round 3 對照表（large）：wall 707.4 → 573.7；B1 578.9 → 444.6（−23.2%）；每次 Cellpose 呼叫（824 次）0.6850 → 0.5229（−23.7%）；VRAM peak 5159 → 2787 MB（−46.0%）；idle_frac 0.190 → 0.370；mean SM 32.9 → 16.6。**② `detect_all_dots`**：控制組 260.1 s/30.7%、ceiling 1.44；overlap 後被 GPU 前段完全遮住（每 tile 0.58 s vs 1.33 s）→ 有效 ceiling ≈ 1.0；round 3 292.9 s/51.1% 但 ceiling 仍 1.013x；round 6 發現原「已 CPU 平行」的描述是錯的。**③ PNG 編碼**：76.8 s/9.05% → 78.5 s/11.1% → 78.7 s/13.7%，絕對成本三輪持平。**④ gc.collect**：36.3 s/4.28% → 36.4 s/6.34%；2026-07-22 更正（讀 source）：`gc.collect()` 在 MAIN 執行緒 `hybrid_pipeline.py:798` 而非背景；以 `gc.freeze()` 解決（83.2 → 1.2 ms/call）。**⑤ Phase A/D**：2.43%/0.60%；round 3 A 20.51 s/3.58%、D 5.05 s/0.88%，不隨 pyvips 3.1.1 → 2.2.3 降版而變；round 4 A 串流化（20.31 → 0.004 s）、D 結構性不可重疊（5.10 s 中可重疊 ≈ 0）。**⑥ model init**：7.3%（25 tile）→ 1.3%（121）→ 0.37%（441）；與 gil-contention-diag 的「UNet++ 14.9% GIL 其中 93% 是 `_init_unet_inferencer` import cascade」互證。**⑦ API 層**：`submit_job` enqueue 2.3 µs、BackgroundTask dispatch ~3.8 ms，~10⁻⁷。**⑧ 閒置 CPU prep**：`enlarge_cell_instances` 19.60 s/3.42% + `build_all_positive_results` 8.81 s/1.54% = 28.4 s/4.96%（large）、8.36 s/5.02%（medium）；round 4 搬走 −8.0%/−5.0%，M2→M3b device-idle gap 10.06 → 1.66 s。**⑨ `detect_all_dots` +22.3%**：239.41 → 292.90 s（+53.5 s）而 cells 僅 +1.8%（12,922 → 13,150）；per cell 18.5 → 22.3 ms（+20.2%）；medium +11.8% vs +2.5% cells；PNG encode 平（78.5 → 78.7 s）；成因未隔離。**記憶體**：VRAM 5.16 GB 與 tile 數無關；RSS 2.82 → 3.07 → 4.04 GB（25 → 121 → 441 tile，17.6x tile、+43% RSS）追蹤累積細胞結果；round 3 VRAM 5159 → 2787/2785 MB、`cuda_reserved` 4.675 → 2.187 GB、RSS 3.94 → 3.90 GB。
  - **round-3/round-4 重新排序優先序、round 6/7/8 anchors** 皆與前面 §7–§9 內容重複（Round-6：`workers=1` 484.7 → 302.7 s（1.60x）；`workers=6` ~0%；Round-7：55.8% 背景、comp24 134.91/65.54 s（2.06x）與 match24 188.8/88.3 s（2.14x）、BG/MAIN 0.527/0.470、Candidate G/F/Cellpose G=16、Phase D 322.7 s（35,840×28,928 14.63 s/14.11 s/GP；71,168×57,344 57.80 s/14.16；141,818×114,366 322.7 s/19.90）、GPU 函式庫環境 gate 表、整片投影 ~2.6/~1.25 h；Round-8：首次整片 3.82 h/1.73 h、三個 crop 導出的 stop-loss 在整片不成立、`tiffsave` 13 config 全關、RLIMIT_NOFILE/resume/worker 計時/allocator knob、7 項文件漂移全關）。

### 20.2 `current-status-comparison-history.md`（42 KB）——baseline vs 現況的逐輪歷史（至 round 8）

- **結構**：§1–§7 為 2026-07-11 的紀錄（`96a28ba` 對 `0e27b20`、config_hash `db2b7e6a`、medium 121 與 large 441）；§8 為 round 3（Cellpose 4.2.1.1）；§9 為 round 4–6 的濃縮鏈；§10 為 round 7；§11 為 round 8。
- **§0 兩個 commit 之間落地的改動**：`010308f`（①方案 (b)：GPU 前段 + 背景 CPU 後段，depth-1 overlap）、`feedfbd`（滑動視窗接縫縫合 + VALIS 文件）、`119ad73`（`draw_tile_seam_edges` 視覺 QA）、`9e618d3`（②③ 元件級：`@lru_cache` 的 morphology `disk()` footprint）。
- **§1 端到端錨點**：medium 243.3 → **208.2 s（−14.5%）**（2.011 → 1.720 s/tile）；large 848.0 → **707.4 s（−16.6%）**（1.923 → 1.604）；無負最佳化；現況 707 s 略低於 `00f2c91` 中間量測 724.7 s（熱雜訊內）。**GPU 利用率機制**（large）：idle_frac 0.459 → **0.190**（idle 砍 ~59%）、mean SM 28.3 → 32.9、busy≥50% 0.252 → 0.292、mem-ctrl 18.1 → 21.5；medium idle 0.477 → **0.163**（砍 ~66%）、mean SM 28.4 → 35.9。
- **§2 時間流向（large 441，% of 該 run wall）**：GPU 前段 386.0（45.5%）→ **578.9（81.8%）**（現為關鍵路徑）；`detect_all_dots` 260.1（30.7%）→ 239.4（33.8%）（同絕對成本，現被遮住）；PNG 76.8（9.1%）→ 78.5（11.1%）；`gc.collect` 36.3（4.3%）→ 36.3（5.1%）；precut A 20.6 → 20.4；stitch D 5.1 → 5.1。注意：overlap 下 self-time ÷ wall 不再是關鍵路徑份額。Harness caveat：①重構後 M2 與 M3b 都呼叫 `segment_windowed`，`B1_m3b_cellpose` bucket 現在合計**兩次** Cellpose 前向；`B_process_precut_tile_TOTAL` 讀 0（函式被拆為 gpu/cpu 兩段）。
- **§3 per-bottleneck 狀態**：① DONE；② 免費解決；③ HIDDEN；④ 不變（5.1%）；⑤ 不變；⑥ 不變；⑦ 不變。**淨結論**：三個 deep-record（①②③）都處理了；關鍵路徑從「GPU idle + 分散的 CPU 工作」移到「GPU 前向本身」。
- **§4 記憶體**：medium RSS 3.07 → 3.06 GB、VRAM 5159 → 5159 MB；large RSS 4.04 → 3.94 GB、VRAM 5159 → 5159 MB；一個異常（非真迴歸）：資源取樣器在現況 medium 記錄 `cuda_alloc_peak 22.2 GB` 但同 run 的 `dmon` framebuffer peak 5159 MB——torch-allocated 欄位不可靠，VRAM 讀 `dmon fb`。
- **§5 整片重新投影（35,700 tile）**：現況線性 `19.4 s + 1.560 s/tile` ⇒ ~15.5 h（原 1.903 → 18.9 h；相對 −18%）。**§6 仍值得優化（當時排序）**：GPU 前段（81.8% wall）；殘餘 GPU idle 16–19%；`cellpose_batch_size` 死 config（VRAM 5.16/32 GB 約 27 GB 閒置）；`gc.collect` 5.1% 與 stitch D 0.7%；停損注記：②③④ 被遮住或低於下限，下一輪紀律是只攻 lever 1。
- **§8 Round 3 詳細**（三向對照）：medium 243.3 → 208.2 → **166.6 s**；large 848.0 → 707.4 → **573.7 s**；large s/tile 1.923 → 1.604 → 1.301；Cellpose 前向（824 次）372.6 → 564.5 → **430.9 s**（−23.7%）、per call 0.4521 → 0.6850 → **0.5229 s**；UNet++（441 次）13.45 → 14.41 → 13.68 s（持平）；VRAM peak 5159 → 5159 → **2787 MB**；`cuda_reserved` 4.68 → 4.68 → **2.19 GB**；idle_frac（large）0.459 → 0.190 → **0.370**（medium 0.477 → 0.163 → 0.293）；mean SM 28.3 → 32.9 → 16.6；arm 模型：MAIN 467.9 → 672.0 → 538.3 s、BG 350.6 → 333.2 → 387.3 s、BG/MAIN 0.749 → 0.496 → **0.719**、outside 28.0 s；Amdahl 表（large、錨點 573.7 s）：GPU forwards → 0 為 1.382x、`gc.collect` 1.083x、⑧ ~1.05x、precut A + stitch D ~1.05x、`detect_all_dots` 1.013x、PNG 1.013x、兩臂都 → 0 理論 4.69x；「`detect_all_dots` 顯示 51.1% wall 卻只值 1.3%；`gc.collect` 6.3% 卻值 8.3%（六倍，儘管看起來小八倍）」；regressions（large）：`detect_all_dots` 239.4 → 292.9 s（+22.3%）、per cell 18.5 → 22.3 ms；`enlarge_cell_instances` 18.29 → 19.60 s；`build_all_positive_results` 7.23 → 8.81 s；PNG、gc、precut/stitch 持平；correctness（非常數）：cells medium 3559/3558/**3647**（+2.5%）、large 12919/12922/**13150**（+1.8%）、tiles success/skipped large 379/62 → **378/63**；記憶體：RSS 3.94 → 3.90 GB（large）、VRAM 2787 MB，idle VRAM ~29.8/32 GB；整片 `12.6 s + 1.2722 s/tile` ⇒ ~12.6 h（−33%）；Round 3 回答：Cellpose 換版買了 −18.9% wall（707.4 → 573.7 s）、整片 ~15.5 → ~12.6 h；機制是 −23.7% per-call Cellpose 前向與 −46% VRAM；對控制組累積 −32.3%（overlap −16.6% + 本換版 −18.9%）；GPU 仍是瓶頸但只剩 34% margin（idle 37%、CPU 後段 72% of 關鍵臂）；新值得優化：`gc.collect`（1.083x、整片 ~44 min）、⑧、precut A + stitch D。重現指令（`uv sync` 對 `uv.lock`、`nvidia-smi` 確認空閒、`perf_measure.py --workers 8 --gpu-dmon`、`aggregate_report.py`、`resource_analyze.py`、`uv pip freeze > pip_freeze_actual.txt`）。
- **§9 rounds 4–6 濃縮鏈**：r3 573.7 s → r4（⑧ + precut 串流）**480.3 s**（−16.3%）→ r5（`workers=3`）**156.1 s**（−67.5%）→ r5b（`workers=6` 推薦）**123.3 s**（−21.0%）→ r6（`dot_detect_n_jobs=1`；建議下修為 `workers=4`/`5`）`workers=1` **302.7 s**、`workers=4` **128.8 s**；累積 `workers=1`：848.0 → 302.7 s（−64.3%）。**§10 Round 7**（組成前提更正：背景 55.82%/組織 44.18%、背景 tile 15,386、組織 12,179；anchors：large 14.1% 302.7/128.8 s 2.35x；comp24 134.9/65.5 s 2.06x、BG/MAIN 0.527、MAIN 需削減 47.3%；match24 188.8/88.3 s 2.14x、0.470、53.0%；per-bottleneck：Phase D 322.7 s 是唯一比估計更糟的；Candidate F 24 ms/tile、7.5% wall、零 wall payoff；GPU codec 依賴 gate；整片重投影 ~2.6/~1.25 h）。**§11 Round 8**（首次整片 3.82 h/1.73 h、2.216x、RSS 61.13/61.67 GB、GPU 2,739/30,439 MB；三個 crop 導出的 stop-loss 在整片不成立；`tiffsave` 消融關閉；RLIMIT_NOFILE、resume、worker 計時、allocator knob；三個重新定位：放行 `workers=4`（帶 VRAM 底線）、Phase D ceiling ~3x（1.239x）、allocator 氣球根因才是活問題）。

### 20.3 `*-zh.md` 翻譯版

- `measurement/bottleneck-list-zh.md`（59 KB，593 行）：`bottleneck-list-history.md` 的**繁體中文翻譯**（標題「效能瓶頸清單 — 混合管線（實測）」；閱讀順序段提到「共 5 輪量測」，說明它翻譯的是 round 5 時點的版本——比 history 英文版少了 round 6–8 的內容）。
- `measurement/current-status-comparison-zh.md`（27 KB，408 行）：`current-status-comparison-history.md` 的繁中翻譯（開頭同樣是 2026-07-11 與 round 3 的內容）。
- 兩者**無新資料**，只是語言版本，且比英文歷史版舊（停在 round 3–5）。

### 20.4 `measurement/perf_report.html`（28 KB）——最初的控制組量測報告（2026-07-07，git `96a28ba`）

- 標題「Hybrid Pipeline — Deep Bottleneck Measurement Report」，執行 `docs/hybrid-pipeline/09-measurement-analysis-plan.md`；量測於 2026-07-07 02:22:02 CST、git `96a28ba556`、config_hash `db2b7e6a`。
- **1 環境與來源**：Python 3.11.15、RTX 5090、driver 580.159.03、CUDA 13.0、CPU Intel Core Ultra 7 265K（20 邏輯核）、RAM 62 GB、torch 2.10.0+cu130（device cap (12,0)/sm_120，已驗證真實 GPU matmul 與 Cellpose 前向）、numpy 2.2.6、cellpose 4.0.8（SAM backbone）、pyvips 3.1.1、scikit-image 0.25.2、scipy 1.16.3、opencv-headless 4.12；環境 caveat（不影響效能數字）：(1) `requirements.txt` 釘 `torch==2.10.0+cu130` 但 PyTorch R2 CDN（`download-r2.pytorch.org`）在此主機 TLS handshake 失敗，wheel 改由 `download.pytorch.org` 直接下載並以 PyPI 解 CUDA-13 依賴；(2) `requirements.txt` 缺 `segmentation_models_pytorch`（`unet_inference.py` 匯入）與 `fastapi/uvicorn`（API 層匯入）；(3) `pip check` 因本機 torch wheel 的 `+cu130` local tag 當掉（pip bug）。
- **2 方法與執行順序（plan §1.4）**：非侵入式 harness `scripts/perf_measure.py`（monkeypatch 計時 shim、0.5 s RAM/VRAM 取樣執行緒、1 s `nvidia-smi dmon`）；先量一個乾淨端到端錨點（不用 cProfile——cProfile 會膨脹 Python 密集項）；這次序列跑本身就是「樸素版本」控制組（`run_batch` 是硬編碼序列迴圈）；所有子項先除以錨點得 %；cProfile 另跑一次僅供函式級 Top 表；輸入規模皆取自 156222×134028 warped WSI 對：25 tile（4096² test ROI）、121 tile（8192² crop）、441 tile（16384² crop）；整片 35,700 tile（~20 h）為投影而非實跑。
- **3 端到端錨點（控制組數字）**：small 25 tile 54.3 s（2.17 s/tile、precut A 1.45 s、`run_batch` 52.8 s、22 cells、3 背景、RSS 2.815 GB、VRAM 4.675 GB）；medium 121 tile 243.3 s（2.01、5.57、237.8 s、103、18、3.074 GB）；large 441 tile 848.0 s（1.92、20.58、827.4 s、379、62、4.041 GB）；整片投影：三個規模最小平方 `total ≈ 9.4 s + 1.903 s/tile × tiles` → 35,700 tile ⇒ ~18.9 h（外推、上限，因為三個 crop 組織密度高 ~14–15% 背景）；**plan 的「1287 tile/39×33」假設 4096 px tile，實際 `default_tile_size=1024` 在 156222×134028 給 204×175 = 35,700 tile**。
- **4 % 排名表**：4.1 階段彙總（small/medium/large %）B1 46.0/48.7/45.5（Amdahl 1.84）、B3 29.3/29.3/33.4（1.5）、B2 9.1/10.8/10.7（1.12）、B4 4.0/4.5/4.6（1.05）、A 2.7/2.3/2.4、init 5.8/1.3/0.4、B-M1 1.3/1.2/1.1、B2r 0.7/0.6/0.8、D 0.7/0.6/0.6、B-stitch 0.4/0.5/0.5、C 0.0；residual 0.1。4.2 large_441tile 子項排名：1 `detect_all_dots` 260.10 s（30.67%，1.44）；2 M2 Cellpose 前向 187.02（22.05%，1.28）；3 M3b Cellpose 前向 185.53（21.88%，1.28）；4 PNG 編碼+寫 76.76（9.05%，1.10）；5 `gc.collect` 36.31（4.28%，1.04）；6 precut A 20.58（2.43%）；7 `enlarge_cell_instances` 16.76（1.98%）；8 M1 UNet++ 前向 13.45（1.59%）；9 `render_overlay_image` 8.49（1.00%）；10 `build_all_positive_results` 6.55（0.77%）；11 tile read 6.54（0.77%）；12 `_stitch_overlay_slide` 5.11（0.60%）；13 per-cell crop 4.08（0.48%）；14 `clear_slide_edge_cells` 3.84（0.45%）；15 `apply_mask_to_ihc_image` 3.68；16 `overlay_ihc_mask_on_dish` 3.64；17 `torch.cuda.empty_cache` 2.69（0.32%）；18 `fuse_masked_ihc_with_dish` 1.91；19–22 model init（UNet++ 1.15、Cellpose M2 1.10、M3b 0.84）與 int32 TIFF 1.07；23–26 其餘 ≤0.10 s。
- **5 橫切——GPU 利用率時間軸與 RAM/VRAM**（頭條發現：**各規模 GPU idle ~48–59% wall，mean SM 僅 ~24–29%**，即使它是唯一的 GPU 工作）：small GPU mean SM 29.4%、median 5%、p90 91%、idle_frac 0.491、busy≥50% 0.309、VRAM 5159 MB、RSS 0.649 → 2.815 → 2.778 GB；medium 28.4%/5%/91%/0.477/0.263/5159 MB/0.649 → 3.074 → 3.074 GB；large 28.3%/5%/92%/0.459/0.252/5159 MB/0.649 → 4.041 → 4.041 GB。記憶體增長宣稱驗證：VRAM peak 在各規模持平 ~5.16/32 GB、RSS 次線性（25 → 441 tile = 17.6x tile、RSS ~2.8 → ~3.3 GB ≈ +18%）。**5.1 相鄰階段供給 vs 消耗**：A → B 嚴格序列（`precut_paired_tiles()` 跑完才開始 `run_batch()`，零重疊；precut 8 執行緒 pool，各規模 ~2.3% wall）；B 內「看起來平行、實為序列」的陷阱——三個模型共用一個 CUDA context、一次一個 tile，per tile GPU 前向與 CPU dot 偵測依序執行，GPU 與 20 個 CPU 核從不同時飽和（= ~48% GPU idle）；B → D 序列（`_stitch_overlay_slide()` 在所有 tile 分析完之後跑，單一非平行 pyvips join+lzw+tiffsave，各規模 ≤0.75% wall）。
- **6 cProfile 函式級 Top（25-tile）**：`run_batch` cumulative 54.942 s；`process_precut_tile` 48.101 s；`_process_one_chunk` 42.620 s；`_run_m3_analysis_stage` 27.551 s；`segment_windowed` 23.323 s（48 次）；cellpose `eval` 23.140；`_run_net` 14.649；`detect_all_dots` 14.891（24 次）；`vit_sam.py` forward 11.906；`image_encoder.py` forward 11.667/11.460（2,304 次）；`joblib` `_retrieve` 11.547；`time.sleep` 11.528（主執行緒等 subprocess worker）。**確認舊 03 文件排名仍成立**（SAM backbone 的 `get_rel_pos`/`add_decomposed_rel_pos` 主導前向）。
- **7 瓶頸清單（①–⑦，格式：現象/佔比/Amdahl/規模/分類/信心）**：與 `bottleneck-list-history.md` 的 control 時期 ①–⑦ 相同（① GPU 序列 idle ~46%，B1 46%、1.84，Class 3+6；② `detect_all_dots` 260.1 s、31%、1.44，Class 1+3；③ PNG 76.8 s、9.1%、1.1，Class 5；④ `gc.collect` 36.3 s、4.3%、1.04，Class 4+6；⑤ precut A 2.3%/stitch D 0.6–0.75%，Class 5（D 亦 Class 3）；⑥ model init 7.3%（25 tile）→ 1.3%（121）→ <1%（441）；⑦ API 層 2.3 µs enqueue、~3.8 ms dispatch）。
- **8 文件/code/config 落差（不混進效能結論）**：**G-A** 整片 tile 數被估低 ~28x（plan/03-doc 的 1287 tile 是 4096 px tile 假設；實際 35,700）；**G-B** `cellpose_batch_size` 仍死（`config_example.py` 無該欄位；兩個 segmenter `getattr(config,"cellpose_batch_size",16)`；VRAM 5/32 GB 顯示 16 對 5090 填不滿）；**G-C** `requirements.txt` 缺 `segmentation_models_pytorch`、`fastapi`、`uvicorn`；**G-D** torch cu130 wheel 安裝路徑（R2 CDN TLS 失敗）；**G-E** `config_example` G2 gotcha 已解（`compute_config_hash()`/`config=Config()` 尾段存在）。原始 artifacts：`_metrics/`（JSON timings、cProfile `.prof`、`nvidia-smi dmon` trace、resource CSV、pip freeze、env stamp）。**No pipeline code was modified.**
- ⚠️ 這份 HTML 是**舊架構（pre-precut → 其實是 precut 後 round 1 時點）的控制組報告**，已被 `bottleneck-list.md` 取代為現況來源。

### 20.5 `pipeline-flow.html`（18 KB）——流程與迴圈視覺化（⚠️ 畫的是 pre-precut 架構）

- 標題「Hybrid Pipeline 流程與迴圈全景 — 交接視覺化」（`cell_mask/hybrid` 時期，2026-07-02 整理；事實來源 perf_report.html / CLAUDE.md / 現行 code）。
- **先看數字（perf_report.html · 2026-06-29 · 3 tiles 4096²）**：39.3 s 平均/tile、29% GPU 平均利用率（峰值 99%）、4.7/32 GB VRAM（僅 15%）、~57% 時間在 Cellpose ViT-SAM 前向、8h 43m WSI 全圖估算（70% 組織）；結論：GPU 平均只有 29% 但峰值 99% + VRAM 只用 15% → **不是算力不足，是單 tile 序列、GPU 被餓著；多 tile 平行是最大結構性機會**。
- **1 主流程** M0 讀取 → M1 疊合 → M2 分割 → M3 細胞/點位 → M0 縫合 → M4 匯出；M0 出現兩次（讀在最前、縫在 M3 之後，是包在 M1–M3 外的「分塊殼」）；各模塊 IN/OUT：M0 讀取 IN IHC/DISH 路徑 OUT `Chunk(ihc,dish,abs_x,abs_y)`（1024,1024,3）uint8（`iter_paired_chunks()`）；M1 IN ihc/dish OUT overlay + core_mask（`generate_ihc_core_mask()`）；M2 IN overlay OUT instance_mask int32（`segment_masked_dish()`）；M3 IN instance_mask + dish overlay + core_mask OUT results/all_dots/per_cell_dots/nucleus_mask（`detect_all_dots()`）；M0 縫合 IN 逐塊 ChunkResult OUT StitchedTile（`StitchAccumulator.add/finalize`）；M4 匯出 IN StitchedTile OUT CSV + overlay PNG + 逐細胞 crop（`export_cell_dot_annotations()`）。
- **2 巢狀迴圈**：`run_batch()` for tile in paired_tiles（模型只載一次）→ `process_single_tile()`（read_size → chunk_offsets → 建 StitchAccumulator）→ `for chunk in iter_paired_chunks()`（M1 → M2 → M3 → `cr = ChunkResult` → `acc.add(cr)`，`cr` 隨即出作用域、numpy 立即可 GC）→ `stitched = acc.finalize()` → M4 匯出；每 tile 結束 `torch.cuda.empty_cache()` + `gc.collect()`。**三層記憶體防線**：模型（batch 級）→ 整圖畫布（tile 級）→ chunk 中間產物（chunk 級）。
- **3 StitchAccumulator 核心區與去重**：相鄰 chunk 沿 overlap/2 切出互不重疊的核心區，一顆細胞只算在「其質心落在哪塊核心區」；去重從 O(n²) 降到 O(n)（`bisect_right(cuts, 質心)`）。**4 記憶體生命週期（鋸齒循環）**：每個 chunk 讀入 + 中間產物 → 峰值；`acc.add` 後 `cr` 出作用域 → GC 谷底；整圖畫布是常駐基線；WSI 真正天花板在「常駐畫布」，156k×134k slide 時 `StitchAccumulator` 一次 allocate 6 張 full-size numpy（輸出端）——解法是 ROI-only/停止整片縫合（04 的 L2；**已被 precut-to-folder 架構取代**）。

### 20.6 `measurement/` 原始資料資料夾（`_metrics_r6`、`_r7`、`_r8`、`_r9`，無 `_r10`–`_r15` 於此資料夾內列出者另見 `_metrics_r12`/`_r14`/`_r15` 的引用）

- **`_metrics_r6/`**（round 6，2026-07-25；git `a4a6254`；env：RTX 5090/580.173.02/32607 MiB、20 核、torch 2.11.0+cu130、cellpose 4.2.1.1、numpy 1.26.4、skimage 0.24.0、joblib 1.5.3；checkpoint hash：`best_model_unet_b4.pth` 76943268569ddc54、`cellpose_ihc_dish_best` febfa4d2e123c6b1、`cellpose_dish_best` 6855a1d02a7b10de）：`a1_w{3,4,5,6,8}_nj1_r{1,2}_{gpu_dmon.txt,resource.csv,timings.json}`（A1 worker 掃描）、`w1_nj1_r{1,2}_*`、`w1_njm1_r{1,2}_*`（`workers=1`，`n_jobs=1` vs `-1`）、`w6_nj1_r{1,2}_*`、`w6_njm1_r{1,2}_*`（`workers=6`）、`a2_contention_njobs_default.json` 與 `a2_contention_njobs_sweep.json`（A2 CPU 競爭稽核）、`b1_cellpose_batch_probe.json`（B1 Cellpose 跨 tile 批次）、`b1b_unet_batch_probe.json`（B1b UNet++）、`b4_gil.raw`/`b4_wall.raw`（py-spy raw profile，786 KB/1.39 MB）、`b4_gil_resource.csv`/`b4_gil_timings.json`/`b4_wall_resource.csv`/`b4_wall_timings.json`/`b4_trace_summary.json`（B4 對 4.2.1.1 的追蹤）、`env_stamp_r6.txt`、`pip_freeze_r6.txt`；注意兩個超大 dmon 檔（`a1_w6_nj1_r2_gpu_dmon.txt` 18.3 MB、`w6_nj1_r1_gpu_dmon.txt` 18.6 MB）——對應當 `workers=6` OOM 的 run。
- **`_metrics_r7/`**（round 7，2026-07-26；git `025f9a5`；config hash `3d1087f2`）：`control_w{1,4}_r{1,2}_*`/`candidate_w{1,4}_r{1,2}_*`（Candidate G 的 A/B：每個含 `gpu_dmon.txt`、`resource.csv`、`stdout.log`、`timings.json`）、`keeprun_w4_*`、`match_w{1,4}_r1_*`（composition-matched anchor）、`runs/{control,candidate,match}_*/{output_bytes.txt,report.csv,summary.txt}`（各 run 的 `report.csv` ~190 KB/287 KB；`output_bytes.txt` 51 B、`summary.txt` 678 B）、`b1_cellpose_batch_probe_g16.json`（G=16）、`comp24_crop.json`/`dense8_crop.json`/`match24_crop.json`（crop 描述）、`core_mask_map.json`/`core_mask_map.npz`（全格 core-mask 圖，45.8 KB）、`tissue_calibration.json`/`tissue_calibration_n800.json`（`tissue_calibrate.py`：32.8 KB/104 KB）、`fs_write_probe.json`（mkdir/blank-write/hardlink）、`gpu_codec_spike.json`/`gpu_codec_spike_projectvenv.json`（環境 gate）、`stitch_probe_35840x28928.json`、`stitch_probe_71168x57344.json`、`stitch_probe_141818x114366.json`、`stitch_probe_1gp.json`（Phase D 規模）、`wsi_projection_{control,match}_w1_r1.json`（整片投影）、`env_stamp_r7.txt`、`pip_freeze_r7.txt`。
- **`_metrics_r8/`**（round 8，2026-07-27）：`fullwsi_result.json`（`full_wsi_validate.py` 的 preflight + 兩次 run 結果：slide 141658×114366、16.201 GP、27,565 tile、grid 185×149、projected 399.7 GB/run（`instance_mask` 115.6、`dish_nucleus_mask` 115.6、`overlay_annotated` 55.5、`_precut_scratch` 65.3、`masked_ihc` 17.1、`dish_mask_overlay` 10.7）、free 9,693.2 GB、`RLIMIT_NOFILE` 1,048,576、`fullwsi_w1` `end_to_end_total_s` 13,762.468、`stats` 10,801/16,764、analysis_output 346,970,388,138 B；`timings` 各 bucket n/t）、`fullwsi_w{1,4}_{gpu_dmon.txt.gz,resource.csv,timings.json}`（`resource.csv` 717 KB/279 KB）、`alloc_conf_w6.json`（12-run allocator sweep）、`stitch_ablate_4gp.json`（13-config `tiffsave` 消融）、`worker_timings_probe.json`、`pip_freeze.txt`。
- **`_metrics_r9/`**（round 9）：`gc_refreeze_probe_full.json`（Exp 0/1/2 的 gc re-freeze 合成 harness；16 KB）、`L_{prefetch,noprefetch}_r{1,2,3}_{resource.csv,timings.json}`（Option L 的 3 次交錯 A/B）、`match24_r9_{control,prefetch,refreeze}_{resource.csv,timings.json}`（Exp 3 與 Option L 步驟 2 量測）、`phase_d_container_spike.json`（Phase D 容器 spike，2.3 KB）。
- **（`_metrics_r12`、`_metrics_r14`、`_metrics_r15` 在文件中被引用——doc 39 §8、doc 43、doc 46 §6——但不在目前這個 `measurement/` 資料夾列表內；它們的資料以文件本身的表格為準。）**

---

## 21. 跨文件綜合：整條優化歷程的主線、關鍵數字演進、方法論教訓與目前還開著的事

### 21.1 主線敘事（15 輪、一條故事）

1. **起點（round 1，2026-07-07）**：舊文件（01–04）的巢狀 chunk 迴圈、整圖畫布、`ProcessPoolExecutor` 平行全是 pre-precut 時期的敘述；HEAD 已是 precut-to-folder 架構。重新規劃量測（doc 09），得到控制組：large 441 tile **848.0 s**，GPU idle ~46–49%、mean SM ~28%（GPU starvation，非算力不足）。
2. **Round 2**：單 process 兩段式 overlap（GPU 前段在主執行緒、CPU 後段在背景執行緒）：wall −14.5%～−18.5%、idle 0.49 → 0.15–0.19；之後 GIL 診斷（81% 是主執行緒自己）推翻「`detect_all_dots` 搶 GIL」的假設，②③ 被 GPU 前段「免費」遮住。
3. **Round 3**：同事把 Cellpose 換成 4.2.1.1（DINOv3 `cpdino` + bfloat16）並重訓兩個 checkpoint：large 707.4 → **573.7 s**（−18.9%）、VRAM 5159 → 2787 MB；但 GPU idle 反升到 0.370——更 bursty。**排序方法因此改成「兩條臂的關鍵臂貢獻」**（不是 self-time %）；細胞數 +1.8–2.5%，臨床簽核至今未完成。
4. **Round 4**：`gc.freeze()`（83.2 → 1.2 ms/call）、⑧ CPU prep 搬離 MAIN（−8.0%）、precut 串流化（−3.1%）；`cellpose_batch_size` 接線但掃描平坦（1024² = 恰好 16 patch）。large **480.3 s**。
5. **Round 5**：跨 tile 多行程（Candidate D，`spawn` + 動態佇列）：`workers=3` **3.09x**（遠高於預估 1.23–1.7x，因為真正回收了兩臂間 GIL 競爭）；MPS 與 CPU 後端 pool 停損。
6. **Round 6**：`detect_all_dots` 的 joblib fan-out 是淨損失（20 threads 比序列慢 2.77x）→ `dot_detect_n_jobs=1`：`workers=1` **1.60x**（484.7 → 302.7 s），MAIN 臂本身快 43.4%（隔離了 +192.9 s 的 GIL 異常）；建議 worker 數下修到 4。
7. **Round 7**：玻片背景比例實測 **55.8%**（推翻 39%）；BG 臂 slack 47–53% → B/D/E/F 全部 wall 天花板 1.00x；Phase D 實測 322.7 s（比外推高 1.8x、超線性）；GPU 函式庫環境 gate（CuPy 在本機跑不起來；nvImageCodec 可用但不產 tile）。
8. **Round 8（2026-07-27）**：**第一次完整整片跑批**：`workers=1` 3.82 h、`workers=4` 1.73 h（2.216x）；三個 crop 導出的 stop-loss（gc、tile read、Phase D）在整片規模不成立；`tiffsave` 旋鈕全死（zstd 被 QuPath/BioFormats 否決）；resume、`RLIMIT_NOFILE`、worker 計時、allocator knob 上線。
9. **Round 9–11（2026-07-27→07-29）**：週期性 gc re-freeze（整片 2,218.4 → 19.5–58.8 s）、tile-read prefetch（Option L；真實成本最終結清為 4.21% wall、99.8% 被隱藏）、Phase D GPU 化先 spike 再反轉；順帶發現並修好**每個 round 9 前的 `overlay_slide.tiff` 金字塔層解成雜訊**；fail-fast 後 hang（`task_q.cancel_join_thread()`）、`PrecutStream` 中止後仍切整片、`expandable_segments:True` 在 `workers=4` 4/12 → 0/12 OOM（預設翻開）。
10. **Round 12–13（2026-07-29）**：重量 `workers=4`——只有 **1.745x**（非 2.216x），全因 Phase D 成長到 32.3% wall；batch-claiming 關閉（0.006%）；**Phase D 管線化讀取上線**（`stitch_backend="tifffile"`）：Phase D 1.913x、e2e 1.200x、peak RSS 45.6 → 17.0 GB。
11. **Round 14（2026-07-30）**：`workers>1` 的 GPU↔CPU 搬運三候選全部負向（磁碟 1.4%、`_from_device` 96.3% 是 GPU-wait、pinned memory 0.42%）。
12. **Doc 44（2026-07-30）**：發現 round 8–13 的整片全在**未配準畫布**（`conform_to_intersection()` 讓尺寸相等但座標未對齊，繞過 `PrecutStream` 守衛）；正確輸入是 `module4_thumbnail.py` 產生的 `*_warped_lv0.tiff`。
13. **Round 15（2026-07-31～08-01）**：正確畫布 **35,700 tile、20.938 GP、65.92% 背景**（組織 tile 幾乎沒變，多出的全是背景）；組織/背景每 tile 係數 **0.7298/0.0460 s（15.9x）**；首次誤差棒（CV <1%）；Phase D page-cache 懸崖（5.41x）；整片 `workers=1` **3.023 h**、`workers=4` **1.478 h**（2.05x）、peak RSS 13.66 GB；交付 `scripts/eta_estimate.py`（尚未接進 API/UI）。

### 21.2 關鍵數字演進速查

| 指標 | 起點 | 現況（最新） | 備註 |
| --- | --: | --: | --- |
| large/441 crop `workers=1` | 848.0 s（控制組） | **294.6–302.7 s**（−64.3%～−65%） | round 15 在同批 crop 檔上重測 294.6 s（2.88x），確認 302.7 s |
| large/441 crop `workers=4` | — | 128.8 s（round 6）→ **143.2 s**（round 15 重測，對 round 1 5.92x） | |
| match24（576 tile，真實組成）`workers=1`/`4` | — | 188.8 s / 88.3 s（2.14x） | |
| 整片（舊畫布 27,565 tile）`workers=1` | 3.82 h（round 8） | 2.84 h（round 10）→ 2.96 h（round 11） | **舊畫布，僅計時資料** |
| 整片（舊畫布）`workers=4` | 1.73 h（round 8） | **4,877.4 s = 1.355 h**（round 13） | **舊畫布** |
| **整片（正確畫布 35,700 tile）`workers=1`/`4`** | round 1 投影 18.87 h | **3.023 h / 1.478 h**（round 15） | **官方現況**；對 round 1 投影 6.24x / 12.77x |
| 整片 peak RSS | 61.13 GB（round 8） | **13.66 GB**（round 15） | −77.6%；README「64 GB RAM」過時 |
| `workers=4` peak GPU | 30,439 MB（93.3%） | 92.2% of card（`expandable_segments` 預設）| VRAM 是 `workers≥5` 的硬限制 |
| 單 process GPU idle_frac（large） | 0.459 | 0.06–0.37（隨配置） | SM==0 指標是 knife-edge，改看 cuda-Event gap |
| `gc.collect`（整片） | 2,218.4 s（16.1%） | **19.5 s** | 週期 re-freeze |
| `B2r_tile_read`（整片） | 2,368.5 s（17.2%，被污染） | **449.3 s（4.21%）**；被 prefetch 隱藏 99.8% | |
| Phase D（整片） | 322.7 s（probe）/1,185.4 s（round 8 實測） | 987.8 s（round 13，舊畫布）/ 1,282–1,332 s（round 15，正確畫布，24.1% of `workers=4` wall） | cached 11.26 vs 冷讀 60.90 s/GP |
| 整片 tile 數 / 背景比例 | 35,700（round 1 假設）→ 27,565 / 55.8%（錯誤畫布） | **35,700 / 65.92%** | 「35,700」其實一直是對的 |

### 21.3 這個資料夾反覆驗證的方法論教訓（可直接當 checklist）

1. **量測要先於最佳化**（Amdahl）；但 **self-time % 在多臂 overlap 下沒有意義**——要看「在自己那條臂上歸零後的 ceiling」；slack 臂上的項目 ceiling ≈ 1.00。
2. **crop 規模的數字常在整片規模不成立**（`gc.collect`、tile read、Phase D）；每個 crop 導出的 stop-loss 在整片出現前都只是暫時的。**每個數字都要標出它是在哪個規模、哪個畫布、哪個 `workers`、哪個 process 制度下量的**（doc 42 的規則）。
3. **bucket 總量上升而 wall 下降是「工作被搬到平行臂」的簽名**（Option L）；bucket 是主執行緒呼叫的牆鐘計時，會被併發競爭膨脹（0.513 s/s）。
4. **單次運氣的「勝過天花板」結果要重複確認**（gc 1.118x 被兩次重複推翻；round 12 的 −52% Phase D 成長在 CV 0.9% 下確認為真）。
5. **A/B 一律平衡順序（Latin square）並從暖機狀態開始**（round 11 因 A,B,A,B 的 slot 順序與熱漂移丟掉兩次 sweep）；**對照組（最笨基準）永遠保留**。
6. **spike 在整片規模可能整個反轉**：Phase D GPU 化（記憶體 slab 2.15x → 真實磁碟 0.884x → 管線化讀取 1.581x）。spike 必須包含完整 I/O 半邊，並以「讀取占比是否轉移」驗證其可信度。
7. **correctness 一票否決**（zstd、Predictor 錯誤、pyramid 層位移）；**programmatic audit 不能取代人眼渲染檢查**（QuPath 兩次抓到 audit 抓不到的缺陷；`overlay_pyramid_audit.py` 仍只測亮度）。
8. **會讓所有 benchmark 都看不見的缺陷**（fail-fast 後行程不退出）需要從外部量測（`exit_latency_probe.py`）。
9. **量測工具本身會 bitrot**（M0 split 後 `perf_measure.py`/`stitch_probe.py`/`core_mask_map.py` 多處失效）；`wrap()` 對缺失符號只印 skip 而不報錯 → 會產生「看似完成的空洞量測」。
10. **輸入正確性優先於一切效能數字**（doc 44：尺寸相等 ≠ 座標對齊）；「守衛通過」不等於「輸入正確」。

### 21.4 目前仍開著的事項（截至 round 15，彙整）

| # | 項目 | 狀態 / 備註 |
| --- | --- | --- |
| 1 | **Round-3 Cellpose checkpoint 重訓的臨床/病理醫師簽核** | 仍 pending（自 round 3 起），所有 round 3 以後的效能勝利都建立在此之上；非工程任務，需向臨床端提出 |
| 2 | **Phase D 冷讀 regime 的量測**（9,216–35,700 tile 之間） | ETA 模型的 Phase D 子模型失準 −56%；最高優先 |
| 3 | **整片數字的誤差棒** | 中型 anchor 已有（CV <1%）；整片仍 n=1（`workers=1` 與 `workers=4` 各一次，Set F 時數不足以各跑兩次） |
| 4 | **ETA 接進 API/UI** | `scripts/eta_estimate.py` 已交付但未接；需先過 frontend-backend-boundary 審查；逐 tile 進度 callback 受 `BackgroundTasks` 框架限制 |
| 5 | **跨玻片驗證** | 只有一張真實玻片；同一張玻片換畫布定義背景比例就差 10 pp；跨病人/染色批次變異未知；模型對此敏感（組織 tile 貴 15.9x） |
| 6 | **`workers=4` 的 VRAM OOM 風險** | round 14 在 crop 規模 1/10 次 OOM（即使 `expandable_segments:True`）；round 15 Set F 單次成功但 n=1 不足以說風險消失；VRAM 餘裕 ~2.5 GB 是四 worker 合計 |
| 7 | **`overlay_pyramid_audit.py` 強化** | 目前只測各層亮度；應加金字塔自洽（各層為上一層 2×2 box shrink）與 source-tile-size 檢查 |
| 8 | **QuPath 渲染回報** | 某些 slide 有些 tile 渲染錯誤的回報在 `r12_fullwsi_w4/overlay_slide.tiff` 無法重現；level 0 與 predictor-broken 檔逐位元相同，故非 round 9 pyramid 修復所致；停放待重現 slide |
| 9 | **真實解析度校準** | 兩個 stitch backend 都寫佔位 1 px/mm（QuPath 中 µm 測量無意義） |
| 10 | **`perf_measure.py:769` 死路徑** | 非 `--stream-precut` 分支呼叫已不存在的 `HP.precut_paired_tiles`；一行 import 修或刪 flag |
| 11 | **README 硬體需求過時** | 「64 GB RAM／~60 GB peak RSS」已過時三輪；實測 13.66 GB |
| 12 | **`bottleneck-list.md`/`current-status-comparison.md` 逐項列仍停在 round 11** | 需以 round 12–15 補齊（本檔 §19 已摘要） |
| 13 | **Candidate F 的 ~157 GB/slide 儲存決策** | 速度面零收益；`os.link` 去重是儲存論點，從未決定 |
| 14 | **Cellpose `_from_device`、整塊 `detect_all_dots` 向量化、`torch.compile`/DALI/TensorRT 等** | 全部停損或背口袋點子，需新證據才重提 |
| 15 | **API 並發（Phase E）與常駐 worker pool** | 從未量測；取決於「並發玻片服務是否在部署計畫內」 |
| 16 | **`run_batch` 預設已為 `workers=4`（CLI `--workers` 預設也為 4）；`backend/api/hybrid.py:89` 明確傳 `workers=1`** | 已決定（doc 40 §1.6）；API 單 tile 請求的 `workers=1` 覆蓋比以往更承重 |

### 21.5 本彙整涵蓋的檔案清單（共 80+ 個檔／資料夾）

README.md；01–08（基礎）；09–46（輪次文件：規劃/落地/結果配對 + 單項報告）；`19-open-backlog.md`、`26-remaining-work-implementation-plan.md`、`DISCOVERED-NOT-IMPLEMENTED.md`（清單類）；`PERFORMANCE_BOTTLENECK_PLAYBOOK.md`、`PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md`（方法論）；`pipeline-flow.html`；`measurement/`：`bottleneck-list.md`、`bottleneck-list-history.md`、`bottleneck-list-zh.md`、`current-status-comparison.md`、`current-status-comparison-history.md`、`current-status-comparison-zh.md`、`detect-all-dots-result.md`、`gil-contention-diag.md`、`pipeline-overlap-result.md`、`perf_report.html`，以及 `_metrics_r6`、`_metrics_r7`、`_metrics_r8`、`_metrics_r9` 四個原始資料夾（JSON/CSV/log/dmon/py-spy raw、`.npz`、`runs/*/report.csv`）。

> 彙整註記：兩個超大檔（`_metrics_r6/a1_w6_nj1_r2_gpu_dmon.txt` 18.3 MB、`w6_nj1_r1_gpu_dmon.txt` 18.6 MB；`b4_*.raw` 0.8–1.4 MB）與各 `report.csv`（~190–290 KB）屬原始資料，未逐行讀取；其意義由對應文件（doc 23、25、27、31–37、39、43、46）的表格所記錄。`-zh` 版本是英文歷史版的繁中翻譯，未發現額外資料。`README.md` 本身（43 KB）已是各輪 TL;DR 與導覽表，本彙整 §0–§1 吸收其內容。
