# 45 — UI 預估時間估算基準：量測與分析計畫

> 本文件是**任務與需求規格**，比照 [09-measurement-analysis-plan.md](./09-measurement-analysis-plan.md)
> 的規格：只定義「要量什麼、怎麼量、量完怎麼建模」，**不寫任何程式碼、不改動 pipeline / API /
> UI**。模型完成、驗證通過之後，要不要接進 UI、怎麼接，是下一份「落地」文件的工作（且會牽涉
> [frontend-backend-boundary](../../.claude/rules/frontend-backend-boundary.md) 的邊界審查）。
>
> 撰寫時已核對現況（見第 0 節）：README.md 的 TL;DR 只更新到 round 12，本計畫額外核對了
> round 13（[40](./40-round-13-phase-d-pipelined-stitch-plan.md)/[41](./41-round-13-phase-d-pipelined-stitch-implementation.md)）、
> round 14（[42](./42-round-14-gpu-cpu-transfer-plan.md)/[43](./43-round-14-gpu-cpu-transfer-implementation.md)）、
> 以及尚未併入 round 敘事的 [44-conform-intersection-shift-investigation.md](./44-conform-intersection-shift-investigation.md)。

---

## 0. 為什麼要做這件事

前後端交接文檔 [`../UI/13-numpy2-migration-and-analysis-ui.md`](../UI/13-numpy2-migration-and-analysis-ui.md)
§7 與 [`../BACKLOG.md`](../BACKLOG.md)（"Analysis (hybrid) UI progress + cancellation"）已經記錄一個
開著、卡在演算法層的缺口：`run_batch` 執行中完全不回報進度，UI 目前只能顯示
`pending/running/done/error` 四種狀態。`frontend/src/components/HybridPanel.tsx` 對此有明確的
設計原則（第 10–13 行原文）：

> "The backend reports one job status for the whole chain — there is no per-stage or per-tile
> signal — so this list documents what is happening, and only the overall status is live.
> **Faking a stage cursor here would be inventing progress.**"

本計畫的目的**不是**繞過這個原則，而是把它延伸到「預估時間」這個新需求上：使用者想在送出分析
（尤其是整張玻片、耗時以小時計的那種）之前或期間，看到一個**有實測依據**的預估秒數／區間，而
不是憑感覺編一個數字。要做到這件事，需要先有一份可信、涵蓋不同規模與不同組織/背景比例的
**時間基準資料**——這正是本計畫要產出的東西。

**與現有進度回報缺口的關係**：本計畫產出的模型設計成兩段式可用：

1. **送出前的靜態估計**（不需要 callback）：只靠 ROI/整片的像素尺寸（`frontend` 既有的
   `estimateTiles()`，見第 1.4 節）+ 本計畫的模型係數，算出一個「預估 X 分鐘～Y 分鐘」。
2. **執行中的動態校正**（需要 callback，不在本計畫範圍）：一旦逐 tile 進度回報做出來，可以用
   「已完成 tile 的實際平均速率」取代模型速率，讓估計隨執行過程收斂變準——但這是另一份文件（且
   要先解決 `BackgroundTasks` 不能回報進度的框架限制）的工作。

**本計畫也是給醫師/教授看的報告的資料基礎**：因此第 6 節明列了報告需要的誠實揭露項目（樣本數、
外推範圍、信賴區間），不是只求一個能用的公式。

---

## 1. 現況盤點（已有什麼，不必重做）

### 1.1 既有的手臂模型（arm model）

`bottleneck-list.md`、doc 38/39 已經建立並驗證過（誤差 &lt;1.3%，`bottleneck-list-history.md` round-3
節）：

```
wall ≈ max(MAIN, BG) + outside
outside = Phase_A(precut) + Phase_D(stitch) + model_init
```

- **MAIN 臂**（主執行緒，序列，關鍵路徑）：3 次 GPU 前向（UNet++ core mask + M2/M3b 兩次
  Cellpose）、`_read_rgb`（tile 讀取）、M1 疊合、`clear_slide_edge_cells`、
  `build_all_positive_results`、`enlarge_cell_instances`、`gc.collect`/`empty_cache`。
- **BG 臂**（背景執行緒，與 MAIN 重疊）：`detect_all_dots`、dot 合併、PNG/TIFF 編碼、
  `render_overlay_image`、per-cell crop 匯出、`filter_and_absolutize`。
- **outside**：Phase A（precut 預切，跑在分析迴圈之前）、Phase D（overlay 縫合，跑在之後）、
  三個模型的一次性 init——這三項**不隨 `workers` 縮放**（Phase A/D 目前是序列，跑在 parent
  process，`workers>1` 對它們沒有加速）。

### 1.2 既有工具（全部可重用，不必重寫）

| 工具 | 用途 | 對本計畫的角色 |
| --- | --- | --- |
| `scripts/perf_measure.py` | 非侵入式量測 harness（monkeypatch 計時），輸出每個 bucket 的 wall/呼叫數 | 每組量測的主要執行工具 |
| `scripts/composition_crop.py` | 從真實玻片裁出 **G×G tile、指定背景比例（`--target-bg`）** 的裁切，背景比例判定用 brightness proxy（需與 `run_batch` 實際 skipped/total 對照校正） | Set A（固定比例、變尺寸）與 Set B（固定尺寸、變比例）的資料產生器——`--grid` 控制尺寸、`--target-bg` 控制比例，兩者正交 |
| `scripts/core_mask_map.py` | 用 pipeline 自己的判準（UNet++ core mask 是否為空）逐格量整張玻片的組織/背景分佈 | 產生 `composition_crop.py` 需要的 map；也用來在 doc 44 校正後的畫布上重算 grid |
| `scripts/wsi_projection.py` | 把一次量測結果依 bucket 所屬母體（all/tissue/background）拆解速率，投影到任意 `--slide-tiles`/`--background-share` | 現有的「速率拆解＋外推」雛型，本計畫第 4 節要把它擴充成 workers>1、非線性 Phase D、多點迴歸版本 |
| `scripts/full_wsi_validate.py` | 整片跑批 runner（`--workers`、`--worker-timings`、`--resume`）；已依 doc 44 移除 `conform_to_intersection()`，改吃 `module4_thumbnail.py` 的 `warp_slide(level=0, non_rigid=True, crop="overlap")` 輸出 | Set F 的執行工具 |
| `scripts/stitch_probe.py` | 獨立量 Phase D（含 `--pipelined` 版本） | Set D 量 Phase D scaling 用，不必跑整條 pipeline |
| `scripts/generate_report.py` | 產生 `measurement/perf_report.html` | 第 6 節報告產出可重用，但**先跑一次確認 schema 沒有跟不上目前 `perf_measure.py` 的輸出**（未獨立驗證過） |

### 1.3 doc 44：畫布尺寸錯誤，round 8–13 的絕對數字需要重新校準

`44-conform-intersection-shift-investigation.md` 發現 round 8–13 全部整片跑批餵的是
`HER2_processed.tiff`/`DISH_processed.tiff`——這是**配準前**、各自 modality 自己座標系的
mosaic 輸出（141818×114366 vs 141658×114415），`full_wsi_validate.py` 舊版的
`conform_to_intersection()` 把兩者裁到交集，讓 `PrecutStream` 的「尺寸須相等」檢查通過，但
**尺寸相等 ≠ 座標對齊**。真正配準好的畫布（`module4_thumbnail.py` 的 `warp_slide` 輸出）是
**156222×134028**——比舊畫布寬 +10.2%、高 +17.2%。

doc 44 的結論（原文轉述）：round 8–13 的 wall-clock 數字**作為「時間量測」本身仍然有效**（同一份
code、同一批 arm 之間的比例關係、`workers=1→4` 的相對加速比都不受影響），**但作為「這張玻片對應
幾個 tile」的基準已經不對**——正確配準的畫布 tile 數還沒有人重算過。粗估（面積比例，未經
`core_mask_map.py` 實測，**只是本文件的推算，不是量測值**）落在 **35,500–36,000 tile** 之間，
巧合地接近 round 6 之前被推翻的「35,700 tile」舊假設——這個巧合本身值得在 Set F 跑之前先用
`core_mask_map.py` 實測一次來排除，不要假設它是對的。

**對本計畫的直接影響**：round 8–13 的 full-WSI 數字（含最新的 round 13 `workers=4` 4,877.4 s）
**可以當作「這套 code 在這個尺度量級的行為」參考，但不能當作本計畫模型的最終校準錨點**——第 4
節的 Set F 明確排在 doc 44 的修正已經跑過一次乾淨全片驗證之後，且**優先序最高的前置動作**是先
用 `core_mask_map.py` 在校正後的畫布上重算一次真正的 tile grid，而不是直接假設 35,500–36,000。

### 1.4 既有的「一半」估算：`estimateTiles()`

`frontend/src/components/HybridPanel.tsx:22-34` 已經有一個送出前的 tile 數估算，鏡射
`backend/algorithms/hybrid/config_example.py` 的 `default_tile_size=1024`/`window_overlap_px=256`：

```ts
const TILE_PX = 1024
const OVERLAP_PX = 256
function estimateTiles(w: number, h: number) {
  const along = (extent: number) =>
    extent <= TILE_PX ? 1 : Math.ceil((extent - TILE_PX) / (TILE_PX - OVERLAP_PX)) + 1
  return along(w) * along(h)
}
```

這是估算模型的「尺寸」輸入端，**已經做好、不必重做**；缺的是（a）背景/組織比例的輸入（目前
`estimateTiles` 把每個 tile 都當一樣貴），和（b）乘上一個有實測依據的「每 tile 秒數」係數。本
計畫的產出就是要餵這兩塊給它。

### 1.5 tile 尺寸恆定：跨玻片尺寸的模型可遷移性成立的前提

`config_example.py:207` (`default_tile_size: int = 1024`) 與 `window_overlap_px: int = 256`
（同檔 216 行附近）是全域常數，不隨玻片實際尺寸變動——玻片越大，只是 tile **數量**（grid 的
列數×欄數）變多，每個 tile 本身的成本剖面不變。這意味著：**只要「每 tile 秒數」係數是用
tissue/background 分群量出來的，它在跨玻片尺寸時原則上可以直接套用**，不需要對每個新尺寸重新
量測整條曲線——需要重新驗證的只有「這個新尺寸/新玻片的背景比例是多少」，而這件事目前只有一張
真實玻片的資料（55.8%），跨玻片的變異程度未知（見第 7 節限制）。

### 1.6 現有的完整量測 anchor 清單

| 規模 | tiles | 背景比例 | workers=1 | workers=4 | 來源 |
| --- | --: | --: | --: | --: | --- |
| large（tissue-dense，非真實比例） | 441 | 14.1% | 302.7 s | 128.8 s | round 6 |
| match24（比例對齊真實玻片） | 576 | 55.9%（真實量到 55.8%） | 188.8 s | 88.3 s | round 7 |
| comp24（比例刻意拉高） | 576 | 73.4% | 134.9 s | 65.5 s | round 7 |
| **完整玻片（舊畫布，見 §1.3）** | 27,565 | 55.8% | 10,666.1 s（round 11，乾淨對照） | **4,877.4 s（round 13，最新）** | round 8/11/13 |

**這張表裡沒有任何一列改變了「玻片本身的大小」**——每一列都是同一片真實玻片裁出來的不同大小
子區域，或整片本身。這正是使用者提出的疑慮（"我們不知道醫師想算多大的圖"）在資料上的具體體現：
**本專案至今從未量過第二張不同物理尺寸的真實玻片**。你已經確認以「同一片裁切＋純幾何合成大
畫布」的方式處理這個缺口（見第 4 節 Set D），而不是等待更多真實案例。

### 1.7 DISCOVERED-NOT-IMPLEMENTED.md 核對

已核對過 `DISCOVERED-NOT-IMPLEMENTED.md` 全文：**沒有任何一條提過「預估時間」/「ETA」/「進度條」
的想法**——本計畫是全新範疇，不是重啟某個被擱置的舊構想，也因此沒有既有的停損理由需要先繞過。

---

## 2. 目標與非目標

### 2.1 目標

1. 產出一組**依 arm、依 tissue/background 分群**的「每 tile 秒數」係數，並用迴歸而非單一 anchor
   外推的方式估計信賴區間。
2. 產出 Phase A（precut）與 Phase D（stitch）的 scaling law——這兩段是唯一「跟整張圖的像素/tile
   **總量**」而非「tile **內容**」相關的成本，且 Phase D 已知在真實冷讀取下是超線性
   （doc 39 §4.4：cached 15.4 s/GP vs 真實冷讀 116.6 s/GP，含讀檔佔 51.9%）——Phase A 從未被
   系統性量過 scaling 曲線（09 §3.1 只列了量測項目，從未有後續文件把它的 s/GP 曲線量出來）。
3. 產出 `workers=1..4` 的 multiprocessing scaling 曲線（目前只有 1 和 4 兩個點，doc 38/39 的
   Amdahl 帳是兩點外推，不是擬合）。
4. 用重複跑批（replicate runs）取得真正的變異數/信賴區間——目前所有整片數字都是「n=1 vs n=1」
   （doc 39/40/41 皆自陳此限制）。
5. 產出一套可重複執行的量測腳本組合（沿用第 1.2 節既有工具，只加薄的分析/擬合層）與一份可交給
   臨床/教授看的報告骨架。

### 2.2 非目標

- **不寫任何 UI/API 程式碼**。模型如何接進 `backend/schemas/common.py`（`JobStatus` 目前沒有
  progress/eta 欄位）、`backend/api/jobs.py`、`frontend/src/components/HybridPanel.tsx`，是下一份
  「落地」文件的事，且會需要先過一次 [frontend-backend-boundary](../../.claude/rules/frontend-backend-boundary.md)
  審查（往 `run_batch`/`_mp_tile_worker` 加逐 tile callback 屬於 `backend/algorithms/hybrid/`，
  對純前端 agent 是唯讀範圍）。
- **不設計逐 tile 進度 callback**——`../BACKLOG.md` 已列的、卡在框架層的另一個缺口，本計畫的模型
  設計成未來可以接上它（見第 0 節），但 callback 本身的設計不在此文件範圍。
- **不驗證醫學正確性**（沿用 09 §1.3 的既有原則）。
- **不用新增真實病人案例做尺寸母體抽樣**——依你的選擇，尺寸方向的資料完全來自同一片玻片的裁切
  （Set A）與純幾何合成（Set D），這是本計畫最大的一項已知限制，第 7 節會明確揭露、不會包裝成
  已解決。

---

## 3. Discover：待確認的現況落差

在排量測順序之前，以下幾點必須先核對，順序不可跳過（呼應 09 §1.4「先量總數、再拆分」的紀律，
這裡是「先確認基準乾淨、再花 GPU 時數」）：

1. **doc 44 的修正是否已在任何量測中被驗證過**：`full_wsi_validate.py` 目前的原始碼已移除
   `conform_to_intersection()`（研究已確認），但**尚未見到任何文件記錄過在校正後畫布上重跑過
   一次全片驗證**——README 最後更新到 round 12，doc 44 本身沒有標 round 編號，看起來是新發現、
   尚未有對應的 implementation 文件。**Set F 開始前必須先確認這件事的最新狀態**（有可能在本計畫
   撰寫與執行之間，另一輪已經把它跑掉了——先查 `docs/hybrid-pipeline/` 底下有沒有比 44 更新的
   文件，或直接跑一次 `core_mask_map.py` preflight 確認畫布尺寸）。
2. **Phase A 從未有 scaling 曲線**：09 §3.1 規劃過要量，但翻遍 10–44 全部文件，Phase A
   （`precut_paired_tiles`）只在個別 anchor 裡出現過總秒數，從未被拿來跟 tile 數/GP 做迴歸——
   這是本計畫要補的空白，不是延伸既有結論。
3. **MAIN 臂上的 GPU 前向從未被拆成顯式的「tissue tile 的 s/tile」常數**：目前唯一做過
   tissue/background 分群速率的是 `B2r_tile_read`（doc 43 §2.1：tissue 7.01 ms/read vs
   background 3.41–4.72 ms/read）。UNet++/Cellpose 兩次前向理論上 background tile 會被 core
   mask 短路成本趨近 0（round 7 §2.3/§2.4 的假設），但**這是假設，從未被單獨量出一個顯式係數並
   驗證過**——Set B 要直接解出這個係數，不是繼續假設。
4. **背景比例是否隨玻片而變，未知，本計畫無法回答**——只有一張真實玻片可用；這點列在第 7 節
   限制，不當作可解決的待辦。

---

## 4. Plan：量測資料集設計（Set A–F）

六組量測，設計原則：**每組只變動一個自變量**，避免像過去單一 anchor 外推那樣把「尺寸效應」和
「比例效應」和「workers 效應」混在一起算。所有組別沿用 `perf_measure.py` 的計時 bucket 與現有
config hash / 環境戳記規範（09 §4.6）。

### Set A — 尺寸 sweep（固定真實比例 ≈55.8%，變 tile 數）

**目的**：解出 Phase A/Phase D 的 s-vs-規模 曲線，並確認 MAIN/BG 每 tile 速率在不同尺寸下是否
穩定（若穩定，證實「每 tile 速率跨尺寸可遷移」這個第 1.5 節的假設；若不穩定，代表有目前未知的
固定開銷或快取效應需要另外建模）。

- **產生方式**：`scripts/composition_crop.py --target-bg 0.558`，`--grid` 從小到大：
  `11`(121 tile，既有)、`21`(441 tile，需重新配比例）、`24`(576 tile，既有 match24)、
  約 `32`(~1000 tile)、`64`(~4000 tile)、`96`(~9000 tile)，最後接上 Set F 的完整玻片。
- **workers**：全部固定 `workers=1`（隔離 tile-parallel 的變因，避免跟 Set C 的 workers 效應
  混疊）。
- **量測項目**：`perf_measure.py` 的完整 bucket 輸出（Phase A 總秒數、每個 MAIN/BG bucket 總秒數
  與呼叫數、Phase D 總秒數）。
- **產出**：一張「規模 → 各 phase 秒數」的表，供第 5 節擬合。

### Set B — 組成 sweep（固定尺寸，變背景比例）

**目的**：直接解出 tissue tile 與 background tile 各自的「每 tile 秒數」係數（迴歸，而非現有
`wsi_projection.py` 那種單一 anchor 重加權）。

- **產生方式**：固定 `--grid 24`（576 tile，與既有 match24/comp24 同尺寸，方便對照驗證舊數字）
  ，`--target-bg` 取多點：`0.10`、`0.25`、`0.50`、`0.558`（既有 match24）、`0.734`（既有
  comp24）、`0.90`。**先跑一次快速 preflight 確認這張真實玻片上，這幾個目標比例在 24×24
  窗口尺度下是否真的存在對應區域**（`composition_crop.py` 是在玻片內滑窗搜尋最接近目標的窗口，
  極端比例如 0.10/0.90 不保證找得到；找不到就如實記錄「這張玻片的可行範圍是 X–Y%」，不要硬湊）。
- **workers**：固定 `workers=1`。
- **量測項目**：同 Set A；額外記錄 `composition_crop.py --report` 產出的「目標比例 vs
  `run_batch` 實際 skipped/total 比例」的落差（brightness proxy 的已知誤差來源，doc 24
  的既有提醒）。
- **分析**：對每個 MAIN/BG bucket，用 `(n_tissue, n_background)` 對 `bucket_wall` 做線性迴歸，
  解出兩個母體各自的秒/tile 係數，並算 R²。

### Set C — workers sweep（固定尺寸與比例，變 workers）

**目的**：把 doc 38/39 現有的兩點（`workers=1,4`）外推換成四點（`1,2,3,4`）擬合，取得
`parallel_portion`/`serial_portion` 的 Amdahl 曲線，而非假設兩點連線。

- **產生方式**：沿用 Set A 已經產出的 2–3 個中型 anchor（例如 576 tile 與 ~4000 tile 兩組），
  `--workers 1,2,3,4`（`full_wsi_validate.py` 本身就支援逗號分隔的 workers 列表）。
- **量測項目**：整體 wall、`perf_measure.py --worker-timings`（26 個 worker 內 bucket）。
- **分析**：`wall(workers) ≈ serial_portion + parallel_portion/workers`（`serial_portion` 理論上
  應該約等於 Phase A + Phase D + init，可跟 Set A/D 的獨立估計互相印證，不吻合就要查原因，不要
  直接採信擬合值）。

### Set D — 合成放大畫布（純幾何，不跑完整 pipeline）

**目的**：把 Phase A / Phase D 的 scaling 曲線延伸到**超過這張真實玻片本身的 16.2–19 GP**，
用來支撐「醫師可能上傳更大的圖」這個疑慮——但**只測幾何/I/O 相關的兩段，不重跑整條分析**，因為
(a) 重複貼上的組織內容對 MAIN/BG 臂的 tissue/background 統計沒有意義（會污染 Set B 已經解出的
係數，不是新資訊），(b) 省下大量不必要的 GPU 小時數。

- **產生方式**：把校正後的 `warp_slide` 輸出（見 §1.3）用 2×2、3×3 鏡射/平鋪拼成 4×/9× 面積的
  合成大 TIFF（純檔案幾何操作，pyvips 的 join，不需要 GPU）。
- **量測項目**：只跑 `scripts/stitch_probe.py`（Phase D，含 `--pipelined`）與 Phase A 的獨立量測
  （若 `perf_measure.py` 沒有「只跑 Phase A、不跑後面」的模式，需要一支薄的呼叫
  `precut_paired_tiles` 並計時的 wrapper——檢查 `scripts/tile_generator.py`/`crop_roi.py` 是否已
  經可以直接拿來用，不要重新設計一支新工具）。
- **產出**：Phase A、Phase D 的 s-vs-GP 曲線延伸到 4×/9× 目前最大實測規模，並標註這是**幾何外推
  ，不是臨床規模的實測**（第 7 節限制會再次強調這個區分）。

### Set E — 重複跑批（誤差界）

**目的**：目前每一個整片級數字都是 n=1，doc 39/40/41 都自陳這個限制。取得真正的執行間變異數，
才能在報告裡給「預估 X 分鐘 ± Y%」而不是假裝精確的單一數字。

- **選點**：**不要選整片級 anchor**（一次 2–3 小時，重複 3 次就是接近一整天 GPU 時數）——選
  Set A/B 已經產生的中型 anchor（分鐘等級，例如 576 tile 或 ~4000 tile），`workers=1` 與
  `workers=4` 各重複 **n=3**（若時間允許，n=5 更好）。
- **量測項目**：同一 anchor 的完整 wall 與各 bucket 秒數的執行間標準差。
- **分析**：把這個相對變異數（例如 ±X% of wall）當作模型信賴區間的一個分量，疊加到第 5 節的
  迴歸殘差上。

### Set F — 校正畫布的完整玻片驗證（gated，最貴，排最後）

**目的**：在 doc 44 修正後的正確配準畫布上，重新跑一次「官方」全片基準，取代 round 8–13 建立在
錯誤畫布上的數字，作為本計畫模型的**驗證用 held-out 點**（不用來 fit 参数，只用來檢查模型
預測誤差）。

- **前置條件（gate）**：先完成第 3 節第 1 點的核對——確認 `full_wsi_validate.py` 現況、且已用
  `core_mask_map.py` 在校正後畫布上重算過一次真正的 tile grid（不要沿用第 1.3 節「35,500–36,000」
  這個推算值）。
- **執行**：`workers=1` 與 `workers=4` 各一次（若 GPU 時數預算允許，各兩次，用來跟 Set E 的
  中型-anchor 變異數互相印證「變異數會不會隨規模放大」這個目前未知的問題）。
- **量測項目**：同 round 8/13 的完整 bucket 輸出 + 正確性投票（`report.csv` 列數、
  `overlay_pyramid_audit.py`）——沿用既有的驗收 gate，不新設計。

### 各組定位一覽

| Set | 自變量 | 依變量（要解出什麼） | 尺度 | 相對代價 |
| --- | --- | --- | --- | --- |
| A | tile 數（固定 55.8% 背景） | Phase A/D 的 scaling 曲線；每 tile 速率是否隨尺寸穩定 | 分鐘～小時 | 中 |
| B | 背景比例（固定 576 tile） | tissue/background 各自的每 tile 秒數係數 | 分鐘 | 低 |
| C | workers（固定尺寸與比例） | Amdahl 曲線的 serial/parallel 拆分 | 分鐘 | 低 |
| D | 合成面積（純幾何，無 GPU 分析） | Phase A/D 延伸到 &gt;16 GP 的曲線形狀 | 分鐘 | 低 |
| E | 重複次數 | 執行間變異數 → 信賴區間 | 分鐘×N | 低 |
| F | （驗證用，不變自變量） | 模型在真實全片規模的預測誤差 | 小時 | **高** |

---

## 5. Analyze：分析與建模方法

1. **Set B 迴歸**：對每個 MAIN/BG bucket，`bucket_wall ≈ a + b_tissue · n_tissue + b_bg · n_bg`，
   最小平方解出 `b_tissue`/`b_bg`，報告 R² 與殘差。**這是本計畫相對於現有
   `wsi_projection.py`（單一 anchor 重加權）的核心改進**——`wsi_projection.py` 本身要在這裡擴充
   成讀取 Set B 的多點資料做迴歸，而不是沿用它目前的單點外推邏輯。
2. **Set A / Set D 曲線擬合**：Phase A、Phase D 各自的 `s` vs `tile 數`或`GP` 先試線性擬合，若
   殘差呈系統性彎曲（round 7/12 已經在 Phase D 觀察到超線性：cached 15.4 s/GP → 真實冷讀
   116.6 s/GP），改試 power law（`s = k · GP^p`）或分段模型（例如以 OS page cache 能裝下的檔案
   大小為轉折點——這張玻片的來源檔案是 46 GB，已知會超出 page cache，round 12 doc 39 §4.4 已經
   量到這個轉折的存在，只是還沒有一個連續函數描述它）。
3. **Set C 的 Amdahl 擬合**：`wall(workers) ≈ serial_portion + parallel_portion / workers`，四點
   最小平方，並與 Phase A + Phase D + init 的獨立估計互相印證（見第 4 節 Set C）。
4. **合成模型**：
   ```
   estimate(n_tissue, n_bg, workers, total_GP) =
       max(MAIN(n_tissue, n_bg), BG(n_tissue, n_bg)) / parallel_speedup(workers)
       + Phase_A(total_GP) + Phase_D(total_GP) + init
   ```
   注意 Phase A/D **不除以 workers**（現行架構下两者都是序列，發生在 parent process，不隨
   `workers` 縮放——這是既有事實，不是本計畫的新假設）。
5. **Set F 驗證**：把 Set F 的實測值代入自變量，比較模型預測 vs 實測的誤差百分比，**不吻合要
   回頭檢查是哪個子模型（組成係數／尺寸曲線／workers 曲線）失準，不要調整模型去遷就這一個
   驗證點**（避免用一個資料點過度擬合，重蹈 09 §1.4 引用的「CupY 案例」覆轍）。
6. **信賴區間合成**：最終呈現給 UI/報告的「預估時間」= 模型點估計 ± (Set E 的執行間變異數
   分量 ⊕ 迴歸殘差分量)，兩個誤差來源獨立假設下用平方和開根號合成。**超出 Set A/D 已驗證規模
   範圍的預測，額外標注「外推，誤差界未知」**，不要跟已驗證範圍給一樣的信賴區間寬度。

---

## 6. 產出與報告格式

1. **量測結果原始資料**：每次跑批的 `perf_measure.py` JSON/CSV 原始輸出全部保留（不只摘要），
   標好 git commit hash、config hash、硬體/driver 版本、輸入規模——沿用 09 §4.6 的環境戳記規範。
2. **係數表**：`b_tissue`、`b_bg`（依 bucket、依 arm）、Phase A/D 的擬合函數與參數、Amdahl
   `serial_portion`/`parallel_portion`，連同各自的 R²/殘差/信賴區間，存成一份結構化資料
   （json/yaml），供將來的 API 層（落地文件的工作）直接讀取，不必重新查文件。
3. **參考實作**（屬本計畫產出，**不算落地**）：一支 `scripts/eta_estimate.py`，輸入
   `(tile_grid or roi_pixels, background_share, workers)`，輸出預估秒數區間——純函式、不碰
   `backend/api`/`backend/schemas`/`frontend`，只是把第 5 節的模型變成一支可測試、可重跑的腳本，
   放在既有 `scripts/` 慣例下，跟 `wsi_projection.py` 平行存在（甚至可以是它的擴充版本）。
4. **給醫師/教授的報告骨架**：方法（量測工具與資料集設計）、資料表（Set A–F 全部原始數字）、
   模型公式與擬合優度、信賴區間、**限制與外推風險章節**（第 7 節內容原樣搬過去，不要在報告裡
   淡化）、與 `HybridPanel.tsx` 既有「不假裝有進度」設計原則的一致性聲明。

---

## 7. 風險與已知限制（誠實揭露，供報告使用，不要包裝成已解決）

1. **只有一張真實玻片**：背景比例 55.8%、每 tile 速率剖面是否具代表性，完全未知——這是本計畫
   最大的限制，且是刻意選擇的範圍（見開頭問答：你選擇了合成尺寸縮放而非等待更多真實案例）。跨
   尺寸的可遷移性建立在 tile pixel size 恆定 1024px 這個**已驗證**的事實上（第 1.5 節），但跨
   **玻片**（不同病人、不同染色批次）的可遷移性**未驗證**，模型上線後應該持續用新案例回測，
   而不是視為一次性完工。
2. **Set D 的超大尺寸曲線是幾何外推，不是臨床規模實測**：報告需要清楚標「已驗證範圍」
   （目前最大真實規模，16.2–19 GP）與「外推範圍」（Set D 的 4×/9× 合成規模）的邊界，超出這個
   邊界的預測（例如比 Set D 還大的假設玻片）誤差界完全未知，不應該顯示信賴區間，只能顯示
   「超出已驗證範圍」的警示。
3. **doc 44 的畫布修正若尚未實跑過，Set F 就是本計畫第一次在正確畫布上驗證**——若第 3 節第 1
   點核對後發現修正還沒有人實際跑過，Set F 的優先序要拉到最前面，因為它同時是「official 基準
   更新」與「模型驗證錨點」兩件事，其他組別的分析可以先用舊畫布的比例關係佐證方法論，但最終
   報告的絕對數字要以 Set F 為準。
4. **本計畫不改動任何 pipeline/API/UI 程式碼**：模型完成後要接進 UI，需要另一份落地規劃文件，
   且會牽涉 `backend/algorithms/hybrid/` 的 callback 設計（`../BACKLOG.md` 已列的、卡住的另一件
   事）與 [frontend-backend-boundary](../../.claude/rules/frontend-backend-boundary.md) 的邊界
   審查。
5. **單一硬體型號**：全部係數都在 RTX 5090 / CUDA 13.0 / torch 2.11.0+cu130 上量測，換硬體
   （例如較舊的 GPU、VRAM 較小的卡）之後，MAIN 臂的每 tile 秒數會完全不同，模型不可跨硬體
   直接套用——這點沿用本資料夾一貫的機器戳記慣例，報告需要明確寫出硬體型號。

---

## 8. 執行順序與代價估計（GPU 時數預算）

建議順序（便宜、風險低的先做，貴的排最後、且有 gate）：

1. 第 3 節的 Discover 核對（doc 44 現況、Phase A 曾否量過曲線）——幾乎零代價，純讀文件/跑
   preflight。
2. Set B（組成 sweep，576 tile，分鐘級）——最快解出 tissue/background 係數，且能立刻跟既有
   match24/comp24 數字對照驗證方法論本身沒有問題。
3. Set C（workers sweep，沿用 Set A/B 的中型 anchor，分鐘級）。
4. Set A（尺寸 sweep，從分鐘級爬到數十分鐘級）。
5. Set D（合成幾何，只跑 Phase A/D 探針，不跑完整分析，便宜）。
6. Set E（重複跑批，選中型 anchor 而非整片——**不要**在全片規模做 n=3，光是這樣就要
   `(3×2.96h)+(3×1.35h)≈12.9h` GPU 時數，遠超預算；中型 anchor 的 n=3 只要數十分鐘）。
7. Set F（gated，全片級，`workers=1`+`workers=4` 各一次起跳，若時數允許各兩次；這是全計畫
   單一最貴的一組，且必須排最後，因為它同時要等 doc 44 的前置修正確認，也適合當「拿前六組
   模型去驗證」的最後一步）。

---

## 9. 與既有 `wsi_projection.py` 的關係（避免重複造輪子）

`scripts/wsi_projection.py` 已經實作了「依 bucket 母體拆解速率 → 投影到任意規模/比例」的核心
邏輯（第 1.2 節）。本計畫**不是**要另起爐灶，而是把它從「單一 anchor 外推、`workers=1` only、
Phase D 線性像素比外推」擴充成：

- 讀取 Set B 的多點資料做迴歸（而非單點重加權）；
- 讀取 Set A/D 的擬合曲線取代 Phase D 目前的線性假設；
- 新增 `workers` 參數，接上 Set C 的 Amdahl 擬合；
- 輸出信賴區間，不只是點估計。

第 6 節的 `scripts/eta_estimate.py` 可以直接是 `wsi_projection.py` 的下一個版本，或是它的薄
wrapper——實作細節留給下一份落地文件決定，本文件只確認「不必重寫這套邏輯，只需要擴充」。
