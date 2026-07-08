# ① stage 2 — 方案 (a) GIL 競爭診斷結果

> 執行 [../11-gpu-pipeline-stage2-plan.md](../11-gpu-pipeline-stage2-plan.md) §4(a)（**診斷優先**，
> 不動生產代碼）與 §5.5/§6.5（(a) 的診斷結果必須記錄，即使結論是「不值得做 (b)/(c)」，
> negative result 也要留下）。doc 11 的 §5 決策樹以本文件的量測數字收斂。
>
> 環境：git HEAD `3d7d91c`；RTX 5090 / CUDA 13.0 / torch 2.10.0+cu130 / Python 3.11.15（傳統單
> GIL，無 free-threading，故 GIL 競爭分析完全成立）；venv `/data/taro_Projects/tsgh/.venv`。
> 量測輸入：medium 121-tile 錨點（`test_picture/_roi_crops/med_{ihc,dish}.tiff`，8192²，11×11）。
> 工具：`py-spy 0.4.2`，`record --format raw`（collapsed stacks），rate 200 Hz。
> 原始資料（collapsed stacks，已 gzip）：[`runs/stage2_gil_diag/gil.raw.txt.gz`](./runs/stage2_gil_diag/)
> （`--gil`，只採「持有 GIL」的堆疊）+ [`runs/stage2_gil_diag/wall.raw.txt.gz`](./runs/stage2_gil_diag/)
> （無 `--gil`，全活躍執行緒）+ 解析腳本 [`runs/stage2_gil_diag/analyze_gil.py`](./runs/stage2_gil_diag/analyze_gil.py)
> （吃 `.gz` 或純文字）。

## 方法

跑兩次同一條 121-tile pipeline（方案 (b) 現行碼，未改動）：

1. **GIL run** — `py-spy record --gil`：每個取樣點只記錄「當下持有 GIL 的那一條執行緒」的堆疊。
   → 得到 **GIL 的歸屬分佈**（誰在拖住 GIL）。28,807 個 GIL-holding 樣本。
2. **Wall run** — `py-spy record`（無 `--gil`、無 `--idle`）：記錄所有活躍執行緒。
   → 得到 **wall 時間分佈**（時間實際花在哪）。537,986 個樣本。

執行緒歸類：`threading._bootstrap_inner` 在堆疊中 = 背景執行緒（`tile-cpu` ThreadPoolExecutor
worker + joblib `prefer='threads'` 的 thread pool）；否則 = 主執行緒。主執行緒「空葉節點」
（py-spy 無法解析出 Python 頂框）= 在 native/CUDA C 中執行、**GIL 已釋放**。

> ⚠️ **py-spy 顯著膨脹 wall-clock**（GIL run 513 s、wall run 866 s，對比原生 207.9 s）——ptrace
> 取樣對這種多執行緒 + CUDA 的行程開銷很大。**本文件的數字只用於「分佈 / 佔比」，絕不當 wall-clock
> 用**；doc 11 §6 的 wall-clock 驗收保留給真正的代碼改動 ablation（原生、不掛 py-spy）。
> 兩次 run 的 `report.csv` 都是 **3557 cells**，與 pipeline-overlap-result.md 的雜訊地板一致，
> 確認掛 py-spy 不改變管線行為。

## 結果 1：GIL 歸屬分佈（誰在拖住 GIL）

| 執行緒 / 來源 | 佔全部 GIL-holding 樣本 | 說明 |
|---|--:|---|
| **主執行緒 合計** | **81.4%** | 拖住 GIL 的絕大多數是主執行緒本身，不是背景的 detect_all_dots |
| ↳ `gc.collect()`（`hybrid_pipeline.py:805`，tile 邊界） | **33.6%** | **單一最大 GIL 持有者**；99.9% 的 run_batch-葉樣本落在這一行 |
| ↳ Cellpose Python 後處理（M2/M3b） | 26.1% | `fill_holes_and_remove_small_masks`/`_stats`/`_extend_centers_gpu`/`get_masks_torch`/`steps_interp`/`get_rel_pos` — 模型內在的 numpy/scipy mask 重建 |
| ↳ UNet++ Python（M1） | 14.9% | 同屬模型內在成本 |
| **背景執行緒 合計** | **18.6%** | detect_all_dots + 落地寫檔 |
| ↳ `detect.red_black`（`_detect_red_dots`/`_detect_black_dots`/`_detect_one_cell`） | 12.5% | 背景 GIL 競爭**集中**於此 |
| ↳ `regionprops` + per-region Python 迴圈 | 4.0% | 同上，Python-level 逐區塊迴圈 |

## 結果 2：Wall 時間分佈（時間實際花在哪）

| 分段 | 佔 wall | 可否重疊 / 影響 |
|---|--:|---|
| 主執行緒 **native/CUDA**（GIL 已釋放） | **61.0%** | ✅ 真正的 GPU 前向；背景可在此窗口重疊 |
| 主執行緒 **Python**（持有 GIL） | **29.0%** | ❌ 此時 GPU 無 kernel 在跑（→ GPU idle）**且**背景執行緒被 GIL 卡住 |
| ↳ 其中 `gc.collect()`（tile 邊界） | 3.7% | 主執行緒序列 stall，GPU idle + 背景雙重浪費 |
| ↳ 其中 Cellpose/UNet mask 重建 | ~19% | 模型內在，散在十幾個函式 |
| 背景執行緒（detect_all_dots + PNG encode） | **10.0%** | 本來就 <路徑外，且大多已與 GPU 前向重疊 |

## 結論（收斂 doc 11 §5 決策樹）

1. **doc 11 的原始假設被數據推翻**：doc 11 §1 假設剩餘 idle 主因是背景 `detect_all_dots`
   的 joblib `prefer='threads'` 與主執行緒搶 GIL。實測相反——**拖住 GIL 的 81% 是主執行緒自己**
   （`gc.collect` 33.6% + Cellpose/UNet Python 重建 41%），背景 detect_all_dots 只佔 18.6%。

2. **殘餘 GPU idle（build (b) 的 15.4%）≈ 主執行緒 29% 的 GIL-held Python 中沒與 GPU kernel 重疊
   的部分**。GPU 只由主執行緒餵；主執行緒卡在 Python（Cellpose 重建、gc）時就沒有 kernel 在跑。

3. **方案 (b)（detect_all_dots 換 process 後端）判定不值得做**：背景工作只佔 10% wall、且大多已
   重疊；它的 18.6% GIL 份額即使完全釋放，也**動不到 GPU idle**——因為 idle 是主執行緒的 29%
   Python 造成的，不是背景執行緒。這正落在 doc 11 §4(a) 預留的逃生出口：「如果發現真正瓶頸是
   主執行緒自己的 Python 開銷，那換 detect_all_dots 後端不會有幫助」。**故本輪不執行 (b)，也
   不需要 §7.1 的 spawn-safety 腳本 / §7.2 的 memmap 量測**（那些是 (b) 的前提，(b) 已被否決）。

4. **診斷浮現一個 doc 11 (a)/(b)/(c) 都沒列到的、更便宜的槓桿——tile 邊界的 `gc.collect()`**：
   它是**單一最大 GIL 持有者（33.6%）**、佔 3.7% wall、且是主執行緒在 tile 邊界的序列 stall
   （此時 GPU idle、背景也被卡）。它就是 bottleneck-list.md 的 ④（當時記為 4.28% wall、在 Amdahl
   停損線以下、未深入）——但在 pipeline (b) 的重疊結構下，它從「4% 的順序開銷」升級成「**#1 的
   GIL / 重疊殺手**」。依 playbook「讓瓶頸定義解法、便宜優先、(b) 移出關鍵路徑」，把它移出 tile
   邊界主執行緒關鍵路徑（例如丟到背景 CPU 後段執行、或降頻）是比 (b)/(c) 更划算的下一步。
   **但它動到既有「記憶體有界」invariant（doc 10 §5.4/§7.2：per-tile `gc.collect`+`empty_cache`
   是記憶體有界的手段之一），任何改動必須依 doc 11 §6 重新驗證 RSS/VRAM，不能只看 wall-clock。**

5. **其餘 ~19% wall 的 Cellpose/UNet Python mask 重建屬模型內在成本**（散在十幾個函式，非單點可
   移除）→ 屬 doc 10 §3(c) / doc 11 §4(c) 的架構/模型替換範疇，本輪不處理（需精度取捨、另立案）。

## 方案 (d) ablation 結果 — gc 重定位到背景執行緒（**負向，已還原**）

依上節建議實作了「把 tile 邊界 `gc.collect()` 下放到背景 `_process_precut_tile_cpu` 尾端、
`empty_cache` 留主執行緒」，並依 doc 11 §6 做 A/B ablation（baseline (b) HEAD vs modified，同一台
RTX 5090、同一 `scripts/perf_measure.py --gpu-dmon` харness、同輸入、同 session、各跑一次；
metrics 見 [`runs/stage2_gil_diag/ablation_metrics/`](./runs/stage2_gil_diag/ablation_metrics/)）。

> 註：本 A/B 用 `perf_measure.py`（會 monkeypatch 熱點函式計時），wall/idle 比 plain run 略膨脹
> （baseline harness 218 s vs plain 207.9 s；idle 0.183 vs plain 0.154）——**兩臂同 harness，比的是
> delta**，不是絕對值。

| 指標 | baseline (b) | modified (gc→bg) | delta / 判定 |
|---|--:|--:|---|
| end_to_end wall | 218.03 s | 217.09 s | **-0.4%（在 run-to-run 雜訊內，未達 §6.1「明顯低於」門檻）** |
| GPU idle_frac | 0.183 | **0.221** | **未改善、反而略升** |
| sm_mean | 32.9% | 31.9% | 持平 |
| VRAM peak | 5159 MB | 5159 MB | ✅ 有界不變 |
| RSS peak | 3.109 GB | 3.092 GB | ✅ 有界不變 |
| 正確性（report.csv） | 3559 cells | 3558 cells | ✅ 質心漂移 1、reddot/blackdot/score **0 筆不符**，在雜訊地板內 |
| gc.collect | 121 calls / 9.67 s | 121 calls / 9.80 s | 每塊一次語意保留 |

**結論：負向（實為持平偏差）。gc 重定位沒有降低 wall-clock，也沒降 idle。** 原因（實作後才看清）：
在 depth-1、`max_workers=1` 的管線裡，主執行緒每塊會在 `_collect(pending)` **阻塞等待背景 CPU 後段
（tile N-1）完成**——即重疊率是 **背景 CPU 綁定**（呼應 pipeline-overlap-result.md：detect_all_dots
是重的那一極）。把 gc 移到背景後段尾端，等於把它加到「主執行緒正在等的那一段」，讓背景那一極更長，
所以 idle 不降反微升。gc 在主執行緒版本雖是序列 stall，但那段 GPU 本來就要 idle（tile 邊界）；移到
背景反而去和「下一塊 GPU 前向的主執行緒 Python（Cellpose 重建）」搶 GIL。**依 playbook「零貢獻的
改動要砍」+ karpathy_rule「不留無法證明價值的複雜度」，此改動已還原（`git restore`）。**

正確性補記：gc.collect 不參與任何運算、只回收記憶體，對輸出零影響；3559 vs 3558 的 1 顆差異純為
GPU 前向非決定性（與 pipeline-overlap-result.md 的雜訊地板一致），非本改動所致。

## 下一步建議（供 Choose 決策）— 依 ablation 結果修正

- **不做（已驗證無效）**：方案 (b)（detect_all_dots 後端，§4(a) 診斷判定）+ (d) gc 重定位（本節
  ablation 判定）。兩者都對 GPU idle 無效。
- **停損（建議）**：殘餘 idle 的主因是 **管線重疊為背景 CPU 綁定 + 主執行緒 Cellpose/UNet 內在
  Python mask 重建（~19% wall，散在十幾個函式）**，兩者都不是單點便宜可解。按 doc 11 §2 的 Amdahl
  停損——剩餘天花板 ~1.18x、且唯一的便宜槓桿（gc）已證實無法轉成 wall-clock 收益——**本輪到此為止，
  記錄現況並停手**，不為了榨最後幾 % 引入跨 tile batching（§4(a)）或換 backbone（§4(c)）——那兩項
  的成本/風險在 doc 10 已評估過，需另立案 + 精度取捨。
- **（低優先、若日後仍要追）**：唯一未試的方向是**降 gc 頻率**（每 N tiles 一次，直接減少 ~9.8 s
  gc 總量的大部分，而非搬家）。但它比重定位更動到「記憶體有界」invariant（doc 10 §5.4：sweep 之間
  RSS 會漲），full-WSI 尺度風險較高，須以更大規模（441-tile / 更長批次）重驗 RSS 才能採用——投報比
  低於直接停損，列為 backlog。

## 追加深挖：「Cellpose/UNet Python 41%／~19% wall」拆解與逐函式演算法/GPU 可行性（2026-07-08）

> 動機：使用者要求進一步分析「CPU 到底在做什麼工作佔用 GIL」，目的是判斷每一塊能否用**演算法
> 改善**或**搬到 GPU 平行運算**解決，而不是停在「model-inherent，非單點可解」這句定性結論。上面
> 「結論」第 5 點與「下一步建議」都把這塊當一個整體處理；重新用 `analyze_gil.py` 的原始 leaf/stage
> 統計＋直接讀 `.venv/lib/python3.11/site-packages/cellpose`／`segment_anything` 原始碼逐一核對後，
> 發現這個整體其實可以拆成兩件精度不同的事，且其中一半（UNet++ 那一半）是量測假象。

### 修正 1：UNet++ 14.9% GIL／原「~19% wall」份額中，93% 是一次性 import／模型載入，不是逐 tile 成本

重跑 `analyze_gil.py` 的 stage 統計，`gpu.unet` bucket = 4291 個 GIL 樣本（= Result 1 表的 14.9%），
但其中 **4004 個（93.3%）落在 `_init_unet_inferencer`**（`hybrid_pipeline.py:748`，在 tile 迴圈**之前**
呼叫一次，內部 import `segmentation_models_pytorch`/`torch`/`timm`/`torchvision`/`triton` 一長串模組
鏈）——不是逐 tile 重複執行的後處理。**wall run** 直接證實這點：`gpu.unet` bucket 在 wall 分佈只佔
**2.6%**（13,790/537,986 樣本），與 bottleneck-list.md ⑥「模型初始化一次性、441-tile 時降到 0.37%」
完全對得上，**不是**新的逐 tile 瓶頸。

**結論：UNet++ 真正逐 tile 的 Python 後處理幾乎可忽略（GIL 樣本扣掉 import 後只剩 ~1.0%，wall 更低）。
本文件「結論」第 5 點與 doc 11 §6 提到的「Cellpose/UNet mask 重建 ~19% wall」，那 19% 幾乎全部是
Cellpose，不是「Cellpose/UNet」聯合——UNet++ 不該再被算進「model-inherent 不可解」這塊。**

### 修正 2：真正的 ~19% wall／26.1% GIL 全部落在第三方套件（cellpose 4.0.8 + segment_anything），
但其中 4/5 個具名函式其實**已經在 GPU 上跑**，卡住的是 Python for-loop 的 kernel-launch 開銷，不是
「該不該搬去 GPU」

逐一讀原始碼確認（路徑、行號、device 是否真的傳到 cuda）：

| 函式 | 檔案 | 是否已在 GPU 上跑 | 實際卡住的地方 | 演算法改善 | GPU 可行性 |
|---|---|---|---|---|---|
| `_extend_centers_gpu` | `cellpose/dynamics.py:21` | ✅ `device=self.device`，由 `CellposeModel` 一路傳入（`models.py` `_compute_masks`→`resize_and_compute_masks`→`compute_masks`→`follow_flows`），非其文件字串預設的 CPU | 200 次迭代的 Python `for` 迴圈，每輪對 GPU tensor 做極小的 in-place 更新——kernel-launch bound，GIL 卡在迴圈本體而非算力 | 把 200 輪 diffusion 迭代向量化/融合 | **已在 GPU**；真正槓桿是 CUDA graph capture 或 `torch.compile` 把固定形狀的迴圈捕捉成一次 launch，不是換裝置 |
| `get_masks_torch` | `cellpose/dynamics.py:488` | ✅ 同上（`device=pt.device`） | 兩個 Python `for k in range(n_seeds)` 迴圈，逐一 candidate cell 做 slice/scatter | 用批次 scatter（`index_put`/向量化）取代逐 seed 迴圈 | 已在 GPU，同上 |
| `steps_interp` | `cellpose/dynamics.py:311` | ✅ | `niter=200` 的 Euler 積分迭代迴圈 | 同上（融合迭代） | 已在 GPU，同上 |
| `get_rel_pos` | `segment_anything/modeling/image_encoder.py:308`（Cellpose SAM ViT backbone） | ✅ | 每個 attention block 呼叫一次，Python 層 `F.interpolate` 調度開銷 | 若多個 block 共用同一組相對位置編碼，可預先算好快取 | 已在 GPU |
| `fill_holes_and_remove_small_masks` | `cellpose/utils.py:619` | ❌ **唯一真正的 CPU-only** | 用 `fill_voids.fill`（CPU C 擴充，無 GPU 版本），Python 迴圈逐一 instance 呼叫 | 用單次 whole-image 的 labeled 填洞取代逐 instance 迴圈 | 現成套件沒有 GPU 路徑，需自己接 `cucim`/`cupy` 的 GPU 連通元件填洞——五個裡面成本最高的一個 |

**這修正了「model-inherent、非單點可解」的定性判斷本身**：5 個函式裡有 4 個根本不是「CPU vs GPU」
的問題——資料早就在 GPU 上，卡住的是 Python 迴圈逐輪 launch 小 kernel 的開銷。真正的槓桿是 CUDA
graph capture / 向量化迴圈，不是 doc 10 §3(c) 假設的「換模型架構」。只有 `fill_holes_and_remove_small_masks`
是真的沒有 GPU 路徑、需要另找函式庫或自己刻。

### 但 Amdahl 天花板幾乎沒有改變，且改動面是「patch 一個釘死版本的第三方套件」

即使把這 19% wall 完全歸零（不可能，只是理論上限），天花板是 `1/(1-0.19) ≈ 1.23x`——比 doc 11 §2
已經判定的 ~1.18x 停損線只高一點點，量級沒變。且任何修法都要在 `cellpose==4.0.8`/`segment_anything`
這兩個**釘死版本的第三方套件**裡打 patch（不是本專案程式碼），需要：
1. 對照 upstream GitHub 是否已有更新版本解決同樣問題（先查，不要重新發明）；
2. patch 後用 doc 10 §5.3 的 `report.csv`/質心比對法重新驗證正確性（fastremap/scatter 等價寫法必須
   bit-exact 或落在雜訊地板內）；
3. 追蹤「釘死版本 + 本地 patch」的長期維護成本（套件升級時 patch 要重打或重新驗證相容性）。

**Choose 結論：維持本輪停損，但理由更正為「天花板太小、改動面是第三方套件」，不是「model-inherent
不可解」**（後者不準確——見上表，4/5 已可行，只是投報比不夠）。列為 backlog，若之後 Amdahl 天花板
因為其他改動而讓這 19% 的相對佔比上升到更值得做的門檻，或發現 upstream 已有現成修法，再重啟。

### 對「CPU 到底在做什麼佔用 GIL」問題的直接回答（供下一階段規劃）

- **背景執行緒（`detect_all_dots`，18.6% GIL／10% wall）**：**本專案自己的程式碼**
  （`m3_module/m3_dot_detection.py`/`m3_dot_kernels.py`），非第三方。理論上可演算法改善（例如
  `regionprops`/`_compute_ring_stats` 的逐 blob Python 迴圈向量化），GPU 搬遷則因為每次呼叫都是
  單一 cell 的小 patch（joblib 逐 cell 平行），搬去 GPU 需要先把所有 cell 的 patch 一次性 batch 起來
  才划算，否則 kernel-launch overhead 會蓋過收益——改動面不小。但**這與目前的瓶頸無關**：doc 11 §4(a)
  已量到背景執行緒的 18.6% GIL 本來就已與 GPU 前向重疊、動不到 idle，本輪不需要為了這個目標去動它。
- **主執行緒 `gc.collect()`（33.6% GIL／3.7% wall）**：非演算法/GPU 範疇（記憶體管理），已於上方
  ablation 判定負向。
- **主執行緒「UNet++ Python」（14.9% GIL）**：**修正 1 — 93% 是一次性 import，非逐 tile 成本，
  不是候選。**
- **主執行緒「Cellpose Python 後處理」（26.1% GIL／19.0% wall）**：**修正 2 — 第三方套件，4/5 個
  函式已在 GPU、卡在 Python 迴圈的 kernel-launch 開銷（槓桿＝CUDA graph/向量化），1/5
  （`fill_holes_and_remove_small_masks`）是真正的 CPU-only、需另尋 GPU 演算法。天花板僅 ~1.23x，
  改動面是釘死版本的第三方套件，本輪判定投報比不足，列 backlog。**

## 重現方式

```bash
cd /data/taro_Projects/tsgh
DIAG=docs/hybrid-pipeline/measurement/runs/stage2_gil_diag
ROI="$PWD/backend/algorithms/hybrid/test_picture/_roi_crops"
# GIL 歸屬：
.venv/bin/py-spy record --gil --rate 200 --format raw --output "$DIAG/gil.raw.txt" \
  -- .venv/bin/python backend/algorithms/hybrid/hybrid_pipeline.py \
     --ihc "$ROI/med_ihc.tiff" --dish "$ROI/med_dish.tiff" --output <out>
# Wall 分佈（無 --gil）：同上去掉 --gil，輸出到 wall.raw.txt
# 解析（gz-aware）：
.venv/bin/python "$DIAG/analyze_gil.py" "$DIAG/gil.raw.txt.gz" "$DIAG/wall.raw.txt.gz"
```
