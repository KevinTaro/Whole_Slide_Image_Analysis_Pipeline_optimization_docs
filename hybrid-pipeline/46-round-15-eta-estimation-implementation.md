# 46 — Round 15：UI 預估時間估算基準（量測執行與效能報告）

> 本文件是 [45-eta-estimation-measurement-plan.md](./45-eta-estimation-measurement-plan.md) 的
> **執行記錄與結果報告**，比照 40/41、42/43 的 plan → implementation 配對慣例。
>
> 量測日期：**2026-07-31 ~ 2026-08-01**。
> 機器：**RTX 5090 (32,607 MiB) / driver 580.173.02 / CUDA 13.0 / torch 2.11.0+cu130 / Python 3.11.15 / 20 cores / 62 GB RAM**。
> git HEAD `8c3e8a3`，`config_hash d2ccc46b`（與 round 13/14 相同，未改動任何 pipeline 程式碼）。
> 總 GPU 時數：**約 7.95 小時**（成功步驟加總，不含兩次因 driver script 缺陷重跑的損耗）。

---

## 0. 摘要

doc 45 的六組量測（Set A–F）**全部執行完成**，另加兩組計畫外但必要的量測（legacy 錨點重測、
Phase A 獨立曲線）。主要結果：

1. **doc 44 的正確畫布首次被實測**：156222×134028，**35,700 tile**（204×175 grid，20.938 GP），
   背景比例 **65.92%**——不是沿用多輪的 55.8%。
2. **首次解出顯式的 tissue/background 每 tile 係數**（doc 45 §3.3 指出從未做過）：
   組織塊 **0.7298 s**、背景塊 **0.0460 s**，比值 **15.9×**，R² **0.99694**。
3. **首次有誤差棒**：n=3 重複跑批，run-to-run CV **0.90%（workers=1）/ 0.49%（workers=4）**。
   本專案此前每一個整片數字都是 n=1 vs n=1。
4. **Phase D 的 page-cache 轉折首次被量到**：同一幾何下，來源在 page cache 內 **11.26 s/GP**，
   冷讀 **60.90 s/GP**，**5.41× 差距**。
5. **模型在 held-out 的 Set F 上，core 準確（+4.3%/−6.7%）、Phase D 失準（−56%）**——依 doc 45
   §5.5 的紀律，**不回頭去遷就這個驗證點**，而是指認失準的子模型。

**與最舊資料的比較（使用者要求的重點，詳見 §5）**：同一批 crop 檔案、round 1 對今日程式碼，
`workers=1` 快 **2.85–2.88×**，`workers=4` 快 **5.92×**；整片而言，round 1 當年的投影是
**18.87 小時**，今日實測 `workers=4` 是 **1.478 小時**，**12.77×**。

---

## 1. Discover（doc 45 §3）——三個必須先確認的項目

### 1.1 doc 44 的修正從未被實跑過（確認）

`docs/hybrid-pipeline/` 底下沒有比 45 更新的文件，`measurement/` 也沒有比 `_metrics_r14` 更新的
metrics 目錄。唯一存在的正確畫布產物是
`backend/algorithms/hybrid/output/full_slide_run/`（2026-07-30，`_resume` 有 35,701 筆），
但**沒有任何 timings JSON**——那是一次產品跑批，不是量測。**Set F 因此確實是正確畫布上的第一次
有計時的全片跑批。**

### 1.2 正確畫布的 tile grid（實測，取代 doc 45 §1.3 的推算）

`her2_warped_lv0.ome.tiff` / `dish_warped_lv0.ome.tiff` 兩者皆為 **156222×134028**（尺寸相等，
`PrecutStream` 的檢查自然通過，不需要任何 conform）。以 pipeline 自己的 `chunk_offsets` 計算：

| | 舊畫布（round 8–13） | **正確畫布（round 15）** | 變化 |
| --- | --: | --: | --: |
| 尺寸 | 141658×114366 | **156222×134028** | +10.3% / +17.2% |
| gigapixels | 16.20 | **20.938** | +29.3% |
| tile 數 | 27,565 | **35,700**（204×175） | **+29.5%** |
| 背景比例 | 55.8% | **65.92%** | +10.1 pp |
| **組織 tile 數** | 12,184 | **12,167** | **−0.14%** |
| 背景 tile 數 | 15,381 | 23,533 | +53.0% |

**關鍵發現：組織 tile 數幾乎完全沒變（−0.14%）。** 正確畫布多出來的 +29.5% tile **全部是背景**
——配準的 `crop="overlap"` 把同一塊組織包在更大的畫布裡。由於背景塊只有組織塊 1/15.9 的成本
（§3.1），畫布變大付出的代價遠小於 tile 數的增幅。這個預測在 Set F 得到驗證（§4）。

另註：doc 45 §1.3 提到 round 6 之前被推翻的「35,700 tile」舊假設，並要求實測排除巧合。實測結果
**恰好就是 35,700**。這不是巧合——當年的假設是對著**真正配準後的畫布**算的，被推翻是因為它被拿去
和錯誤的 conformed 畫布（27,565）比較。舊假設一直是對的，錯的是比較對象。

### 1.3 Phase A 從未有 scaling 曲線（確認，且已補上）

確認 doc 45 §2.2 的說法。更進一步：**在出貨組態下 Phase A 根本無法從一般跑批讀出**——
`--stream-precut` 把預切重疊進分析迴圈，`phaseA_precut_s` 只涵蓋讀檔頭與算格線
（實測各尺度都是 0.03–0.05 s）。曲線見 §3.4。

### 1.4 量測過程中發現的程式碼落差（依 doc 09 §1.2 另行記錄，不混進效能結論）

| # | 位置 | 問題 | 處置 |
| --- | --- | --- | --- |
| 1 | `scripts/full_wsi_validate.py:81` | `from m0_reader import chunk_offsets`——`m0_reader` 已移入 `m0_module/`，preflight 直接 `ModuleNotFoundError` | **已修**（改為 `m0_module.m0_reader`） |
| 2 | `scripts/core_mask_map.py:36` | 同上 | **已修** |
| 3 | `scripts/core_mask_map.py:79` | `HP.generate_ihc_core_mask` 已搬到 `m1_overlay.py:53` | **已修**（改為直接 import） |
| 4 | `scripts/perf_measure.py:769` | 非 `--stream-precut` 分支呼叫 `HP.precut_paired_tiles`，**該函式已完全不存在**（被 `PrecutStream` 取代，只剩 `hybrid_pipeline.py:133` 的 docstring 提及）。round 13 起每次都帶 `--stream-precut`，所以這條路徑很久沒被執行過 | **未修**——本輪沒有任何量測走這條路徑，修它等於未經驗證地改動量測工具。列此備查 |
| 5 | `run_batch` 的 `stats["skipped"]` | **不等於背景 tile 數**。b10 實測：`F_write_blank_tile` = 58（core mask 全空），但 `stats.skipped` = 77——多出的 19 是「跑完偵測但沒有可用細胞」的 tile | 見 §1.5 |

### 1.5 `stats.skipped` ≠ 背景比例（解釋了 round 13 記錄裡的一個矛盾）

doc 45 §1.6 引用 round 13 時記錄「10,800 success / 16,765 skipped」＝ 60.8%，卻同時說背景是
55.8%——差 5 個百分點，從未被解釋。本輪在 b10 上直接拆開：

```
576 tiles = 58 背景（core mask 全空 → F_write_blank_tile）
          + 518 進入完整處理，其中 499 產出可用細胞、19 沒有
stats = {success: 499, skipped: 77}      # 77 = 58 + 19
```

**`skipped` 混合了兩個不同的族群。** 建模必須用 `F_write_blank_tile` 的計數，不能用 `skipped`；
本報告與 `scripts/eta_estimate.py` 全部使用前者。Set F 實測 `F_write_blank_tile` = 23,533
= 65.92%，與 `core_mask_map.py` 的獨立測量**完全一致**。

---

## 2. 量測執行摘要（真實花費時間）

| 步驟 | 內容 | 實際耗時 |
| --- | --- | --: |
| core_mask_map | 35,700 tile 全片 UNet 掃描（正確畫布） | 1,170.3 s |
| crops | 11 個 crop（Set A 6 個尺寸 + Set B 6 個組成，共用 b_real） | 2,168 s |
| legacy | round-1 原始 crop 重測，3 尺寸 × 2 worker 數 | 636 s |
| Set B | 組成 sweep，6 × 576 tile，workers=1 | 950 + 367 s |
| Set C | workers sweep，576 與 4,096 tile × w1–w4 | 3,290 s |
| Set A | 尺寸 sweep，121→9,216 tile，workers=1 | 3,177 s |
| Set E | 重複跑批 n=3 × 2 worker 數 | 517 s |
| Set D | Phase D 探針，6 個尺度到 4× 片面積 | 2,254 s |
| **Set F** | **全片驗證，正確畫布，w1 + w4** | **16,212 s** |
| Phase A probe | 獨立預切曲線，6 個尺寸（純 I/O，不用 GPU） | 570 s |
| **合計** | | **≈ 8.11 h** |

驅動腳本 `/home/taro/r15/{run_all,make_crops,run_sets,run_setD,run_setF,run_legacy}.sh`，
比照 round 14 把 campaign driver 放在 scratch 的慣例。兩次重跑源自 driver script 自身的缺陷
（`case "${1:?...{B|C|A|E}}"` 會靜默破壞 pattern matching；crop 命名 `b_real` vs `breal` 不一致），
與 pipeline 無關，已修並加上 fail-fast 檢查。

---

## 3. 各組結果

### 3.1 Set B — 組成 sweep（**本計畫的核心產出**）

固定 576 tile（18688×18688，0.349 GP），`workers=1`，用 `core_mask_map` 的 `--map` 標記
（pipeline 自己的 empty-core-mask 判準，**不是** `composition_crop.py` 的亮度 proxy）。

| run | 背景 tile | 組織 tile | 背景比例 | wall | s/tile |
| --- | --: | --: | --: | --: | --: |
| b10 | 58 | 518 | 10.07% | 381.3 s | 0.6619 |
| b25 | 144 | 432 | 25.00% | 322.3 s | 0.5596 |
| b50 | 288 | 288 | 50.00% | 240.2 s | 0.4170 |
| b_real | 380 | 196 | 65.97% | 164.7 s | 0.2859 |
| b73 | 423 | 153 | 73.44% | 131.9 s | 0.2290 |
| b90 | 518 | 58 | 89.93% | 65.7 s | 0.1140 |

對 core 時間（wall − Phase D）做最小平方：

```
core = 0.7298 s x n_tissue  +  0.0460 s x n_bg      (截距 0.001 s)
R² = 0.99694    殘差 RMS 6.03 s    最大相對殘差 6.47%
```

**一個組織 tile 的成本是背景 tile 的 15.9 倍。** doc 45 §3.3 指出 round 7 只「假設」背景塊會被
core mask 短路成本趨近 0，從未量過顯式係數。實測：**不是 0，是組織塊的 6.3%**。

`--map` 標記的準確度：預測 58 個背景 tile，`F_write_blank_tile` 實測 **58**——**零誤差**。
doc 45 §4 Set B 要求記錄的「亮度 proxy 落差」因此不存在（改用 `--map` 即消除），doc 24 的該項提醒
在此路徑下可以關閉。

**逐 bucket 係數表**（doc 45 §6.2 要求的結構化產出，完整版見 `coefficients.json`）：

| bucket | arm | 母體 | s/組織 tile | s/背景 tile | R² |
| --- | --- | --- | --: | --: | --: |
| `B1_m3b_cellpose` | MAIN | tissue | 0.60168 | −0.01165 | 0.9995 |
| `B3_detect_dots` | BG | tissue | 0.29984 | −0.01924 | 0.9938 |
| `B3_enlarge_cells` | MAIN | tissue | 0.05058 | −0.00297 | 0.9921 |
| `B1_unet_coremask` | MAIN | all | 0.04091 | 0.02887 | 0.9726 |
| `B3_build_results` | MAIN | tissue | 0.03144 | −0.00030 | 0.9998 |
| `B2_render_overlay` | BG | tissue | 0.02346 | −0.00067 | 0.9994 |
| `B2r_tile_read` | MAIN | all | 0.02169 | 0.00755 | 0.9791 |
| `Bs_clear_edge` | MAIN | tissue | 0.01947 | −0.00161 | 0.9795 |
| `F_write_blank_tile` | BG | background | 0.00007 | 0.00310 | 0.9940 |

（負的 s/背景 tile 是最小平方在該 bucket 背景成本≈0 時的雜訊，絕對值都 <0.02 s，不具物理意義。）

**臂模型**：MAIN 0.8016 s/組織 tile、BG 0.3244 s/組織 tile，**BG/MAIN = 0.405**——與
`bottleneck-list.md` 在 match24 量到的 0.470 同量級，MAIN 仍是關鍵路徑，BG 臂還有 ~59% 餘裕。

### 3.2 Set A — 尺寸 sweep（固定組成 65.9%）

| run | tile | GP | 背景 | 組織 | wall | Phase D | Set B 模型預測 | 誤差 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| a11 | 121 | 0.076 | 80 | 41 | 39.8 s | 0.7 s | 34.4 s | +15.9% |
| a21 | 441 | 0.268 | 291 | 150 | 124.6 s | 2.8 s | 125.6 s | −0.8% |
| b_real | 576 | 0.349 | 380 | 196 | 164.7 s | 3.6 s | 164.1 s | +0.3% |
| a32 | 1,024 | 0.617 | 675 | 349 | 280.8 s | 6.4 s | 292.2 s | −3.9% |
| a64 | 4,096 | 2.441 | 2,700 | 1,396 | 1,230.6 s | 25.1 s | 1,168.7 s | +5.3% |
| a96 | 9,216 | 5.474 | 6,074 | 3,142 | 2,723.5 s | 186.5 s | 2,630.2 s | +3.5% |

**每 tile 速率跨尺寸穩定——doc 45 §1.5 的可遷移性前提成立。** 扣掉 Phase D 後的
「core / 組織 tile」：0.953（121）、0.812、0.822、0.786、0.864、0.808 秒。排除 121 tile 那點
（2.5 s 的 model init 佔了 39.8 s 的 6%），**21× 尺寸範圍內只有 ±5% 變動**。只在 576 tile 上擬合的
Set B 係數，外推到其他尺寸誤差在 −3.9%~+5.3%。

### 3.3 Set D + Set A — Phase D 的 page-cache 轉折（**計畫沒預期到會這麼乾淨**）

Set D 用 `stitch_probe.py` 的合成 scratch（doc 45 §4 原提議做 4×/9× 的實體大 TIFF；
`stitch_probe` 以 hard link 重複真實 overlay tile，同樣的曲線、不用寫 ~400 GB）：

| GP | 1.04 | 4.08 | 16.22 | 20.94 | 41.88 | 83.75 |
| --- | --: | --: | --: | --: | --: | --: |
| Phase D | 11.2 s | 44.7 s | 182.6 s | 241.8 s | 556.7 s | 1,240.9 s |
| s/GP | 10.80 | 10.94 | 11.26 | 11.55 | 13.29 | 14.82 |

cached regime 擬合：**Phase D = 10.135 × GP^1.0665，R² = 0.98981**——到 84 GP 幾乎是線性。

**但 hard link 讓來源常駐 page cache。** 把它和真實跑批對照：

| 情境 | 幾何 | Phase D | s/GP |
| --- | --- | --: | --: |
| Set D 探針（hard link，cache 常駐） | 141818×114366，27,565 tile，16.22 GP | 182.6 s | 11.26 |
| **round 13 實跑（真實相異 tile，冷讀）** | **同一幾何** | **987.8 s** | **60.90** |
| | | | **5.41× 差距** |

這是**同一幾何的對照**，比 doc 39 §4.4 用不同尺度的 spike 估出的 15.4 vs 116.6 s/GP 更乾淨。
它也解釋了 Set A 的 a96 異常：5.474 GP、9,216 個真實相異 tile、~18 GB scratch，實測 34.07 s/GP，
是 cached 曲線的 **3.0×**——不是面積造成的超線性，而是同一個 cache 轉折在較小尺寸就到達，因為
資料是真的。

**結論：Phase D 的正確模型是分段的——scratch 裝得進 page cache 時 ~10.1 s/GP × GP^1.07，
裝不進時約 5× 之。轉折點由主機可用 RAM 決定，不由玻片大小決定**（本機 62 GB，轉折落在
scratch 8 GB 與 18 GB 之間）。

### 3.4 Phase A 獨立曲線（doc 45 §2.2 的空白，首次量測）

| crop | tile | GP | Phase A | s/tile |
| --- | --: | --: | --: | --: |
| a11 | 121 | 0.076 | 5.74 s | 0.04746 |
| a21 | 441 | 0.268 | 19.58 s | 0.04439 |
| b_real | 576 | 0.349 | 22.01 s | 0.03821 |
| a32 | 1,024 | 0.617 | 40.86 s | 0.03990 |
| a64 | 4,096 | 2.441 | 156.75 s | 0.03827 |
| a96 | 9,216 | 5.474 | 322.02 s | 0.03494 |

```
Phase A = 0.06268 x n_tiles^0.9364      R² = 0.9995
```

**指數 0.936——略為次線性**，每 tile 成本隨規模緩慢下降（0.0475 → 0.0349 s），與 Phase D 的
超線性行為相反。這是本專案第一次有 Phase A 的 scaling 曲線（doc 45 §2.2 列為空白）。

外推到 35,700 tile：**約 1,149 s ≈ 19.1 分鐘**——**如果它是一個獨立的序列階段**。

**但在出貨組態下它不是**：Set F 的 `phaseA_precut_s` 實測 **0.03 s**。doc 18 §4.2 的
Phase A 串流化，從未在整片尺度被量化過：**實測價值約 19 分鐘，佔 `workers=1` wall 的 10.6%、
佔 `workers=4` wall 的 21.6%**——後者尤其可觀，因為 Phase A 是序列的，不會隨 workers 縮放。

這也修正了 doc 45 §5.4 的模型公式：`estimate = ... + Phase_A(total_GP) + Phase_D(total_GP)`
在串流組態下會**重複計算 Phase A**。`scripts/eta_estimate.py` 因此沒有獨立的 Phase A 項。

### 3.5 Set C — workers scaling（四點擬合，取代兩點外推）

| anchor | tile | w1 | w2 | w3 | w4 | e2e 加速 | core 加速 | Amdahl serial 比例 | R² |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| b_real | 576 | 164.7 s | 110.2 s | 94.8 s | 91.0 s | 1.81× | 1.84× | 36.8% | 0.9938 |
| a64 | 4,096 | 1,230.6 s | 691.1 s | 559.6 s | 499.7 s | 2.46× | 2.54× | 17.1% | 0.9947 |
| **full slide** | 35,700 | 10,883.4 s | — | — | 5,321.4 s | **2.05×** | **2.36×** | (23.0%) | n=2 |

**serial 比例不是常數**——576 tile 時 36.8%，4,096 tile 時 17.1%。doc 38/39 的 Amdahl 帳是在單一
尺度上用兩點外推、並把 parallel portion 當成 pipeline 的固定性質；**它不是**。ETA 模型若用單一
`speedup(workers)` 常數，兩端必有一端錯。

**doc 45 §4 Set C 要求的交叉印證失敗（並已查明原因）**：

| anchor | Phase A + Phase D + init | Amdahl 擬合的 serial | 比值 |
| --- | --: | --: | --: |
| b_real | 0.04 + 3.60 + 2.52 = 6.16 s | 62.6 s | **10.2×** |
| a64 | 0.03 + 25.10 + 3.52 = 28.64 s | 229.8 s | **8.0×** |

擬合的 serial 項比架構上所有序列階段加總還大 **8–10 倍**。依 doc 45 的指示不直接採信擬合值：
原因是 **`workers>1` 不會讓 GPU 吞吐變成 4 倍**——只有一張 RTX 5090，四個 worker 行程仍然把
kernel 排到同一個裝置上，Amdahl 歸給「serial」的絕大部分其實是 **GPU 裝置競爭**，不是
Phase A/D/init。這與 round 14 從另一端獨立量到的現象一致（doc 43 §3：Cellpose `_from_device`
每次呼叫從 38.7 ms 升到 101.8 ms＝2.63×，工作量完全相同）。

**因此 doc 38/39 推估的「~2.5×–2.8× 組成上限」不是 `workers=4` 的實際天花板**；實測是隨規模變動的
1.81×（576）→ 2.46×（4,096）→ 2.05×（全片，被 Phase D 拖累，core 本身 2.36×）。

### 3.6 Set E — 誤差棒（本專案第一次）

固定 b_real（576 tile，65.97% 背景），n=3：

| workers | 三次 wall | 平均 | 標準差 | **CV** |
| --: | --- | --: | --: | --: |
| 1 | 164.7 / 162.6 / 165.5 s | 164.2 s | 1.48 s | **0.90%** |
| 4 | 91.0 / 91.5 / 91.9 s | 91.5 s | 0.45 s | **0.49%** |

**run-to-run 變異低於 1%。** 兩個推論：

1. ETA 的信賴區間由**迴歸殘差**（6.5%）主導，不是執行雜訊——區間是模型準確度的陳述，不是抖動。
2. **回頭看，本專案很多 n=1 的歷史結論其實是可信的**。round 12 的 Phase D「1,239.2 → 1,889.8 s」
   （+52%）與 round 11 未解釋的 530 s 殘差都曾被標註「可能只是變異」；在中型錨點 0.9% 的 CV 下，
   run-to-run 雜訊**不可能**解釋 52% 的變動，那些是真實效應。
   **限制**：此 CV 量在 576 tile；變異是否隨規模放大**仍未知**（Set F 每個 worker 數只跑一次，
   本輪無法回答，見 §7）。

### 3.7 Set F — 正確畫布的全片驗證（held-out）

`workers=1` 與 `workers=4` 各一次，35,700 tile / 20.938 GP，出貨組態
（`--stream-precut`、`tifffile` stitch backend、prefetch 開）：

| | wall | Phase D | core | Phase D 佔比 | peak RSS | 正確性 |
| --- | --: | --: | --: | --: | --: | --- |
| `workers=1` | **10,883.4 s = 3.023 h** | 1,332.1 s | 9,551.3 s | 12.2% | 13.66 GB | 352,881 cells |
| `workers=4` | **5,321.4 s = 1.478 h** | 1,282.0 s | 4,039.4 s | **24.1%** | 13.24 GB | 352,874 cells（−0.002%） |

- 背景 tile 實測 23,533/35,700 = **65.92%**，與 `core_mask_map.py` 獨立測量完全一致。
- 正確性 veto **通過**（−0.002%，與 round 8/12/13 同量級的 GPU 非確定性）。
- **peak RSS 13.66 GB**——對比 round 8 的 61.13 GB（−77.6%）、round 13 的 16.95 GB（−19.4%）。
  README.md:174–182 仍寫「full-slide 需 64 GB RAM／實測約 60 GB peak RSS」，**這個數字已經過時三輪**。
- Phase D 在 `workers=4` 佔 24.1% wall——重現 round 12 的結構（tile 平行臂本身縮放良好，
  端到端被序列的 Phase D 拖住）。

---

## 4. 模型與 held-out 驗證

模型（`scripts/eta_estimate.py`，doc 45 §6.3 的參考實作）：

```
wall = core(n_tissue, n_bg) / speedup(workers, scale) + phase_d(n_tiles) + init
core    = 0.7298 x n_tissue + 0.0460 x n_bg
phase_d = 0.001783 x n_tiles^1.209          (擬合於 Set A，R² = 0.778)
speedup = 1/(f + (1-f)/w)，f 依 log(tile 數) 在量測錨點間內插，範圍外夾住不外推
init    = 2.52 s
```

**沒有獨立的 Phase A 項**（§3.4）。Set F **完全沒有參與任何係數的擬合**。

| | 分項 | 預測 | 實測 | 誤差 |
| --- | --- | --: | --: | --: |
| `workers=1` | core | 9,962.8 s | 9,551.3 s | **+4.3%** |
| | Phase D | 569.6 s | 1,332.1 s | **−57.2%** |
| | **wall** | 10,534.9 s | 10,883.4 s | −3.2% |
| `workers=4` | core | 3,770.7 s | 4,039.4 s | −6.7% |
| | Phase D | 569.6 s | 1,282.0 s | −55.6% |
| | **wall** | 4,342.8 s | 5,321.4 s | **−18.4%** |

**診斷（doc 45 §5.5：不調整模型去遷就驗證點，而是指認失準的子模型）**：

1. **core 子模型健全**：外推到 3.9 倍於擬合尺度、且是擬合時沒看過的組成，誤差 +4.3%/−6.7%。
2. **Phase D 子模型失準 ~56%**。原因明確：它擬合在 Set A 上，而 Set A 最大的一點（9,216 tile）
   **正好落在 page-cache 轉折上**，轉折之後沒有任何資料。用一條穿過轉折的冪律再外推 3.9 倍，
   必然低估——R² 只有 0.778 就是當場可見的警訊。
3. **`workers=1` 的 −3.2% 是假象**：+4.3% 的 core 誤差與 −57% 的 Phase D 誤差恰好相消。
   `workers=4` 時 core 縮小、Phase D 不變，相消消失，真實的 −18.4% 才顯現。
   **把 −3.2% 當成模型準確度會是錯的。**

**要修好 Phase D 需要 9,216–35,700 tile 之間、冷讀 regime 的量測**，本輪沒有編列。
在那之前，玻片尺度的 ETA 在 Phase D 這一項帶有已知的 −56% 偏誤。

---

## 5. 與最舊資料的比較（使用者要求的重點）

### 5.1 同一批 crop 檔案：round 1 對今日程式碼（like-for-like）

round 1 control（git `96a28ba`、`config_hash db2b7e6a`、完全序列 `run_batch`）用的
`small`/`med`/`large` crop 檔案**至今仍在磁碟上**，因此這是同樣的位元組、同樣的工作量：

| anchor | tile | round 1 `w1` | **round 15 `w1`** | 加速 | **round 15 `w4`** | 對 round 1 加速 |
| --- | --: | --: | --: | --: | --: | --: |
| small | 25 | 54.3 s | **22.5 s** | **2.41×** | 26.3 s | 2.06× |
| med | 121 | 243.3 s | **85.2 s** | **2.85×** | 54.6 s | **4.46×** |
| large | 441 | 848.0 s | **294.6 s** | **2.88×** | **143.2 s** | **5.92×** |

三點附註：

1. **`bottleneck-list.md` 的招牌數字 302.7 s 在今日程式碼上被確認**——實測 294.6 s，快 2.7%，
   在雜訊內。那個數字自 round 6 以來從未重新驗證過。
2. **小於約 100 tile 時多行程會反效果**：25 tile 時 `workers=4`（26.3 s）比 `workers=1`
   （22.5 s）**慢**，因為 4 份 model init 攤不掉。UI 送出小 ROI 時不應該用 `workers=4` 的加速比
   去報 ETA——模型透過 Amdahl 的 serial 項會自然反映，但這是實際會被使用者看到的效應。
3. **加速比隨規模成長**（2.06× → 4.46× → 5.92×），因為固定成本被攤薄。**任何單一的「我們快了 N 倍」
   如果不講清楚是在哪個規模量的，就沒有意義。**

### 5.2 整片：從第一次投影到今天的實測

| 里程碑 | 畫布 | tile | workers | wall | 對 round 1 投影 |
| --- | --- | --: | --: | --: | --: |
| **round 1 投影**（`9.4 + 1.903 s/tile`，從未實跑） | （假設 35,700） | 35,700 | 1 | **67,946 s = 18.87 h** | 1.00× |
| round 8 首次實跑 | 舊（錯） | 27,565 | 1 | 13,762.5 s = 3.82 h | — |
| round 8 | 舊（錯） | 27,565 | 4 | 6,211 s = 1.73 h | — |
| round 11（乾淨對照） | 舊（錯） | 27,565 | 1 | 10,666.1 s = 2.96 h | — |
| round 13（最佳舊數字） | 舊（錯） | 27,565 | 4 | 4,877.4 s = 1.355 h | — |
| **round 15（正確畫布）** | **正確** | **35,700** | **1** | **10,883.4 s = 3.023 h** | **6.24×** |
| **round 15（正確畫布）** | **正確** | **35,700** | **4** | **5,321.4 s = 1.478 h** | **12.77×** |

**重要：round 8–13 那幾列的 tile 數比 round 15 少 29.5%**，因為它們跑在錯誤的（未配準、被裁到交集的）
畫布上。直接把 4,877.4 s 和 5,321.4 s 相比會**低估**今日的效能。

**把 6.24×／12.77× 拆成兩個獨立來源**（避免把功勞算錯地方）：

| 來源 | 計算 | 倍數 |
| --- | --- | --: |
| **程式碼改進** | round 1 與 round 15 用**同一種投影法**（441 crop 的 blended 速率 × 35,700）：67,946 s → 23,849 s | **2.85×** |
| **組成修正** | 真實玻片是 65.9% 背景，不是 441 crop 的 14.1%；實測 23,849 s → 10,883 s | **2.19×** |
| **多行程** | `workers=1` → `workers=4` | **2.05×** |
| **合計（w4）** | 67,946 s → 5,321.4 s | **12.77×** |

也就是說：**約 2.85× 來自真正的工程優化，約 2.19× 來自「當年用組織密集的 crop 去外推整片，本來就
高估了」**（doc 24/25 已發現這件事，此處給出整片尺度的量化），另外 2.05× 來自多行程。
**只報 12.77× 而不做這個拆解會誇大工程貢獻。**

### 5.3 每組織 tile 的成本（跨畫布唯一可比的口徑）

兩個畫布的**組織 tile 數幾乎相同**（12,184 vs 12,167，−0.14%），因此「每組織 tile 秒數」是唯一
不受畫布錯誤影響的比較口徑：

| 里程碑 | wall | 組織 tile | s/組織 tile |
| --- | --: | --: | --: |
| round 8 `w1` | 13,762.5 s | 12,184 | 1.1296 |
| round 11 `w1` | 10,666.1 s | 12,184 | 0.8754 |
| **round 15 `w1`** | 10,883.4 s | 12,167 | **0.8945** |
| round 8 `w4` | 6,211.0 s | 12,184 | 0.5098 |
| round 13 `w4` | 4,877.4 s | 12,184 | 0.4003 |
| **round 15 `w4`** | 5,321.4 s | 12,167 | **0.4374** |

以這個口徑看，round 15 比 round 11/13 **略慢**（+2.2% / +9.3%）。這**不是回歸**，而是正確畫布
多出的 8,152 個背景 tile 的真實成本（8,152 × 0.046 s ≈ 375 s）加上更大畫布的 Phase D
（20.94 GP vs 16.22 GP）。**在正確的輸入上，這就是真實的成本。**

### 5.4 記憶體

| 里程碑 | peak RSS（全片） |
| --- | --: |
| round 8 | 61.13 GB |
| round 13 | 16.95 GB |
| **round 15** | **13.66 GB（−77.6% vs round 8）** |

**README.md:174–182 的「64 GB RAM／實測約 60 GB peak RSS」已過時三輪**，應更新為約 16 GB。

---

## 6. 產出

| 產出 | 位置 |
| --- | --- |
| 係數檔（doc 45 §6.2） | `/home/taro/r15/coefficients.json` |
| 參考實作（doc 45 §6.3） | `scripts/eta_estimate.py`（`fit` / `estimate` 兩個子命令） |
| Phase A 探針（doc 45 §4 Set D 授權的薄 wrapper） | `scripts/phase_a_probe.py` |
| 原始量測資料 | `docs/hybrid-pipeline/measurement/_metrics_r15/` |
| campaign driver | `/home/taro/r15/*.sh` |
| Set F 完整輸出 | `/home/taro/r15_fullwsi/` |

`scripts/eta_estimate.py` **不碰** `backend/api/`、`backend/schemas/`、`frontend/`——
接進 UI 是下一份落地文件的工作，且需要先過 frontend-backend-boundary 審查（doc 45 §2.2）。

用法：

```bash
.venv/bin/python scripts/eta_estimate.py fit \
    --metrics-dir /home/taro/r15/_metrics --out /home/taro/r15/coefficients.json

.venv/bin/python scripts/eta_estimate.py estimate \
    --coefficients /home/taro/r15/coefficients.json \
    --roi-px 156222x134028 --background-share 0.6592 --workers 4
```

---

## 7. 限制（doc 45 §7 原樣延續，並依本輪結果更新）

1. **只有一張真實玻片**（未解決，且**比 doc 45 寫的更嚴重**）。doc 45 假設背景比例 55.8%；
   實測正確畫布是 65.92%。**同一張玻片、換一個（正確的）畫布定義，背景比例就差 10 個百分點。**
   跨病人、跨染色批次的變異**完全未知**，而模型對這個輸入很敏感（組織塊貴 15.9 倍）。
   模型上線後必須用新案例持續回測。
2. **Phase D 子模型已知失準 −56%**（§4）。玻片尺度的 ETA 目前帶有這個偏誤。修正需要
   9,216–35,700 tile 之間、冷讀 regime 的 Phase D 量測。
3. **Set D 的大尺寸曲線是 cached regime**，不是臨床規模實測。hard link 讓來源常駐 page cache；
   形狀（指數 1.07）可以遷移，**每 GP 的絕對成本不行**（同幾何實測差 5.41×）。
   已驗證的真實規模上限是 **20.94 GP**；41.88 GP 與 83.75 GP 兩點只有 cached 數字。
4. **變異數只在 576 tile 量過**（CV <1%）。**變異是否隨規模放大仍然未知**——Set F 每個 worker 數
   只跑一次。doc 45 §4 Set F 提到「時數允許各兩次」以互相印證，本輪時數不允許。
5. **單一硬體型號**：全部係數量在 RTX 5090 / CUDA 13.0 / torch 2.11.0+cu130。§3.5 顯示
   `workers>1` 的行為由 **GPU 裝置競爭**主導，因此換卡（尤其不同 SM 數量或 VRAM）之後
   **workers 曲線會整條改變**，不只是每 tile 秒數。模型不可跨硬體套用。
6. **本輪未改動任何 pipeline/API/UI 程式碼**。§1.4 修的三處都在 `scripts/` 的量測工具內，
   且都是「已經壞掉、無法執行」的 bitrot，不影響任何被量測的行為（`config_hash` 全程 `d2ccc46b`，
   與 round 13/14 相同）。
7. **`workers=4` 的 VRAM 風險未在本輪重測**。round 14 記錄過 `expandable_segments:True` 下
   1/10 次 crop 規模跑批會 OOM；本輪 Set F 的 `workers=4` 單次成功，但 n=1 不足以說這個風險消失。

---

## 8. 仍然開放的項目

1. **Phase D 冷讀 regime 的量測**（§4 診斷出的唯一失準子模型）——最高優先，且是把 ETA 用在整片
   尺度前的前提。
2. **`perf_measure.py:769` 的死路徑**（§1.4 #4）——非 `--stream-precut` 分支呼叫已不存在的函式。
3. **README.md 的硬體需求已過時三輪**（§5.4）——RSS 從 60 GB 降到 13.66 GB。
4. **逐 tile 進度 callback** 仍卡在框架層（`BackgroundTasks` 無法回報進度），是把本模型從
   「送出前靜態估計」升級成「執行中動態校正」的前提——doc 45 §0 已列，不在本輪範圍。
5. **跨玻片驗證**——只有一張玻片，見 §7.1。
