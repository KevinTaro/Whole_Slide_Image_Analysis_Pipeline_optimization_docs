# 目前狀態 vs. 原始基線 — 實測比較

> **目的。** 在 **目前 HEAD** 上重新量測混合管線，並與保留的 **串列「dumb-version」控制基線** 做 apples-to-apples 比較，接著交叉參照 [`bottleneck-list.md`](./bottleneck-list.md)，逐項說明 **哪些已完成**、**哪些已不再構成問題**、以及 **哪些仍值得最佳化**。本文只做量測，遵守方案 §1.2 / playbook 的紀律；文中不提出新修復建議。
>
> **原始（控制組）：** git `96a28ba`、串列 `run_batch`（一次只處理一個地磚）、`_metrics/`（2026-07-07）。已保留，不覆蓋。
> **目前：** git `0e27b20`、config_hash `db2b7e6a`、`_metrics_current/`（2026-07-11）。同一台機器（RTX 5090 / CUDA 13.0 / torch 2.10.0+cu130）、同一套非侵入式 `scripts/perf_measure.py` 採集器、同一批真實 WSI 裁切（`test_picture/_roi_crops/{med,large}`）、`--gpu-dmon`、`--workers 8`，沒有 py-spy。規模：**中 121 地磚（11×11）**、**大 441 地磚（21×21）**。
>
> **2026-07-22 已量測第三輪**（Cellpose 4.2.1.1 / `cpdino` 更換，git `f95a573`）。下方
> §1–§7 原封保留為 2026-07-11 的紀錄；**要看目前狀態請直接跳到 §8**。重點：大/441 wall
> **707.4 → 573.7 s（−18.9%）**，全 WSI 外推 **~15.5 小時 → ~12.6 小時**。

## 兩次提交之間落地了什麼（程式碼，`backend/algorithms/hybrid/`）

| commit | 變更 | 瓶頸項目 |
|---|---|---|
| `010308f` | **① 方案 (b)**：把每地磚工作切成 **GPU 前段（主執行緒）** + **CPU 後段（背景執行緒）**，在 `run_batch` 中做深度 1 的重疊 | ① |
| `feedfbd` | segmentation 的滑動視窗接縫拼接（+ VALIS 文件） | — |
| `119ad73` | `draw_tile_seam_edges`（視覺 QA 輔助） | — |
| `9e618d3` | **②③ 元件級**：`@lru_cache` morphology `disk()` footprint（唯讀、共享） | ② (a) |

完整落地紀錄：[`pipeline-overlap-result.md`](./pipeline-overlap-result.md)（①）、[`detect-all-dots-result.md`](./detect-all-dots-result.md)（②③）。

---

## 1. 總覽 — 端到端錨點（控制組 vs 目前）

| 規模 | 地磚數 | **原始 wall** | **目前 wall** | Δ wall | 原始 s/地磚 | 目前 s/地磚 |
|---|--:|--:|--:|--:|--:|--:|
| 中 | 121 | 243.3 s | **208.2 s** | **−14.5%** | 2.011 | 1.720 |
| 大  | 441 | 848.0 s | **707.4 s** | **−16.6%** | 1.923 | 1.604 |

任何規模都沒有出現負最佳化；規模愈大改善略增（一次性初始化被攤提）。目前大規模（707 s）略低於 `00f2c91` 的中間量測值（724.7 s）——一致，落在熱噪音範圍內，後續 commit 沒有回退。

### GPU 使用率 — 機制（nvidia-smi dmon，每秒）

| 規模 | 指標 | 原始 | 目前 | 變化 |
|---|---|--:|--:|---|
| 大  | **idle_frac**（sm==0） | 0.459 | **0.190** | **idle 約減少 59%** |
| 大  | 平均 SM % | 28.3 | **32.9** | +4.6 pt |
| 大  | busy≥50% frac | 0.252 | 0.292 | +0.04 |
| 大  | mem-ctrl 平均 % | 18.1 | 21.5 | +3.4 pt |
| 中 | **idle_frac** | 0.477 | **0.163** | **idle 約減少 66%** |
| 中 | 平均 SM % | 28.4 | **35.9** | +7.5 pt |

**這就是①的全部故事。** 串列管線在主執行緒做 CPU 工作（dot detection、PNG encode、GC）時，讓 GPU 閒置了約 46–48% 的 wall。現在兩階段重疊把這些 CPU 後段放到背景執行緒，去蓋在下一個地磚的 GPU 前段上，所以 GPU 閒置降到約 16–19%，wall 也下降約 15–17%。

---

## 2. 時間去了哪裡 — 階段位移（大 441，佔該次執行 wall 比例）

| bucket | 原始 s（% wall） | 目前 s（% wall） | 解讀 |
|---|--:|--:|---|
| **GPU 前段**（UNet + 2× Cellpose fwd） | 386.0 (45.5%) | **578.9 (81.8%)** | **現在是關鍵路徑** |
| `detect_all_dots`（B3） | 260.1 (30.7%) | 239.4 (33.8%) | 絕對成本差不多，**現在已被重疊/遮蔽** |
| PNG encode+write（B2） | 76.8 (9.1%) | 78.5 (11.1%) | 絕對成本差不多，**現在已被重疊/遮蔽** |
| `gc.collect`（B4） | 36.3 (4.3%) | 36.3 (5.1%) | 不變（現在放在背景階段） |
| precut A | 20.6 (2.4%) | 20.4 (2.9%) | 不變（獨立階段，未重疊） |
| stitch D | 5.1 (0.6%) | 5.1 (0.7%) | 不變（串列、分析後） |

CPU 項目的「% of wall」上升了，但絕對秒數沒變——因為分母（wall）縮小了。 在重疊架構下，「self-time ÷ wall」**不再**代表關鍵路徑占比：這些階段和 GPU 前段同時執行，它們的 *有效* 端到端 Amdahl 上限接近 1.0。

> 採集器註解（與 `detect-all-dots-result.md` §5 相同）：在 ① 重構之後，M2 與 M3b 都會呼叫 `segment_windowed`，所以 `B1_m3b_cellpose` bucket 現在會把 **兩次** Cellpose 前向都算進去；上面的「GPU 前段」= `B1_unet_coremask` + `B1_m3b_cellpose`（+ 舊版的 `B1_m2_cellpose`，若有）。`B_process_precut_tile_TOTAL` 讀成 0，是因為那個函式已拆成 gpu/cpu chunk 子函式，舊 wrapper 不再包住整段。端到端 wall 與 GPU idle_frac（主要比較指標）不受影響。

---

## 3. 與 [`bottleneck-list.md`](./bottleneck-list.md) 對照的瓶頸狀態

| # | 項目 | 類別 | 原始 | **目前狀態** |
|---|---|---|---|---|
| ① | GPU 未充分利用 / 串列管線 | 3+6 | idle 45.9%，GPU 前段 45.5% wall | **DONE（方案 b）**。idle→19.0%，wall −16.6%；GPU 前段現在成為穩定關鍵路徑（81.8% wall）。 |
| ② | `detect_all_dots`（M3 dots） | 1+3 | 30.7% wall，上限 1.44 | **RESOLVED「免費」完成。** 在背景階段執行，**完全被遮蔽**於 GPU 前段之後（每地磚 0.58 s ≪ GPU 前段 1.33 s）。有效上限約 1.0。`disk()` 上移已 bit-exact 落地；process/regionprops 重寫 **停損**。 |
| ③ | PNG encode+write | 5 | 9.1% wall，上限 1.10 | **HIDDEN。** 現在在背景 CPU 階段，與 GPU 前段重疊；雖然有 ~11% self-time，但對關鍵路徑幾乎沒有貢獻。未獨立處理。 |
| ④ | 每地磚 `gc.collect` | 4+6 | 4.3%，上限 1.04 | **不變**（5.1% self-time）。現在在背景階段 → 部分被遮蔽。低於 Amdahl floor，只記錄。 |
| ⑤ | precut A / stitch D | 5(+3) | 2.4% / 0.6% | **不變。** 兩者都是重疊 B loop 之外的獨立串列階段；仍低於 floor。D 是目前唯一剩下的串列但理論上可重疊候選。 |
| ⑥ | model init（一次性） | 6 | 0.37% @441 | **不變**，規模放大後可忽略。 |
| ⑦ | API / job 層 | 6 | ~10⁻⁷ | **不變**，可忽略。 |

**總結：** 三個深度記錄候選項（①②③）都已被處理——①是直接修正，②③則是因為被重疊遮蔽。關鍵路徑已經從「GPU 閒置 + 分散 CPU 工作」轉移到 **GPU 前向本身**。

---

## 4. 記憶體（有界 — 這個說法仍成立）

| 規模 | peak RSS 原始 → 目前 | VRAM（dmon fb）原始 → 目前 |
|---|---|---|
| 中 | 3.07 → **3.06 GB** | 5159 → **5159 MB** |
| 大 | 4.04 → **3.94 GB** | 5159 → **5159 MB** |

VRAM 的實體峰值在兩個規模、兩個版本都**固定在 5.16 GB / 32 GB**——沒有隨地磚數增加。雙緩衝重疊只保留兩個地磚在飛行中（RSS 在目前大規模甚至還略低）。「記憶體有界，不隨地磚線性成長」這個不變量在重構後仍成立。

> 一個異常，**不是真正回退**：資源採樣器在目前中規模執行中記到 `cuda_alloc_peak 22.2 GB`，但同一輪的 driver `dmon` framebuffer 峰值是 **5159 MB**——這在物理上不可能超過。22 GB 是沒有其他證據支持的 `torch.cuda.memory_allocated()` 採樣假象（CPROFILE 基線也曾出現同類尖峰）。實體 VRAM 仍然有界；請把 torch-allocated 欄位視為不可靠，改看 `dmon fb` 的 VRAM。

---

## 5. 全 WSI 重新投影（35,700 地磚 @ 1024px）

對兩個目前錨點做線性擬合：`wall ≈ 19.4 s + 1.560 s/地磚`。

| | s/地磚 斜率 | 全 WSI（35,700） | 備註 |
|---|--:|--:|---|
| 原始（控制組） | 1.903 | ~18.9 h | 上界 |
| **目前** | **1.560** | **~15.5 h** | 上界，規模下 **−18%** |

這仍然只是 **上界**，原因和之前一樣：裁切內容組織密度很高（~85%）；真正的玻片大多是白色背景，空核心地磚能走更快路徑。相對的 −18% 才是有意義的數字。

---

## 6. 目前還值得最佳化的項目（已排序，只做量測分類）

依目前關鍵路徑占比排序；**只做分類，不在此設計修復方案。**

1. **GPU 前段 — M1 UNet + 2× Cellpose forwards（441 地磚時佔 wall 81.8%）。首要。**
   既然重疊已把 CPU 工作遮蔽掉，這個就是 wall 本身。這是 **① 的下一階段**（CuPy / GPU-kernel 層級），範圍已在
   [`../11-gpu-pipeline-stage2-plan.md`](../11-gpu-pipeline-stage2-plan.md) 說明，也在 [`gil-contention-diag.md`](./gil-contention-diag.md) 標註（Cellpose `dynamics.py` / SAM `get_rel_pos` 是 kernel-launch-bound，已經是 GPU resident）。類別 1（演算法）+ 2（硬體/kernel）。

2. **殘餘 GPU idle ~16–19%。** 重疊是深度 1，而且 `detect_all_dots` 用的是 joblib `prefer='threads'`，所以它的 CPU 工作會和主執行緒 torch 在 GIL 上競爭——把重疊上限卡在理論 ~50% 以下。可動槓桿：更深的 pipeline，或把 CPU 階段換成 process backend。**但** 在 CPU 階段已被遮蔽的情況下，端到端價值有限——只有在 GPU 前段（槓桿 1）縮小、重新把它露出來之後才值得做。類別 3（並行）。

3. **`cellpose_batch_size` — 死配置（Class 7）。尚未啟用的結構性槓桿。**
   VRAM 目前只有 **5.16 / 32 GB** 被用掉，還剩約 27 GB headroom。這個欄位還沒接到 `Config`（`getattr(..., 16)` fallback），因此無法做跨地磚 / 更大 GPU batch。要先把它接好，之後的 batch-size sweep 才有意義。這是前置條件，不是本身就能直接改善 wall 的建議。（屬於配置正確性，而不是 wall 最佳化提案。）

4. **每地磚 `gc.collect`（5.1%）** 與 **串列 stitch D（0.7%）**——兩者都低於 Amdahl floor；已記錄，不深入分析。D 是唯一剩下「串列但原則上可重疊」的階段，但在一個約 15 小時的執行中還不到 1%。

**停損註記（不變）：** ②③④ 已被遮蔽或低於 floor——在槓桿 1 縮小 GPU 前段之前，追它們不會降低 wall。下一輪應該只看槓桿 1。

---

## 7. 重現

```bash
cd /data/taro_Projects/tsgh
ROI="$PWD/backend/algorithms/hybrid/test_picture/_roi_crops"
MC=docs/hybrid-pipeline/measurement/_metrics_current
for s in "med:medium_121tile:medium" "large:large_441tile:large"; do
  IFS=: read pre lab out <<<"$s"
  .venv/bin/python scripts/perf_measure.py \
    --ihc "$ROI/${pre}_ihc.tiff" --dish "$ROI/${pre}_dish.tiff" \
    --output docs/hybrid-pipeline/measurement/runs_current/$out \
    --label $lab --workers 8 --gpu-dmon --metrics-dir "$MC"
done
.venv/bin/python scripts/aggregate_report.py "$MC"
.venv/bin/python scripts/resource_analyze.py "$MC"
# 與保留的控制組基線 docs/hybrid-pipeline/measurement/_metrics/ 比較
```

原始工件（保留）：控制組在 `_metrics/`、目前在 `_metrics_current/`（`*_timings.json`、`*_agg.json`、`*_resource_summary.json`、`*_gpu_dmon.txt`、`*_resource.csv`、`*_stdout.log`）。管線輸出則分別在 `runs/`（控制組）與 `runs_current/`（目前）。

---

# 8. 第三輪（2026-07-22）— Cellpose 4.2.1.1 / `cpdino` 更換

> **上方 §1–§7 原封保留為 2026-07-11 的紀錄。** 本節新增第三輪量測。
> **第三輪：** git `f95a573`、config_hash `db2b7e6a`（**未變** — 參數完全相同）、
> `_metrics_cellpose421/`（2026-07-22）。同一台機器（RTX 5090、driver 580.159.03）、
> 同一套 `scripts/perf_measure.py`、同一批裁切（`test_picture/_roi_crops/{med,large}`）、
> `--gpu-dmon`、`--workers 8`、無 py-spy。啟動前已確認 GPU 空閒（89 MiB、0%、無其他
> compute process）— 遵守文件 13 §0 的共用伺服器紀律。環境戳記 + `pip freeze` +
> 模型權重 SHA256 見 `_metrics_cellpose421/env_stamp.txt`。

## 8.1 第二輪到第三輪之間變了什麼

**管線程式碼：幾乎沒變。** `git diff 0e27b20..f95a573 -- backend/algorithms/hybrid/` 只有兩處：
刪除已無用的 `_process_one_chunk` 同步包裝、以及 `_stitch_overlay_slide` 的 `tiffsave` 拿掉
明寫的 `tile_width/tile_height=256`。雙臂重疊、`run_batch`、M0–M4 與所有閾值皆未動。

**模型與環境：變很多。** 組員把 Cellpose 換成更快的 DINOv3 backbone 版本，且 venv 依
`uv.lock` 重建：

| | 第 1–2 輪（`_metrics`，2026-07-07） | 第 3 輪（2026-07-22） |
|---|---|---|
| cellpose | 4.0.8（`cpsam`、fp32） | **4.2.1.1**（`cpdino` / DINOv3、預設 `use_bfloat16=True`） |
| M2 + M3b 權重 | 先前訓練 | **重新訓練**（各 343 MB；SHA256 見 `env_stamp.txt`） |
| torch / torchvision | 2.10.0+cu130 / 0.25.0 | 2.11.0+cu130 / 0.26.0 |
| numpy | 2.2.6 | **1.26.4** ↓ |
| scikit-image | 0.25.2 | **0.24.0** ↓ |
| pyvips | 3.1.1 | **2.2.3** ↓ |
| opencv | 4.9/4.10/4.12 混雜 | **4.8.1.78** ↓ |
| timm / scipy | 1.0.22 / 1.16.3 | 1.0.26 / 1.17.1 |

> **這是「一包」變更，不是單變數對照實驗。** 下方的歸因只做到「逐 timing bucket」這個
> 程度，這也是資料誠實能支撐的極限。另外 `_metrics_current/`（第 2 輪）**沒有 `pip freeze`**，
> 所以第 2 輪的實際套件版本並無紀錄，只能「假設」與第 1 輪相同。往後每一輪都應在 timings
> 旁留一份 `pip freeze`；第 3 輪已照做。

## 8.2 總覽 — 三輪錨點

| 規模 | 地磚數 | 控制組（r1） | 重疊（r2） | **第 3 輪** | Δ r2→r3 | Δ r1→r3 |
|---|--:|--:|--:|--:|--:|--:|
| 中 | 121 | 243.3 s | 208.2 s | **166.6 s** | **−20.0%** | **−31.6%** |
| 大 | 441 | 848.0 s | 707.4 s | **573.7 s** | **−18.9%** | **−32.3%** |
| 大 s/地磚 | | 1.923 | 1.604 | **1.301** | −18.9% | −32.3% |

任何規模都沒有負最佳化；兩個規模差距在 1.1 個百分點內，所以這是 regime-stable 的結果，
不是小樣本假象。

### 改善來自哪裡 — 完全侷限在 Cellpose forward

| bucket（大/441） | r1 控制組 | r2 重疊 | **r3** | Δ r2→r3 |
|---|--:|--:|--:|--:|
| **Cellpose forwards**（824 次呼叫） | 372.6 s | 564.5 s | **430.9 s** | **−23.7%** |
| — **每次呼叫** | 0.4521 s | 0.6850 s | **0.5229 s** | **−23.7%** |
| UNet++ forward（441 次） | 13.45 s | 14.41 s | **13.68 s** | −5.1%（持平） |
| **VRAM 峰值**（dmon fb） | 5159 MB | 5159 MB | **2787 MB** | **−46.0%** |
| `cuda_reserved` 峰值 | 4.68 GB | 4.68 GB | **2.19 GB** | −53% |

UNet++ 持平，其他非 Cellpose 的 bucket 不是持平就是變差，所以 −18.9% 的 wall 可以明確歸因
到 Cellpose forward。−46% VRAM 與 4.2 版文件所述的 `use_bfloat16=True` 預設一致。

> **§2/① 的「B1 絕對秒數異常成長」問題大致自行解決了。** B1 總量走勢為
> 386.0（r1）→ 578.9（r2）→ **444.6 s**（r3）。當初無法解釋的 +192.9 s 多數隨著套件升級而
> 回退，因此 §2 想追的那個未解問題已可退場。harness 重標記的但書不變：`B1_m3b_cellpose`
> 仍然把**兩個** Cellpose forward 加總（n=824 = 2 × 412），`B_process_precut_tile_TOTAL` 仍為 0。

## 8.3 GPU 反而變得更閒 — 這才是真正的重點

| 指標（dmon，每秒） | r1 控制組 | r2 重疊 | **r3** |
|---|--:|--:|--:|
| 大 **idle_frac**（sm==0） | 0.459 | 0.190 | **0.370** |
| 大 平均 SM % | 28.3 | 32.9 | **16.6** |
| 大 busy≥50 比例 | 0.252 | 0.292 | **0.130** |
| 中 **idle_frac** | 0.477 | 0.163 | **0.293** |
| 中 平均 SM % | 28.4 | 35.9 | **17.4** |

閒置率大約翻倍、平均 SM 砍半。**這不是退步** — 同時間 wall 掉了 18.9%。它的意思是 GPU
現在更早做完自己那份工作，然後花更多時間等 CPU 臂。管線正從 GPU-bound 漂移向 CPU-bound。

### 雙臂模型（現在真正決定 wall 的東西）

`run_batch`（`hybrid_pipeline.py:766-810`）跑兩條並行的臂 — 成員是**讀原始碼確認**的，
不是從 timing 推測：

- **MAIN**（主執行緒）：3 個 GPU forward、`_read_rgb`、M1 overlay、`clear_slide_edge_cells`、
  `build_all_positive_results`、`enlarge_cell_instances`、**`gc.collect` + `empty_cache`**。
- **BG**（單一背景執行緒）：`detect_all_dots`、merge、PNG/TIFF 編碼、`render_overlay_image`、
  per-cell crops、`filter_and_absolutize`。

`wall ≈ max(MAIN, BG) + outside`（outside = 預切 A + 拼接 D + init）：

| 大/441 | r1 控制組 | r2 重疊 | **r3** |
|---|--:|--:|--:|
| MAIN 臂 | 467.9 s | 672.0 s | **538.3 s** |
| BG 臂 | 350.6 s | 333.2 s | **387.3 s** |
| **BG / MAIN** | 0.749 | **0.496** | **0.719** |
| 重疊外 outside | — | — | 28.0 s |
| 模型檢核：`max(arm)+outside` | — | — | 566.3 s vs **實測 573.7 s**（−1.3%） |

**本輪最重要的一個數字：MAIN 臂只要再減 151.0 s（= GPU forward 的 34.0%），
背景 CPU 臂就會變成新的關鍵路徑。** 中規模的餘裕更薄，只有 36.8%。第 2 輪還有約 100%
的餘裕（BG/MAIN 0.496），第 3 輪只剩 39%。**再來一次 Cellpose 等級的改善，② 和 ③ 就會重新浮出水面。**

## 8.4 雙臂模型下的 Amdahl 天花板（大/441，第 3 輪錨點 573.7 s）

在重疊架構下，「self-time ÷ wall」**不是**關鍵路徑佔比。天花板以
`wall / (max(MAIN', BG') + outside)` 計算：

| 槓桿 | self-time | % wall | 臂 | **歸零後天花板** |
|---|--:|--:|---|--:|
| GPU forwards → 0 | 444.6 s | 77.5% | MAIN | **1.382x** |
| `gc.collect` → 0 | 36.4 s | 6.3% | MAIN | **1.083x** |
| ⑧ CPU 前處理移出 MAIN | 28.4 s | 5.0% | MAIN | ~1.05x |
| 預切 A + 拼接 D → 0 | 25.6 s | 4.5% | outside | ~1.05x |
| `detect_all_dots` → 0 | 292.9 s | 51.1% | BG | **1.013x** |
| PNG 編碼 → 0 | 78.7 s | 13.7% | BG | **1.013x** |
| 兩臂同時 → 0（理論值） | — | — | — | 4.69x |

**在相信任何「% of wall」數字之前，先看這張表。** `detect_all_dots` 顯示佔 51.1% 的 wall，
實際價值只有 **1.3%**；`gc.collect` 顯示 6.3%，實際價值 **8.3%** — 看起來小 8 倍，實際大 6 倍。
這就是為什麼對「位於關鍵臂上的項目」要暫停那條「<10% 就丟掉」的硬性下限
（見 `bottleneck-list.md` 修訂後的停損規則）：在約 12.6 小時的全 WSI 執行中，
單是 `gc.collect` 就是 **約 44 分鐘** 的 wall。

## 8.5 哪些變差了

| bucket（大/441） | r2 | **r3** | Δ |
|---|--:|--:|--:|
| `detect_all_dots` | 239.4 s | **292.9 s** | **+22.3%** |
| — 每顆細胞 | 18.5 ms | **22.3 ms** | +20.2% |
| `enlarge_cell_instances` | 18.29 s | 19.60 s | +7.2% |
| `build_all_positive_results` | 7.23 s | 8.81 s | +21.9% |
| PNG 編碼 | 78.54 s | 78.72 s | 持平 |
| `gc.collect` | 36.33 s | 36.39 s | 持平 |
| 預切 A / 拼接 D | 20.39 / 5.08 s | 20.51 / 5.05 s | 持平 |

細胞數只增加 **+1.8%**（12,922 → 13,150），所以 `detect_all_dots` 是**每顆細胞慢約 20%**。
主要假說是 **scikit-image 0.25.2→0.24.0 / numpy 2.2.6→1.26.4 / opencv→4.8.1.78 的降版**
（該路徑正是建立在這幾個函式庫上的 LAB + H-morphology）；競爭假說則是重新訓練的權重造成
細胞幾何改變。兩者是一起變的，**成因尚未隔離** — 記為 `bottleneck-list.md` 的 ⑨。
這些今天都不會影響 wall（在 BG 臂、天花板 1.013x）；它重要的原因是它在吃掉 §8.3 算出的 39% 餘裕。

值得注意的是：儘管 **pyvips 3.1.1 → 2.2.3** 降版，預切 A 與拼接 D **沒有**變差。

## 8.6 正確性 — 本輪並未維持不變

方案 §1.3 把精度列在範圍外，但前兩輪是可逐位元比對的、這一輪不是，所以在把 −18.9%
當成「免費」之前必須先標註：

| | r1 控制組 | r2 重疊 | **r3** |
|---|--:|--:|--:|
| 細胞數，中 | 3559 | 3558 | **3647**（+2.5%） |
| 細胞數，大 | 12919 | 12922 | **13150**（+1.8%） |
| 大：成功/略過地磚 | 379 / 62 | 379 / 62 | **378 / 63** |

r1 與 r2 的差距在 ±3 顆細胞內（GPU 非決定性的噪音底線）；r3 不是。重新訓練的權重產生了
不同的分割結果，且有一個地磚從成功翻轉為略過。
**這是需要由病理醫師另行驗證的模型品質變更 — 效能結果並不能說明新的 mask 是否更好。**

## 8.7 記憶體

| 規模 | RSS 峰值 r2 → r3 | VRAM（dmon fb）r2 → r3 |
|---|---|---|
| 中 | 3.06 → **3.09 GB** | 5159 → **2785 MB** |
| 大 | 3.94 → **3.90 GB** | 5159 → **2787 MB** |

記憶體有界的不變量仍然成立。VRAM 現在有 **32 GB 中約 29.8 GB 閒置**，這強化（而非改變）
既有的 `cellpose_batch_size` 無效設定發現：`hybrid_pipeline.py:206,218` 仍在對一個沒有該
欄位的 `Config` 呼叫 `getattr(config, "cellpose_batch_size", 16)`。

## 8.8 全 WSI 重新外推（35,700 地磚 @ 1024px）

以第 3 輪兩個錨點做線性擬合：`wall ≈ 12.6 s + 1.2722 s/地磚`。

| 輪次 | s/地磚斜率 | 全 WSI（35,700） | Δ vs 控制組 |
|---|--:|--:|--:|
| r1 控制組 | 1.903 | ~18.9 小時 | — |
| r2 重疊 | 1.560 | ~15.5 小時 | −18% |
| **r3 cpdino** | **1.2722** | **~12.6 小時** | **−33%** |

理由不變，這仍是**上界**（裁切的組織密度約 85%；真實玻片多為白色背景，空核心地磚走快速路徑）。
有意義的數字是相對的 −33%。

## 8.9 回答本輪量測要回答的問題

- **換 Cellpose 換到多少？** **大/441 wall −18.9%、中/121 −20.0%** — 707.4 → 573.7 s，
  全 WSI 外推從 ~15.5 小時降到 ~12.6 小時。機制是 **每次 Cellpose forward −23.7%**
  （0.6850 → 0.5229 s）加上 **VRAM −46%**。
- **相對於最初的控制組基線**，累計改善為 **−32.3%**（848.0 → 573.7 s），其中 ① 重疊貢獻
  −16.6%、本次更換貢獻 −18.9%。
- **GPU 還是瓶頸嗎？** 是，但只剩一點點 — 餘裕只有 34%。GPU 現在有 37% 的時間是閒置的，
  而 CPU 後段已達關鍵臂的 72%。
- **有哪些以前不值得、現在值得優化？** `gc.collect`（天花板 1.083x、全 WSI 約 44 分鐘）、
  卡在 MAIN 臂上的 CPU 前處理（⑧）、以及預切 A + 拼接 D — 這些以 self-time 看都「不到 10%」，
  但都在關鍵路徑上，而且都可以壓到接近零。詳見 `bottleneck-list.md` →
  「第 3 輪後重新排序的優先順序」。

## 8.10 重現第 3 輪

```bash
cd /data/taro_Projects/tsgh
uv sync                                  # 環境需與 uv.lock 一致；`uv sync --frozen --dry-run` 應回報無變更
ROI="$PWD/backend/algorithms/hybrid/test_picture/_roi_crops"
MC=docs/hybrid-pipeline/measurement/_metrics_cellpose421
nvidia-smi                               # 共用伺服器：先確認 GPU 空閒（文件 13 §0）
for s in "med:medium_121tile:medium" "large:large_441tile:large"; do
  IFS=: read pre lab out <<<"$s"
  .venv/bin/python scripts/perf_measure.py \
    --ihc "$ROI/${pre}_ihc.tiff" --dish "$ROI/${pre}_dish.tiff" \
    --output docs/hybrid-pipeline/measurement/runs_cellpose421/$out \
    --label $lab --workers 8 --gpu-dmon --metrics-dir "$MC"
done
.venv/bin/python scripts/aggregate_report.py "$MC"
.venv/bin/python scripts/resource_analyze.py "$MC"
uv pip freeze > "$MC/pip_freeze_actual.txt"   # 每一輪都要做
```

原始工件（保留，三輪並列）：`_metrics/`（r1 控制組）、`_metrics_current/`（r2 重疊）、
`_metrics_cellpose421/`（r3）。管線輸出在 `runs/`、`runs_current/`、`runs_cellpose421/`。

---

# 第 4–8 輪摘要（2026-07-22 至 2026-07-27）— 完整英文版見 [`current-status-comparison.md`](./current-status-comparison.md)

> 本檔中文版的逐輪紀錄停在第 3 輪；第 4–8 輪只在此處做濃縮摘要。完整細節請讀英文版
> [`current-status-comparison.md`](./current-status-comparison.md) §9–§11 與各輪原始文件
> （18、21、23、25、27）。

| 輪次 | 變更 | large/441 wall（或整片） | Δ | 來源 |
|---|---|--:|--:|---|
| r4 | ⑧ 純 CPU 前處理搬出 MAIN 臂 + 預切 A 串流化 | 480.3 s | −16.3% | doc 18 |
| r5 | 跨 tile 多行程落地（採用 `workers=3`） | 156.1 s | −67.5% | doc 21 |
| r5b | 找到 worker 數上限；建議改用 `workers=6` | 123.3 s | −21.0% | doc 21 §4.7 |
| r6 | 拔掉 `detect_all_dots` 的 joblib 派工（`dot_detect_n_jobs=1`）；`workers=1` 免費拿到 1.60x；建議值下修到 `workers=4`/`5`（`workers≥6` 有 OOM 風險） | `workers=1`：302.7 s；`workers=4`：128.8 s | −37.5%（`workers=1`） | doc 23 |
| r7 | **組成前提推翻**：玻片實測 55.8% 背景（非先前假設的 39%），Phase D 拼接實測 322.7 s（外推值的 1.8 倍、超線性）。全 WSI 估算改為 **~2.6h（`workers=1`）/ ~1.25h（`workers=4`）** | — | — | doc 25 |
| **r8** | **本專案第一次跑完整真實 WSI**（見下） | `workers=1`：3.82h；`workers=4`：1.73h | 見下 | doc 27 |

累計（`workers=1`）：**848.0 s → 302.7 s（−64.3%）**（裁切錨點），六輪皆未把多行程放行到生產，
直到第 8 輪的完整 WSI 驗證通過。

## 第 8 輪（2026-07-27）— 真實整片實測，三個「已關閉」的裁切數字重新開啟

第 8 輪跑了本專案第一次真正的完整 WSI 端到端驗證（`workers=1` 與 `workers=4` 各一次，配準階段
每個 modality 畫布尺寸不同，`PrecutStream` 會 fail-fast，需先用 `--conform` 裁到交集
99.86%）：

| | `workers=1` | `workers=4` | 第 7 輪推算值 |
|---|--:|--:|--:|
| 端到端 wall | **13,762 s = 3.82h** | **6,211 s = 1.73h** | 2.6h / 1.25h |
| 相對推算值 | **+47%** | **+38%** | — |
| 實測加速比 | — | **2.216x** | 2.06x–2.17x（預測區間內） |
| `report.csv` 列數 | 356,255 | 356,221（**−0.01%**，正確性投票通過） | — |
| 峰值 RSS | 61.13 GB | 61.67 GB | 裁切規模僅 ~4 GB |
| 峰值 GPU | 2,739 MB | 30,439 MB（32,607 的 93.3%） | — |

組成預測準到只差一格 tile，超出推算的原因**完全在逐 tile 速率，不在組成**——三個先前用裁切
量過、以為已關閉的成本在整片規模重新浮現：`gc.collect`（16.1% wall，非近乎 0）、tile read
（17.2%，非 1.22%）、Phase D 拼接（8.6%/19.3%，非 3.5%/7.3%）。**19-open-backlog item 7 已關閉，
`workers>1` 放行 gate 已滿足**——但帶一個 VRAM 但書：`workers=4` 峰值用掉卡的 93.3%（餘裕僅
~2.2 GB）。**建議放行 `workers=4`**，32 GB VRAM 應視為硬底線，`workers≥5` 仍應等
`workers≥6` 的 allocator 氣球問題根治後再開放。完整記錄見
[`../27-remaining-work-implementation.md`](../27-remaining-work-implementation.md) §6。
