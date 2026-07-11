# 目前狀態 vs. 原始基線 — 實測比較

> **目的。** 在 **目前 HEAD** 上重新量測混合管線，並與保留的 **串列「dumb-version」控制基線** 做 apples-to-apples 比較，接著交叉參照 [`bottleneck-list.md`](./bottleneck-list.md)，逐項說明 **哪些已完成**、**哪些已不再構成問題**、以及 **哪些仍值得最佳化**。本文只做量測，遵守方案 §1.2 / playbook 的紀律；文中不提出新修復建議。
>
> **原始（控制組）：** git `96a28ba`、串列 `run_batch`（一次只處理一個地磚）、`_metrics/`（2026-07-07）。已保留，不覆蓋。
> **目前：** git `0e27b20`、config_hash `db2b7e6a`、`_metrics_current/`（2026-07-11）。同一台機器（RTX 5090 / CUDA 13.0 / torch 2.10.0+cu130）、同一套非侵入式 `scripts/perf_measure.py` 採集器、同一批真實 WSI 裁切（`test_picture/_roi_crops/{med,large}`）、`--gpu-dmon`、`--workers 8`，沒有 py-spy。規模：**中 121 地磚（11×11）**、**大 441 地磚（21×21）**。

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
