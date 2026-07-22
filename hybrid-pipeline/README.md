# backend/algorithms/hybrid Pipeline — 優化交接文檔

> 這份資料夾是給「接手效能優化」的人看的。目標：**30 分鐘內掌握瓶頸與現況**。
> 事實（效能數字、版本、設計決策）皆已由前一輪研究驗證，本文檔只做組織與導覽。
>
> ⚠️ **路徑遷移提醒**：本資料夾內較舊的文件（01–08 與本檔）撰寫時程式碼還在
> `cell_mask/hybrid/`；UI Phase 1 目錄重構後已搬到 **`backend/algorithms/hybrid/`**
> （`cell_mask/` 現在只剩舊的 `dish_mask/`/`unet_mask/` 訓練資料，`cell_mask/hybrid/`
> 與 `cell_mask/docs/6_30_report.md` 已不存在）。09 之後的文件已經在用新路徑；讀到
> 舊文件裡的 `cell_mask/hybrid/...` 一律換算成 `backend/algorithms/hybrid/...`。

## TL;DR（現況，2026-07-22 第 4 輪量測後）

1. **定位**：IHC(Her2)＋DISH 雙染色 WSI 的逐 tile 分析流水線 —— 讀圖 → 疊合 → Cellpose 分割 → 逐細胞 HER2/CEP17 點位計數與擴增判定 → 縫回整圖 → 匯出 CSV/視覺化。
2. **最大瓶頸**：**Cellpose GPU 前向**（UNet++ core mask + M2/M3b 兩次 Cellpose）仍是 #1，但已歷經四輪優化：① 單 process 兩段式 pipeline/overlap（GPU 主執行緒 + CPU 背景執行緒重疊）、Cellpose 4.0.8→4.2.1.1（DINOv3 `cpdino` backbone + bfloat16）、`gc.freeze()`、CPU 前處理搬離 MAIN 臂、precut 串流化。累計 large/441-tile 錨點從 **848.0 s → 480.3 s（−43.4%）**；全 WSI 估算從 ~18.9h 壓到 **~10.5h**（皆為上限估計，見下）。目前 MAIN（GPU）臂只剩 **~15.9%** 餘裕可縮，縮更多就會把瓶頸換到 BG（CPU：`detect_all_dots` + PNG 編碼）臂。**最新排名、餘裕與下一步全部以 [measurement/bottleneck-list.md](./measurement/bottleneck-list.md) 為準**，本段只是導覽用摘要，過幾輪就會再過時。
3. **GPU 限制（先讀）**：機器是 **RTX 5090（Blackwell, sm_120）**，**只能跑 cu130 的 torch**（實測 `torch 2.11.0+cu130`）。不能任意降 torch 版本，也不能用舊 CUDA wheel —— 改動依賴前務必看 [06-versions-dependencies.md](./06-versions-dependencies.md)。
4. **還沒做完什麼**：見 [../BACKLOG.md](../BACKLOG.md) —— 未做/半做的優化項目、文件與 code 的已知落差，集中列在那一份，不散在各文件裡找。

## 要接手優化，先讀這份（導覽表）

| 你想做的事 | 先讀 | 重點 |
| --- | --- | --- |
| 搞懂整條流水線怎麼跑、資料怎麼流（**注意：巢狀迴圈的敘述是 pre-precut 架構**，見 09 的落差表） | [01-architecture-dataflow.md](./01-architecture-dataflow.md) | M0→M1→M2→M3→M0→M4；batch 迴圈 vs chunk 迴圈的巢狀；記憶體生命週期 |
| 查某個模塊的輸入輸出/函式/瓶頸（同上，部分模塊已在後續重構中改名/搬動） | [02-module-reference.md](./02-module-reference.md) | 逐檔詳解 + I/O shape/dtype + 關鍵演算法 + 實測套件版本 |
| 看實測效能數字、找瓶頸排名（**舊數字，2026-06-29，pre-refactor**） | [03-benchmarks-bottlenecks.md](./03-benchmarks-bottlenecks.md) | perf_report.html 的 cProfile Top、GPU/CPU/VRAM 利用率 |
| 決定優先改哪裡、投報比（**舊數字，pre-refactor**） | [04-optimization-roadmap.md](./04-optimization-roadmap.md) | 短/中/長期優化 + 設計決策的深層原因 + 已棄用嘗試 |
| 為什麼要重新規劃量測、新架構跟舊文件差在哪 | [08-problem-analysis.md](./08-problem-analysis.md) → [09-measurement-analysis-plan.md](./09-measurement-analysis-plan.md) | 只定位問題/規劃量測，不下解法結論 |
| **看目前 HEAD 的實測瓶頸排名（最新一輪，取代 03/04 的舊數字）** | [measurement/bottleneck-list.md](./measurement/bottleneck-list.md) | RTX 5090 實測，25/121/441-tile 真實 WSI；**已滾動 4 輪**（control→overlap→Cellpose 4.2.1.1→⑧/precut），每輪都保留在同一份文件裡，不要只看第一段 |
| 三輪（含前兩輪）的完整 before/after 對照 | [measurement/current-status-comparison.md](./measurement/current-status-comparison.md) | control vs overlap vs round-3，逐項目狀態表 |
| 針對 GPU 序列瓶頸（①）的下一步方案設計 | [10-gpu-serial-pipeline-plan.md](./10-gpu-serial-pipeline-plan.md) | 承接 bottleneck-list.md①，playbook Analyze→Plan→Choose，含驗收標準 |
| 方案 (b) 的實作與量測結果（-18.5%，idle 0.494→0.154） | [measurement/pipeline-overlap-result.md](./measurement/pipeline-overlap-result.md) | 驗收 doc 10 §5 標準；含「為何沒到理論上限」的根因分析 |
| stage 2：detect_all_dots 與主執行緒 GIL 競爭的下一步 | [11-gpu-pipeline-stage2-plan.md](./11-gpu-pipeline-stage2-plan.md) | 承接 pipeline-overlap-result.md 的剩餘 15.4% idle，先診斷再決定要不要動 joblib 後端 |
| stage 2 (a) py-spy GIL 診斷 + (d) gc ablation（**推翻 doc 11 假設 → 停損**） | [measurement/gil-contention-diag.md](./measurement/gil-contention-diag.md) | 拖 GIL 的 81% 是主執行緒（gc.collect 33.6% + Cellpose/UNet 重建）；(b) 與 (d) gc 重定位皆實測對 idle 無效 → 本輪停損、保持方案 (b) 現狀 |
| `detect_all_dots`（②）優化方案設計 | [12-detect-all-dots-optimization-plan.md](./12-detect-all-dots-optimization-plan.md) | Analyze→Plan→Choose；結論被自己的落地結果推翻，見下一列 |
| ② 落地結果：**被 GPU 前段完全遮住，不用改** | [measurement/detect-all-dots-result.md](./measurement/detect-all-dots-result.md) | `disk()` 去重複已 bit-exact 落地；process/regionprops 重寫全部停損 |
| Cellpose 換模型後的下一步優先序（round 3 之後） | [13-next-optimization-plan.md](./13-next-optimization-plan.md) | 雙臂模型（MAIN/BG）取代 self-time%排序；`gc.collect` 頻率、⑧ CPU 前處理搬離 MAIN、precut/stitch overlap、`cellpose_batch_size` 接線排序 |
| `gc.collect` 頻率優化設計 + 落地 + 結果 | [14-gc-collect-frequency-plan.md](./14-gc-collect-frequency-plan.md) → [15-...-implementation.md](./15-gc-collect-frequency-implementation.md) → [16-...-result.md](./16-gc-collect-frequency-result.md) | **DONE**：不是原計畫的 batching，是 `gc.freeze()`；1.069–1.077x，符合預測天花板 |
| GPU starvation 還剩什麼、三個「大槓桿」值不值得做之前要先關掉什麼 | [17-gpu-starvation-prerequisites-plan.md](./17-gpu-starvation-prerequisites-plan.md) | ⑧ 搬離 MAIN、precut/stitch overlap、`cellpose_batch_size` 接線 —— 三個都要先做，才能把跨 tile multiprocessing 這類大改動的天花板量準 |
| 上面那份的落地與量測（**round 4，目前最新**） | [18-gpu-starvation-prerequisites-implementation.md](./18-gpu-starvation-prerequisites-implementation.md) | ⑧ 搬離 MAIN（-8.0%/-5.0%）、precut 串流化（再 -3.1%/-3.8%）、`cellpose_batch_size` 接線但掃描結果**負向**（tile size 下已無可加速空間）；跨 tile multiprocessing 是唯一還沒動、天花板最高的槓桿 |
| 在本地把它跑起來、驗證沒改壞（**部分路徑已過時，見檔內提醒**） | [05-dev-testing-guide.md](./05-dev-testing-guide.md) | `cp config_example.py config.py` → 編 config → CLI 跑；回歸基準 |
| 動依賴前確認相容性（**部分版本已在後續輪更新，見 measurement 文件**） | [06-versions-dependencies.md](./06-versions-dependencies.md) | venv 實測版本 vs requirements vs pyproject 三方衝突；Blackwell 限制 |
| 踩到怪坑（config 跑不動、幻影檔案…） | [07-gotchas-appendix.md](./07-gotchas-appendix.md) | codegraph 過期、config gitignore、未接線參數、失聯 spec docs（G2 已修復，見檔內更新） |
| 看視覺化的流程與迴圈關係（**畫的是 pre-precut 架構**） | [pipeline-flow.html](./pipeline-flow.html) | M0 複雜迴圈、StitchAccumulator 去重、記憶體生命週期圖 |
| **還沒做完 / 半做的優化與文件落差總表** | [../BACKLOG.md](../BACKLOG.md) | 跨 hybrid-pipeline 與 UI 兩個資料夾，唯一的「還欠什麼」清單 |

**建議閱讀順序**：先 [measurement/bottleneck-list.md](./measurement/bottleneck-list.md)（現況與四輪演進，含每輪的「Re-sorted priority」）→ 依你要接手的項目挑對應的 10–18 → 需要架構/模塊細節再翻 `01/02`（記得套用上面的路徑遷移提醒）。`03/04` 只有歷史對照價值，不要當現況用。

## 參考地圖（既有資源，本文檔不重寫）

| 資源 | 位置 | 內容 |
| --- | --- | --- |
| 模塊級規格（權威、簡潔） | `backend/algorithms/hybrid/CLAUDE.md` | 各 M 的職責、Invariants、跑法 —— **與本文檔並存，不衝突** |
| 效能報表（實測，**pre-refactor 舊版**） | `docs/hybrid-pipeline/measurement/perf_report.html` | 3 tiles cProfile、資源 timeline、WSI 估算、優化建議；已被 `measurement/bottleneck-list.md` 取代為現況來源 |
| 核配對 v3 說明（視覺） | `docs/algo/elastic_matching_v3_explainer.html` | M3 DISH 核配對與計數邏輯（⚠️ 描述的是舊「以核為中心」變體，與現行 `m3_elastic_matching.py`（以細胞為中心）不同，見 07 的版本漂移提醒 —— **以 code 為準**） |
| Sliding-window 縫合說明 | `docs/algo/sliding-window-seam-stitch.html` | 重疊視窗分割 + 接縫縫合的原始設計 |
| 前後端傳圖片 vs 傳路徑的取捨討論 | `docs/algo/frontend_backend_split_architecture.html` | API 邊界設計討論；對應 UI 護欄「邊界一律檔案路徑 + JSON」（`docs/UI/04-guardrails-red-lines.md`） |
| UI 交接文檔（**Phase 0 共識文件，Phase 1–3 已完工，現況見該資料夾 10/11**） | [`../UI/README.md`](../UI/README.md) | FastAPI+React+pywebview；`backend/api/*`、`frontend/` 已存在且能跑，不再是「未來式」 |

> 已移除的死連結：`cell_mask/docs/6_30_report.md`（該檔已在 `44fb5ba` commit 被清理，不要再找）。

## 一句話現況

管線功能完整、單 tile 正確性有回歸基準（單塊輸入 bit-identical 於 pre-M0 路徑）；
**記憶體已靠分塊/串流架構壓下來**（RSS 隨累積細胞數成長、非隨 tile 數線性成長，VRAM 於 441-tile 錨點穩定在 ~2.8GB/32GB）；
GPU 前向已歷經四輪優化（overlap pipeline + Cellpose 換模型 + `gc.freeze()` + CPU 前處理搬離 MAIN 臂 + precut 串流化），
剩下的**速度**問題集中在：GPU 前向本身（唯一還有大天花板的槓桿是跨 tile multiprocessing，但需解 fork-under-CUDA、風險最高）、
以及 BG 臂（`detect_all_dots` + PNG 編碼）在 MAIN 臂繼續縮小後遲早被重新暴露。全 WSI 全圖單線程估算已從 ~18.9h 壓到 **~10.5h**（皆為 upper bound，真實切片組織密度較低會更快；且**從未在真正完整 WSI 上實測過**，見 [../BACKLOG.md](../BACKLOG.md)）。目前完整、按優先序排列的待辦清單見 [../BACKLOG.md](../BACKLOG.md)。
