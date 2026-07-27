# backend/algorithms/hybrid Pipeline — 優化交接文檔

> 這份資料夾是給「接手效能優化」的人看的。目標：**30 分鐘內掌握瓶頸與現況**。
> 事實（效能數字、版本、設計決策）皆已由前一輪研究驗證，本文檔只做組織與導覽。
>
> ⚠️ **路徑遷移提醒**：本資料夾內較舊的文件（01–08 與本檔）撰寫時程式碼還在
> `cell_mask/hybrid/`；UI Phase 1 目錄重構後已搬到 **`backend/algorithms/hybrid/`**
> （`cell_mask/` 現在只剩舊的 `dish_mask/`/`unet_mask/` 訓練資料，`cell_mask/hybrid/`
> 與 `cell_mask/docs/6_30_report.md` 已不存在）。09 之後的文件已經在用新路徑；讀到
> 舊文件裡的 `cell_mask/hybrid/...` 一律換算成 `backend/algorithms/hybrid/...`。

## TL;DR（現況，2026-07-27 第 8 輪量測後）

0. **第 8 輪重點：完成了本專案第一次真正的完整 WSI 端到端跑批**（`workers=1` 與 `workers=4` 各一次，皆通過正確性投票），`workers>1` 的生產放行 gate（19-open-backlog item 7）**已關閉**——但帶一個 VRAM 但書，不是速度但書：`workers=4` 在真實整片上量到 **93.3% 卡用量（30,439 / 32,607 MB，餘裕 ~2.2 GB）**。同一輪也發現**三個先前用裁切量出、以為已經關閉的數字在整片規模下不成立**：`gc.collect`（`gc.freeze()` 的效益在整片規模消失，退回 16.1% wall）、tile read（1.22% 停損值在整片是 17.2%）、Phase D 縫合（3.5%/7.3% 在整片是 8.6%/19.3%）。細節見 [`27-remaining-work-implementation.md`](./27-remaining-work-implementation.md)。
1. **定位**：IHC(Her2)＋DISH 雙染色 WSI 的逐 tile 分析流水線 —— 讀圖 → 疊合 → Cellpose 分割 → 逐細胞 HER2/CEP17 點位計數與擴增判定 → 縫回整圖 → 匯出 CSV/視覺化。
2. **單 process（`workers=1`，生產預設）瓶頸**：**Cellpose GPU 前向**（UNet++ core mask + M2/M3b 兩次 Cellpose）仍是單 process 下的 #1，已歷經五輪優化：① 單 process 兩段式 pipeline/overlap（GPU 主執行緒 + CPU 背景執行緒重疊）、Cellpose 4.0.8→4.2.1.1（DINOv3 `cpdino` backbone + bfloat16）、`gc.freeze()`、CPU 前處理搬離 MAIN 臂、precut 串流化、**round 6：拔掉 `detect_all_dots` 的 joblib 平行派工**（`dot_detect_n_jobs=1`，一行 config 改動）。第 6 輪這個改動實測 **1.60x**（large/441 錨點 484.7 s → 302.7 s）—— 原因不在 CPU 那段變快，而是背景執行緒少了 19 條搶 GIL 的 thread，讓 MAIN 臂的 Cellpose 前向自己快了 43.4%；也順帶坐實了 round 3 記錄過、當時只是假設的 +192.9 s GIL 競爭異常。累計 large/441-tile 錨點從 **848.0 s → 302.7 s（−64.3%）**；單 process 全 WSI 估算（校正後 grid 為 **27,565 tile**，不是先前假設的 35,700）從 ~8.1h 壓到 ~5.3h；**round 7 再把這個估算換成量出來的組成**——用 pipeline 自己的判準（UNet++ core mask 全空）逐格量完整張玻片，**背景佔 55.8%、組織只佔 44.2%**（round 6 由亮度縮圖估的「~61% 組織」是錯的），加上實測的 Phase D 縫合成本後，單 process 估算降到 **~2.6h**（`workers=4` 約 **~1.25h**；仍是逐族群速率的線性外推，見下）。MAIN（GPU）臂餘裕因此變寬（②③ 比以往任何一輪都更難被重新暴露），單 process 這條路線目前的下一個槓桿仍是跨 tile 多行程。
3. **第 5／6 輪跨 tile 多行程（`workers` 參數）**。round 5（doc 20 規劃 → doc 21 落地）建置了 `run_batch(..., workers=N)`，round 6 在 `dot_detect_n_jobs=1` 落地後重掃了 worker 數（doc 23 §6）：**建議值從 round 5b 的 `workers=6` 下修到 `workers=4`（無人看管長跑）/`workers=5`（可接受重跑成本時）** —— `workers=4` 只比實測地板慢 8.0%，而 `workers≥6` 這輪新測到 **6 次跑有 2 次因 CUDA allocator 碎片化 OOM**（同一個 24.76 GiB 的整數倍氣球，隨機 victim tile），fail-fast 下會讓整批全毀。`workers=3`（round 5 原推薦）在 large/441 錨點量到 **3.09x**（482.8 s → 156.1 s，效率 103%），`workers=4` 疊加 round 6 改動後量到 **128.8 s**（−6.3% vs 未疊加）；遠超原估的 1.23x–1.7x —— 原估只算了裝置閒置，漏算了 MAIN/BG 兩臂之間的 GIL 競爭（round 6 也證實：`workers≥6` 下這個 GIL 增益已被多行程吃掉，`dot_detect_n_jobs` 改動在高 worker 數下幾乎沒有額外增益，兩者是同一筆帳的兩條路，不會疊加）。正確性投票、fail-fast + 兄弟行程終止皆驗證通過；`workers=1` 仍是預設值，`backend/api/hybrid.py` 的單 tile 請求路徑不受影響。round 7 在**組成對齊真實玻片**的 576-tile 錨點上覆測：`workers=4` 買到 **2.14x**（188.8 s → 88.3 s），且兩臂模型量到 **BG/MAIN = 0.47–0.53**（背景 arm 還有 47–53% 餘裕），這也是 round 7 把所有落在背景 arm 的 GPU 候選（doc 24 的 B/D/E/F）全部以「wall 天花板 1.00x」關掉的直接依據。**round 8（2026-07-27）：完整 WSI 規模驗證已完成，放行 gate 已滿足。** 在真實 27,565-tile 整片上，`workers=1` 3.82 h、`workers=4` 1.73 h（分別比推算值 2.6h/1.25h 多 47%/38%），實測加速比 **2.216x**（落在 round 7 預測的 2.06x–2.17x 區間內），正確性投票通過（356,255 vs 356,221 列，−0.01%）。**建議：放行 `workers=4` 到生產**，但 `workers=4` 在整片上量到 **93.3% 卡用量（30,439 / 32,607 MB，餘裕僅 ~2.2 GB）**，32 GB VRAM 應視為硬底線，`workers≥5` 在 `workers≥6` 的 allocator 氣球問題被根治前不應開放。詳見 [`27-remaining-work-implementation.md`](./27-remaining-work-implementation.md) §5–§6。**最新排名、餘裕與下一步全部以 [measurement/bottleneck-list.md](./measurement/bottleneck-list.md) 為準**，本段只是導覽用摘要，過幾輪就會再過時。
4. **GPU 限制（先讀）**：機器是 **RTX 5090（Blackwell, sm_120）**，**只能跑 cu130 的 torch**（實測 `torch 2.11.0+cu130`）。不能任意降 torch 版本，也不能用舊 CUDA wheel —— 改動依賴前務必看 [06-versions-dependencies.md](./06-versions-dependencies.md)。
5. **還沒做完什麼**：見 [19-open-backlog.md](./19-open-backlog.md) —— 本資料夾範圍內未做/半做的優化項目、待驗證的正確性問題、文件與 code 的已知落差，集中列在那一份，不散在各文件裡找。（跨 hybrid-pipeline 與 UI 兩個資料夾的總表在 [../BACKLOG.md](../BACKLOG.md)。）

## 要接手優化，先讀這份（導覽表）

| 你想做的事 | 先讀 | 重點 |
| --- | --- | --- |
| 搞懂整條流水線怎麼跑、資料怎麼流（**注意：巢狀迴圈的敘述是 pre-precut 架構**，見 09 的落差表） | [01-architecture-dataflow.md](./01-architecture-dataflow.md) | M0→M1→M2→M3→M0→M4；batch 迴圈 vs chunk 迴圈的巢狀；記憶體生命週期 |
| 查某個模塊的輸入輸出/函式/瓶頸（同上，部分模塊已在後續重構中改名/搬動） | [02-module-reference.md](./02-module-reference.md) | 逐檔詳解 + I/O shape/dtype + 關鍵演算法 + 實測套件版本 |
| 看實測效能數字、找瓶頸排名（**舊數字，2026-06-29，pre-refactor**） | [03-benchmarks-bottlenecks.md](./03-benchmarks-bottlenecks.md) | perf_report.html 的 cProfile Top、GPU/CPU/VRAM 利用率 |
| 決定優先改哪裡、投報比（**舊數字，pre-refactor**） | [04-optimization-roadmap.md](./04-optimization-roadmap.md) | 短/中/長期優化 + 設計決策的深層原因 + 已棄用嘗試 |
| 為什麼要重新規劃量測、新架構跟舊文件差在哪 | [08-problem-analysis.md](./08-problem-analysis.md) → [09-measurement-analysis-plan.md](./09-measurement-analysis-plan.md) | 只定位問題/規劃量測，不下解法結論 |
| **看目前 HEAD 的實測瓶頸排名（現況精簡版，取代 03/04 的舊數字）** | [measurement/bottleneck-list.md](./measurement/bottleneck-list.md) | RTX 5090 實測；已精簡為「現況清單」——每個瓶頸一行、標結果與來源文件，不含逐輪敘事；完整 7 輪演進史搬到 [measurement/bottleneck-list-history.md](./measurement/bottleneck-list-history.md) |
| baseline 與現況的完整 before/after 對照 | [measurement/current-status-comparison.md](./measurement/current-status-comparison.md) | 只留 control baseline vs 目前 HEAD 兩欄；逐輪歷史搬到 [measurement/current-status-comparison-history.md](./measurement/current-status-comparison-history.md) |
| 已發現但從未落地／已停損的優化清單（一次性稽核，非常態維護） | [DISCOVERED-NOT-IMPLEMENTED.md](./DISCOVERED-NOT-IMPLEMENTED.md) | 讀完 01–25 全部文件整理出的「討論過但沒改 code」清單，逐項標狀態（open/gated/stop-lossed）與來源 |
| 針對 GPU 序列瓶頸（①）的下一步方案設計 | [10-gpu-serial-pipeline-plan.md](./10-gpu-serial-pipeline-plan.md) | 承接 bottleneck-list.md①，playbook Analyze→Plan→Choose，含驗收標準 |
| 方案 (b) 的實作與量測結果（-18.5%，idle 0.494→0.154） | [measurement/pipeline-overlap-result.md](./measurement/pipeline-overlap-result.md) | 驗收 doc 10 §5 標準；含「為何沒到理論上限」的根因分析 |
| stage 2：detect_all_dots 與主執行緒 GIL 競爭的下一步 | [11-gpu-pipeline-stage2-plan.md](./11-gpu-pipeline-stage2-plan.md) | 承接 pipeline-overlap-result.md 的剩餘 15.4% idle，先診斷再決定要不要動 joblib 後端 |
| stage 2 (a) py-spy GIL 診斷 + (d) gc ablation（**推翻 doc 11 假設 → 停損**） | [measurement/gil-contention-diag.md](./measurement/gil-contention-diag.md) | 拖 GIL 的 81% 是主執行緒（gc.collect 33.6% + Cellpose/UNet 重建）；(b) 與 (d) gc 重定位皆實測對 idle 無效 → 本輪停損、保持方案 (b) 現狀 |
| `detect_all_dots`（②）優化方案設計 | [12-detect-all-dots-optimization-plan.md](./12-detect-all-dots-optimization-plan.md) | Analyze→Plan→Choose；結論被自己的落地結果推翻，見下一列 |
| ② 落地結果：**被 GPU 前段完全遮住，不用改** | [measurement/detect-all-dots-result.md](./measurement/detect-all-dots-result.md) | `disk()` 去重複已 bit-exact 落地；process/regionprops 重寫全部停損 |
| Cellpose 換模型後的下一步優先序（round 3 之後） | [13-next-optimization-plan.md](./13-next-optimization-plan.md) | 雙臂模型（MAIN/BG）取代 self-time%排序；`gc.collect` 頻率、⑧ CPU 前處理搬離 MAIN、precut/stitch overlap、`cellpose_batch_size` 接線排序 |
| `gc.collect` 頻率優化設計 + 落地 + 結果 | [14-gc-collect-frequency-plan.md](./14-gc-collect-frequency-plan.md) → [15-...-implementation.md](./15-gc-collect-frequency-implementation.md) → [16-...-result.md](./16-gc-collect-frequency-result.md) | **DONE**：不是原計畫的 batching，是 `gc.freeze()`；1.069–1.077x，符合預測天花板 |
| GPU starvation 還剩什麼、三個「大槓桿」值不值得做之前要先關掉什麼 | [17-gpu-starvation-prerequisites-plan.md](./17-gpu-starvation-prerequisites-plan.md) | ⑧ 搬離 MAIN、precut/stitch overlap、`cellpose_batch_size` 接線 —— 三個都要先做，才能把跨 tile multiprocessing 這類大改動的天花板量準 |
| 上面那份的落地與量測（round 4） | [18-gpu-starvation-prerequisites-implementation.md](./18-gpu-starvation-prerequisites-implementation.md) | ⑧ 搬離 MAIN（-8.0%/-5.0%）、precut 串流化（再 -3.1%/-3.8%）、`cellpose_batch_size` 接線但掃描結果**負向**（tile size 下已無可加速空間）；跨 tile multiprocessing 是唯一還沒動、天花板最高的槓桿 |
| 跨 tile 多行程的候選方案設計（承接上一列） | [20-cross-tile-multiprocessing-plan.md](./20-cross-tile-multiprocessing-plan.md) | Candidate A–E 設計空間、7 條正確性不變量、實驗順序；**純規劃文件，未動 pipeline code** |
| 下一輪要試什麼、怎麼量、什麼算成功（round 5 之後） | [22-next-optimization-cycle-plan.md](./22-next-optimization-cycle-plan.md) | Track A（多行程調校 / CPU 核心競爭稽核）與 Track B（把 CPU 迴圈搬上 GPU 批次）的候選清單與停損條件；**純規劃文件，未動 pipeline code** |
| 上面那份的落地與量測（round 6） | [23-next-optimization-cycle-implementation.md](./23-next-optimization-cycle-implementation.md) | `detect_all_dots` 的 joblib 平行派工實測**比序列慢 2.77x**，改 `dot_detect_n_jobs=1` 後 `workers=1`（生產預設）**1.60x**（484.7 → 302.7 s）；順帶隔離出 ① 那筆 +192.9 s 的 GIL 異常；Cellpose / UNet++ 跨 tile 批次兩條線皆實測後停損 |
| encode/decode 與逐項迴圈還有什麼能搬上 GPU（純調查，未動 code） | [24-gpu-encode-decode-loop-acceleration-plan.md](./24-gpu-encode-decode-loop-acceleration-plan.md) | Candidate A（Phase D 縫合）/ B（逐塊 debug 影像編碼）/ D·E（`detect_all_dots` 等搬 GPU）/ F（背景塊空白檔）/ G（重複 mkdir）的調查與排序；**純規劃文件** |
| 上面那份的落地與量測（**round 7，目前最新**） | [25-gpu-encode-decode-loop-acceleration-implementation.md](./25-gpu-encode-decode-loop-acceleration-implementation.md) | 逐格量出玻片真實組成（**55.8% 背景**，推翻 round 6 的 ~61% 組織），據此關掉 B/D/E/F（全在餘裕 47–53% 的 BG arm，天花板 1.00x）；Candidate G 做出來、量完 **0.056% wall** 後 revert；Phase D 在真實 16.2 GP 上實測 **322.7 s**（外推值的 1.8 倍且超線性），是唯一還值得動的項目；GPU codec 環境閘門：nvTIFF 無 Python binding、cuCIM 不能寫、CuPy 在本機跑不起來 |
| 上面那份的落地與量測（round 5） | [21-cross-tile-multiprocessing-implementation.md](./21-cross-tile-multiprocessing-implementation.md) | 落地 Candidate D：`run_batch(..., workers=N)`、`spawn` worker pool、動態工作佇列；實測 **3.09x**（`workers=3`）、正確性投票與 fail-fast 皆通過；MPS（Candidate C）與加深 CPU 後段（Candidate A）皆停損；**尚未放行到生產**（卡在完整 WSI 規模驗證） |
| 把 19 / DISCOVERED / bottleneck-list 三份重疊的待辦壓成一份有排序的計畫 | [26-remaining-work-implementation-plan.md](./26-remaining-work-implementation-plan.md) | 把 quickref 的 Discover→Analyze→Plan→Choose 套用在**待辦清單本身**：Tier 0（阻擋性 gate）→ Tier 1（便宜的效能）→ Tier 4（文件漂移）…，並明列「不要再提」的排除清單；**純規劃文件** |
| 上面那份的落地與量測（**round 8，目前最新**） | [27-remaining-work-implementation.md](./27-remaining-work-implementation.md) | 落地 Tier 0.3 斷點續跑（`run_batch(checkpoint=True)`，輸出逐位元相同）、Tier 0.4 `RLIMIT_NOFILE` 護欄、Tier 1.3 worker 內 per-stage 計時（4 桶 → 26 桶）、Tier 0.2 allocator 旋鈕與掃描、Tier 1.1 `tiffsave` 旋鈕消融、Tier 4 七項文件漂移全部關掉（含新增 46 個測試）；**新發現**：配準階段每個 modality 的畫布尺寸不同，`PrecutStream` 會 fail-fast，這是整片驗證從沒被擋住過的原因（前七輪都用同座標 crop） |
| 在本地把它跑起來、驗證沒改壞（**部分路徑已過時，見檔內提醒**） | [05-dev-testing-guide.md](./05-dev-testing-guide.md) | `cp config_example.py config.py` → 編 config → CLI 跑；回歸基準；自動化測試見 `tests/`（round 8 新增） |
| 動依賴前確認相容性（**部分版本已在後續輪更新，見 measurement 文件**） | [06-versions-dependencies.md](./06-versions-dependencies.md) | venv 實測版本 vs requirements vs pyproject 三方衝突；Blackwell 限制 |
| 踩到怪坑（config 跑不動、幻影檔案…） | [07-gotchas-appendix.md](./07-gotchas-appendix.md) | codegraph 過期、config gitignore、未接線參數、失聯 spec docs（G2 已修復，見檔內更新） |
| 看視覺化的流程與迴圈關係（**畫的是 pre-precut 架構**） | [pipeline-flow.html](./pipeline-flow.html) | M0 複雜迴圈、StitchAccumulator 去重、記憶體生命週期圖 |
| **本資料夾還沒做完 / 半做的優化、待驗證正確性、文件落差** | [19-open-backlog.md](./19-open-backlog.md) | 只引用本資料夾內的文件；效能項目依天花板排序、正確性待簽核項目、文件↔code 落差 |
| 跨 hybrid-pipeline 與 UI 的總表 | [../BACKLOG.md](../BACKLOG.md) | 涵蓋範圍更廣（含 UI Phase 4/5），本資料夾內細節仍以 19 為準 |

**建議閱讀順序**：先 [measurement/bottleneck-list.md](./measurement/bottleneck-list.md)（現況與五輪演進，含每輪的「Re-sorted priority」）→ 依你要接手的項目挑對應的 10–21 → 需要架構/模塊細節再翻 `01/02`（記得套用上面的路徑遷移提醒）。`03/04` 只有歷史對照價值，不要當現況用。

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
**記憶體已靠分塊/串流架構壓下來**（RSS 隨累積細胞數成長、非隨 tile 數線性成長，VRAM 於 441-tile 錨點單 process 下穩定在 ~2.8GB/32GB，多 process 下隨 N 近線性成長，見 doc 21 §4.4）；
GPU 前向單 process 已歷經五輪優化（overlap pipeline + Cellpose 換模型 + `gc.freeze()` + CPU 前處理搬離 MAIN 臂 + precut 串流化 + round 6 拔掉 `detect_all_dots` 的 joblib 派工），round 4 量到的「餘裕僅 15.9%」觸底已被 round 6 打破 —— 拔掉背景執行緒的 GIL 競爭讓 MAIN 臂本身變快，②③（`detect_all_dots`/PNG 編碼）目前比任何一輪都更難被重新暴露；
round 5 **已建置並實測**跨 tile 多行程這個唯一剩下的大天花板槓桿，round 6 在 `dot_detect_n_jobs=1` 落地後重掃了 worker 數並把建議值從 round 5b 的 `workers=6` 下修到 **`workers=4`**（無人看管長跑）/`workers=5`（可接受重跑成本時）—— `workers≥6` 這輪新測到 6 次跑 2 次因 CUDA allocator 碎片化 OOM，見 [23-next-optimization-cycle-implementation.md](./23-next-optimization-cycle-implementation.md) §6/§4.6。正確性投票通過；**round 8（2026-07-27）完成了本專案第一次完整 WSI 端到端跑批，放行到生產的 gate 已滿足**——真實 27,565-tile 整片上 `workers=1` 3.82h、`workers=4` 1.73h，實測加速比 2.216x，正確性投票通過，詳見 [27-remaining-work-implementation.md](./27-remaining-work-implementation.md) §6。**建議放行 `workers=4`，但 VRAM 餘裕只剩 ~2.2 GB（93.3% 卡用量）**，`workers≥5` 仍應等 allocator 氣球問題根治後再開放。單 process 全 WSI 全圖估算（校正後 grid 為 27,565 tile，不是先前假設的 35,700）已從 ~18.9h 壓到 ~5.3h（round 6）、再到 ~2.6h（round 7 推算值）；round 8 **實測**出的真實數字是 **3.82h**（`workers=1`）/ **1.73h**（`workers=4`）——比推算值高 47%/38%，原因不在組成估算（差距不到一格 tile），而是三個先前用裁切量過、以為已關閉的成本在整片規模下重新浮現：`gc.collect`（16.1% wall，`gc.freeze()` 效益未撐到整片規模）、tile read（17.2%，非停損時的 1.22%）、Phase D 縫合（8.6%/19.3%，非先前的 3.5%/7.3%）。目前完整、按優先序排列的待辦清單見 [../BACKLOG.md](../BACKLOG.md)。
