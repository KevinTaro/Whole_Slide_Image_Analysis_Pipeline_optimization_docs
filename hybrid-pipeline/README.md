# cell_mask/hybrid Pipeline — 優化交接文檔

> 這份資料夾是給「接手效能優化」的人看的。目標：**30 分鐘內掌握瓶頸與現況**。
> 事實（效能數字、版本、設計決策）皆已由前一輪研究驗證，本文檔只做組織與導覽。

## TL;DR（三行）

1. **定位**：IHC(Her2)＋DISH 雙染色 WSI 的逐 tile 分析流水線 —— 讀圖 → 疊合 → Cellpose 分割 → 逐細胞 HER2/CEP17 點位計數與擴增判定 → 縫回整圖 → 匯出 CSV/視覺化。
2. **最大瓶頸**：**Cellpose 的 ViT-SAM backbone**（`run_net` 32.3% + `get_rel_pos` 25.1%，合計 >57% 時間都在 GPU 前向）；其次是 debug PNG 寫出（13.4%，production 可直接關）。GPU 平均利用率僅 **29%**，代表流水線是「單 tile 序列、GPU 常在等 CPU/IO」，多 tile 平行是最大結構性機會。
3. **GPU 限制（先讀）**：機器是 **RTX 5090（Blackwell, sm_120）**，**只能跑 cu130 的 torch**（實測 `torch 2.11.0+cu130`）。不能任意降 torch 版本，也不能用舊 CUDA wheel —— 改動依賴前務必看 [06-versions-dependencies.md](./06-versions-dependencies.md)。

## 要接手優化，先讀這份（導覽表）

| 你想做的事 | 先讀 | 重點 |
| --- | --- | --- |
| 搞懂整條流水線怎麼跑、資料怎麼流 | [01-architecture-dataflow.md](./01-architecture-dataflow.md) | M0→M1→M2→M3→M0→M4；batch 迴圈 vs chunk 迴圈的巢狀；記憶體生命週期 |
| 查某個模塊的輸入輸出/函式/瓶頸 | [02-module-reference.md](./02-module-reference.md) | 逐檔詳解 + I/O shape/dtype + 關鍵演算法 + 實測套件版本 |
| 看實測效能數字、找瓶頸排名 | [03-benchmarks-bottlenecks.md](./03-benchmarks-bottlenecks.md) | perf_report.html 的 cProfile Top、GPU/CPU/VRAM 利用率 |
| 決定優先改哪裡、投報比（**舊數字，pre-refactor**） | [04-optimization-roadmap.md](./04-optimization-roadmap.md) | 短/中/長期優化 + 設計決策的深層原因 + 已棄用嘗試 |
| 看目前 HEAD 的實測瓶頸排名（**最新一輪，取代 03/04 的舊數字**） | [measurement/bottleneck-list.md](./measurement/bottleneck-list.md) | RTX 5090 實測，25/121/441-tile 真實 WSI，只分類不提解法 |
| 針對 GPU 序列瓶頸（①）的下一步方案設計 | [10-gpu-serial-pipeline-plan.md](./10-gpu-serial-pipeline-plan.md) | 承接 bottleneck-list.md①，playbook Analyze→Plan→Choose，含驗收標準 |
| 在本地把它跑起來、驗證沒改壞 | [05-dev-testing-guide.md](./05-dev-testing-guide.md) | `cp config_example.py config.py` → 編 config → CLI 跑；回歸基準 |
| 動依賴前確認相容性 | [06-versions-dependencies.md](./06-versions-dependencies.md) | venv 實測版本 vs requirements vs pyproject 三方衝突；Blackwell 限制 |
| 踩到怪坑（config 跑不動、幻影檔案…） | [07-gotchas-appendix.md](./07-gotchas-appendix.md) | codegraph 過期、config gitignore、未接線參數、失聯 spec docs |
| 看視覺化的流程與迴圈關係 | [pipeline-flow.html](./pipeline-flow.html) | M0 複雜迴圈、StitchAccumulator 去重、記憶體生命週期圖 |

**建議閱讀順序**：先 `pipeline-flow.html`（看圖建立直覺）→ `03`（知道瓶頸在哪）→ `04`（知道怎麼改）→ 需要細節再翻 `01/02`。

## 參考地圖（既有資源，本文檔不重寫）

| 資源 | 位置 | 內容 |
| --- | --- | --- |
| 模塊級規格（權威、簡潔） | `cell_mask/hybrid/CLAUDE.md` | 各 M 的職責、Invariants、跑法 —— **與本文檔並存，不衝突** |
| 效能報表（實測） | `cell_mask/hybrid/output/perf_report.html` | 3 tiles cProfile、資源 timeline、WSI 估算、優化建議 |
| 6/30 進度筆記 | `cell_mask/docs/6_30_report.md` | 細胞飄移 / patch 分割的當時狀態與「WSI 太大爆記憶體」的原始痛點 |
| 核配對 v3 說明（視覺） | `docs/elastic_matching_v3_explainer.html` | M3 DISH 核配對與計數邏輯（註：描述以核為中心變體，見 07 的版本漂移提醒） |
| Sliding-window 縫合說明 | `docs/sliding-window-seam-stitch.html` | 重疊視窗分割 + 接縫縫合的原始設計 |
| 下一階段 UI 規劃 | `docs/next-phase-ui-architecture.md` | FastAPI+React+pywebview（Phase 1 待醫師驗證後啟動，與本管線解耦） |

## 一句話現況

管線功能完整、單 tile 正確性有回歸基準（單塊輸入 bit-identical 於 pre-M0 路徑）；
**記憶體已靠 M0 分塊讀取 + 增量縫合壓下來**（原本 20k² ROI 峰值 ≈31GB），
剩下的是**速度**問題 —— WSI 全圖單線程估 8h43m（70% 組織），瓶頸集中在 Cellpose GPU 前向與 debug IO。
