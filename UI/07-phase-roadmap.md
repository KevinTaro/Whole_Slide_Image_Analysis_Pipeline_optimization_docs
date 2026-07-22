# 07 · Phase 路線圖

> 上一篇：[06 開發環境](06-dev-setup.md)　·　下一篇：[08 陷阱與未決事項](08-pitfalls-open-decisions.md)　·　上層：[README](README.md)
>
> ⚠️ **本文件的狀態欄已過時（原寫於 Phase 0 共識階段，全部標「未啟動」）**。實際上
> Phase 1–3 都已完工並可跑：`backend/algorithms/{hybrid,thriple_image_layer}/`、
> `backend/api/{alignment,hybrid,jobs,tiles}.py`、`frontend/` 整包都已存在且已 commit
> （`835975e`，2026-07-21）。下表狀態欄已更新為現況；細節見
> [10-viewer-ui-implementation.md](10-viewer-ui-implementation.md)（實作清單）與
> [11-runbook-teammate.md](11-runbook-teammate.md)（怎麼跑起來）。「啟動 Phase 1 的條件」
> 一節保留作為當初的決策記錄，不代表現在還沒啟動。

分階段、**每階段都能獨立 ship**，不中斷現有演算法開發。

## Phase 0–5 總表

| Phase | 內容 | 狀態 | 進度說明 |
|---|---|---|---|
| **0 · 現況（原始共識階段）** | 演算法以 script / notebook 運作，醫師驗證中 | 🟢 **已結束，被下方 Phase 取代** | 當初的「不動任何東西」共識已被 Phase 1–3 的實作取代。 |
| **1 · 目錄重構** | 建 `backend/algorithms/`，把 `thriple_image_layer/` + hybrid pipeline **搬入**（只改 import path）。`scripts/` 保留 CLI 入口 | 🟢 **已完工** | `backend/algorithms/hybrid/`、`backend/algorithms/thriple_image_layer/` 已存在；hybrid pipeline 原路徑 `cell_mask/hybrid/` 已不存在。 |
| **2 · 薄 API 包裝** | 每個 pipeline 建 FastAPI endpoint（只驗證＋呼叫＋回結果檔路徑）；加 `/api/tiles/...` DeepZoom 服務 | 🟢 **已完工** | `backend/api/alignment.py`、`backend/api/hybrid.py`、`backend/api/jobs.py`、`backend/api/tiles.py`、`backend/io/pyramid.py` 皆已存在，`backend/main.py` 已掛上對應 router。 |
| **3 · 最小可用 UI** | OpenSeadragon 看單張切片（讀 `/api/tiles`）；能觸發 pipeline、看進度、看疊合圖 | 🟢 **已完工，瀏覽器實測通過** | `frontend/`（Vite+React 19+TS）已建好單張檢視、三層透明度疊合、pipeline 觸發面板（含任務輪詢）。細節見 [10](10-viewer-ui-implementation.md)。**尚未做**：ROI 畫框（屬 Phase 4）、桌面封裝（屬 Phase 5）。 |
| **4 · ROI 與參數微調** | Annotorious 畫 ROI → POST → 重跑該區域；參數面板連 Pydantic schema 自動生成；即時驗證（改參數→快速重算小區→疊圖重繪） | ⚪ **未啟動** | 依賴 [05](05-dataflow-api-contract.md) 的座標轉換與型別同步。 |
| **5 · 打包** | PyInstaller 把 backend 打成單一可執行檔；前端 build 成靜態檔 embed 進 backend；pywebview launcher 啟動 | ⚪ **未啟動** | 目前是 dev server 形態（uvicorn + Vite dev server，見 [11](11-runbook-teammate.md)）。驗收：**乾淨的 Windows 機器雙擊 `.exe` 能正常使用**。pyvips / OpenSlide 原生依賴要處理。 |

---

## 啟動 Phase 1 的條件（歷史決策記錄，Phase 1 已實際啟動並完工）

> 下面兩條是當初決定「什麼時候可以開始搬檔」的判準，**保留供回顧**；Phase 1–3 目前已完工，
> 不代表這兩個條件在啟動當下就已 100% 成立——實際啟動的觸發時機未在文件中另行記錄。

**兩個條件同時成立**才啟動：

1. **醫師驗證通過** —— 演算法輸出被臨床端接受。
2. **演算法進入維護期** —— 以 bugfix 為主、無大改。

> 至於「演算法進入維護期」是否已完全成立：hybrid pipeline 仍在持續做效能優化（見
> `docs/hybrid-pipeline/measurement/bottleneck-list.md` 的第 4 輪量測），且 round 3
> 換 Cellpose backbone 之後的輸出**尚待病理/臨床驗證**（`13-next-optimization-plan.md` §3）。
> UI Phase 1–3 是在演算法仍有變動的情況下並行推進的，不是嚴格等兩個條件都塵埃落定才動手。

---

## 現在該怎麼做（Phase 3 完工後）

- Phase 1–3 的「零重寫、搬檔不改邏輯」原則在往後**仍然適用**——之後任何搬檔/重構都比照同樣紀律。
- 下一個要做的是 **Phase 4（ROI 與參數微調）**，依賴 [05](05-dataflow-api-contract.md) 的座標轉換合約；啟動前建議先看 [08](08-pitfalls-open-decisions.md) 的未決事項（尤其 #4 多 case 管理、#6 `module3_roi_evaluation.py` 歸類，會影響 Phase 4 scope）。
- Phase 5（打包）啟動前需要先確認 [08](08-pitfalls-open-decisions.md) 未決事項 #1/#2（醫師電腦 OS 分布、有無 GPU），這兩項直接決定打包 target 與是否要做 CPU-only fallback。

---

## 本文檔刻意**不**處理的事（避免過早決策）

- 演算法內容的任何變更（仍在驗證 + 迭代）。
- 資料庫選型（filesystem 就夠，真有需要再加 SQLite）。
- 多使用者 / 權限（單機單使用者，不需要）。
- 雲端同步 / 跨機器（違反硬性約束）。
- 自動更新機制（先讓它跑得起來再說）。
- 具體 UI 視覺設計（shadcn/ui 先上，細節等 Figma / 醫師回饋）。
