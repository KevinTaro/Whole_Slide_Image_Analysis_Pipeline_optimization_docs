# 07 · Phase 路線圖

> 上一篇：[06 開發環境](06-dev-setup.md)　·　下一篇：[08 陷阱與未決事項](08-pitfalls-open-decisions.md)　·　上層：[README](README.md)

分階段、**每階段都能獨立 ship**，不中斷現有演算法開發。

## Phase 0–5 總表

| Phase | 內容 | 狀態 | 進度說明 |
|---|---|---|---|
| **0 · 現況** | 演算法以 script / notebook 運作，醫師驗證中 | 🟢 **進行中** | **本階段不動任何東西**。只有本文檔這份共識。 |
| **1 · 目錄重構** | 建 `backend/algorithms/`，把 `thriple_image_layer/` + `cell_mask/hybrid/` **搬入**（只改 import path）。`scripts/` 保留 CLI 入口 | ⚪ 未啟動 | **零重寫**（護欄 6）。驗收：既有所有 script 跑得起來、輸出 **bit-identical**。不做：改演算法 / 換資料結構 / 加 abstraction。 |
| **2 · 薄 API 包裝** | 每個 pipeline 建 FastAPI endpoint（只驗證＋呼叫＋回結果檔路徑）；加 `/api/tiles/...` DeepZoom 服務 | ⚪ 未啟動 | 驗收：用 curl / HTTPie 跑完整 pipeline，結果與 CLI **一模一樣**。 |
| **3 · 最小可用 UI** | OpenSeadragon 看單張切片（讀 `/api/tiles`）；能觸發 pipeline、看進度、看疊合圖 | ⚪ 未啟動 | 驗收：醫師能從頭到尾跑完一個 case **不用開 terminal**。 |
| **4 · ROI 與參數微調** | Annotorious 畫 ROI → POST → 重跑該區域；參數面板連 Pydantic schema 自動生成；即時驗證（改參數→快速重算小區→疊圖重繪） | ⚪ 未啟動 | 依賴 [05](05-dataflow-api-contract.md) 的座標轉換與型別同步。 |
| **5 · 打包** | PyInstaller 把 backend 打成單一可執行檔；前端 build 成靜態檔 embed 進 backend；pywebview launcher 啟動 | ⚪ 未啟動 | 驗收：**乾淨的 Windows 機器雙擊 `.exe` 能正常使用**。pyvips / OpenSlide 原生依賴要處理。 |

---

## 啟動 Phase 1 的條件

**兩個條件同時成立**才啟動：

1. **醫師驗證通過** —— 演算法輸出被臨床端接受。
2. **演算法進入維護期** —— 以 bugfix 為主、無大改。

在此之前，演算法還在迭代，**搬檔＝找罪受**（搬完又大改，等於白搬）。

---

## 現在（演算法迭代期間）該怎麼做

不是什麼都不能做。做這幾件事能把未來 Phase 1 的遷移成本壓到最低：

- **維持現有結構**，不預先建 `backend/` / `frontend/`（避免空殼腐化）。
- **新模組放在對的位置**：前處理 / 對齊類放 `thriple_image_layer/`，DISH 類放 `cell_mask/hybrid/`——反正 Phase 1 會整包搬，現在歸對位就少一次搬。
- **保持演算法可被 CLI 獨立呼叫**（別現在就依賴任何 web 概念），這樣 Phase 1 的「零重寫」才成立。
- **不手寫任何 UI code**。本文檔只是共識，等演算法驗證完再啟動 Phase 1。

---

## 本文檔刻意**不**處理的事（避免過早決策）

- 演算法內容的任何變更（仍在驗證 + 迭代）。
- 資料庫選型（filesystem 就夠，真有需要再加 SQLite）。
- 多使用者 / 權限（單機單使用者，不需要）。
- 雲端同步 / 跨機器（違反硬性約束）。
- 自動更新機制（先讓它跑得起來再說）。
- 具體 UI 視覺設計（shadcn/ui 先上，細節等 Figma / 醫師回饋）。
