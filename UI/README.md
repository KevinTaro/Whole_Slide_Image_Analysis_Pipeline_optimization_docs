# UI 交接文檔

> **一句話**：下一階段 UI＝**FastAPI（只綁 localhost）+ React Web UI + pywebview 打包成桌面 app**。
> **現況（2026-07-22）：Phase 1–3 已完工**——`backend/` 與 `frontend/` 都已存在並可跑（tile
> server、單張/疊合檢視、pipeline 觸發面板），詳見 [07](07-phase-roadmap.md) 的狀態表、
> [10](10-viewer-ui-implementation.md) 實作清單、[11](11-runbook-teammate.md) 跑法。
> 未完成的是 **Phase 4（ROI/參數微調）** 與 **Phase 5（打包成桌面 app）**。

這份文檔是給接手 UI 的成員的交接包。目標：**30 分鐘內掌握全貌**，知道「為什麼這樣設計」「哪些不能碰」「現在做到哪、下一步是什麼」。

---

## Phase 0 現況（**歷史記錄，已過時，保留供對照**）

> ⚠️ **本節敘述的是 Phase 0（共識階段）的狀態，早已過時**：`backend/` 與 `frontend/` 都已存在，
> tile server + React 切片工作台已能跑，Phase 1–3 已完工（見上方與 [07](07-phase-roadmap.md)）。
> UI 現況以 [10](10-viewer-ui-implementation.md)、跑法以 [11](11-runbook-teammate.md) 為準；
> 下方保留原文供歷史對照，**不要照下面的敘述行動**。

- ~~repo 內還沒有 `backend/` 或 `frontend/` 資料夾~~ —— 兩者皆已存在，見上方。
- ~~目前所有功能都是 Python script / CLI（`thriple_image_layer/`、`cell_mask/hybrid/`）~~ —— 已搬進 `backend/algorithms/`，且已有 FastAPI 包裝（見 [07](07-phase-roadmap.md) Phase 2）。
- 演算法仍在**醫師驗證 + 迭代**中，**尚未定版**（round 3 的 Cellpose 換模型結果仍待病理驗證，見 `docs/hybrid-pipeline/13-next-optimization-plan.md` §3）——這點依然成立，UI 開發是與演算法迭代**並行**推進的，不是等演算法定版才開始。

---

## 導覽表（建議閱讀順序）

| 順序 | 檔案 | 用途 | 什麼時候看 |
|---|---|---|---|
| 1 | [01-architecture.md](01-architecture.md) | 三層架構、為什麼選 FastAPI+React+pywebview、否決了哪 5 種方案 | 一定要先看 |
| 2 | [04-guardrails-red-lines.md](04-guardrails-red-lines.md) | 7 條硬性護欄 + AI code review checklist | 一定要先看（動手前的紅線） |
| 3 | [07-phase-roadmap.md](07-phase-roadmap.md) | Phase 0–5 路線圖、Phase 1 啟動條件 | 想知道「現在該做什麼」 |
| 4 | [03-directory-structure.md](03-directory-structure.md) | 現況 vs 目標目錄樹、遷移對照、未對應項目 | 準備 Phase 1 搬檔前 |
| 5 | [05-dataflow-api-contract.md](05-dataflow-api-contract.md) | 一次請求的完整路徑、長任務、型別同步、座標轉換 | 寫 API / 前端前 |
| 6 | [02-tech-stack-versions.md](02-tech-stack-versions.md) | 已釘 / 未釘版本、版本真相來源層級 | 要 `uv add` / 裝套件前 |
| 7 | [06-dev-setup.md](06-dev-setup.md) | 現在能跑什麼、未來開發指令、除錯備註 | 要動手跑東西時 |
| 8 | [08-pitfalls-open-decisions.md](08-pitfalls-open-decisions.md) | 9 個常見陷阱 + 8 個未決事項 | 隨時翻，踩雷前查 |
| 9 | [09-viewer-tiff-subifd.md](09-viewer-tiff-subifd.md) · [09-…pm-brief.md](09-viewer-subifd-pm-brief.md) | 為什麼 viewer 一定要 subifd=False 副本（技術版＋PM 決策版） | 產 / 顯示切片前 |
| 10 | [10-viewer-ui-implementation.md](10-viewer-ui-implementation.md) | **實作現況**：tile server + React 切片工作台改了哪些檔 | 想知道現在有什麼 |
| 11 | [11-runbook-teammate.md](11-runbook-teammate.md) | **照做把 UI 跑起來**（後端 + 前端 + 操作 + 踩雷表） | 要在自己機器上跑 |
| 12 | [12-env-setup.md](12-env-setup.md) | **從零重建 `tsgh311` 環境**（conda 原生層 + pip 層，Windows） | 機器上還沒有環境時 |
| — | [architecture-diagram.html](architecture-diagram.html) | 視覺化：分層圖、請求流程、座標邊界、護欄紅線 | 想看圖的時候（瀏覽器開） |

---

## 素材來源

本文檔由 [`docs/next-phase-ui-architecture.md`](../next-phase-ui-architecture.md) 拆分、組織、擴寫而成，並以**實際 repo 狀態**（`pyproject.toml`、`uv.lock`、`.venv`、既有目錄）校正版本號與檔案路徑。原始草稿仍保留，可對照。

> **核心原則（貫穿全文）**：能不手搓輪子就不搓，工具用好用滿。每一層都優先採用業界成熟工具，不自建框架、不造通訊協定、不手寫 UI 元件。
