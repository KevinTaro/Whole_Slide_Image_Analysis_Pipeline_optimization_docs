# UI 交接文檔

> **一句話**：下一階段 UI＝**FastAPI（只綁 localhost）+ React Web UI + pywebview 打包成桌面 app**，但**現在還在 Phase 0**——只有共識、還沒寫任何一行 UI code。

這份文檔是給接手 UI 的成員的交接包。目標：**30 分鐘內掌握全貌**，知道「為什麼這樣設計」「哪些不能碰」「什麼時候開始動手」。

---

## Phase 0 現況（重要）

- **repo 內還沒有 `backend/` 或 `frontend/` 資料夾**。本文所有目錄樹、endpoint、指令都是 **提議 / 未來式**，不是現況。
- 目前所有功能都是 Python script / CLI（`thriple_image_layer/`、`cell_mask/hybrid/`）。
- 演算法仍在**醫師驗證 + 迭代**中，**尚未定版**。UI 要等演算法進維護期才啟動（見 [07](07-phase-roadmap.md)）。
- 所以：**現在讀這份文檔＝先建立共識**，不是叫你馬上開工。

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
| — | [architecture-diagram.html](architecture-diagram.html) | 視覺化：分層圖、請求流程、座標邊界、護欄紅線 | 想看圖的時候（瀏覽器開） |

---

## 素材來源

本文檔由 [`docs/next-phase-ui-architecture.md`](../next-phase-ui-architecture.md) 拆分、組織、擴寫而成，並以**實際 repo 狀態**（`pyproject.toml`、`uv.lock`、`.venv`、既有目錄）校正版本號與檔案路徑。原始草稿仍保留，可對照。

> **核心原則（貫穿全文）**：能不手搓輪子就不搓，工具用好用滿。每一層都優先採用業界成熟工具，不自建框架、不造通訊協定、不手寫 UI 元件。
