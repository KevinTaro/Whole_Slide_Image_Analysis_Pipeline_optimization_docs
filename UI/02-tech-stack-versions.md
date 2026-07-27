# 02 · 技術棧與版本

> 上一篇：[01 架構](01-architecture.md)　·　下一篇：[03 目錄結構](03-directory-structure.md)　·　上層：[README](README.md)

## 版本真相來源層級（先讀這段）

```
pyproject.toml（tracked，寫「約束」）
        +
uv.lock / .venv（resolved，寫「實際解出的版本」）
        │  以上兩者為準
        ▼
────────────────────────────────────────
requirements.txt（一次性 pip freeze 快照，已 drift，別信）
```

- **要知道現在跑的是哪版** → 查 `uv.lock` 或直接 `source .venv/bin/activate && pip show <pkg>`。
- **要改約束** → 改 `pyproject.toml`，然後 `uv lock` / `uv sync`。
- **Phase 1/2 新套件**（FastAPI 等）的確切版本，在 **`uv add <pkg>` 的當下**由 uv 依約束解出來決定，**不是現在**先寫死。
- `uv.lock` 目前被 `.gitignore` 忽略（本機 resolved 真相，與 `.venv` 一致），所以 repo 內**可追蹤的約束來源是 `pyproject.toml`**。

---

## 已釘版本（後端，已在 repo / venv）

以下為 `.venv` 實測 = `uv.lock` 解出的值（**權威**）：

| 套件 | 釘住版本 | 鎖在哪 / 約束 |
|---|---|---|
| Python | **3.11.15**（`>=3.11,<3.12`） | `pyproject.toml` → `requires-python` |
| pyvips | **2.2.3** | `pyproject`（無 pin）→ `uv.lock` 解出 |
| openslide-python | **1.4.3** | `pyproject` `>=1.4.3` |
| openslide-bin | **4.0.0.13** | `pyproject` `>=4.0.0.13`（原生 OpenSlide binary） |
| pydantic | **2.13.3** | `uv.lock`（目前是 transitive；FastAPI 進來時 v2 已就位） |
| numpy | **1.26.4** | `pyproject` **`numpy<2`**（硬約束，見下） |
| valis-wsi | **1.2.0** | `pyproject` `>=1.2.0`（對齊主力） |
| cellpose | **4.1.1** | `uv.lock` |
| scikit-image | **0.24.0** | `pyproject` override `>=0.22,<0.25` |
| pillow | **12.2.0** | `pyproject` `>=10` |
| opencv-contrib-python-headless | **4.8.1.78** | `pyproject` override `<4.9` |

---

## 未釘版本（前端 / 打包，**尚未進 repo**）

還沒 `uv add` / `pnpm add`，所以沒有確切版本號。進 repo 時才由 uv / pnpm 依當下最新穩定版解出：

| 類別 | 工具 | 為什麼選它 |
|---|---|---|
| Web 框架 | **FastAPI** | 自動產生 OpenAPI schema → 前端可自動生成 typed client |
| ASGI server | **Uvicorn** | FastAPI 標配 |
| 金字塔切片 | **openslide DeepZoomGenerator** 或 **pyvips `dzsave`** | 不手刻 tiling |
| UI 框架 | **React + Vite** | 生態最大、AI 訓練資料最多 |
| 元件庫 | **shadcn/ui** | 複製貼上式元件，不被鎖定 |
| 影像檢視器 | **OpenSeadragon** | 病理影像業界標準 |
| ROI 標註 | **Annotorious（OSD plugin）** | 不手刻 canvas 互動 |
| API client | **openapi-typescript** 或 **orval** | 從 OpenAPI 自動生成 TS 型別 |
| 資料流 | **TanStack Query** | 不手寫 fetch / cache / retry |
| 樣式 | **Tailwind CSS** | shadcn/ui 原生搭配 |
| 原生視窗 | **pywebview** | 把 FastAPI + 瀏覽器包成單一視窗 |
| 打包 | **PyInstaller** 或 **Briefcase** | 產出 `.exe` / `.app`（pyvips、OpenSlide 原生依賴要處理） |
| Python 環境 | **uv** | 已用於本專案 |
| 前端套件管理 | **pnpm** | 磁碟空間 / 速度 |

開發品質工具：**ruff**（Python lint/format）、**mypy / pyright**（Python type）、**tsc --noEmit**（TS type）、**ESLint + Prettier**（前端）、**pytest**（演算法）、**Playwright**（前端 E2E）。

---

## 注意事項（黑名單 / 陷阱版本）

- **PyQt6 6.10.1 不是我們的框架**。它出現在 `requirements.txt`，是 cellpose / napari 桌面 GUI 相依（連帶 `superqt` / `pyqtgraph` / `QtPy`）拖進來的。我們做的是 **pywebview + web**，不是 Qt app。別被誤導。
- **`numpy<2` 是硬約束**，不能為了新套件放寬——VALIS / numba / opencv 相容鏈綁在 1.26.x。
- **opencv `<4.9`**、**scikit-image `0.22–0.25`** 也是 override 鎖住的相容區間，動它們會炸 VALIS。
- **`requirements.txt` 已 drift，不要照它裝**。它是 2026-04 的一次性快照，數字對不上實際 venv：它寫 `pyvips 3.1.1`（實際 2.2.3）、`numpy 2.2.6`（**直接違反 `numpy<2`**）、`cellpose 4.0.8`（實際 4.1.1）、`pydantic 2.12.5`（實際 2.13.3）。這正是 [08](08-pitfalls-open-decisions.md) 的「requirements.txt 不同步」陷阱的實例——**查版本一律看 `uv.lock` / venv**。
