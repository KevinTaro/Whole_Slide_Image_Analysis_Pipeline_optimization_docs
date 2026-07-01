# 下一階段 UI 架構規劃

> **狀態**：預先規劃（pre-planning）。本文件只處理「殼」的長期結構，不處理演算法內容 — 演算法層仍在迭代中，待醫師驗證後才會定版。
>
> **建立日期**：2026-04-23
> **作者**：diadia0000 + Claude Opus 4.7 協作討論
> **適用範圍**：`thriple_image_layer/` 與 `cell_mask/hybrid/` 在醫師驗證通過後，要走向可長期維護、可給醫師在自己電腦操作的階段

---

## 1. 背景與約束

### 1.1 目的

醫師驗證完演算法後，需要一個可長期維護、可部署到醫師電腦的使用介面。目前所有功能都是 Python script / CLI，無法交給非工程背景的使用者。

### 1.2 硬性約束

| 約束 | 影響 |
|---|---|
| 病理影像**不得離開本機** | 不可做 remote server，所有處理必須在醫師電腦完成 |
| 醫師電腦部署 | 成品必須是「雙擊開啟」的程度，不能要求裝 Python / node |
| 中高互動（疊圖、ROI 圈選、參數微調、即時驗證） | UI 框架必須能處理 GB 級影像的流暢瀏覽 |
| UI 與演算法必須**結構性分離** | 當 coding AI 修改 UI 時，不應碰到演算法程式碼；反之亦然 |

### 1.3 核心原則

> **能不手搓輪子就不搓，工具用好用滿。**

每一層都優先採用業界成熟工具，不自建框架、不造通訊協定、不手寫 UI 元件。

---

## 2. 架構決策

### 2.1 選定方案：**FastAPI (localhost-only) + Web UI，用 pywebview 打包成桌面 app**

```
doctor-pc (雙擊 .exe)
        │
        ▼
┌─────────────────────────────────────────┐
│  pywebview native window                │
│  ┌───────────────────────────────────┐  │
│  │  Chromium 內嵌視圖                 │  │
│  │  ↕ HTTP (127.0.0.1:PORT)          │  │
│  │  frontend/ (OpenSeadragon + React) │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
              │ HTTP
              ▼
┌─────────────────────────────────────────┐
│  backend/ FastAPI (只綁 127.0.0.1)       │
│  ├─ api/         ← HTTP endpoints       │
│  ├─ algorithms/  ← 你的演算法（純函式）   │
│  └─ io/          ← WSI 讀取、金字塔切片   │
└─────────────────────────────────────────┘
              │
              ▼
        local filesystem
        （影像 + 結果，不進 git）
```

### 2.2 為什麼不選其他方案

| 替代方案 | 為何不選 |
|---|---|
| **PyQt / PySide 純桌面** | UI 與演算法靠紀律分離（signal/slot 容易黏合），AI 訓練資料相對少，金字塔影像瀏覽要自己刻 viewport。|
| **Electron (Node 後端)** | 要把 Python 演算法重寫成 Node，不合 "能不手搓就不搓" 原則。|
| **Streamlit / Gradio** | ROI 圈選與即時參數互動超出其擅長範圍，客製化到中高互動會變成反向手搓。|
| **Remote web server** | 違反「影像不得離開本機」。|
| **Tauri (Rust 殼)** | Rust 學習成本高於效益，pywebview 已足夠。|

### 2.3 UI / 演算法分離的硬性保證

**HTTP 是物理防火牆**：

- `frontend/` 內不存在 `import backend.algorithms` 的可能 — 它是 TypeScript。
- `backend/algorithms/` 內不得 `import fastapi` — code review 與 lint 規則會擋。
- 兩層之間只有 `backend/api/` 作為轉譯層，負責把 HTTP request 翻譯成演算法的函式呼叫。

這條規則會寫進 `CLAUDE.md`，coding AI 在任一層工作時只能看到該層檔案。

---

## 3. 技術棧（工具清單）

### 3.1 後端

| 類別 | 工具 | 為什麼 |
|---|---|---|
| Web 框架 | **FastAPI** | 自動產生 OpenAPI schema → 前端可自動生成 typed client |
| ASGI server | **Uvicorn** | FastAPI 標配 |
| 資料驗證 | **Pydantic v2** | 不手寫 request/response 驗證 |
| WSI 讀取 | **pyvips** / **OpenSlide** | 專案已用，繼續用 |
| 金字塔切片 | **openslide-python DeepZoomGenerator** 或 **pyvips dzsave** | 不手刻 tiling |
| 現有演算法 | **保持原狀** | VALIS、`module1-4`、`cell_mask/hybrid` 不重寫 |

### 3.2 前端

| 類別 | 工具 | 為什麼 |
|---|---|---|
| UI 框架 | **React + Vite** | 生態最大、AI 訓練資料最多 |
| 元件庫 | **shadcn/ui** | 複製貼上式元件，不被鎖定，客製容易 |
| 影像檢視器 | **OpenSeadragon** | 病理影像業界標準（QuPath 以外幾乎所有新工具都用它） |
| ROI 標註 | **Annotorious OpenSeadragon plugin** | 不手刻 canvas 互動 |
| API client | **openapi-typescript** 或 **orval** | 從 FastAPI 的 OpenAPI 自動生成 TS 型別 |
| 資料流 | **TanStack Query** | 不手寫 fetch / cache / retry |
| 樣式 | **Tailwind CSS** | shadcn/ui 原生搭配 |

### 3.3 打包與部署

| 類別 | 工具 | 為什麼 |
|---|---|---|
| 原生視窗 | **pywebview** | 把 FastAPI + 瀏覽器包成單一視窗 |
| 打包 | **PyInstaller** 或 **Briefcase** | 產出 .exe / .app；pyvips、OpenSlide 等原生依賴會需要處理 |
| Python 環境 | **uv** | 已用於本專案 |
| 前端套件管理 | **pnpm** | 磁碟空間 / 速度 |

### 3.4 開發品質

| 類別 | 工具 |
|---|---|
| Python lint/format | **ruff** |
| Python type check | **mypy** 或 **pyright** |
| TypeScript type check | **tsc --noEmit** |
| 前端 lint | **ESLint + Prettier** |
| 測試 | **pytest**（演算法） / **Playwright**（前端 E2E） |

---

## 4. 目錄結構（提議）

```
tsgh/
├── backend/                      ← 新增
│   ├── api/                      ← FastAPI endpoints（唯一 HTTP 介面）
│   │   ├── alignment.py
│   │   ├── cell_detection.py
│   │   ├── roi.py
│   │   └── tiles.py              ← DeepZoom 切片服務
│   ├── algorithms/               ← 從現有目錄搬入
│   │   ├── preprocess/           ← 原 thriple_image_layer/module1_preprocess.py
│   │   ├── alignment/            ← 原 module2_alignment.py + VALIS wrapper
│   │   ├── thumbnail/            ← 原 module4_thumbnail.py
│   │   └── hybrid/               ← 原 cell_mask/hybrid/
│   ├── io/
│   │   ├── wsi_reader.py
│   │   └── pyramid.py
│   ├── schemas/                  ← Pydantic models（API 合約）
│   ├── main.py                   ← FastAPI app entrypoint
│   └── launcher.py               ← pywebview 啟動器
│
├── frontend/                     ← 新增
│   ├── src/
│   │   ├── viewer/               ← OpenSeadragon + ROI
│   │   ├── params/               ← 參數微調面板
│   │   ├── pipeline/             ← 跑 pipeline 的 UI
│   │   ├── api/                  ← 自動生成的 typed client
│   │   └── components/           ← shadcn/ui
│   ├── package.json
│   └── vite.config.ts
│
├── scripts/                      ← 保留：CLI 入口，不經過 UI
├── docs/                         ← 本文件所在
├── packaging/                    ← PyInstaller spec、圖示
└── pyproject.toml                ← 既有
```

**關鍵不變式**：

- `backend/algorithms/*.py` **永遠可以被 CLI 獨立呼叫**，不依賴 FastAPI 啟動。
- 因此 `scripts/` 內的 CLI 入口是演算法的第二個使用者（UI 是第一個），保證演算法不會被 UI 框架綁架。

---

## 5. API 合約設計原則

### 5.1 邊界定義（這層是 AI 最需要理解的契約）

**演算法輸入**：檔案路徑 + JSON 參數
**演算法輸出**：檔案路徑（結果檔）+ JSON metadata

**禁止**：

- 演算法函式回傳 numpy array 給 API 層（太大、序列化痛）
- API 層直接操作 numpy array（應該交給演算法層）
- 前端直接讀取本機檔案路徑（要透過 `/api/tiles` 等 endpoint）

### 5.2 長時間任務

Pipeline 一跑可能數十分鐘。不用 WebSocket 自己刻，用：
- FastAPI BackgroundTasks + polling `/api/jobs/{id}` 查狀態
- 若未來需要即時進度條：Server-Sent Events (SSE)

### 5.3 ROI 座標系

- 瀏覽器端：DeepZoom viewport 座標
- 演算法端：原始影像像素座標
- `backend/api/roi.py` 負責換算，演算法層只認像素座標。

---

## 6. 遷移策略（Migration Plan）

分階段、每階段都可獨立 ship，不中斷現有演算法開發。

### Phase 0 — **現況**

- 演算法以 script / notebook 形式運作。
- 醫師驗證中。
- 本階段**不動任何東西**。

### Phase 1 — **目錄重構（lift-and-shift，零重寫）**

1. 建立 `backend/algorithms/` 並把 `thriple_image_layer/` 和 `cell_mask/hybrid/` **搬入**（import path 調整即可，函式內容完全不動）。
2. 在 `scripts/` 保留 CLI 入口以確認遷移後仍可獨立執行。
3. 驗收：既有所有 script 都跑得起來，輸出結果 bit-identical。
4. **不做**的事：改演算法、換資料結構、加 abstraction。

### Phase 2 — **薄 API 包裝**

1. 為每個 pipeline 建立 FastAPI endpoint，endpoint 只做參數驗證 + 呼叫 `algorithms/` 的既有函式 + 回傳結果檔路徑。
2. 加 `/api/tiles/{slide_id}/{level}/{col}_{row}.jpg` 的 DeepZoom 服務。
3. 驗收：用 curl / HTTPie 能跑完整個 pipeline，產生跟 CLI 一模一樣的結果。

### Phase 3 — **最小可用 UI**

1. OpenSeadragon 檢視單張切片（從 `/api/tiles` 讀）。
2. 能觸發 pipeline、看進度、看結果疊合圖。
3. 驗收：醫師能從頭到尾跑完一個 case 不用開 terminal。

### Phase 4 — **ROI 與參數微調**

1. Annotorious 畫 ROI → POST 給 backend → 重跑該區域的 pipeline。
2. 參數面板（sliders / form）連到 Pydantic schema，自動生成 UI。
3. 即時驗證：改參數 → 快速重算小區域 → 疊合圖重繪。

### Phase 5 — **打包**

1. PyInstaller 把 backend 打成單一可執行檔。
2. 前端 build 成靜態檔，embed 進 backend。
3. pywebview launcher 啟動 backend + 開視窗。
4. 驗收：在一台乾淨的 Windows 機器上雙擊 `.exe` 能正常使用。

---

## 7. 本文件**不**處理的事

這些刻意留白 — 避免過早決策：

- **演算法內容的任何變更**。演算法仍在醫師驗證 + 迭代階段。
- **資料庫選型**。目前 filesystem 就夠，真的有需要再加 SQLite。
- **多使用者 / 權限**。單機單使用者，不需要。
- **雲端同步 / 跨機器**。違反硬性約束，不考慮。
- **自動更新機制**。先讓它跑得起來再說。
- **具體 UI 視覺設計**。shadcn/ui 先上，視覺細節等有 Figma 或醫師回饋再調。

---

## 8. 未決事項（需要後續確認）

- [ ] 醫師電腦 OS 分布（Windows / macOS）→ 決定 PyInstaller target
- [ ] 醫師電腦規格（有無 GPU）→ 決定是否需要 CPU-only fallback
- [ ] 影像格式範圍（.svs / .ndpi / .tiff 各多少）→ 決定 WSI reader 需支援哪些
- [ ] 是否需要多 case 管理 / 患者清單 → 影響 Phase 3 UI scope
- [ ] 醫師對 UI 語言偏好（繁中 / 英文 / 雙語）→ 決定 i18n 要不要進 Phase 3

---

## 9. 下一步（本文件之後的動作）

1. **不動手寫 code**。本文件只是共識，等演算法驗證完再啟動 Phase 1。
2. 演算法迭代期間：維持現有結構，任何新模組**放在對的位置**（preprocess 類放 `thriple_image_layer/`，後續會一起搬），減少 Phase 1 遷移工作量。
3. Phase 1 啟動時機：醫師驗證通過 + 演算法進入維護期（bugfix 為主、無大改）。
