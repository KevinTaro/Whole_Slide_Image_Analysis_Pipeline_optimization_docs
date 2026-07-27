# 10 · Viewer UI 實作紀錄（我做了什麼）

> **給誰看**：UI（游能舜）自存 + 交接給組員。
> **一句話**：Phase 1–3 的 viewer 已經從「提議」變成「能跑」——後端多了一支
> **DeepZoom tile server**，前端建了一個 **React + OpenSeadragon** 的切片工作台，
> 支援「單張檢視 / 三層透明度疊合 / 觸發 pipeline 任務」三件事。
> 日期：2026-07-12
> 狀態：**已實作、瀏覽器實測通過，但整包仍在工作區、尚未 commit。**

這份是「改了哪些檔、每個檔負責什麼」的清單型紀錄。**怎麼把它跑起來**見
[`11-runbook-teammate.md`](11-runbook-teammate.md)；為什麼 viewer 一定要 subifd=False
副本見 [`09-viewer-tiff-subifd.md`](09-viewer-tiff-subifd.md)。

> ⚠️ 文件版本落差：本 repo 的 [`README.md`](README.md) 與 [`06-dev-setup.md`](06-dev-setup.md)
> 仍寫「Phase 0，尚無 backend/ frontend/」。那是舊敘述，**已不成立**——兩者都已存在。
> 本篇是目前唯一反映 UI 現況的文件。

---

## 1. 後端：DeepZoom tile server（新增）

前端要用 OpenSeadragon 看 WSI，就得有人把金字塔切成 tile 供給。這條是新加的：

| 檔案 | 狀態 | 負責 |
|---|---|---|
| `backend/io/pyramid.py` | 新增 | **無框架**的取圖核心：用 OpenSlide + `DeepZoomGenerator` 讀 slide、切 tile。tile 256 / overlap 1 / JPEG Q80。`slide_id` → `<SLIDES_DIR>/<slide_id>.<ext>` 解析，含 path-traversal 防護（拒 `/`、`\`、`..`），generator 有 cache + lock。 |
| `backend/io/__init__.py` | 新增 | package 標記。 |
| `backend/api/tiles.py` | 新增 | **薄** FastAPI 包裝（只驗證 + 呼叫 + 回傳，不做影像處理）。三個 endpoint：`GET /api/tiles`（slide_id 清單，給 picker）、`GET /api/tiles/{id}.dzi`（DeepZoom XML）、`GET /api/tiles/{id}_files/{lvl}/{col}_{row}.jpeg`（單張 tile）。 |
| `backend/main.py` | 修改 | 掛上 tiles router。 |

**slide 從哪來**：`TSGH_SLIDES_DIR` 環境變數指定的資料夾（預設 `<repo>/slides`）。
放進去的必須是 **OpenSlide 讀得到金字塔的檔** = `subifd=False` + tiled 的 viewer 副本
（原因見 09）。現階段指向 `D:\tsgh_output\thriple_image_layer\viewer`。

**遵守的護欄**（[`04-guardrails-red-lines.md`](04-guardrails-red-lines.md)）：
- 護欄 2：前端只認 `slide_id`，永遠不碰檔案系統路徑。
- 護欄 1/3：`pyramid.py` 不 import 任何 web 框架；`tiles.py` 不做影像處理。

> 其餘 backend 的改動（`api/alignment.py`、`api/hybrid.py`、`api/jobs.py`、
> `schemas/*`、`algorithms/thriple_image_layer/*`）屬於 pipeline / 任務層，
> 是為了讓 UI 的「觸發 pipeline」面板有 endpoint 可打，不在本篇 UI 主軸細列。

---

## 2. 前端：React 切片工作台（新增整包 `frontend/`）

Vite + React 19 + TypeScript。技術選型全部沿用既有共識（不手搓輪子）：

| 用途 | 工具 |
|---|---|
| WSI 檢視 | **OpenSeadragon 6** |
| 任務輪詢 / 資料抓取 | **@tanstack/react-query** |
| 型別安全 API client | **openapi-fetch** + 由後端 OpenAPI 生成的 `src/api/schema.ts` |
| 樣式 | **Tailwind v4**（`@tailwindcss/vite` plugin，無 tailwind.config） |
| TypeScript | 釘 **5.9**（Vite 給的 TS 6 會撞 openapi-typescript peer） |

### 2.1 元件一覽

| 檔案 | 負責 |
|---|---|
| `src/App.tsx` | 版面 + **「單張／疊合」模式切換**。左側欄：標題、模式 toggle、（單張時）切片清單、pipeline 面板。 |
| `src/components/SlidePicker.tsx` | 打 `GET /api/tiles` 列出切片，點選即檢視。有「重新整理」。 |
| `src/components/SlideViewer.tsx` | **單張** OSD 檢視，tileSource 指向 `/api/tiles/{id}.dzi`。停用 nav-control 按鈕（離線無 CDN 圖），滾輪縮放、拖曳平移、右下角導覽縮圖。 |
| `src/components/OverlayViewer.tsx` | **三層疊合**：把 SLIDES_DIR 內每張切片用 `addTiledImage` 疊成一個 OSD 的多圖層，每層一支 **透明度滑桿**（0–100%）。前提是各層像素對齊（同尺寸同原點，crop="overlap"）。 |
| `src/components/PipelinePanel.tsx` | 觸發 **對齊 pipeline**（前處理／對齊／ROI 評估／疊合縮圖）與 **Hybrid 細胞分割**（選 IHC + DISH 切片），拿 `job_id` 後用 `useJob` 輪詢狀態、顯示計時與結果路徑。完成時自動 refresh 切片清單。 |
| `src/api/client.ts` | 單一 typed client（same-origin `/api/*`）。 |
| `src/api/jobs.ts` | `useJob(jobId)`：輪詢 `/api/jobs/{id}` 到 done/error（1.5s，非 WebSocket，符合 05 資料流合約）。 |
| `src/api/schema.ts` | 由後端 `openapi.json` 生成的型別，**不手改**。 |
| `vite.config.ts` | dev 時把 `/api` proxy 到 `http://127.0.0.1:8000`。 |

### 2.2 已驗證的行為（瀏覽器實測）

- 單張：選 HER2/DISH/HE 任一 viewer 副本，流暢縮放不爆記憶體（`level_count=11`）。
- 疊合：三張像素對齊，拉滑桿可透出底層。
- Pipeline 面板：能送任務、輪詢狀態、完成後清單自動刷新。

---

## 3. 這份實作依賴的前置（不是我在這輪做的，但缺了跑不動）

1. **viewer 副本必須先產好**：aligned TIFF（`subifd=True`）→ pyvips 轉 `subifd=False + tiled`
   金字塔副本，放進 SLIDES_DIR。三張已轉好（HER2 2.03GB / DISH 1.60GB / HE 2.95GB，
   皆 155807×133474）。轉檔邏輯與成本見 [`09-viewer-tiff-subifd.md`](09-viewer-tiff-subifd.md) §3.2。
2. **conda `tsgh311` 環境**（含 openslide、Node 26/npm 11 裝在 env root）。

---

## 4. 已知落差 / 待辦

- 這整包 **尚未 commit**（`frontend/`、`backend/io/`、`backend/api/tiles.py` 都是 untracked）。
- README / 06-dev-setup 的「Phase 0」敘述過時，尚未回頭更新。
- viewer 副本目前是**手動**一次性轉檔；「pipeline 跑完自動吐一份副本」尚未落地
  （09-pm-brief §5 的待決事項，等 PM 拍板由誰在哪一環產）。
- 打包成桌面 app（pywebview）尚未做——目前是 dev server 形態。
