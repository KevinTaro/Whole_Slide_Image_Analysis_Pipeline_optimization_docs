# 14 · 兩台機器的最終執行方式（Linux 跑運算 · Windows 呈現前端）

> **給誰看**：要 demo 或驗收這套系統的人。
> **統整自**：[01 架構](01-architecture.md)、[04 護欄](04-guardrails-red-lines.md)、
> [05 資料流與 API 合約](05-dataflow-api-contract.md)、[09 viewer subifd](09-viewer-tiff-subifd.md)、
> [11 單機 runbook](11-runbook-teammate.md)、`hybrid_flow_mockup.html`（UI 目標形狀），
> 以及 `docs/algo/frontend_backend_split_architecture.html`（跨機器分析，已被 commit
> `b04cbfd` 從工作樹移除，用 `git show b04cbfd^:docs/algo/...` 取回）。
> **驗證日**：2026-07-28，本文所有指令與數字都在兩台機器上實跑過。

---

## 0. 最終形狀

```
 Windows（.153 / RTX 4080）                    Linux（.154 / RTX 2080 Ti）
┌──────────────────────────────┐              ┌────────────────────────────────┐
│ 瀏覽器 localhost:5173         │              │ uvicorn  127.0.0.1:8000        │
│   └ React + OpenSeadragon    │              │   ├ api/      端點轉譯          │
│ Vite dev server              │              │   ├ algorithms/ GPU 運算        │
│   proxy /api → 127.0.0.1:8000│═════════════▶│   └ io/pyramid  tile server    │
└──────────────────────────────┘  SSH tunnel  │ 影像與結果都在這台的硬碟        │
                                  (加密)      └────────────────────────────────┘
```

前端只送 `slide_id` 與 ROI 座標、只收小 JSON 與 JPEG 圖磚；**原始影像與結果檔從頭到尾
沒有離開 Linux**。

---

## 1. 為什麼用 SSH tunnel，而不是把後端綁 0.0.0.0

[01](01-architecture.md) 的第一條硬性約束是「**病理影像不得離開本機**，FastAPI 只綁
`127.0.0.1`，不開對外埠」。兩台機器天生跟這條有張力，SSH tunnel 是唯一同時滿足的解：

| | 綁 `0.0.0.0` | **SSH tunnel（採用）** |
|---|---|---|
| LAN 上的 listening port | 有，任何人可連 | **沒有**，後端仍只綁 127.0.0.1 |
| 傳輸 | 明文 HTTP | SSH 加密 |
| ufw（該機開著且無 sudo 看不到規則） | 會被擋 | 走 22 埠，不受影響 |
| `vite.config.ts` | 要改 proxy target（動到 git 追蹤的檔） | **一個字都不用改** |

Vite 的 proxy 目標本來就是 `127.0.0.1:8000`（`vite.config.ts:20`）。隧道把 Linux 的 8000
映到 Windows 的 127.0.0.1:8000 之後，前端完全不知道後端在另一台 —— 這也正好落實護欄 2
「前端不認得檔案系統路徑」。

---

## 2. 執行步驟

### A. Linux — 起後端

```bash
ssh yoyo@210.240.160.154
source ~/projects/tsgh-env.sh      # TSGH_STORAGE_DIR / TSGH_SLIDES_DIR / PYTHONPATH / PATH
nvidia-smi                          # ← 先看一眼，這張卡是跟別人共用的
cd $TSGH_ROOT
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**後端不會馬上回應**（import 會把 torch/valis/cellpose 整包拉起來）：實測**冷啟動 ~42 秒、
page cache 熱的時候 ~7 秒**。不要以為它掛了。`TSGH_STORAGE_DIR` 沒設的話後端會直接拒絕 import。

### B. Windows — 開隧道 + 起前端

```powershell
# 視窗 A：隧道（保持開著）。若本機自己的後端在跑，先關掉，否則 8000 會撞埠
ssh -N -L 8000:127.0.0.1:8000 yoyo@210.240.160.154

# 視窗 B：前端
$env:Path = "C:\Users\RCLab\miniconda3\envs\tsgh311;" + $env:Path
Set-Location "C:\Users\RCLab\Desktop\tsgh\frontend"
npm run dev        # → http://localhost:5173
```

### C. 驗證（三層各驗一次，出問題才知道斷在哪）

```powershell
# ① 隧道通不通 + 後端活著（應回 slide_id 陣列）
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/tiles | Select -Expand Content

# ② tile server 真的吐得出圖（應為 HTTP 200 + image/jpeg）
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/api/tiles/<slide_id>_files/8/0_0.jpeg"

# ③ Vite proxy 有生效（同樣回 slide_id 陣列）
Invoke-WebRequest -UseBasicParsing http://localhost:5173/api/tiles | Select -Expand Content
```

三步都過，瀏覽器開 `http://localhost:5173` 就是完整系統。

---

## 3. 資料要放在哪

**全部放 Linux。** 後端讀的是自己硬碟（`frontend_backend_split_architecture.html` 的核心
結論：路徑是本機語意，跨機器無效；控制訊息一律用 id）。

| 用途 | 位置 | 誰在讀 |
|---|---|---|
| viewer 切片來源 | `~/tsgh_data/viewer`（`TSGH_SLIDES_DIR`） | `backend/io/pyramid.py` |
| 對齊 run 的工作區 | `~/tsgh_data/storage`（`TSGH_STORAGE_DIR`） | `backend/schemas/alignment.py` |
| CZI 輸入 / 對齊輸出 | `~/tsgh_data/{picture,thriple_image_layer}` | `thriple_image_layer/config.py` |

**丟進 `TSGH_SLIDES_DIR` 的必須是 OpenSlide 讀得到金字塔的檔**（`level_count > 1`）。
對齊 pipeline 的原輸出是 `subifd=True`、`level_count=1`，直接丟會**把後端讀爆**（見
[09](09-viewer-tiff-subifd.md)），要先用 `scripts/make_viewer_copy.py` 轉副本：

```bash
.venv/bin/python -c "import openslide;print(openslide.OpenSlide('<檔>').level_count)"
# 1 → 不能用，先轉；11（或 >1）→ OK
```

> **例外（2026-07-28 實測）**：hybrid 產出的 `overlay_slide.tiff` **不需要轉**。
> 實測 `subifds: None` / `level_count = 4`，OSD 直接讀得到，而且 `/api/hybrid/result`
> 會自動 `pyramid.register` 成 slide_id `hybrid_overlay`（`backend/api/hybrid.py:45`）。
> mockup 裡「結果需先產 viewer 副本」那句對這個檔已經過時。

---

## 4. mockup 三個畫面 ↔ 實際端點

`hybrid_flow_mockup.html` 是目標 UI 形狀，對應關係如下（都已存在，不是待實作）：

| mockup 畫面 | 實際呼叫 | 備註 |
|---|---|---|
| ① 圈選 ROI | `GET /api/tiles` 取清單 → OSD 開圖 | ROI 是 **slide 像素**座標。已做成可拖曳/縮放的選取框，見 [15](15-roi-box-selection.md) |
| 「開始分析」 | `POST /api/hybrid/tile`<br>`{ihc_slide_id, dish_slide_id, roi_x/y/w/h}` | 四個 ROI 值要嘛全給、要嘛全不給（`HybridTileIn` 驗證）；不給 = 整片 |
| ② 分析中 | `GET /api/jobs/{job_id}` 輪詢 1.5s | BackgroundTasks + polling，不是 WebSocket（[05](05-dataflow-api-contract.md)） |
| ③ 結果 | `GET /api/hybrid/result` → `{summary, has_report, overlay_slide_id}`<br>`GET /api/hybrid/report.csv` | summary.txt 內文直接內嵌回傳；overlay 用 slide_id 給 OSD；CSV 是下載 |

**整片不要用 UI 送**：`HybridTileIn` 的註解寫明整片約 27,500 個 tile／數小時 GPU，
所以 UI 正常情況一定送 ROI。Demo 請圈小範圍。

---

## 4.5 端到端實測紀錄（2026-07-29）

前端在 Windows 的瀏覽器、後端在 Linux，走完一次完整分析。**17 個請求全部 200**，
GPU 在分析期間跑到 **67% / 6087 MiB**，`job` 從送出到 `done` **22 秒**（單一 tile 的 ROI）：

| 階段 | 實際打的端點 |
|---|---|
| 開頁 | `GET /api/tiles`、`/api/alignment/runs`、`/api/alignment/published`、`/api/hybrid/result` |
| 選 IHC 切片 | `GET /api/tiles/{id}.dzi` → `{id}_files/6..10/0_0.jpeg`（OSD 逐層抓金字塔） |
| 開始分析 | `POST /api/hybrid/tile` → `{job_id}` |
| 分析中 | `GET /api/jobs/{id}` 反覆輪詢 |
| 完成 | `GET /api/hybrid/result` → `GET /api/tiles/hybrid_overlay.dzi` → overlay 圖磚 |
| 下載 | `GET /api/hybrid/report.csv` → `text/csv`，欄位正確 |

完成後前端**自動把檢視器切到標註後的 overlay**（`api/hybrid.py` 的 `pyramid.register`
把它登記成 slide_id `hybrid_overlay`），全程沒有任何一張大圖走過 HTTP 之外的路徑。

> ⚠️ **測試資料的注意事項**：若把非金字塔的 tile 轉成 slide 來測，OpenSlide 只認得
> 它支援的格式，用 `pyvips` 轉檔時**存完要等檔案落地再開**（寫完立刻 `OpenSlide()` 會讀到
> 半成品而報 ICC 相關的 `ValueError`）。另外用 JPEG 重新編碼會**改變像素值**，判讀數字
> 會與原始輸入跑出來的不同 —— 要做數值驗收必須用無損輸入。

## 5. 跨機器的已知落差（不是 bug，但 demo 前要知道）

1. **`結果：<路徑>` 顯示的是 Linux 路徑**。`JobStatus.result_path`（`schemas/common.py:17`）
   在 `PipelinePanel.tsx:387` 被當文字印出來，在 Windows 上那個路徑不存在。
   **只是顯示，不影響功能** —— 真正開圖走的是 `metadata.slide_id`（`PipelinePanel.tsx:240`
   只把 `result_path` 當「有沒有產出」的旗標），所以看圖照常。
2. **API 路徑硬編 `workers=1`**（`api/hybrid.py:89`）。這正好符合 2080 Ti 的 22GB VRAM 上限
   —— `workers=4` 需要 30.4GB，這張卡跑不了。走 UI 不會踩到這個限制。
3. **GPU 是跟別人共用的**。曾因他人的 Streamlit 佔卡導致
   `CUDA error: CUDA-capable device(s) is/are busy or unavailable`。Demo 前先 `nvidia-smi`。
4. **隧道斷線 = 前端全紅**。SSH 斷了前端會顯示「無法取得切片清單」，重連隧道即可，
   不用重啟後端或前端。

---

## 6. 踩雷對照表

| 症狀 | 原因 / 解法 |
|---|---|
| 前端「無法取得切片清單(後端未啟動?)」 | 隧道斷了、或後端還在那 42 秒的 import。先跑 §2C ① |
| `Address already in use` / 隧道起不來 | Windows 本機自己的後端佔著 8000。關掉它，或換 `-L 8001:` 但那樣要改 proxy（不建議） |
| 切片清單是空的 | Linux 的 `TSGH_SLIDES_DIR` 指錯，或資料夾裡沒有 tiff/svs |
| 開圖瞬間後端 `MemoryError` | 丟了 `subifd=True` 原檔（`level_count=1`）。轉 viewer 副本（§3） |
| `CUDA ... busy or unavailable` | 別人在用卡。`nvidia-smi` 看一下，等一下重跑 |
| 分析跑很久沒動靜 | 圈太大。整片 ≈ 27,500 tiles／數小時，ROI 圈小一點 |
| `'node' is not recognized`（Windows） | conda env 根目錄沒加進 PATH（§2B 第一行） |

---

## 7. 一頁摘要

```bash
# 【Linux】
ssh yoyo@210.240.160.154
source ~/projects/tsgh-env.sh && nvidia-smi && cd $TSGH_ROOT
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000   # 冷啟動等 ~42s
```
```powershell
# 【Windows 視窗 A】隧道
ssh -N -L 8000:127.0.0.1:8000 yoyo@210.240.160.154
# 【Windows 視窗 B】前端
$env:Path = "C:\Users\RCLab\miniconda3\envs\tsgh311;" + $env:Path
Set-Location "C:\Users\RCLab\Desktop\tsgh\frontend"; npm run dev
# → http://localhost:5173
```
