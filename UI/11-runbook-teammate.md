# 11 · Runbook：把 Viewer UI 跑起來（組員照做）

> **給誰看**：接手 / 要驗收這套 UI 的組員。
> **目標**：照著貼指令，30 分鐘內在自己機器上打開切片工作台、看到單張與三層疊合。
> 對應實作清單見 [`10-viewer-ui-implementation.md`](10-viewer-ui-implementation.md)。
> 環境：Windows 11 + conda `tsgh311`，**PowerShell**。

---

## 0. 名詞與路徑（先看一眼）

| 名稱 | 值 |
|---|---|
| Repo 根目錄 | `C:\Users\RCLab\Desktop\tsgh` |
| conda 環境 | `tsgh311`（含 openslide、pyvips、Node 26 / npm 11） |
| Python | `C:\Users\RCLab\miniconda3\envs\tsgh311\python.exe` |
| Node / npm（**裝在 env 根目錄，不在 Scripts\**） | `C:\Users\RCLab\miniconda3\envs\tsgh311\{node.exe, npm.cmd, npx.cmd}` |
| Slides 來源（viewer 副本） | `D:\tsgh_output\thriple_image_layer\viewer` |

> ⚠️ 舊筆記可能寫 repo 根是 `...\tsgh\tsgh`——那是舊的巢狀版面，**現在是扁平的
> `...\tsgh`**。以上表為準。

---

## 1. 前置：確認 slides 資料夾裡是 viewer 副本

tile server 只吃 **OpenSlide 讀得到金字塔的檔**（`subifd=False` + tiled）。若直接丟
pipeline 原輸出（`subifd=True`）會把後端**讀爆當掉**（原因見 09）。

先確認 `D:\tsgh_output\thriple_image_layer\viewer` 裡有轉好的副本（例：
`HER2_aligned_viewer.tiff` / `DISH_aligned_viewer.tiff` / `HE_aligned_viewer.tiff`）。

驗一張是不是合格副本（`level_count` 要 > 1，通常 11）：
```powershell
$py = "C:\Users\RCLab\miniconda3\envs\tsgh311\python.exe"
& $py -c "import openslide; s=openslide.OpenSlide(r'D:\tsgh_output\thriple_image_layer\viewer\HER2_aligned_viewer.tiff'); print('level_count=', s.level_count)"
# level_count= 11  → OK；若是 1 → 這是 subifd=True 原檔，不能用，得先轉副本
```

**若還沒有副本 / 有新的對齊輸出要轉**：用 repo 內的腳本 `scripts/make_viewer_copy.py`
把 aligned TIFF（`subifd=True`）轉成 `subifd=False + tiled` 金字塔副本（`jpeg Q85 tile256
pyramid bigtiff`）。轉一張約 2 分鐘 / 2 GB，轉完會印出 OpenSlide `level_count` 讓你確認 > 1：

```powershell
$py = "C:\Users\RCLab\miniconda3\envs\tsgh311\python.exe"
# 一次可轉多張；輸出為 <來源檔名>_viewer.tiff，直接吐進 SLIDES_DIR
& $py scripts\make_viewer_copy.py `
    D:\tsgh_output\thriple_image_layer\HER2_aligned_lv0.ome.tiff `
    D:\tsgh_output\thriple_image_layer\DISH_aligned_lv0.ome.tiff `
    --out-dir D:\tsgh_output\thriple_image_layer\viewer
```

> ⚠️ 腳本只做「輸入→輸出」轉換，**不會產生輸入**：你手上得先有 aligned TIFF（跑完對齊
> pipeline 才有）。背景與實測成本見 [`09-viewer-tiff-subifd.md`](09-viewer-tiff-subifd.md) §3.2。

---

## 2. 啟動後端（tile server + pipeline API）

開一個 PowerShell，**cwd 設在 repo 根**（讓 `backend` package import 得到）：

```powershell
$py = "C:\Users\RCLab\miniconda3\envs\tsgh311\python.exe"
$env:PYTHONPATH   = "C:\Users\RCLab\Desktop\tsgh"
$env:TSGH_SLIDES_DIR = "D:\tsgh_output\thriple_image_layer\viewer"   # slides 來源
& $py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**驗一下**（另開一個視窗）：
```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/tiles | Select-Object -Expand Content
# 應回傳 slide_id 的 JSON 陣列，例：["DISH_aligned_viewer","HE_aligned_viewer","HER2_aligned_viewer"]
```
拿不到清單 = 後端沒起來，或 `TSGH_SLIDES_DIR` 指錯。

---

## 3. 啟動前端（Vite dev server）

**Node 裝在 env 根、不在 PATH 上**，先把它加進 PATH，否則 `npm`/`npx` 會報
`'node' is not recognized`。cwd 要在 `frontend\`：

```powershell
$env:Path = "C:\Users\RCLab\miniconda3\envs\tsgh311;" + $env:Path
Set-Location "C:\Users\RCLab\Desktop\tsgh\frontend"

npm install          # 第一次才需要（node_modules 已在時可略）
npm run dev          # Vite 起在 http://localhost:5173
```

瀏覽器開 **http://localhost:5173**。`/api/*` 會由 Vite 自動 proxy 到後端 8000
（見 `vite.config.ts`），所以前端不需要知道後端位址。

---

## 4. 操作：三件事

1. **單張檢視**：左上模式選「**單張**」→ 左側清單點一張切片 → 右側 OSD 開圖。
   滾輪縮放、拖曳平移、右下有導覽縮圖。
2. **三層疊合**：模式選「**疊合**」→ 自動把 SLIDES_DIR 內全部切片疊成多圖層，
   左上角每層一支透明度滑桿，拉一拉可透出底層（前提：各層像素對齊）。
3. **觸發 pipeline**：左側「影像對齊 Pipeline」四顆按鈕（前處理／對齊／ROI 評估／
   疊合縮圖），或「Hybrid 細胞分割」選 IHC + DISH 切片後執行。送出後下方「任務狀態」
   會輪詢顯示 running/done、計時與結果路徑，完成後切片清單自動刷新。

---

## 5. 改了後端 schema 之後：重生前端 typed client

只有動到 Pydantic schema / API 形狀時才要做（後端須在跑）。cwd 在 `frontend\`：
```powershell
$env:Path = "C:\Users\RCLab\miniconda3\envs\tsgh311;" + $env:Path
& "C:\Users\RCLab\miniconda3\envs\tsgh311\npx.cmd" openapi-typescript http://127.0.0.1:8000/openapi.json -o src/api/schema.ts
```
`src/api/schema.ts` 是**生成物，不要手改**。

---

## 6. 踩雷對照表

| 症狀 | 原因 / 解法 |
|---|---|
| 前端「無法取得切片清單(後端未啟動?)」 | 後端沒起 / port 不對 / proxy 沒生效。先用 §2 的 curl 驗後端。 |
| 清單是空的 | `TSGH_SLIDES_DIR` 指錯，或資料夾裡沒有支援副檔名（tiff/svs/...）。 |
| 開圖瞬間後端 `MemoryError` / 整台卡住 | 丟進去的是 `subifd=True` 原檔（`level_count=1`）。**必須換成 viewer 副本**（§1）。 |
| `'node' is not recognized` | 沒把 env 根加進 PATH（§3 第一行）。 |
| `npx openapi-typescript` peer 版本錯 | 前端 TypeScript 要釘 5.9（已在 `package.json` 釘好，別升 6）。 |
| OSD 沒有放大縮小按鈕 | 刻意停用（離線無 CDN 圖）——用滾輪縮放、拖曳平移。 |

---

## 7. 一鍵摘要（照順序貼三段）

```powershell
# 【視窗 A】後端
$py = "C:\Users\RCLab\miniconda3\envs\tsgh311\python.exe"
$env:PYTHONPATH = "C:\Users\RCLab\Desktop\tsgh"
$env:TSGH_SLIDES_DIR = "D:\tsgh_output\thriple_image_layer\viewer"
& $py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
```powershell
# 【視窗 B】前端
$env:Path = "C:\Users\RCLab\miniconda3\envs\tsgh311;" + $env:Path
Set-Location "C:\Users\RCLab\Desktop\tsgh\frontend"
npm run dev
# → 開 http://localhost:5173
```
