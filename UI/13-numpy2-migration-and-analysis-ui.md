# 13 — numpy 2 遷移、Windows 可跑化、分析 UI（含 ROI）

> **一句話**：把整個 stack 從 numpy 1 換到 numpy 2（valis fork HEAD 已強制要求），
> 讓後端在 **Windows** 上能直接 `uvicorn backend.main:app` 起來，修掉影像對準 UI 實測出的
> blocker，並把設計稿裡的**分析（hybrid）UI** 從零做完，加上 **ROI 範圍分析**。
>
> 日期：2026-07-27 ～ 2026-07-28。對應設計稿：[`hybrid_flow_mockup.html`](hybrid_flow_mockup.html)、
> [`alliment_ui_plan.html`](alliment_ui_plan.html)。護欄依 [`04-guardrails-red-lines.md`](04-guardrails-red-lines.md)。

---

## 1. Windows 上把後端跑起來

出發點是「實機測影像對準 UI」，但**後端在這台 Windows 機器上根本 import 不進去**，所以先修這些。
全部都是平台相容性問題，不是功能改動。

| 症狀 | 根因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'fcntl'` | `tuspyserver` 4.2.12 是 POSIX-only，import 時就炸 | 新增 **`backend/api/tus_compat.py`**：`sys.platform == "win32"` 時裝一個 `fcntl` stub |
| 上傳 PATCH 回 500 `[WinError 183]`（檔案已存在） | `tuspyserver/info.py` 用 `os.rename`，Windows 上目標已存在會失敗 | 同檔案用 `_RenameIsReplace` proxy 把該模組的 `os.rename` 換成 `os.replace` |
| `AttributeError: module 'SimpleITK' has no attribute 'TransformixImageFilter'` | 裝到的是 stock SimpleITK 而非 SimpleElastix 時，module 匯入期就掛 | ~~`module2_alignment.py` 加 `hasattr` guard~~ **已還原**（2026-07-28，見下方註記）——`.venv` 裝的是 SimpleElastix 所以不受影響，但**任何 stock SimpleITK 的環境仍會 import 就掛** |
| `ModuleNotFoundError: No module named 'resource'` | `m0_stitch._ensure_nofile_limit` 用 POSIX 的 `resource` | `try/except ImportError` + 早退（Windows 沒有 fd 上限問題） |
| `shutil.rmtree` 撞 `WinError 32` | pyvips 對 `.v` 暫存檔的 mmap handle 尚未釋放 | ~~`module1_preprocess.py` 改 `ignore_errors=True`~~ **已還原**（見下方註記）——Linux 上不會發生此錯誤，且該寫法會把真正的清理失敗無聲吞掉 |

> **整個 `thriple_image_layer/` 已完整還原（2026-07-28）**：依組員要求，該資料夾的演算法行為不得
> 更動，`module2_alignment.py` 與 `module1_preprocess.py` 都已 `git checkout` 回改動前的版本
> ——`git diff dcbf11f -- backend/algorithms/thriple_image_layer/` 為空。跟著回來的是
> **非剛性配準強制開啟、寫死 `SimpleElastixWarper`**（原本加的 `config.valis.non_rigid_method`
> 開關已不存在）。注意 `module4_aligned_layers.py:42` 仍在讀 `config.valis.non_rigid_method`，
> 但它現在管不到 module2 了——兩邊會對不起來，是已知的待處理項。

> **`tus_compat` 的 import 順序是有意義的**：`backend/api/alignment.py` 裡它必須排在
> `from tuspyserver import ...` **之前**，所以帶了 `# noqa: F401` 註記。不要讓 formatter 重排。

結果：`uvicorn backend.main:app` 可直接啟動，CZI 上傳在 Windows 端到端成功
（POST 201 + PATCH 204，檔案落在 `czi_input/{HER2,DISH,HE}_40X.czi`）。

---

## 2. numpy 1 → numpy 2

### 為什麼要換

valis fork（`github.com/diadia0000/valis`）的 HEAD 把自己的需求從 `numpy<2` **翻轉成
`numpy>=2,<3`**。而 `pyproject.toml` 的 `[tool.uv] override-dependencies` 裡釘著 `numpy<2`
—— **override 的優先權高於套件自己的 metadata**，所以那幾行留著就會裝出「numpy 2 的 valis
疊在 numpy 1 上」的組合。這不是「順便升級」，是不換就裝不對。

### 改了什麼

`pyproject.toml`：

- `numpy<2` → `numpy>=2,<3`
- `opencv-contrib-python-headless<4.9` → `>=4.10,<5`（valis HEAD 同步抬高了下限）
- **移除** `override-dependencies` 裡的 `numpy` / `opencv*` / `scikit-image` 五行，原地留註解說明為什麼不能留
- 保留 drop stock simpleitk 的 override（SimpleElastix 由 `simpleitk-simpleelastix` 提供）

流程就是刪掉 `uv.lock` 後 `uv sync`（179 個套件全數重解）。

### 換完的實際版本

```
numpy   2.4.6
torch   2.11.0+cu130      cuda True
valis-wsi 1.2.0  (git 50813e7ce518878374be8c51bad3feee736fe073)
opencv-contrib-python-headless 4.13.0.92
SimpleITK.TransformixImageFilter  存在（SimpleElastix 可用）
```

CUDA torch 沒有被降級，cellpose / smp / dinov3 都跟著解得動。

### 驗證與已知的坑

- `Valis.register()` 用**本專案自己的 config**（DiskFD + LightGlue、非剛性關閉）在 numpy 2.4.6
  上跑得乾淨。
- 另一組探測 config 在 stage B 撞到 `TypeError: expected np.ndarray (got NoneType)`。追下去是
  **fork 自己的 bug**：`feature_matcher.py:1451` 該用剛算出來的 `r_kp2`/`r_desc2`，卻用了
  `kp2_xy`/`desc2`。**與 numpy 2 無關，也不在本專案 pipeline 走的路徑上**，但如果哪天要開別的
  matcher 組合會撞到。
- 用 numpy 2 環境實跑過一整輪 M0→M4 GPU 分析，出得了 `report.csv` / `summary.txt` /
  `overlay_slide.tiff`。

> ⚠️ **環境分裂**：conda 的 `tsgh311` 仍是舊的（numpy 1.26.4 / torch 2.5.1+cu121）。
> 新環境是 repo 內的 **`.venv`**（`uv sync` 產出）。跑後端請用 `.venv`，不要再用 conda 那個。
>
> ⚠️ **測試一定要用 `.venv\Scripts\python.exe -m pytest`。** `uv run pytest` 在這台會**安靜地**
> 掉到 conda `tsgh311` 的 `pytest.exe`（PATH 上找得到），於是整套測試是在 **numpy 1.26.4 +
> stock SimpleITK** 底下跑的，量到的東西跟實際 runtime 無關。`.venv` 原本沒裝 pytest 才會這樣，
> 現已補裝。

> ⚠️ **`config.py` 是 gitignored 的**：`backend/algorithms/hybrid/config.py` 不進版控。這台機器上
> 那份是舊的，害 `test_config_parity` 5 個案例紅。從 `config_example.py` 重新產生後全綠
> （舊檔備份在 `config.py.backup-20260728`，未進版控）。新機器上請照 `config_example.py` 生成。

---

## 3. 影像對準 UI：實測修掉的問題

實機（瀏覽器）跑過一輪，測出來的東西：

### 3.1 前端輪詢在背景分頁會停 —— 這是最嚴重的一個

`frontend/src/api/jobs.ts` 的 `useJob` 只發過**一次** `/api/jobs/{id}` 就不再發，
UI 永遠卡在 `running`，即使後端那個 job 早就 `error` 了。

根因是 TanStack Query 預設 `refetchIntervalInBackground: false`——**分頁不在前景時輪詢會暫停**。
這不只是「畫面不更新」而已：`PipelinePanel` 是等前一步的輪詢回報 `done` 才送出下一步，
所以使用者切走視窗，**整條 pipeline 就停在那裡不往下走**。對準流程動輒數十分鐘到數小時，
使用者一定會切視窗。

修法是加 `refetchIntervalInBackground: true`。

### 3.2 viewer 顯示的到底是哪個 run

`aligned_her2` / `aligned_dish` / `aligned_result` 是**全域 slide_id**，誰最後 publish 就歸誰。
面板上選了另一個 run 並不會換掉畫面上的影像，之前完全沒有任何提示。

- 後端：`backend/api/alignment.py` 加 module 級 `_published_run_id`（跟 `pyramid` 的 registry
  一樣是 in-memory，隨 process 消失），`publish_run()` 與 `run_thumbnail()` 完成時寫入，
  新增 **`GET /api/alignment/published`**（schema `PublishedRun`）。
- 前端：`OverlayViewer` 顯示「來源工作：`<run_id>`」。

### 3.3 介面細節

`PipelinePanel.tsx`：

- 第一步不顯示「重跑」checkbox（第一步本來就沒有前一步可跳過）
- 該 run 已全部完成且沒勾重跑時，按鈕禁用並顯示「此工作已完成」，不再讓人按了沒反應
- 錯誤訊息加 `break-all`，長路徑不再撐破版面

---

## 4. 分析 UI（hybrid）

設計稿裡的分析流程之前**完全沒有前端**，只有 API。這次補完。

### 4.1 前端

新檔 **`frontend/src/components/HybridPanel.tsx`**：

```
選 IHC 切片 → 自動預覽 → （可選）框 ROI → 顯示「約 N 個 tile」
  → 開始分析 → 輪詢 job → 完成後顯示結果卡
```

結果卡：`summary.txt` 內容（分數摘要）、下載 `report.csv`、一鍵切到標註 overlay。

配合的既有元件改動：

- `SlideViewer.tsx`：`export` 了 `ImageRect`，新增 `onViewportChange`。回報的矩形已經
  **clamp 到影像範圍內**（OpenSeadragon 的 viewport 會超出影像邊界），否則使用者稍微縮太遠，
  送出去的 ROI 就會越界被後端擋掉。
- `App.tsx`：改成同時管 `singleSlide` 與 `viewRect`，切換切片時清掉舊的矩形（不然會拿 A 切片的
  座標去分析 B 切片）。
- `OverlayViewer.tsx`：見 3.2。

### 4.2 後端

`backend/api/hybrid.py`：

- `HybridTileIn` 從 `ihc_path`/`dish_path` 改成 **`ihc_slide_id`/`dish_slide_id`**
  （護欄 2：前端不碰檔案系統路徑），端點自己 `pyramid.resolve()`，壞 id 當場 404
  而不是等背景 job 才失敗
- 新增 **`GET /api/hybrid/result`** → `HybridResult{summary, has_report, overlay_slide_id}`。
  還沒跑過時回空欄位而不是 404——「沒有結果」對面板來說是正常狀態
- 新增 **`GET /api/hybrid/report.csv`** → `FileResponse`（病理師要帶去別的工具的那份）
- `OVERLAY_SLIDE_ID = "hybrid_overlay"`，比照 alignment 的 `aligned_result` 註冊給 tile server

---

## 5. ROI 範圍分析

### 為什麼一定要有

整張 WSI 是 **~27,565 個 tile、數小時 GPU**。沒有 ROI 的話這個 UI 對真實切片是不能用的。

### 做法

`PrecutStream(region=(x, y, w, h))`：**在 ROI 的尺寸上算出一般的 tile 格線，再平移到 ROI 的原點**。
所以 tile 檔名維持**切片絕對座標**，下游的切割線、centroid core-ownership 去重、
`filter_and_absolutize` 全都不用改。

不直覺的一點：`m0_stitch._validate_axis` 原本硬性要求 `starts[0] == 0`。那個檢查**不是幾何上必需**
（`core_crop_bounds` 對最外側那一欄用的是 tile 自己的 `abs_x`），它是**「第一欄不見了」的守門員**。
所以 `compute_tile_geometry` 改成收一個明確的 `origin=(x, y)`（預設 `(0, 0)`），
**而不是從 `starts[0]` 推**——用推的會把那個守門員無聲地丟掉。
`hybrid_pipeline.run_batch` 從 `tile_stream.region` 讀這個 origin。

API 端 `HybridTileIn.roi_x/y/w/h`：四個一起給或都不給（`model_validator` 擋），
邊界檢查放在 `PrecutStream`——那是唯一知道切片實際尺寸的地方。

### 已知缺口

ROI 切出來的 tile **目錄**如果之後用 dir-scan 路徑重掃（例如 `--resume`），沒有 origin 可傳，
會過不了格線檢查。今天不影響——API 一律走 stream 路徑——但 `--resume` 哪天遇到 ROI run 就會撞到。

---

## 6. 驗證紀錄

### 6.1 這批工作本身（合併前）

| 項目 | 結果 |
|---|---|
| 測試 | **62 passed, 1 skipped**（`tests/` 51+1、`backend/tests/` 11，**分開跑**） |
| 前端 | `tsc` 與 `oxlint` 皆乾淨 |
| ROI（API 直打） | `{'success': 0, 'skipped': 1}`，`overlay_slide.tiff` 剛好 1024×1024 |
| ROI（UI 操作） | 13 秒 `done`，自動切到標註 overlay |
| tile 數對比 | 整張 2048² 合成切片 = 9 tiles；ROI (512,512,1024,1024) = 1 tile |
| numpy 2 全流程 | M0→M4 GPU 實跑，三個產物齊全 |

### 6.2 併入 main 之後的複驗（2026-07-28）

併進去的時候 `origin/main` 已經往前 8 個 commit（round-9 的 prefetch / gc re-freeze /
stitch pyramid levels / 判讀依據修改），`hybrid_pipeline.py` 與 `m0_stitch.py` 兩邊都動到，
由 git 自動合併。所以整套重驗了一次：

| 項目 | 結果 |
|---|---|
| 測試 | **80 passed, 1 skipped**（`tests/` 69+1、`backend/tests/` 11） |
| 前端 | `npm run build`（`tsc -b` + vite）通過、`oxlint` 乾淨 |
| 後端啟動 | `uvicorn backend.main:app` 正常起來 |
| `GET /api/tiles` | `["TEST_DISH","TEST_IHC","hybrid_overlay"]`；`.dzi` 與實際 tile JPEG 都取得到 |
| `GET /api/alignment/runs` | 5 個 run 的進度正確（`demo-done` 四步齊全） |
| `POST /api/alignment/publish` + `GET /published` | publish `demo-done` 後 `aligned_*` 三個 id 上線，`/published` 回報 `demo-done` |
| **ROI 分析端到端** | 15.1 秒 `done`，`{success: 0, skipped: 1}`，`overlay_slide.tiff` 仍是 1024×1024（ROI 有生效） |
| `GET /api/hybrid/result` / `report.csv` | summary 正確（UTF-8 無誤）、CSV 下得下來 |
| 錯誤路徑 | 壞 slide_id → 404；ROI 只給一個參數 → 422 |

> 合併後 `run_batch` 裡 ROI 的 `origin` 傳遞邏輯完整保留，沒有被 prefetch 那批改動蓋掉；
> 對方那 18 個新測試（寫於 numpy 1 時代）在 numpy 2 上也全綠。
>
> 另外要知道的：這次重解把 **starlette 從 0.4x 拉到 1.3.1、fastapi 到 0.140.5**（`app.routes`
> 現在是 `_IncludedRouter` 包起來的，用舊寫法列路由會看起來像「沒註冊」，實際有）。
> tus 上傳（POST 201 + PATCH 204，含斷點續傳）由 `backend/tests/test_chunked_upload.py`
> 在新 starlette 上驗過。

新增測試 **`tests/test_precut_roi.py`**（10 個案例）：格線平移、tile 不越界、ROI 全覆蓋、
寫出的像素與檔名座標一致、origin-aware 幾何、**少掉第一欄仍然要報錯**、
core region 對 ROI 恰好鋪滿一次（`hits.min() == 1 and hits.max() == 1`）、越界／小於一個 tile 被拒。

`tests/test_stitch_nofile_guard.py` 在 Windows 上改為 skip（那是 POSIX `resource` 的測試）。

> **既有問題（不是這次改出來的）**：`tests/` 和 `backend/tests/` 合在**同一個 process** 跑會
> collection error——`backend/algorithms/thriple_image_layer/config.py` 和 `hybrid/config.py`
> 同名，在 `sys.path` 上互相蓋掉。`tests/conftest.py` 和 `test_run_batch_resume.py` 本次未修改。
> 暫時解法就是兩個目錄分開跑。

---

## 7. 還沒做的

| 項目 | 卡在哪 |
|---|---|
| **逐 tile / 逐階段進度** | `run_batch` 執行中完全不回報，只在結束時回 `{success, skipped}`。要動演算法層加 callback。 |
| **取消任務** | FastAPI `BackgroundTasks` 無法取消。要換執行模型（例如 process pool + 可中斷的 job registry）。 |
| **真 CZI 的完整對準實跑** | `picture/` 裡三張真檔沒有端到端跑過一次。測試用的是 `D:\tsgh_storage_test` 假造的 run 資料夾（進度本來就是從磁碟產物推的）＋ `D:\tsgh_test_slides` 合成切片，因為真跑一次要數小時。 |
| **ROI + `--resume`** | 見 §5 已知缺口。 |

---

## 7b. 改到 Linux 跑要注意的

§1 那些 Windows 相容性問題在 Linux 上都不存在（`tus_compat.py` 整個檔案是 `sys.platform ==
"win32"` 包起來的，Linux 完全不執行），但下面這些是**換平台不會消失、甚至只在 Linux 出現**的：

- **`ulimit -n`（RLIMIT_NOFILE）** ← 只在 Linux 咬人。縫合會把整片的 tile 同時開著（真實切片
  27,565 個），Linux 常見預設 soft limit 是 **1,024**，會在整片分析跑完數小時後的最後一步炸。
  `_ensure_nofile_limit()` 會自己把 soft 提到 hard，但 **hard 不夠就沒救**——上機前確認
  `ulimit -Hn`，Docker 用 `--ulimit nofile=1048576:1048576`。
- **記憶體**：非剛性配準在 32 GB 機器上 OOM 是記憶體問題不是平台問題；整片 hybrid 分析的
  peak RSS 記錄是 **61–62 GB**（BACKLOG §1 item 7）。
- **磁碟**：整片 precut 暫存約 49 GB，加上 `_stitch_scratch`。
- **SimpleElastix 的 wheel 只有三個平台**（macOS arm64 / **linux x86_64** / win amd64）。
  x86_64 照 `uv.lock` 裝沒問題；**aarch64 Linux 裝不起來**，而 `module2_alignment.py` 還原後
  沒有 import guard，後端會 import 就掛。
- **檔名大小寫**：Linux 區分大小寫，`HER2_40X.czi` ≠ `her2_40x.czi`。

---

## 8. 怎麼跑

```powershell
# 後端（用 .venv，不是 conda tsgh311）
$env:PYTHONPATH        = "C:\Users\RCLab\Desktop\tsgh"
$env:TSGH_STORAGE_DIR  = "D:\tsgh_storage_test"
$env:TSGH_SLIDES_DIR   = "D:\tsgh_output\thriple_image_layer\viewer"   # 測試片：D:\tsgh_test_slides
& ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 前端
cd frontend; npm run dev

# 測試（一定要指名 .venv 的 python，別用 uv run pytest；兩個目錄分開跑，見 §6）
& ".\.venv\Scripts\python.exe" -m pytest tests
& ".\.venv\Scripts\python.exe" -m pytest backend/tests
```

> 這台機器上 Chrome 連 `localhost` / `127.0.0.1:5173` 連不上（`curl` 卻是 200），
> 用區網 IP `http://210.240.160.153:5173/` 才進得去。
