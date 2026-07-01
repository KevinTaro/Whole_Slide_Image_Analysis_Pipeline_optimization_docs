# 06 · 開發環境

> 上一篇：[05 資料流與 API 合約](05-dataflow-api-contract.md)　·　下一篇：[07 Phase 路線圖](07-phase-roadmap.md)　·　上層：[README](README.md)

## 現在（Phase 0）能做什麼

目前只有演算法 CLI，**沒有 UI 可跑**。你能做的是把既有 pipeline 跑起來，熟悉輸入輸出。

```bash
# 一律先 source 專案 venv（Python 3.11.15）
source /home/sec312/project/tsgh/.venv/bin/activate

# config.py 是 gitignored，第一次要先複製範本再改路徑/參數
cp cell_mask/hybrid/config_example.py cell_mask/hybrid/config.py

# DISH 混合 pipeline —— 單張大 patch（≥1k 正方形）
python cell_mask/hybrid/hybrid_pipeline.py --ihc PATCH.tiff --dish PATCH.tiff

# DISH 混合 pipeline —— 批次（test_picture 內驗證圖）
python cell_mask/hybrid/hybrid_pipeline.py --batch --test [--output DIR]

# 對齊 / 前處理層（VALIS 配準管線 CLI）
python thriple_image_layer/run_full_pipeline.py --help
```

---

## 環境事實

| 項目 | 現況 |
|---|---|
| Python | **3.11.15**（`.venv`，對齊 `uv.lock`） |
| 套件管理 | **uv**（`pyproject.toml` + `uv.lock`）；查版本見 [02](02-tech-stack-versions.md) |
| GPU | 演算法（Cellpose 4.1.1 / torch cu13）**吃 GPU**；醫師電腦有無 GPU 尚未確認（[08](08-pitfalls-open-decisions.md) 未決事項） |
| Docker 替代方案 | `Dockerfile` + `docker-compose.yml` 已備：NVIDIA CUDA 13.0 + cuDNN + Ubuntu 24.04，`image: tsgh-pytorch:latest`。`docker compose build` → `docker compose run` 可在容器內跑，避免本機環境污染。 |

---

## Phase 1+ 預期開發指令（**未來式，現在還沒有**）

等 `backend/` `frontend/` 建起來後大致長這樣：

```bash
# 後端（熱重載）
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# 前端（Vite dev server）
cd frontend && pnpm install && pnpm dev

# 從 OpenAPI 生成前端 typed client（改完 Pydantic schema 後跑，見 05）
pnpm run generate-client            # openapi-typescript / orval

# 測試 & 品質
pytest backend/                     # 演算法 / API 單元測試
ruff check . && ruff format .       # Python lint / format
tsc --noEmit                        # 前端型別檢查
pnpm run lint                       # ESLint + Prettier

# 桌面殼（打包前本機驗證）
uv run python backend/launcher.py   # pywebview 啟 backend + 開視窗
```

---

## 除錯備註

- **`scripts/tiff_preview_server.py` 是什麼**：一支小小的 **Flask** BigTIFF 本機預覽工具（`pyvips` 讀圖、render 成 HTML 頁）。它是**丟棄式除錯工具**，**不是** UI 的雛型。看 BigTIFF 尺寸/內容時很方便，但**不要把它擴充成正式 UI**——正式 UI 走 FastAPI + React（[01](01-architecture.md)、陷阱見 [08](08-pitfalls-open-decisions.md)）。
- **WSI 記憶體問題結論**（已驗證，別再走冤枉路）：那個 ~400GB 的爆量是 **`m0_stitch` 的縫合輸出畫布**（輸出端），**不是 reader 讀太多**。所以**換 WSI reader 省不了**——2026-07 實測把 warped 檔轉 JPEG(comp=7) 後 cuCIM 理論上可當 reader，但瓶頸在輸出端。真正的解是 **ROI 化 + 停止「整片縫合」**，只縫需要的區域。這也是 [05](05-dataflow-api-contract.md) 為什麼強調「以 ROI / tile 為單位」的實務根據。
