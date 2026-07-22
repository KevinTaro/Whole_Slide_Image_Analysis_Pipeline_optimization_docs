# 06 · 開發環境

> 上一篇：[05 資料流與 API 合約](05-dataflow-api-contract.md)　·　下一篇：[07 Phase 路線圖](07-phase-roadmap.md)　·　上層：[README](README.md)
>
> ⚠️ **本檔「現在（Phase 0）能做什麼」一節已過時**：Phase 1–3 已完工，`backend/`/`frontend/`
> 都已存在且可跑，不再只有 CLI。實際跑法請直接看 [11-runbook-teammate.md](11-runbook-teammate.md)
> （後端+前端+操作，Windows/PowerShell）；下面保留純演算法 CLI 的跑法（仍然有效，適合不需要
> UI、只想跑 pipeline 本體/量測腳本的情境），並已把路徑改為目前實際的
> `backend/algorithms/...`（原 `cell_mask/hybrid/` 已隨目錄重構移除）。

## 純演算法 CLI（不透過 UI，仍然有效）

```bash
# 一律先 source 專案 venv（Python 3.11.15；實際路徑以 `ls -d .venv` 為準，不同機器可能不同）
source .venv/bin/activate

# config.py 是 gitignored，第一次要先複製範本再改路徑/參數
cp backend/algorithms/hybrid/config_example.py backend/algorithms/hybrid/config.py

# DISH 混合 pipeline —— 單張大 patch（≥1k 正方形）
python backend/algorithms/hybrid/hybrid_pipeline.py --ihc PATCH.tiff --dish PATCH.tiff

# DISH 混合 pipeline —— 批次（test_picture 內驗證圖）
python backend/algorithms/hybrid/hybrid_pipeline.py --batch --test [--output DIR]

# 對齊 / 前處理層（VALIS 配準管線 CLI）
python backend/algorithms/thriple_image_layer/run_full_pipeline.py --help

# 效能量測（round 2–4 用的正式腳本，非手動 cProfile；見 docs/hybrid-pipeline/measurement/）
python scripts/perf_measure.py --ihc <A.tiff> --dish <B.tiff> --output <out> \
  --label <tag> --workers 8 --gpu-dmon --metrics-dir <m>
```

---

## 環境事實

| 項目 | 現況 |
|---|---|
| Python | **3.11.15**（`.venv`，對齊 `uv.lock`） |
| 套件管理 | **uv**（`pyproject.toml` + `uv.lock`）；查版本見 [02](02-tech-stack-versions.md) |
| GPU | 演算法（Cellpose，round 3 起換成 4.2.1.1 `cpdino` backbone / torch cu130）**吃 GPU**；醫師電腦有無 GPU 尚未確認（[08](08-pitfalls-open-decisions.md) 未決事項） |
| Docker 替代方案 | `Dockerfile` + `docker-compose.yml` 已備：NVIDIA CUDA 13.0 + cuDNN + Ubuntu 24.04，`image: tsgh-pytorch:latest`。`docker compose build` → `docker compose run` 可在容器內跑，避免本機環境污染。 |

---

## Phase 1–3 開發指令（**已可用，不再是未來式**）

`backend/`、`frontend/` 已存在（[07](07-phase-roadmap.md) Phase 1–3 已完工）。完整的「照做跑起來」
教學（含 Windows/PowerShell 的實際路徑、slide 準備、踩雷表）在
**[11-runbook-teammate.md](11-runbook-teammate.md)**，這裡只列通用指令：

```bash
# 後端（熱重載）
uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# 前端（Vite dev server；package.json 用 npm，見 11 的實測指令，非 pnpm）
cd frontend && npm install && npm run dev

# 從 OpenAPI 生成前端 typed client（改完 Pydantic schema 後跑，見 05）
npx openapi-typescript http://127.0.0.1:8000/openapi.json -o src/api/schema.ts

# 測試 & 品質（尚未確認專案是否已接 CI/pytest 覆蓋率，視 backend/tests/ 現況為準）
pytest backend/                     # 演算法 / API 單元測試
ruff check . && ruff format .       # Python lint / format
tsc --noEmit                        # 前端型別檢查

# 桌面殼（Phase 5，打包前本機驗證）—— 尚未實作，見 07 Phase 5 狀態
uv run python backend/launcher.py   # pywebview 啟 backend + 開視窗（規劃中）
```

---

## 除錯備註

- **WSI 記憶體問題結論**（已驗證，別再走冤枉路）：那個 ~400GB 的爆量是 **`m0_stitch` 的縫合輸出畫布**（輸出端），**不是 reader 讀太多**。所以**換 WSI reader 省不了**——2026-07 實測把 warped 檔轉 JPEG(comp=7) 後 cuCIM 理論上可當 reader，但瓶頸在輸出端。真正的解是 **ROI 化 + 停止「整片縫合」**，只縫需要的區域。這也是 [05](05-dataflow-api-contract.md) 為什麼強調「以 ROI / tile 為單位」的實務根據。
