# 03 · 目錄結構

> 上一篇：[02 技術棧與版本](02-tech-stack-versions.md)　·　下一篇：[04 護欄與紅線](04-guardrails-red-lines.md)　·　上層：[README](README.md)

## 現況（Phase 0）—— 什麼存在、什麼要搬、什麼不動

```
tsgh/
├── thriple_image_layer/          ← 對齊 / 前處理層【Phase 1 會搬入 backend/algorithms/】
│   ├── module1_preprocess.py         → algorithms/preprocess/
│   ├── module2_alignment.py          → algorithms/alignment/
│   ├── module3_roi_evaluation.py     ⚠ 未對應（見下）
│   ├── module4_thumbnail.py          → algorithms/thumbnail/
│   ├── module5_tile_generator.py     ⚠ 未對應（見下）
│   ├── run_full_pipeline.py          ⚠ CLI 入口，搬去哪待定
│   ├── config.py（gitignored）/ config_example.py（tracked）
│   └── artifacts/  output/
│
├── cell_mask/
│   ├── hybrid/                    ← DISH 混合 pipeline【Phase 1 會搬入 backend/algorithms/hybrid/】
│   │   ├── hybrid_pipeline.py  m0_reader.py  m0_stitch.py  m1_overlay.py
│   │   ├── m2_segmentation.py  m3_cell_detection.py  m3_module/
│   │   ├── m4_export.py  m4_module/  unet_inference.py  models/
│   │   ├── config.py（gitignored）/ config_example.py（tracked）
│   │   └── CLAUDE.md（此層專屬規則）
│   ├── dish_mask/                ⚠ 未對應（獨立於 hybrid）
│   └── unet_mask/                ⚠ 未對應（獨立於 hybrid）
│
├── scripts/                      ← CLI / 工具，【保留，不進 UI】
│   ├── tile_generator.py         ⚠ 與 module5_tile_generator.py 疑似重複
│   ├── tiff to png.py  check_tiff_size.py  cuda_test.py
│
├── docs/                         ← 本文件所在【不動】
├── paper/                        ← 論文素材【不動】
├── pyproject.toml  uv.lock  requirements.txt  README.md
└── Dockerfile  docker-compose.yml

（尚無 backend/ 與 frontend/ —— 這是 Phase 0 的關鍵事實）
```

---

## 目標（Phase 1+）—— 提議

```
tsgh/
├── backend/                      ← 新增
│   ├── api/                      ← FastAPI endpoints（唯一 HTTP 介面）
│   │   ├── alignment.py
│   │   ├── cell_detection.py
│   │   ├── roi.py                ← 座標轉換唯一出沒地（見 04 / 05）
│   │   └── tiles.py              ← DeepZoom 切片服務
│   ├── algorithms/               ← 從現有目錄搬入（函式內容不動）
│   │   ├── preprocess/           ← 原 module1_preprocess.py
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
│   ├── package.json  vite.config.ts
│
├── scripts/                      ← 保留：CLI 入口，不經過 UI
├── docs/  packaging/  pyproject.toml
```

**關鍵不變式**：`backend/algorithms/*.py` **永遠能被 CLI 獨立呼叫**，不依賴 FastAPI 啟動。`scripts/` 內的 CLI 入口＝演算法的第二個使用者（UI 是第一個），保證演算法不被 UI 框架綁架。

---

## 遷移對照表

| 現有路徑 | 目標路徑 | 動作 |
|---|---|---|
| `thriple_image_layer/module1_preprocess.py` | `backend/algorithms/preprocess/` | 搬 + 改 import path，內容不動 |
| `thriple_image_layer/module2_alignment.py`（+ VALIS wrapper） | `backend/algorithms/alignment/` | 搬 + 改 import path |
| `thriple_image_layer/module4_thumbnail.py` | `backend/algorithms/thumbnail/` | 搬 + 改 import path |
| `cell_mask/hybrid/`（整個） | `backend/algorithms/hybrid/` | 搬 + 改 import path |
| WSI 讀取邏輯 | `backend/io/wsi_reader.py` | 抽出 |
| 金字塔 / DeepZoom 切片 | `backend/io/pyramid.py` + `backend/api/tiles.py` | 新建（用 openslide/pyvips，不手刻） |
| `scripts/*.py` | `scripts/`（原地） | 保留，當 CLI 第二使用者 |

驗收（Phase 1）：既有所有 script 都跑得起來，輸出 **bit-identical**。

---

## ⚠ 未對應項目清單（需要人類決策）

以下項目**不在**原始遷移表內，歸類有歧義，動手前要先跟作者確認：

| 項目 | 歧義 | 可能歸屬 |
|---|---|---|
| `thriple_image_layer/module3_roi_evaluation.py` | ROI 評估算演算法還是 API 職責？ | `algorithms/roi/` ？或併入 `api/roi.py` 的下游？ |
| `thriple_image_layer/module5_tile_generator.py` | 切片生成與 `io/pyramid.py`、`api/tiles.py` 職責重疊 | `io/` ？還是廢棄（改用 DeepZoom）？ |
| `scripts/tile_generator.py` | 與 `module5_tile_generator.py` **疑似重複**，兩份 tile generator | 需確認哪份是現役、能否合併 |
| `thriple_image_layer/run_full_pipeline.py` | 全流程 CLI 入口 | 搬 `scripts/` 當 CLI 第二使用者？還是留原地？ |
| `cell_mask/dish_mask/` | 獨立於 `hybrid/`，實驗性 / 舊版？ | 搬 `algorithms/` ？還是不搬（棄用）？ |
| `cell_mask/unet_mask/` | 同上，與 `hybrid/unet_inference.py` 關係不明 | 需確認是否為 hybrid 的上游資產 |
| 各層的 `config.py`（gitignored） | 每個 pipeline 各一份、且不進 git | 搬檔後如何統一 / 對接 Pydantic schema（見 05） |
