# 05 — 開發與測試指南

> 怎麼在本地把它跑起來、怎麼確認沒改壞。

## 現有測試方式：沒有自動化測試

- **無 pytest、無 CI**。專案根目錄與 hybrid 目錄下**沒有 test 檔**（codegraph 列出的 `test_*.py` 是幻影，實際不存在 —— 見 [07](./07-gotchas-appendix.md)）。
- 現有驗證方式是**手動 CLI 跑** + 目視檢查輸出圖：
  - `python hybrid_pipeline.py --batch --test`（跑 `test_picture/` 內的小圖）
  - `python hybrid_pipeline.py --ihc <tile> --dish <tile>`（單對）
- `elastic_matching_v3_explainer.html` 底部提到的 `/tmp/test_nucleus_centric.py`、`/tmp/test_detect_wiring.py` 是當時一次性驗證腳本，**放在 /tmp、不在 repo**，別預期能找到。

## 本地跑法（詳細步驟）

```bash
# 0. 進專案 venv（記憶：python/test 一律 source 這個 venv）
source /home/sec312/project/tsgh/.venv/bin/activate

# 1. 建 config（config.py 是 gitignored，不 cp 跑不動）
cd /data/tsgh/cell_mask/hybrid
cp config_example.py config.py

# 2. ⚠️ 編 config.py：cp 完還要補檔尾（見下方警告），並填模型路徑/tile 目錄
#    - unet_model_path / cellpose_model_path / cellpose_dish_model_path
#    - ihc_tile_dir / dish_tile_dir（或 ihc_test_dir / dish_test_dir）
#    - output_dir / slide_id / model_version

# 3. 跑
python hybrid_pipeline.py --batch --test          # 用 test_picture 小圖（最快驗證）
python hybrid_pipeline.py --batch                  # 正式 tile 目錄
python hybrid_pipeline.py --ihc A.tiff --dish B.tiff --output out/   # 單對
```

> **⚠️ `cp config_example.py config.py` 之後還不能直接跑**：`config_example.py` 尾端**缺少** `compute_config_hash()` 函式與 `config = Config()` 這行，而 `hybrid_pipeline.py` 是 `from config import config, compute_config_hash`。直接 cp 會 `ImportError`。要從現有的 `config.py`（若已存在）或參考 [07](./07-gotchas-appendix.md) 手動補回這兩段。若機器上已有可跑的 `config.py`，別覆蓋它。

### 模型檔位置（已在 repo）
```
models/unet/b4/best_model_unet_b4.pth
models/unet/b6/best_model_unet_b6.pth
models/cellpose/cellpose_ihc_dish_best      # M2 IHC-DISH 細胞
models/cellpose/cellpose_dish_best          # M3b DISH 核
```
（注意 `config_example.py` 的預設模型路徑指向舊位置 `models/best_model_unet.pth` 等，與實際 `models/unet/b4/…`、`models/cellpose/…` 不符 —— cp 後要改成實際路徑。）

## 測試資料位置與規模

| 用途 | 位置 | 規模 |
| --- | --- | --- |
| 快速驗證小圖 | `test_picture/{her2,dish,merge}/` | 單 tile 約 2–3MB |
| 正式 tile | `tile/{her2,dish,merge}/` | 依切法 |
| 完整 WSI（實測估算來源） | `her2_warped_lv0.ome.tiff` 等 | 156k×134k px；warped 檔約 **21.5GB / 17.6GB** 級 |
| 輸出 | `output/<tile_id>/` | 每 tile 一子目錄（CSV + 多張 overlay/crop PNG） |

- **驗證 M0 分塊/縫合**建議用「一張略大於 1024（會觸發多 chunk）的 ROI」，比小 tile 更能暴露接縫問題。
- 純資料重組層（`m0_stitch`）可用**合成 numpy** 單測（不需模型）—— 這是最該補自動化測試的地方。

## 回歸基準（怎麼確認沒改壞）

CLAUDE.md 定義的**權威回歸基準**：

> **單塊輸入（≤ `tile_size`=1024）的輸出，必須 bit-identical 於 pre-M0 的單影像路徑。**

- 原理：輸入 ≤ 1024 時只 yield 單一 chunk → 核心區 = 整塊、無接縫、全域 ID == 局部 ID → `StitchAccumulator` 退化成「原樣複製」；pyvips 解碼對 JPEG-TIFF 與 `skimage.io.imread` 逐位元相同。
- **驗法**：拿一張 ≤1024² 的 tile，比對「走 M0 路徑」與「pre-M0 直讀」的 CSV/mask。
- **重要例外**：**GPU 推論本身非決定性**（Cellpose/UNet 前向跨 run 有微小抖動）。所以跨 run 比對要用 **noise floor（容忍閾值）** 而非精確相等；真正該 bit-identical 的是「同一份 mask 進去、縫合/去重/重編號」這段純資料邏輯。
- 改 `m0_stitch` / `m0_reader` / 去重邏輯後，這條基準是你的護欄。

## perf_report.html 怎麼重新產生

- **目前沒有自動化腳本** —— `output/perf_report.html` 是**手動量測**產出的（`--batch --test` 3 tiles + cProfile + 資源取樣，再人工整理成 HTML）。
- 要更新數字：手動包一層 `cProfile` 跑 `run_batch`、同時取樣 GPU/CPU/RAM（如 `nvidia-smi`/`psutil`），再更新 HTML。
- **提醒**：現有數字量測於 2026-06-29，在 M0 記憶體優化 commit 之前（見 [03](./03-benchmarks-bottlenecks.md) 的量測時點說明）；瓶頸排名仍有效，絕對秒數建議重跑取得。
