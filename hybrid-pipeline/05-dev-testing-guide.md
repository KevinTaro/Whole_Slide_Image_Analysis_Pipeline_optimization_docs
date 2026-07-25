# 05 — 開發與測試指南

> 怎麼在本地把它跑起來、怎麼確認沒改壞。

## 現有測試方式：hybrid 目錄本身仍沒有自動化測試

- **`backend/algorithms/hybrid/` 目錄下仍沒有 test 檔**（codegraph 列出的 `test_*.py` 是幻影，實際不存在 —— 見 [07](./07-gotchas-appendix.md)）。**注意這個結論範圍限定在 hybrid 目錄**：專案根目錄的 `backend/tests/`（`test_chunked_upload.py`/`test_module1_strips.py`/`test_openslide_before_pyvips.py`/`test_resume.py`）已有真的 pytest 測試，但涵蓋的是對齊/上傳/resume 相關子系統，不含 hybrid pipeline。
- hybrid pipeline 唯一的自動化檢查是 `scripts/verify_gc_freeze.py`（非 pytest，獨立腳本）——驗證 `gc.freeze()` 的 cadence/pairing invariant（見 [15-gc-collect-frequency-implementation.md](./15-gc-collect-frequency-implementation.md) §4），不是 pipeline 輸出正確性測試。
- 現有 pipeline 輸出驗證方式仍是**手動 CLI 跑** + `report.csv` 比對（見下方回歸基準）：
  - `python hybrid_pipeline.py --test`（跑 `test_picture/` 內建 ROI 範例，走完整 precut+分析路徑）
  - `python hybrid_pipeline.py --ihc <tile/ROI/WSI> --dish <tile/ROI/WSI>`（單對，任意大小）
  - ⚠️ **`--batch` flag 不存在**：現行 `build_arg_parser()`（`hybrid_pipeline.py`）只有 `--test`／`--ihc`／`--dish`／`--output` 四個參數，沒有「對一整個 tile 目錄跑批次」的獨立模式 —— `--ihc`/`--dish` 本身就接受單 tile、ROI 或整張 WSI，內部會自動 `PrecutStream` 切好再送進 `run_batch()`。下面本節與模型檔小節先前寫的 `--batch ...` 是舊介面，已於本輪文件更新中修正為現行參數。
- `docs/algo/elastic_matching_v3_explainer.html` 底部提到的 `/tmp/test_nucleus_centric.py`、`/tmp/test_detect_wiring.py` 是當時一次性驗證腳本，**放在 /tmp、不在 repo**，別預期能找到。

## 本地跑法（詳細步驟）

> ⚠️ **路徑已更新**：程式碼已從 `cell_mask/hybrid/` 搬到 `backend/algorithms/hybrid/`（UI
> Phase 1 目錄重構），venv 也已改在本 repo 根目錄下的 `.venv/`，不再是舊的
> `/home/sec312/project/tsgh/.venv`。下面已更新為目前路徑；若你機器上的實際路徑不同，
> 以你自己 `ls -d .venv` / `git rev-parse --show-toplevel` 的結果為準，不要照抄。

```bash
# 0. 進專案 venv（記憶：python/test 一律 source 這個 venv）
cd <repo 根目錄>          # 例：/data/taro_Projects/tsgh
source .venv/bin/activate

# 1. 建 config（config.py 是 gitignored，不 cp 跑不動）
cd backend/algorithms/hybrid
cp config_example.py config.py

# 2. 編 config.py：填模型路徑/tile 目錄（G2 的「缺檔尾」問題已修復，見下方更新，cp 完可直接 import）
#    - unet_model_path / cellpose_model_path / cellpose_dish_model_path
#    - ihc_tile_dir / dish_tile_dir（或 ihc_test_dir / dish_test_dir）
#    - output_dir / slide_id / model_version

# 3. 跑
python hybrid_pipeline.py --test                                     # 用內建 test_picture ROI（最快驗證）
python hybrid_pipeline.py --ihc A.tiff --dish B.tiff --output out/   # 單一 ROI/WSI 對，任意大小

# 量測（round 3/4 用的正式量測腳本，非手動 cProfile）：
python ../../../scripts/perf_measure.py --ihc <A.tiff> --dish <B.tiff> \
  --output <out> --label <tag> --workers 8 --gpu-dmon --metrics-dir <m>
```

> **更新（已修復）**：舊版 `config_example.py` 尾端曾缺少 `compute_config_hash()` 函式與
> `config = Config()` 這行，導致 `cp` 完直接 `ImportError`（見 [07](./07-gotchas-appendix.md) G2）。
> 現行 `config_example.py`（226 行，與 `config.py` 行數一致）已補上這兩段，`cp` 完可直接
> import，不需再手動補檔尾。若你手上的是更舊的 checkout，仍可能踩到這個坑。

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

## perf_report.html 怎麼重新產生（**已過時，改用 `scripts/perf_measure.py`**）

- `docs/hybrid-pipeline/measurement/perf_report.html` 是**手動量測**產出的（`--batch --test` 3 tiles + cProfile + 資源取樣，再人工整理成 HTML），量測於 2026-06-29，在 M0 記憶體優化 commit 之前（見 [03](./03-benchmarks-bottlenecks.md) 的量測時點說明）。**不要再手動重現這份報告** —— 它已被下面的自動化腳本取代。
- **現行量測方式是 `scripts/perf_measure.py`**（round 2–4 都用它）：非侵入式 monkeypatch 計時 + `nvidia-smi dmon`（`--gpu-dmon`）+ `psutil` 資源取樣，輸出 `*_timings.json` / `*_resource.csv` / `*_gpu_dmon.txt`。配套分析腳本：`scripts/aggregate_report.py`、`scripts/resource_analyze.py`、`scripts/arm_report.py`（雙臂模型 + 餘裕）、`scripts/gc_ablation_report.py`（正確性 veto，最近質心配對）。
- 最新一輪的完整重現指令見 [18-gpu-starvation-prerequisites-implementation.md](./18-gpu-starvation-prerequisites-implementation.md) §9；現況數字一律以 [measurement/bottleneck-list.md](./measurement/bottleneck-list.md) 為準，不要再讀這份舊 HTML 當現況。
