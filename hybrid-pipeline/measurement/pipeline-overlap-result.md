# ① GPU 序列瓶頸 — 方案 (b) 實作與量測結果

> 執行 [../10-gpu-serial-pipeline-plan.md](../10-gpu-serial-pipeline-plan.md) 的推薦方案 **(b)：單
> process 內兩段式 pipeline/overlap**，並依該文件第 5 節驗收標準量測。doc 10 保持不動（作為對照
> 規格）。本文件只記錄「做了什麼 + 量到什麼」，不改 doc 10 的結論。
>
> 環境：git 工作區基於 `ce980d1`；RTX 5090 / CUDA 13.0 / torch 2.10.0+cu130；venv `/data/taro_Projects/tsgh/.venv`。
> 量測輸入：`test_picture/_roi_crops/med_{ihc,dish}.tiff`（8192²，11×11 = **121 tiles**，medium 錨點）。

## 實作摘要（改動只在 `backend/algorithms/hybrid/hybrid_pipeline.py`）

把每個 tile 從中間切成兩段，用單一背景執行緒做深度 1 的 pipeline：

- **GPU 前段（主執行緒）** `_process_one_chunk_gpu` / `_process_precut_tile_gpu`：讀檔 → M1 UNet → M2
  Cellpose → M3b DISH Cellpose（`segment_windowed` 為止）。**三個 GPU 前向全部只在主執行緒 / 單一
  CUDA context 序列執行** —— 不碰跨行程 fork-under-CUDA 限制。
- **CPU 後段（背景執行緒）** `_finish_chunk_cpu` + `_process_precut_tile_cpu`：`detect_all_dots`
  （joblib `prefer='threads'`）+ merge + 核心去重 + 所有落地寫檔（PNG/TIFF/overlay/per-cell crop）。
  **完全不碰 torch**。
- `run_batch` 迴圈：tile N 的 GPU 前段在主執行緒跑時，tile N-1 的 CPU 後段在背景執行緒重疊執行；
  「先收前一塊、再提交本塊」，同時最多兩塊在飛 → 記憶體有界。
- `process_precut_tile` / `_process_one_chunk` 保留為同步 wrapper，行為與拆分前完全一致（單塊 /
  API / 回歸路徑不受影響）。fail-fast 與 tile 間 `empty_cache`/`gc.collect` 語意保留。

> 註：doc 10 §4 原估「只需動 `run_batch`」，但實測程式碼中三個 GPU 前向與 `detect_all_dots` 是
> **交錯在同一個 `process_precut_tile` 內**，故乾淨的重疊必須把該函式切成 GPU/CPU 兩段（見上）。
> 這是實作時對 §4 範圍註記的修正，方案設計方向（(b) 單 process、thread、GPU 留主執行緒）不變。

## 驗收結果（doc 10 §5）

| 標準 | baseline（原碼） | modified（方案 b） | 判定 |
|---|---|---|---|
| §5.1 端到端 wall-clock（121 tiles） | 255.1 / 255.0 / 254.7 s（三次） | **207.9 s** | ✅ **-18.5%（省 47 s）**，明顯低於 baseline，無負向優化 |
| §5.2 GPU idle_frac（`nvidia-smi dmon -s u`，同法量測） | 0.494（mean SM 26.3%） | **0.154（mean SM 34.5%）** | ✅ idle 消掉約 69%；與 wall-clock 改善一致 |
| §5.3 正確性（`report.csv` vs origin） | — | 3557 vs 3557 cells，**全數 <3px 對上，reddot/blackdot/score 0 筆不符** | ✅ 比 run-to-run 雜訊地板還乾淨 |
| §5.4 記憶體有界 | peak RSS ~2.95 GB | peak RSS 3.06 GB（**+4%**） | ✅ 雙緩衝多一塊 numpy，仍有界；VRAM 依設計不變（GPU 仍一次一塊） |

**正確性雜訊地板**（兩次原碼跑互比，作為容忍度基準）：3557 vs 3557 cells，2 顆質心漂移 >3px，
reddot/blackdot 各 2 筆、score 1 筆不符 —— 純 GPU 前向非決定性。modified vs origin 反而 **0 筆不符**，
即改動的輸出落在（且優於）雜訊地板內，正確性未退讓。

## 為何實際省 18.5%，而非理論上限 ~50%

doc 10 §3(b) 概算的上限是 `max(B1, B2+B3+B4)`（理論接近腰斬）。實測 idle 從 0.494 降到 0.154 證明
重疊確實發生，但沒到滿：主因是 `detect_all_dots` 用 joblib **`prefer='threads'`**（執行緒後端），與
主執行緒 torch 的 CPU 端有 **GIL 競爭**，重疊率被壓低 —— 正是 doc 10 §5.2 預先點名要查的情況。這不是
失敗，而是「瓶頸移動」：下一輪若要再擠，方向是讓 `detect_all_dots` 改用行程後端 / 加大 pipeline 深度
（另案評估，需再各自 ablation）。

## 重現方式

```bash
cd /data/taro_Projects/tsgh
ROI=backend/algorithms/hybrid/test_picture/_roi_crops
.venv/bin/python backend/algorithms/hybrid/hybrid_pipeline.py \
  --ihc "$PWD/$ROI/med_ihc.tiff" --dish "$PWD/$ROI/med_dish.tiff" --output <out_dir>
# 對照 report.csv：以最近質心配對（<3px），比對 reddot/blackdot/score 是否落在雜訊地板內。
```
