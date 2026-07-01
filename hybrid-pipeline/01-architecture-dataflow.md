# 01 — 架構與資料流

> 全景 + 巢狀迴圈 + 記憶體生命週期。視覺版見 [pipeline-flow.html](./pipeline-flow.html)。

## Pipeline 全景

```
輸入：一對配對影像 (IHC/Her2 tile, DISH tile)，同尺寸；可為單 tile / 任意 ROI / 整張 WSI
         │
   ┌─────▼──────────────────────────────────────────────────────────┐
   │ M0 讀取  m0_reader.iter_paired_chunks()                          │
   │   pyvips 隨機存取，逐 1024² 重疊視窗吐出對齊的 Chunk(ihc,dish)   │
   └─────┬──────────────────────────────────────────────────────────┘
         │  每個 chunk 依序進入 _process_one_chunk()
   ┌─────▼─────┐   ┌────────────┐   ┌────────────────────────────────┐
   │ M1 疊合   │──▶│ M2 分割    │──▶│ M3 細胞/點位                    │
   │ UNet++    │   │ Cellpose   │   │ 生成細胞→彈性配對 DISH 核→      │
   │ core mask │   │ instance   │   │ 偵測 HER2 黑點/CEP17 紅點→擴增  │
   │ →50/50疊  │   │ mask       │   │ 判定                            │
   └───────────┘   └────────────┘   └─────┬──────────────────────────┘
                                          │ ChunkResult（局部座標）
   ┌──────────────────────────────────────▼──────────────────────────┐
   │ M0 縫合  m0_stitch.StitchAccumulator.add() / finalize()          │
   │   質心 core-ownership 去重 → 全域重編號 → 座標絕對化 → 貼回整圖 │
   └──────────────────────────────────────┬──────────────────────────┘
                                          │ StitchedTile（slide-level 整圖）
   ┌──────────────────────────────────────▼──────────────────────────┐
   │ M4 匯出  m4_export                                               │
   │   CSV（cell_id/座標/reddot/blackdot/score）+ overlay 視覺化 +    │
   │   逐細胞裁切 PNG                                                 │
   └─────────────────────────────────────────────────────────────────┘
```

### 每個 M 的角色（一句話）

| 模塊 | 檔案 | 角色 |
| --- | --- | --- |
| **M0 讀** | `m0_reader.py` | 用 pyvips 把任意大小輸入切成 bounded-memory 的重疊 chunk，IHC/DISH 同 offset 對齊。 |
| **M1** | `m1_overlay.py` | UNet++ 產生 IHC 腫瘤 core mask，套到 IHC＋DISH，做 50/50 alpha blend 當 M2 輸入。 |
| **M2** | `m2_segmentation.py` | Cellpose 在疊合圖上切出細胞 instance mask（重疊視窗 + IoMin 去重）。 |
| **M3** | `m3_module/*` | 逐細胞生成結果、把 DISH 核彈性配給細胞、在核內數 HER2/CEP17 點、算 Score 判擴增。 |
| **M0 縫** | `m0_stitch.py` | 把每個 chunk 的局部結果去重、重編號、絕對化座標，增量貼回 slide-level 整圖。 |
| **M4** | `m4_export.py` + `m4_module/*` | 輸出 CSV、醫師檢視 overlay、逐細胞 crop。 |

> 注意 M0 出現兩次：**讀（reader）在最前、縫（stitch）在 M3 之後**。M0 是包在 M1–M3 外面的「分塊殼」，不是單一階段。

## 巢狀迴圈：batch 外圈 × chunk 內圈

三層呼叫關係（都在 `hybrid_pipeline.py`）：

```
run_batch(ihc_dir, dish_dir, ...)                      # 外圈：掃目錄，逐「配對 tile」
│   模型只在這裡初始化一次（UNet + 2× Cellpose），整個 batch 重用
│
└─ for (ihc_path, dish_path) in paired_tiles:
     │
     process_single_tile(ihc_path, dish_path, ...)     # 中圈：單一 tile/ROI/WSI
     │   read_size() → chunk_offsets() → 建 StitchAccumulator
     │
     └─ for chunk in iter_paired_chunks(...):           # 內圈：逐 1024² chunk
          │
          cr = _process_one_chunk(chunk, ...)           # M1→M2→M3（單塊）
          acc.add(cr)                                    # 立刻縫入、cr 隨即可 GC
     │
     stitched = acc.finalize()                          # 收斂成整圖
     _write_m1_artifacts / _export_tile_outputs(...)    # M4
```

- **外圈（batch）**：`run_batch` 掃 `ihc_dir`/`dish_dir`，`find_paired_tiles` 依檔名排序配對。模型**只載一次**（`_init_unet_inferencer` / `_init_cellpose_segmenter` / `_init_dish_cellpose_segmenter`），避免每 tile 重載 ~582MB 權重。
- **中圈（tile）**：`process_single_tile` 不管輸入實際多大，一律用 `default_tile_size`(1024) 分塊處理，記憶體恆定。單 tile CLI 走 `_run_single_tile_cli` 直接呼叫中圈。
- **內圈（chunk）**：`_process_one_chunk` 對單塊跑完整 M1→M2→M3，回傳帶絕對座標 offset 的 `ChunkResult`。

## Chunk 記憶體生命週期

核心設計：**任一時刻只有「一個 chunk 的中間產物」在記憶體，加上一張正在被增量填的整圖畫布**。

```
iter_paired_chunks 讀入 Chunk        → 峰值：1 塊 IHC + 1 塊 DISH（各 1024²×3 uint8 ≈ 3MB）
  ↓ _process_one_chunk 跑 M1/M2/M3   → 峰值：core_mask / overlay / instance_mask / 各種 dot 結構
  ↓ 回傳 ChunkResult
acc.add(cr)                          → 把該塊「核心區」貼進整圖畫布
cr 離開作用域                        → 該塊所有 numpy 立即可被 GC（谷底）
  ↺ 下一個 chunk 重複
```

- `process_single_tile` 內的迴圈刻意讓 `cr` 在每輪出作用域（原始碼註解：`# cr goes out of scope here; its numpy arrays are freed immediately`）。
- 因此**記憶體不隨 chunk 數線性成長**，只由「整圖畫布大小」決定 —— 這是把 20k² ROI 從 ≈31GB 壓下來的關鍵。
- **注意**：整圖畫布仍是 full-H×full-W（`StitchAccumulator.__init__` 就 allocate 了 6 張整圖 numpy：instance/nucleus mask + core_mask + 3 張 RGB）。對真正 WSI（156k×134k）而言，這張畫布本身才是新的記憶體天花板 —— 見 [04](./04-optimization-roadmap.md) 的長期優化（輸出端縫合畫布）。

## StitchAccumulator 的增量貼圖與去重

`m0_stitch.StitchAccumulator` 是純資料重組（不碰模型，可用合成資料單測）。核心是**質心 core-ownership 去重**：

- 把每個 chunk 沿 `overlap/2` 切出**互不重疊、鋪滿全圖的「核心區」**（`_cut_lines`：相鄰塊在後一塊起點再進 `overlap//2` 處切）。
- 一顆細胞**只算在「其質心落在哪塊核心區」的那一塊**（`add()` 內用 `bisect_right(cuts_x, gxc) != col` 判斷質心是否屬於本塊核心）。
- 重疊帶的重複偵測因此**自動消除** —— 同一顆細胞在相鄰兩塊都被 Cellpose 偵測到，但只有質心所在的那塊會認領它。

**為什麼省掉 IoMin 的 O(n²)**：
- 視窗內部（`segment_windowed`）仍用 IoMin 去重相鄰視窗的重複，因為那是同一塊內、幾何未知。
- 但**跨 chunk** 若也用 IoMin，就要兩兩比對重疊帶所有 instance（面積/交集），成本隨細胞數平方成長。
- core-ownership 改用「質心落點」這個 **O(細胞數) 的單一判斷** 取代兩兩比對 —— 質心是唯一的、確定屬於某一塊核心區，不需要跟鄰塊的細胞互比。這是把跨塊去重從 O(n²) 降到 O(n) 的關鍵設計。

貼圖時：`add()` 只把該塊「核心區範圍」的像素貼進整圖（`core_mask/masked_ihc/dish_mask_overlay/overlay_image` 各自一次切片賦值），互不重疊、不重複塗。同時做全域重編號（細胞 1..N、DISH 核 1..M、`assigned_dish_ids` 一併改寫）與座標絕對化（質心/每個 dot `+(abs_x, abs_y)`）。

## 記憶體/資源相關的關鍵設定

### pyvips cache 關閉的原因
`m0_reader.py` 頂層 `pyvips.cache_set_max(0)`：
- pyvips 預設會把每次 `crop()` 運算結果快取在 C heap。
- 對 WSI 逐塊掃數千次時，這些快取**不會釋放**，導致 RAM 緩慢單調成長。
- 關掉 cache → 每塊讀完即丟，換取記憶體可預測（讀取本來就是 IO-bound，重算成本可接受）。

### batch 迴圈之間的 GPU/GC 清理
`run_batch` 每處理完一個 tile 就跑：
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()   # 釋放 PyTorch allocator 快取回給驅動
gc.collect()                    # 回收整圖畫布等大 numpy 的循環參照
```
- 目的：長 batch（上千 tile）跨 tile 不要記憶體單調成長。
- `empty_cache()` 釋放的是 allocator 保留但未用的 VRAM；`gc.collect()` 收 Python 端大物件。
- 這是 tile 邊界的「重置點」，配合 chunk 邊界的 GC，兩層一起把峰值壓在「單塊處理 + 單張整圖畫布」。

## 單塊退化 = 回歸基準

當輸入 ≤ `tile_size`（1024）時，只會 yield 單一 chunk：
- 核心區 = 整塊、無接縫、全域 ID == 局部 ID → `StitchAccumulator` 退化成「原樣複製」。
- pyvips 解碼對 JPEG-TIFF 與 `skimage.io.imread` 逐位元相同。
- 故**單塊輸出 bit-identical 於 pre-M0 的單影像路徑**（GPU 推論本身非決定性，跨 run 比對用 noise floor 而非精確相等）。這是驗證「M0 沒改壞既有行為」的回歸基準，詳見 [05](./05-dev-testing-guide.md)。
