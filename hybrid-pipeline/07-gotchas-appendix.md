# 07 — 踩坑附錄

> 每個坑：現象 → 診斷法 → 解決方式。都是實測驗證過的。

## G1. Codegraph 索引過期 —— 有「幻影檔案」

- **現象**：`codegraph_files cell_mask/hybrid` 列出 **30 個檔**，但磁碟上實際只有 **20 個 .py**（19 個 git-tracked + gitignored 的 `config.py`）。index 停在某個舊狀態，包含已刪除/重命名/從未存在的檔。
- **驗證的幻影檔案清單（11 個，codegraph 有、磁碟沒有）**：
  ```
  heatmap_visualizer.py          image_io.py
  m0_tile_generator.py           m0_wsi_reader.py
  models.py                      roi_tile_generator.py
  test_m0_roi_tile_generator.py  test_roi_tile_generator.py
  test_wsi_reader.py             types.py
  wsi_reader.py
  ```
  （前一輪研究估「~9 個」；實測比對後為 **11 個**。數字不重要，重點是 index 不可信。）
- **診斷法**：`git ls-files 'cell_mask/hybrid/**/*.py'`（真實檔案）對照 `codegraph_files`（index 認為的）。差集就是幻影。
- **解決方式**：**檔案是否存在一律以 `git ls-files` / `find` 為準**，不要相信 codegraph 的檔案列表。查符號/呼叫關係時，若命中的檔在上面幻影清單裡，直接忽略。（codegraph 對「存在的檔」內的符號查詢仍可用；只是檔案集合過期。）
- **附帶提醒**：`heatmap_visualizer.py` 同時被 `cell_mask/hybrid/CLAUDE.md` 的 Architecture 段落引用（描述成 standalone 驗證工具），但**該檔實際不存在** —— 是文件 + index 雙重失聯，別去找它。

## G2. `config.py` 是 gitignored —— 不 cp 跑不動，且 cp 完還不夠

- **現象**：clone 後直接 `python hybrid_pipeline.py ...` → `ModuleNotFoundError: No module named 'config'`（或 import 失敗）。
- **診斷法**：`git check-ignore cell_mask/hybrid/config.py` → 命中（確認被 ignore）；`git ls-files ... config.py` → 空（確認沒進 git）。
- **解決方式**：`cp config_example.py config.py`，再填模型路徑/tile 目錄。
- **⚠️ 但 cp 之後還會壞（重要）**：`config_example.py` 尾端**缺少** `compute_config_hash()` 函式與 `config = Config()` 這行（只到 `slide_id` 就結束），而 `hybrid_pipeline.py` 是 `from config import config, compute_config_hash`。所以**光 cp 會 `ImportError: cannot import name 'config'`**。
  - **驗證**：`config_example.py` 212 行，結尾是 `slide_id`；正式 `config.py` 229 行，多出 `compute_config_hash(cfg)` 與 `config = Config()`。
  - **解法**：cp 後在檔尾補上這兩段（或直接沿用機器上已存在、可跑的 `config.py`，別覆蓋）。附帶：`config_example.py` 用 `import torch`、正式 `config.py` 用 `from torch import cuda` —— 功能等價，別困惑。
  - **順手可改善**：讓 `config_example.py` 補回檔尾兩段，使「cp 即可跑」名副其實。（本文檔不改 code，只標示。）

## G3. `cellpose_batch_size` 沒接線到 Config

- **現象**：perf_report.html 建議「把 `cellpose_batch_size` 從 16 調到 32–64」，但你在 `config.py` 怎麼設都沒效果。
- **根因**：`Config` dataclass（`config.py` / `config_example.py`）**根本沒有 `cellpose_batch_size` 欄位**。`hybrid_pipeline._init_cellpose_segmenter` / `_init_dish_cellpose_segmenter` 讀的是 `getattr(config, "cellpose_batch_size", 16)` —— 欄位不存在，永遠 fallback 到 **16**。
- **診斷法**：在 `config.py` 搜 `cellpose_batch_size`（找不到）；看 `hybrid_pipeline.py` 的 `getattr(config, "cellpose_batch_size", 16)` 確認是「軟讀取 + 預設」。
- **解決方式**：先在 `Config` 加 `cellpose_batch_size: int = 32`（或想要的值），欄位存在後 `getattr` 才會拿到你設的值。**沒補欄位就調 config 是無效操作** —— 這是做 [04](./04-optimization-roadmap.md) S3 的前置。

## G4. 程式碼註解引用的 spec docs 已不存在（+ 文件/code 漂移）

- **現象**：註解叫你「see docs/…」，但檔案找不到。
- **失聯清單**：
  - `docs/sdd-elastic-dish-matching.md` —— `m3_elastic_matching.py` docstring 引用，**不存在**。
  - `docs/dish_dot_detection_spec.md` —— `config.py` / `config_example.py` 的 dot 參數註解引用（`see docs/dish_dot_detection_spec.md v0.2`），**不存在**。
- **診斷法**：`find /data/tsgh -name 'sdd-elastic-dish-matching.md' -o -name 'dish_dot_detection_spec.md'` → 空。
- **解決方式**：**以 code 為單一事實來源**，忽略失聯引用。
- **⚠️ 更深的漂移（別被誤導）**：`m3_elastic_matching.py` 的 docstring 自己就註明「該文件描述舊『以核為中心』版本」。現行 code 是 **以細胞為中心 + 重疊優先 + reach**；但 `docs/elastic_matching_v3_explainer.html` 描述的是 **以核為中心 (v3)**。也就是**現存的 HTML 說明也和現行 code 對不齊**。改 M3 配對邏輯時，**讀 `m3_elastic_matching.py` 本體，不要照 HTML/失聯 spec**。同理 `dish_elastic_expand_factor` 在 HTML 標「已棄用」，但 code 仍用它算 reach —— 以 code 為準。

## G5. `generate_ihc_core_mask` 參數名叫 `ihc_tile_path`，實際傳的是 ndarray

- **現象**：`m1_overlay.generate_ihc_core_mask(ihc_tile_path: Path, ...)` 形參型別寫 `Path`，但 `hybrid_pipeline._process_one_chunk` 呼叫時傳的是 `chunk.ihc`（**numpy ndarray**）。型別標註對不上，新人會以為是 bug。
- **這不是 bug**：函式內部把該參數交給 `unet_inferencer.predict_single(...)`，而 `predict_single` 的簽名是 `image: Union[np.ndarray, Path, str]` —— ndarray 完全支援（會走「已是陣列」分支）。呼叫端也已標 `# pyright: ignore[reportArgumentType]`，代表作者知道型別對不上但刻意為之。
- **診斷法**：對照 `generate_ihc_core_mask` 簽名（`m1_overlay.py` L54）與呼叫點（`hybrid_pipeline.py` L290 `generate_ihc_core_mask(chunk.ihc, ...)`），再看 `predict_single` 接受 `Union[np.ndarray, Path, str]`（`unet_inference.py` L210）。
- **解決方式**：**功能面不用改**。若要消除誤導，把形參更名為 `ihc_image`、型別改 `Union[np.ndarray, Path]`（純可讀性，非修 bug）。

## 一句話總結

這些坑的共通根源是**「文件/索引/範本」與「現行 code」漂移**。遇到任何指向外部檔或範本的引用，先用 `git ls-files`/`find` 確認存在、再以 **code 本體為準**。動優化前，G2（config 補檔尾）與 G3（補 `cellpose_batch_size` 欄位）是兩個會讓你「以為改了其實沒改」的隱形絆腳石，優先處理。
