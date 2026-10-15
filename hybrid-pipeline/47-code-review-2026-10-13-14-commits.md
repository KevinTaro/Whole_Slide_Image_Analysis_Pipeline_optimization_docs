# 47 — Code review：2026-10-13～14 的三個 commit（2e2129b / d0ccce8 / 83b8305）

> **更正（2026-10-15，見 [48 §4](./48-memory-time-measurement-before-fix.md#4-對-doc-47-的更正)）**：本文 P0-2 的
> 24,463 MB 與 P1-1 的 16,912 MB 是 200 ms 取樣的**瞬間尖峰**（< 2 s），不是持續用量。穩態實測：
> `workers=1` ≈ 2.2–2.6 GB、`workers=4` ≈ 8.9–10.5 GB，新舊版相同。**P1-1 不成立為持續用量，降為 P2**；
> P0-1（崩潰）、P1-2（scripts）、P2-1（進度條）不受影響。

> 審查日期：**2026-10-15**。審查對象：作者 `diadia0000` 最近三個 commit。
> 依據：本目錄 doc 13 / 20 / 25 / 27 / 28 / 33 / 37 / 39 / 46、`.claude/rules/no-regression.md`、
> `.claude/rules/frontend-backend-boundary.md`。
> **本文件只做審查與量測，未修改任何程式碼。** 量測用的舊版程式碼是以 `git archive 2e2129b`
> 匯出到 scratchpad 跑的，repo 工作區保持 clean。
> 機器：RTX 5090 (32,607 MiB)，GPU 量測時無其他行程（idle 40 MiB）。

---

## 0. 結論（TL;DR）

| commit | 內容 | 嚴重度 | 建議 |
|---|---|---|---|
| `2e2129b` | overlay 兩個顏色常數 | 低 | **保留**，不需 rollback |
| `d0ccce8` | worker 數改由 config 驅動、刪除單行程路徑、API 不再傳 `workers=1` | **P0（目前壞掉）** | **今天就要 hotfix**；不建議整個 `git revert`（見 §4） |
| `83b8305` | 背景塊 off-population 旗標（預設關）、逐塊比對 UI、2 支新端點 | 低～中 | **保留**，補幾個小問題（§3） |

**最重要的一句話**：在任何 `config.py` 是舊版的機器上（`config.py` 是 gitignored，每台機器各自一份，
**本機就是**），`d0ccce8` 之後 **UI 的每一次分析與不帶 `--workers` 的 CLI 都會立刻
`AttributeError` 崩潰**。作者自己的機器大概已經手動加了 `workers` 欄位，所以沒發現。

---

## 1. `2e2129b` — overlay 顏色常數

```diff
-_COLOR_BOUNDARY = (19, 69, 139)      # BGR → RGB #8B4513 咖啡色
+_COLOR_BOUNDARY = (0, 140, 255)      # BGR → RGB #FF8C00 橘色
-_COLOR_DISH_MATCHED = (0, 0, 255)    # 純紅
+_COLOR_DISH_MATCHED = (180, 105, 255)  # 粉紅 #FF69B4
```

**評估：合理，保留。**

- 「被配對的 DISH 核」原本是純紅 `(0,0,255)`，跟 CEP17 紅點 `_COLOR_CEP17 = (0,0,220)` 幾乎同色，
  紅色輪廓會吃掉核邊緣的紅點——改粉紅是**改善**判讀。
- 細胞邊界原本是咖啡色，畫在 DAB 咖啡色的 IHC 底圖上對比度差；改橘色對比較好。

**要注意的事（不是 rollback 理由）**：

1. `e9fe22b` 曾經**刻意**把 her2-dish 邊界改成咖啡色。這次又改掉，請確認看片的醫師／使用者知道
   顏色換了（沒有找到 UI 圖例或文件寫死顏色，`grep` 前端與 docs 皆無）。
2. `overlay_slide.tiff` 的像素會變 → `tests/test_hybrid_golden.py` 的 overlay 檢查（≤1% 像素差）
   在已錄過 baseline 的機器上會失敗，需要刻意重錄 baseline。
3. 既有的過時註解（不是這個 commit 造成的）：`m0_tile_runner.py` 的 `_finish_chunk_cpu` 寫「overlay 的
   **橘色**（未配對）核」，但未配對核實際是綠色 `_COLOR_DISH_BBOX`。現在邊界剛好變成橘色，這句註解更容易誤導。

---

## 2. `d0ccce8` — 「改 worker 由 config 驅動」

### 2.1 改了什麼

1. `config_example.py` 新增 `workers: int = 4`，並把 `workers` 加進 `_HASH_EXCLUDE`。
2. `run_batch(workers=None)` → `workers = workers or config.workers`（`hybrid_pipeline.py:149`）。
3. **刪除整條單行程分析迴圈**（~100 行），`workers=1` 也改走 `_run_tiles_multiprocess`（spawn 一個 worker）。
4. `backend/api/hybrid.py` 的 `run_batch(..., workers=1)` → 不再傳 `workers`（`api/hybrid.py:231-232`）。
5. CLI `--workers` 預設由 `4` 改成 `None`（讀 config）。
6. 同一個 commit 夾帶 `report/` 下 ~1,050 行無關文件，以及 `run-ui.sh` 的 chmod。

### 2.2 問題（依嚴重度）

#### P0-1　舊 `config.py` 上，UI 分析與 CLI 預設路徑全部崩潰（已實測）

`run_batch` 第一行就讀 `config.workers`，**沒有 `getattr` 退路**。`config.py` 是每台機器各自的
gitignored 檔案，不會跟著 `git pull` 拿到新欄位。本機的 `config.py` 與 `config_example.py` 欄位差異：

```
< off_population_on_background_tiles
< workers
```

實測：

```text
$ python -c "... run_batch(Path('/x'), Path('/y'), Path('/tmp/z'))"
run_batch() -> AttributeError 'Config' object has no attribute 'workers'

$ hybrid_pipeline.py --ihc small_ihc.tiff --dish small_dish.tiff --output ...   # 不帶 --workers
WALL=1.13 EXIT=1
AttributeError: 'Config' object has no attribute 'workers'
```

API 的 `_analysis()` 現在不傳 `workers` → **前端按「開始分析」與佇列裡的每一項都會失敗**。
諷刺的是同一位作者在下一個 commit（`83b8305`）的新旗標就有用 `getattr(config, ..., False)` 並寫了
「舊的 config.py 沒有此欄位 → 維持舊行為」——同一個保護在這裡漏了。
這直接違反 `no-regression.md`（「預設加而不是改」「不傳參數時行為與現在完全相同」）。

#### P0-2　API 從 1 個 worker 變成 4 個：VRAM 4–6 倍、OOM 風險（已實測）

這個 `workers=1` override 是**刻意保留**的（2026-07-29 決定把預設改成 4 時特別留下：單塊／ROI 的
API 請求不該付 N 份模型初始化成本，而且預設變 4 之後這個 override 更重要）。`d0ccce8` 把它拿掉時
commit 訊息與 docstring 都沒有提到這個取捨。

同一個 small ROI（25 塊）的實測：

| 組態 | wall | peak VRAM（`nvidia-smi` 200 ms 取樣） | 結果 |
|---|---|---|---|
| **舊版 `2e2129b`，`workers=1`（舊 API 行為）** | 29.16 s | **4,444 MB** | 成功 25／跳過 0 |
| HEAD，`--workers 4`（新 API 行為，config 補上欄位後） | 30.05 s | **24,463 MB** | 成功 25／跳過 0 |

- **小 ROI 沒有變快**（30.05 vs 29.16 s），因為 4 個 worker 各自載三個模型，25 塊根本攤不掉初始化成本，
  卻多吃了 **~20 GB VRAM**。
- 整片時 `workers=4` 確實快約 2×（doc 46：3.023 h → 1.478 h），所以「API 跑整片想用 4」本身是合理的需求，
  **但**：doc 13 記錄這台是多人共用伺服器；doc 39/43 記錄 `workers=4` 下 VRAM 餘裕只剩 ~2.2–2.5 GB，
  round 14 在裁切規模上 **10 次 OOM 1 次**，而 README 標記「`workers=4` VRAM OOM 風險未重測」仍是 open item。
  這等於把一個已知、尚未關閉的 OOM 風險直接推給醫師按的那顆按鈕。

#### P1-1　`workers=1` 不再是「低 VRAM 模式」：4.4 GB → 16.9 GB（已實測）

刪掉單行程路徑後，`workers=1` 也走 spawn 路徑，而那條路徑會在 spawn 前套用
`config.cuda_alloc_conf = "expandable_segments:True"`（`m0_multiprocess.py:280-282`）。
doc 27 §5 與 doc 37 已經量過 **`expandable_segments` 會拉高 peak VRAM**，當初是為了壓 `workers≥4` 的
24.76 GiB 配置器氣球才開的。實測把它歸因清楚：

| 組態 | wall | peak VRAM |
|---|---|---|
| 舊版 `workers=1`（單行程，不套 alloc conf） | 29.16 s | 4,444 MB |
| HEAD `--workers 1`（spawn，`expandable_segments:True`） | 27.25 s | **16,912 MB** |
| HEAD `--workers 1`，`PYTORCH_CUDA_ALLOC_CONF=` 強制關掉 | 26.48 s | **2,738 MB** |

也就是說：**現在任何人想「用 `--workers 1` 省 VRAM」都會拿到 3.8 倍的 VRAM**。原因不在 spawn 本身，
而在 alloc conf 不分 worker 數一律套用（這點在單行程路徑還在時被它擋掉了，所以以前沒人發現——見 §5-1）。

#### P1-2　`scripts/` 裡 8 支量測／驗證工具壞掉

`hybrid_pipeline.py` 移除了 `_init_unet_inferencer`、`_init_cellpose_segmenter`、
`_process_precut_tile_gpu`、`_process_precut_tile_cpu`、`prefetch_tile_reads`、`_frozen_gc_generation`
等 re-export。實測 `hasattr(HP, ...)` 全部為 `False`。以 `import hybrid_pipeline as HP` 使用這些名字的腳本：

`perf_measure.py`、`verify_gc_freeze.py`、`cellpose_batch_probe.py`、`unet_batch_probe.py`、
`core_mask_map.py`、`tissue_calibrate.py`、`cpu_contention_probe.py`、`mp_concurrency_probe.py`

其中 `perf_measure.py` 是 doc 09 以來所有量測的主力工具（`wrap_tile_boundary()` 第 182 行
`orig = HP._process_precut_tile_gpu` 直接 `AttributeError`）。**光把 import 加回去不夠**：分析迴圈已經
搬到 `m0_multiprocess`，包 `HP._process_precut_tile_gpu` 就算不報錯也**量不到任何東西**（靜默失效）。
另外 doc 25 指出 `workers=1` 單行程是「唯一能拿到 per-function bucket 的模式」——刪掉它之後，
逐函式量測只剩 worker 內的 probe hook 一條路，`PERFORMANCE_BOTTLENECK_PLAYBOOK.md` 的流程要跟著改。

#### P2-1　前端進度條少了「載入模型」階段

模型改在 worker 內載入，父行程不再開 `載入模型` 那條 tqdm。測試有跟著改，但前端
`frontend/src/lib/hybridProgress.ts` 的 `PHASES` 仍分配 4% 給 `載入模型`、
`backend/api/hybrid.py:82` 的註解也還寫著四個階段。結果：模型載入的那幾秒會顯示成「分析中 0/N 塊」、
沒有 ETA，然後進度條從 4% 起跳。不影響正確性，但 UI 文案不準。

#### P2-2　小問題

- `workers or config.workers`：`--workers 0` 會被靜默當成「用 config」；負數會變成 0 個行程，
  最後拋出誤導性的「所有 worker 已結束，但只收到 0/N 塊」。以前 `workers<=1` 走單行程，所以沒有這個坑。
- `_run_tiles_multiprocess` 不會把 worker 數壓在 tile 數以下：ROI 只有 2 塊也會 spawn 4 個各載三個模型。
- commit 夾帶 `report/` 一千多行無關文件，讓這個 commit 無法乾淨地 revert。

### 2.3 做對的地方

- **輸出正確性沒有退化（已實測）。** 以 golden test 同一套容差（質心 ≤2 px 且點數／分數／分類相同）比對：

  | 對照 | 細胞數 | 一致率 |
  |---|---|---|
  | 舊版 vs **舊版重跑**（GPU 雜訊基準） | 2380 vs 2379 | 99.92%（2 顆不同） |
  | 舊版 vs HEAD `workers=1` | 2380 vs 2379 | 99.92%（2 顆不同） |
  | 舊版 vs HEAD `workers=4` | 2380 vs 2379 | 99.92%（2 顆不同） |

  差異跟「同一版本跑兩次」一模一樣，屬於 GPU 非決定性，遠高於 99.5% 門檻。`summary.txt` 的差異（±1–2）
  在重跑基準裡同樣出現。
- 刪掉重複迴圈這個方向本身是對的：doc 33 follow-up 3 就踩過「兩條迴圈要手動同步，Option L 只接到了
  生產不跑的那條」的坑，單一路徑消除了這類漂移。
- `workers` 放進 `_HASH_EXCLUDE` 正確（只影響平行度，不影響每塊結果，不該讓 checkpoint 失效）。
- spawn 一個 worker 的額外開銷在實測中不明顯（27.25 vs 29.16 s，屬雜訊範圍）。

---

## 3. `83b8305` — 「改了一坨測試用的東西」

### 3.1 改了什麼

1. **演算法**：新旗標 `off_population_on_background_tiles: bool = False`。開啟時，IHC core 全空的背景塊
   不再直接 `return None`，而是跳過 M2、照跑 DISH Cellpose（M3b），讓這些塊也能產出 off-population。
2. **API**（`backend/api/` 與 `backend/schemas/`，屬前端可寫範圍）：`GET /api/hybrid/precut`、
   `GET /api/hybrid/precut/{name}.tiff`，以及 `PrecutTile` schema；`schema.ts` 已重新產生。
3. **前端**：`TileCompare.tsx`（原始 DISH tile ↔ 標註 overlay 同步平移縮放）、`lib/tiff.ts`（瀏覽器端
   TIFF 解碼）、`HybridPanel` 的入口按鈕。
4. **測試**：`tests/test_background_tile_off_population.py`（2 passed）。

### 3.2 評估：保留

- 旗標**預設關**，且用了 `getattr` 退路 → 不開就與舊行為完全相同，符合 no-regression。
- 端點是**新增**的，沒有改到既有路徑；`item_id` 走 `_item_dir()` 的 32-hex 驗證、`name` 用
  `tile_x(\d+)_y(\d+)` fullmatch，沒有路徑穿越問題。
- `TileCompare` 的座標換算是對的：`PrecutStream` 的 tile 檔名是**全片座標**（`m0_reader.py:161-163`
  加了 `origin_x/origin_y`），前端用 `tile.x - origin.x` 換到 overlay 座標，兩邊一致。
- `tiff.ts` 支援的格式（8-bit RGB、strip、deflate、predictor 1/2）正好涵蓋 `PrecutStream`
  的 `_TILE_COMPRESSION = "deflate"`，不支援的格式會明確報錯而不是畫出垃圾。

### 3.3 問題

1. **commit 訊息誤導**：「改了一坨測試用的東西」實際包含一個**演算法行為旗標**、兩支新 API、一個新 UI 元件。
   日後追「off-population 數字為什麼變了」的人不會想到要看這個 commit。建議以後拆成 3 個 commit。
2. **測試會在舊 config 上「假通過」**：`monkeypatch.setattr(..., raising=False)` 讓測試在 config 根本沒有
   這個欄位時也能通過；而且只測到 GPU 前段（`_process_one_chunk_gpu`），**沒有測到這個功能真正的產出**——
   背景塊走完 `_finish_chunk_cpu` 之後，`build_off_population_results` 確實吐出 off-population 細胞、
   `_process_precut_tile_cpu` 不再走 `tg.chunk is None` 的空白塊分支。
3. **開啟旗標的效能代價沒有被量過，而且很大。** doc 46 量到：整片 **65.92% 是背景塊**，背景塊
   **0.0460 s/塊**、組織塊 **0.7298 s/塊**（15.9×）。開旗標等於把 ~23,500 個背景塊從快速路徑拉進
   M1 + DISH Cellpose + 全套 CPU off-population 偵測 + overlay 繪製。上限估計（背景塊變成組織塊成本）：
   23,533 × (0.7298 − 0.0460) ≈ **+4.5 h 的 tile 計算時間**（`workers=1` 整片目前才 3.0 h）。
   實際值會低於上限（少了 M2），但數量級是「整片時間可能翻倍」。**開啟前必須照 doc 45/46 的方法量一次。**
4. **開了之後的科學意義要先確認**：「IHC core 全空」包含真正 HER2 陰性的組織，也包含玻片空白、碎屑。
   DISH Cellpose 在空白／碎屑上的假核會直接變成 off-population 細胞進到 `report.csv` 與 `summary.txt`。
5. **整片時 tile 清單太大**：`/api/hybrid/precut` 對整片會回 **35,700 筆**，`TileCompare` 把它們全部渲染成
   `<select>` 的 `<option>`（`TileCompare.tsx:161`）——DOM 很重，而且其中 ~66% 是沒東西可看的背景塊。
6. 開啟旗標後 `m3_dot_detection.py:138` 的註解「core_mask 全空時 M1 直接 short-circuit 整塊」不再成立。
7. 之後有人把 `off_population_on_background_tiles` 補進自己的 `config.py` 時，**就算值是 `False`**，
   `compute_config_hash` 也會因為多一個欄位而改變 → 既有的 `_resume/` checkpoint 全部失效。
   不是 bug，但要事先告知組員。
8. `TileCompare` 依賴 `_precut_scratch/`。目前它「永不自動刪除」，但若日後有人為了磁碟空間加上清理，
   這個功能會靜默變成「沒有 tile 可以比對」。建議在 `m0_reader` / CLAUDE.md 註明這個依賴。

---

## 4. 要不要 rollback？

**建議：不要整個 `git revert`，改成今天就送一個小的 forward-fix。**

理由：

- `2e2129b`、`83b8305` 沒有破壞任何既有行為，沒有 rollback 的必要。
- `d0ccce8` 確實壞了，但：(a) 它跟 `report/` 一千多行文件綁在一起，revert 會連文件一起丟；
  (b) 刪除重複迴圈的方向是對的、輸出經實測沒有退化；(c) 真正的問題都在**邊緣**（config 退路、API 呼叫端、
  alloc conf、scripts），每一項都是幾行的修正。

**forward-fix 的最小內容（給負責修的人，本文件不動 code）**：

| # | 修什麼 | 為什麼 |
|---|---|---|
| 1 | `hybrid_pipeline.py:149` 改成 `workers = workers or getattr(config, "workers", 4)` | P0-1：舊 config 不再崩潰 |
| 2 | API 呼叫端恢復明確的 worker 數（`workers=1`，或新增 `api_workers` config 欄位、預設 1） | P0-2：恢復刻意保留的低 VRAM API 行為；「API 跑整片要用 4」另開決策 |
| 3 | `cuda_alloc_conf` 只在 `workers > 1`（或 `>= 4`）時套用 | P1-1：`workers=1` 回到 ~2.7–4.4 GB |
| 4 | 8 支 `scripts/` 改成從 `m0_slide` / `m0_multiprocess` 取用並包裝正確的函式 | P1-2：恢復量測工具，避免靜默量不到 |
| 5 | `hybridProgress.ts` 移除 `載入模型`、重新分配權重；更新 `api/hybrid.py:82` 註解 | P2-1 |

修完後依 `no-regression.md` 跑煙霧測試、`report.csv` 對照，以及 `test_hybrid_golden.py`。

**如果團隊不想 forward-fix**，退而求其次的做法是只 revert `d0ccce8` 的程式碼部分（保留 `report/`）：
`git revert -n d0ccce8 && git checkout HEAD -- report/ run-ui.sh`，再 commit
（README 與 hybrid/CLAUDE.md 的說明跟著程式碼一起還原）。

---

## 5. 之前就漏掉的（這次 review 順帶發現，不是這三個 commit 造成的）

1. **`cuda_alloc_conf` 不分 worker 數一律套用。** 這是 P1-1 的根因。單行程路徑還在時，`workers=1` 剛好
   繞過它，所以以前沒有人在 `workers=1` 量到這 3.8 倍的 VRAM。doc 27/37 的 alloc conf 量測都只在
   `workers≥4`，`workers<4` 時「開它是否還划算」從來沒有被問過。
2. **新增 config 欄位沒有機制同步到各機器的 `config.py`。** 這次 P0-1 的本質是流程問題：`config_example.py`
   長出新欄位，每台機器的 gitignored `config.py` 不會知道。目前的防線只有「每個讀取點記得 `getattr`」，
   而這次就漏了一個。建議加一個啟動時檢查：比對 `config` 與 `config_example.Config` 的欄位集合，缺欄位就
   **警告並套用 example 的預設值**（或讓 `config.py` 繼承 example 只覆寫路徑）。
3. **`_run_tiles_multiprocess` 不依 tile 數調整 worker 數**（`min(workers, total)`）。小 ROI 永遠白付初始化成本。
4. **沒有任何 API 層的 VRAM 前置檢查。** 在多人共用的卡上，`workers=4` 開跑前沒有確認剩餘 VRAM 是否足夠
   （需要 ~24–30 GB），只能等 OOM 之後 fail-fast。

---

## 6. 效能改善建議（依投報率排序）

1. **API 依 tile 數決定 worker 數**：`workers = min(config_workers, n_tiles, vram_budget // per_worker)`。
   小 ROI 用 1（實測 4 個 worker 沒變快、多 20 GB），整片才用 4（doc 46：2.05× end-to-end）。
   `PrecutStream.positions` 在開跑前就知道總塊數，資訊是現成的。
2. **`workers<4` 時不套 `expandable_segments`**：實測 `workers=1` 16.9 GB → 2.7 GB，wall 不變（26.5 vs 27.3 s）。
3. **若要開 off-population 背景旗標，先加便宜的組織預檢**：背景塊先用 DISH 影像的白色比例（或既有的
   `dot_background_l_threshold` 的 L 通道）判斷是否為空白玻片，空白才整塊跳過、有組織才跑 DISH Cellpose。
   另外 core 全空時 `_run_m1_overlay_stage` 產出的 `m2_input_overlay` 沒有被用到，可以跳過那段融合。
   這兩項可以把 §3.3-3 的「可能翻倍」壓回一小部分；但一樣要先照 doc 45/46 量。
4. **`TileCompare` 前端**：
   - tile 清單改成只列「有細胞的塊」（後端可用 `report.csv` 的質心或 heatmap 格子過濾），或改用座標跳轉
     + 虛擬化清單，不要一次渲染 35,700 個 `<option>`。
   - 解碼後不必再經 `canvas.toBlob()`（`TileCompare.tsx:125`）編一次 PNG 再讓 OSD 解回來；可以直接把
     canvas 當 OSD 的 image source，或把解碼丟到 Web Worker，換塊時 UI 不會卡。

---

## 附錄 A：量測方法

- 輸入：`backend/algorithms/hybrid/test_picture/_roi_crops/small_{ihc,dish}.tiff`（25 塊）。
  `no-regression.md` 指定的 `~/tsgh_data/smoke/` 在本機不存在；`test_hybrid_golden.py` 的輸入是整片 WSI，
  本機也沒有已錄的 baseline，因此改用 small crop 做 A/B。
- 舊版：`git archive 2e2129b backend` 匯出到 scratchpad，沿用本機 `config.py` 與 `models/`（symlink）。
- 每一組：`/usr/bin/time` 量 wall，背景以 `nvidia-smi --query-gpu=memory.used -lms 200` 取 peak VRAM。
- 正確性：沿用 `test_hybrid_golden.py` 的容差規則（質心 ≤2 px、其餘欄位完全相同、門檻 99.5%），
  並以「舊版跑兩次」作為 GPU 雜訊基準。
- 每組 n=1（重跑基準除外）；wall 差異在 ±2 s 內者視為雜訊，結論只採用遠大於雜訊的差異（VRAM 4–6×、崩潰）。

## 附錄 B：本文件引用的位置

- `backend/algorithms/hybrid/hybrid_pipeline.py:149` — `workers = workers or config.workers`
- `backend/api/hybrid.py:231-232` — `run_batch(...)` 不再傳 `workers`
- `backend/api/hybrid.py:82` — 過時的四階段註解
- `backend/algorithms/hybrid/m0_module/m0_multiprocess.py:280-282` — alloc conf 套用點
- `backend/algorithms/hybrid/m0_module/m0_multiprocess.py:291-299` — 依 `workers` spawn，無 tile 數上限
- `backend/algorithms/hybrid/config_example.py:101` / `:185` / `:271` — 兩個新欄位與 `_HASH_EXCLUDE`
- `backend/algorithms/hybrid/m3_module/m3_dot_detection.py:138` — 開旗標後過時的註解
- `frontend/src/lib/hybridProgress.ts` — `PHASES` 仍含 `載入模型`
- `frontend/src/components/TileCompare.tsx:125` / `:161` — PNG 重編碼、全量 `<option>`
- `tests/test_background_tile_off_population.py:16` — `raising=False`
- `scripts/perf_measure.py:182` — `HP._process_precut_tile_gpu` 已不存在
