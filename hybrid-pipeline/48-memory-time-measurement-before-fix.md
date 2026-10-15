# 48 — 修正前量測：時間、平均與峰值記憶體（RAM / VRAM），以及「舊版只用 10 GB」「改 config 沒變化」的查證

> 量測日期：**2026-10-15**。本文件是 [47](./47-code-review-2026-10-13-14-commits.md) 的後續，
> 目的是在擬定修正計畫**之前**，先把現況量清楚。**未修改任何程式碼。**
> 機器：RTX 5090 (32,607 MiB) / 20 cores / 62 GB RAM。GPU 上另有一個 `/data/tsgh` 的行程常駐
> 598 MiB（另一份部署），量測已扣除（見附錄 A）。
> 原始資料與量測工具：[`measurement/48-memory-matrix/`](./measurement/48-memory-matrix/)
> （`results.jsonl` 每組一行、`samples/` 每 0.5 s 一筆的時間序列、`memrun.py` / `matrix.sh`）。

---

## 0. 結論（TL;DR）

1. **組員看到的「~10 GB」是對的，而且新舊版一樣。** `workers=4` 時：
   - **RAM**（整個行程樹）峰值 **10.6–11.2 GB**，三種輸入大小都一樣；
   - **VRAM 穩態**（中位數）**8.9 GB**（`expandable_segments` 開）／**10.5 GB**（關）。
   用 `htop` 看 RAM、或隨手開 `nvidia-smi` 看 VRAM，看到的都會是「大約 10 GB」。
2. **doc 47 裡的 24.5 GB / 16.9 GB 是瞬間尖峰，不是持續用量——在此更正。** 這些尖峰每次只出現在
   **1–4 個取樣點（< 2 秒）**，人眼開 `nvidia-smi` 幾乎不可能看到。doc 47 把「200 ms 取樣的最大值」當成
   代表值，誇大了 P0-2 與 P1-1 的嚴重度（見 §4 的更正對照）。**但尖峰是真的**，而且會造成 OOM（§3.3）。
3. **新舊版在相同 worker 數下的時間、RAM、VRAM 都沒有差別**（差異都在雜訊內），輸出正確性全數 ≥ 99.86%
   （門檻 99.5%）。`d0ccce8` 唯一實際改變行為的是：**不傳 `--workers` 時會崩潰**，以及 **API 從 1 個 worker
   變成 4 個**。
4. **「改 config 沒變化」有四個各自成立的原因**（§5），其中最可能的是：**舊版程式根本不讀
   `config.workers`**——舊版 CLI 的 `--workers` 預設寫死 `4`，config 裡改什麼都不會影響 worker 數。
5. **`expandable_segments` 在 `workers=4` 是必要的**：關掉它的那組在中型輸入上**實際 OOM 了**，
   簽章正是 round 11 的「單一 worker 漲到 24.76 GiB」配置器氣球（§3.3）。修正計畫不能用「關掉它省 VRAM」。
6. **4 個 worker 只在輸入夠大時才划算**：small（25 塊）**慢 13–15%**、med（121 塊）快 **1.66–1.70×**、
   large（441 塊）快 **2.21–2.23×**。原因是 4 個 worker 的模型載入要 **19.5–23.9 s**（1 個 worker 只要 6.5 s），
   小輸入攤不掉。

---

## 1. 量了什麼、怎麼量

### 1.1 版本與組態

| 代號 | 程式碼 | 說明 |
|---|---|---|
| `old_*` | `2e2129b`（`d0ccce8` 之前） | 以 `git archive` 匯出到 scratchpad，沿用本機 `config.py` |
| `new_*` | `HEAD 83b8305`（repo 本身，未修改） | 本機 `config.py` **沒有** `workers` 欄位 |
| `cfg4_*` / `cfg1_*` | HEAD 匯出副本 | 副本的 `config.py` 加上 `workers: int = 4` / `= 1`，模擬「已正確更新 config.py 的機器」 |
| `exonly_*` | HEAD 匯出副本 | 只改 `config_example.py` 的 `workers`，`config.py` 不動，模擬「改錯檔案」 |

後綴：`_w1` / `_w4` = CLI 傳 `--workers 1/4`；`_default` = 不傳 `--workers`；`_off` = 以
`PYTORCH_CUDA_ALLOC_CONF=`（空字串）讓 `config.cuda_alloc_conf = "expandable_segments:True"` 不生效。

### 1.2 輸入

`backend/algorithms/hybrid/test_picture/_roi_crops/`：small 4096²（**25 塊**）、med 8192²（**121 塊**）、
large 16384²（**441 塊**）。每塊 1024 px。

### 1.3 指標

| 指標 | 定義 |
|---|---|
| wall | 指令開始到結束（含 Python import、模型載入、Phase D 縫合） |
| RAM RSS 峰值 | 整個行程樹（父 + resource tracker + worker）RSS 加總的最大值。**會重複計算共享頁**，最接近 `htop` 逐行加總看到的數字 |
| RAM PSS 峰值／平均 | 同上，但用 `/proc/<pid>/smaps_rollup` 的 PSS（共享頁按比例攤）——**真實的系統記憶體占用** |
| VRAM 峰值 | 每 0.5 s 取樣的最大值，取兩個計數器中較大者：整卡 `memory.used` 扣掉起跑前基準，以及本行程樹各 PID 的 `used_memory` 加總 |
| VRAM 中位數（穩態） | 有 VRAM 用量的取樣點的中位數；這才是「平常開 `nvidia-smi` 會看到的數字」 |

> 為什麼要兩個 VRAM 計數器：`expandable_segments` 透過 CUDA VMM 對映記憶體，`nvidia-smi` 的**每行程**
> 統計有時看不到這部分，**整卡**統計看得到；反過來整卡統計會混入別人的用量。兩者取大、並扣掉基準。
> 第一輪量測只用了每行程計數器，與整卡數字對不上，已整輪作廢重跑（原始結果保留在 scratchpad，不採用）。

---

## 2. 結果

### 2.1 時間

| 輸入 | old `w1` | new `w1` | old `w4`（= 舊 CLI 預設） | new `w4` | 4 workers 加速 |
|---|---|---|---|---|---|
| small（25 塊） | **26.1 s** | 27.7 s | 30.6 s | 30.6 s | **0.85×（變慢）** |
| med（121 塊） | 102.5 s | 103.6 s | 61.8 s | 61.1 s | **1.66–1.70×** |
| large（441 塊） | 351.7 s | 344.6 s | 157.9 s | 158.0 s | **2.21–2.23×** |

模型載入時間（log 的「N 個 worker 模型載入完成，耗時」）：1 worker **6.4–6.8 s**；4 workers **19.2–23.9 s**——
四個行程同時載入會互搶 CPU / 磁碟 / PCIe，並不是平行的 6.5 s。small 跑 4 workers 時，
**整段 30 s 裡有 ~20 s 在載模型**。

**新舊版在相同 worker 數下的時間差都在 ±2%**（spawn 一個 worker 的額外成本量不出來）。

### 2.2 RAM（系統記憶體，整個行程樹）

| 輸入 | `w1` RSS 峰值 | `w1` PSS 平均 | `w4` RSS 峰值 | `w4` PSS 峰值 | `w4` PSS 平均 |
|---|---|---|---|---|---|
| small | old 2.87 / new 3.36 GB | 2.35 / 2.49 GB | old 10.77 / new 10.86 GB | 9.2–9.3 GB | 5.2 GB |
| med | old 3.00 / new 3.44 GB | 2.60 / 2.84 GB | old 10.94 / new 10.90 GB | 9.3 GB | 6.8 GB |
| large | old 3.76 / new 3.55 GB | 2.79 / 3.05 GB | old 11.11 / new 11.15 GB | 9.5 GB | 7.9–8.0 GB |

- `w4` 的 RAM 峰值**幾乎不隨輸入大小變化**（10.8 → 11.1 GB）：主成分是 4 份 torch + 三個模型的常駐量，不是資料。
- new `w1` 比 old `w1` 多 ~0.3–0.5 GB：多了一個父行程 + resource tracker（spawn 路徑），量級很小。
- 整片（doc 46：35,700 塊）的 RAM 峰值是 13.66 GB，與本表趨勢一致（多出的是 `per_tile_owned` 累積的細胞物件）。

### 2.3 VRAM

| 輸入 | 組態 | **穩態（中位數）** | p95 | **峰值** | 峰值持續 |
|---|---|---|---|---|---|
| small / med / large | old `w1`（單行程） | **2.63 GB** | 2.63 | 2.63 | 完全平坦 |
| small / med / large | new `w1` | **2.23 GB** | 2.23 | 7.2–17.0 | 0–1 個取樣點 |
| small / med / large | new `w1` `_off` | **2.63 GB** | 2.63 | 2.63–4.7 | 平坦 |
| med / large | old / new `w4` | **8.9 GB** | 8.9 | 25.3–30.1 | **1–3 個取樣點（< 1.5 s）** |
| med / large | new `w4` `_off` | **10.5 GB** | 10.5 | 11.1–18.1 | 0–1 個取樣點 |
| small | old / new `w4` | 4.6 GB | 14.2–22.5 | 24.7–30.3 | 2–4 個取樣點 |

要點：

1. **穩態用量：1 個 worker ≈ 2.2–2.6 GB，4 個 worker ≈ 8.9–10.5 GB。** 新舊版完全相同。
2. **`expandable_segments` 讓穩態變低、尖峰變高**：穩態 2.63 → 2.23 GB（w1）、10.5 → 8.9 GB（w4），
   但會冒出 7–30 GB 的瞬間尖峰。這與 round 11 的觀察一致（「它會長進所有存在的餘裕」）。
3. **old `w1` 完全沒有尖峰**，因為舊的單行程路徑不套用 `cuda_alloc_conf`；new `w1` 走 spawn 路徑才會套到。
   這是 `d0ccce8` 在 VRAM 上**唯一真實的行為差異**，但影響的是尖峰而不是持續用量。

### 2.4 正確性

所有成功的組合都以 golden test 的容差（質心 ≤ 2 px、其他欄位完全相同）對照同尺寸的 old `w1`：
**最低 99.857%（med new `w4_off`），其餘 99.89–99.96%**，全部高於 99.5% 門檻。細胞數差異 ≤ 7/34,751。
與 doc 47 的結論一致：差異是 GPU 非決定性，不是程式改動造成的。

---

## 3. 分析

### 3.1 組員的「舊版只用 10 GB」——對的，而且不矛盾

舊版 CLI 不帶 `--workers` 時**就是 4 個 worker**（`2e2129b` 的 argparse `default=4`）。
這個組態的 RAM 峰值 10.8–11.1 GB、VRAM 穩態 8.9 GB（若他的 `config.py` 沒有 `cuda_alloc_conf`
欄位或是空字串，則是 10.5 GB）。三種看法都會落在「~10 GB」：

| 他可能看的是 | 會看到 |
|---|---|
| `htop` / `free` 的系統記憶體 | 10.6–11.2 GB |
| 隨手開的 `nvidia-smi`（`expandable_segments` 開） | ~8.9 GB |
| 隨手開的 `nvidia-smi`（他的 config 沒開 `expandable_segments`） | ~10.5 GB |

doc 47 的 24.5 GB 是**每 200 ms 取樣、取最大值**得到的，只存在不到 2 秒。兩邊數字都是真的，量的是不同的東西。
另外，**他的卡如果比較小或上面有別的行程**，`expandable_segments` 能長進去的餘裕就比較少，尖峰也會比較矮。

### 3.2 那 doc 47 的擔憂還成立嗎？

部分成立，需要降級表述：

- **API 從 1 變 4 個 worker（doc 47 P0-2）**：持續用量從 **~2.6 GB 變成 ~9–10.5 GB**（約 4×，不是 doc 47
  寫的 5.5×）；RAM 從 ~3 GB 變成 ~11 GB；**小 ROI 反而慢 15%**。結論不變——這是需要明確決策的行為改變，
  但嚴重度的依據應改用穩態數字。
- **`workers=1` VRAM 3.8×（doc 47 P1-1）**：**不成立為「持續用量」**。持續用量反而略降（2.63 → 2.23 GB），
  多出來的只是 0–1 個取樣點的尖峰。降為 P2（在共用 GPU 上，尖峰仍可能撞到別人的 OOM）。

### 3.3 尖峰會不會出事？會——而且開關 `expandable_segments` 兩邊都有風險

`med_old_w4_off`（舊版、4 workers、**關掉** `expandable_segments`）在第 23 秒 **OOM 中止**：

```
CUDA out of memory. Tried to allocate 110.00 MiB. ... 38.50 MiB is free.
Process 299294 has 24.76 GiB memory in use. ...
RuntimeError: process_precut_tile 於 tile_x768_y0 失敗 ...；整批 fail-fast 中止。
```

**單一 worker 漲到 24.76 GiB**——這正是 round 11（doc 37）記錄的配置器氣球簽章，也就是當初把
`expandable_segments:True` 設為預設的理由。同樣的組態在 new 版跑 med / large 時都**沒有**撞到（隨機發生，
round 11 量到 4/12）。所以：

- `workers=4` **必須**開 `expandable_segments`，否則有機率 OOM；
- 開了之後換成「瞬間尖峰 25–30 GB」，在 32 GB 卡上只剩 ~2 GB 餘裕（doc 43 的 1/10 OOM 就是這個）。

**這是 `workers=4` 本身的已知風險，跟 `d0ccce8` 無關**；`d0ccce8` 只是把它擴大到 API 路徑。

### 3.4 4 個 worker 的損益平衡點

以 2.1 的數字粗估：4 workers 固定多付 ~13–17 s 的模型載入，換來每塊時間從 ~0.80 s（w1，large）降到
~0.30 s。損益平衡大約在 **30–40 塊** 左右；small（25 塊）在平衡點以下，所以變慢。
這是「API 依 tile 數決定 worker 數」（doc 47 §6-1）的直接依據。

---

## 4. 對 doc 47 的更正

| doc 47 原文 | 更正 |
|---|---|
| P0-2：「HEAD `--workers 4` peak VRAM **24,463 MB**」 | 那是瞬間尖峰；**穩態 ~8.9 GB**（med/large），尖峰 25–30 GB 只持續 < 2 s |
| P1-1：「`workers=1` 不再是低 VRAM 模式：4.4 GB → **16.9 GB**（3.8 倍）」 | **不成立**。舊版 4.4 GB 是整卡數字（含其他行程 ~0.6 GB + 取樣誤差），本機單行程實際 2.63 GB；新版穩態 **2.23 GB**，16.9 GB 是單一取樣點的尖峰。降為 P2 |
| §6-2：「`workers<4` 時不套 `expandable_segments`：16.9 → 2.7 GB」 | 改為：可消除 w1 的瞬間尖峰（穩態 2.23 → 2.63 GB，反而略升）。仍建議，但優先度低 |
| P0-1（崩潰）、P1-2（scripts）、P2-1（進度條） | **不變**，本輪再次實測確認 P0-1（`new_default`、`exonly_default` 都在 1.1 s 內 `AttributeError`） |

doc 47 開頭已加註指向本節。

---

## 5. 「改了 config 沒變化」——四個原因，依可能性排序

| # | 情境 | 實測 | 為什麼 |
|---|---|---|---|
| 1 | **跑舊版、改 config 的 worker 數** | 舊版程式碼裡**沒有** `config.workers` | 舊版 worker 數只來自 CLI：`--workers` 預設寫死 `4`。config 裡加什麼欄位、改什麼值都不會被讀 |
| 2 | **改的是 `config_example.py`** | `exonly_default`：**exit 1，1.2 s**，`AttributeError` | 程式只 import `config.py`（gitignored、每台機器一份）。`config_example.py` 只是範本，`run-ui.sh` 只在 `config.py` 不存在時才複製一次 |
| 3 | **改了 `config.py`，但指令有帶 `--workers`** | `cfg1_cli4`：config 設 1，實際 **6 個行程（4 workers）**、RAM 10.6 GB | CLI 參數優先於 config（`workers or config.workers`） |
| 4 | **改了 `cuda_alloc_conf`，但 shell 已經 export 過 `PYTORCH_CUDA_ALLOC_CONF`** | （程式碼路徑） | `m0_multiprocess.py:281`：環境變數已存在就**不覆蓋**。另外 API server 要重啟才會重新讀 config |

反證——**正確改 `config.py` 且不帶 `--workers` 時，新版確實有變化**：

| 組態 | worker 數 | 行程數 | RAM RSS 峰值 | wall（small） |
|---|---|---|---|---|
| `cfg4_default`（config.py `workers=4`） | 4 | 6 | 10.87 GB | 32.2 s |
| `cfg1_default`（config.py `workers=1`） | 1 | 3 | 3.47 GB | 29.4 s |

建議請組員確認三件事：他跑的是哪個版本、改的是哪個檔案、指令有沒有帶 `--workers`。

---

## 6. 給修正計畫的輸入（只列事實，計畫另文擬定）

1. **P0-1 仍是第一優先**：`new_default` 實測 1.1 s 崩潰，所有舊 `config.py` 的機器都會中。
2. **API worker 數應依工作量決定**：小 ROI（< ~30 塊）用 1 個 worker 更快、RAM 少 7–8 GB、VRAM 穩態少 ~6–8 GB；
   整片才值得 4 個（2.2×）。
3. **不要用「關 `expandable_segments`」來省 VRAM**：`workers=4` 關掉會 OOM（本輪實際發生 1 次）。
   只考慮在 `workers=1` 時不套用（消尖峰，穩態不變）。
4. **模型載入是小輸入的主要成本**：4 workers 載入 19.5–23.9 s。若未來 API 要常駐 worker pool，
   這是可省下的固定成本。
5. **VRAM 監控請看穩態與尖峰兩個數字**，不要只看最大值；本文件的 `memrun.py` 可直接重用。

---

## 附錄 A：量測細節

- 25 組，n=1（崩潰組除外每組都成功或在預期內失敗）。wall 差異 ±2 s 內視為雜訊。
- GPU 起跑前基準 646 MiB（`/data/tsgh` 行程 598 MiB + 桌面），整卡數字已扣除；兩組之間間隔 5 s 讓 VRAM 釋放。
- 第一輪（只有每行程 VRAM 計數器）在跑到 `small_new_w1_off` 時發現兩個計數器不一致，整輪中止重跑；
  第一輪數字不採用。
- `old_*` 是從 git 匯出的副本，`models/` 以 symlink 指回 repo；repo 工作區全程 clean。
- 峰值以 0.5 s 取樣，**可能漏掉更短的尖峰**——實際最大值可能比表中更高；穩態中位數不受此影響。

## 附錄 B：完整結果表

| label | exit | wall s | RSS 峰值 GB | PSS 峰值 GB | PSS 平均 GB | VRAM 整卡峰值 | VRAM 整卡平均 | VRAM 行程峰值 | VRAM 行程平均 | 行程數 | 塊數（成功/跳過） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small_old_w1 | 0 | 26.1 | 2.87 | 2.84 | 2.35 | 2.63 | 2.17 | 2.63 | 2.19 | 1 | 25/0 |
| small_old_default | 0 | 30.6 | 10.77 | 9.27 | 5.21 | 19.98 | 5.76 | 25.78 | 5.76 | 6 | 25/0 |
| small_old_w4_off | 0 | 29.8 | 10.76 | 9.20 | 5.19 | 29.11 | 6.06 | 11.25 | 5.70 | 6 | 25/0 |
| small_new_default | **1** | 1.1 | 0.48 | 0.47 | 0.24 | 0 | 0 | 0 | 0 | 1 | AttributeError |
| small_new_w1 | 0 | 27.7 | 3.36 | 3.05 | 2.49 | 9.27 | 2.19 | 2.97 | 1.99 | 3 | 25/0 |
| small_new_w1_off | 0 | 26.5 | 3.40 | 3.11 | 2.57 | 2.63 | 2.26 | 2.63 | 2.22 | 3 | 25/0 |
| small_new_w4 | 0 | 30.6 | 10.86 | 9.20 | 5.23 | 29.53 | 6.60 | 30.29 | 6.25 | 6 | 25/0 |
| small_new_w4_off | 0 | 29.6 | 10.73 | 9.07 | 5.21 | 15.75 | 6.03 | 10.71 | 5.61 | 6 | 25/0 |
| small_cfg4_default | 0 | 32.2 | 10.87 | 9.13 | 5.27 | 19.41 | 5.75 | 24.68 | 5.92 | 6 | 25/0 |
| small_cfg1_default | 0 | 29.4 | 3.47 | 3.17 | 2.54 | 2.35 | 1.91 | 7.45 | 2.12 | 3 | 25/0 |
| small_cfg1_cli4 | 0 | 31.1 | 10.62 | 9.09 | 5.26 | 24.82 | 5.72 | 14.53 | 5.42 | 6 | 25/0 |
| small_exonly_default | **1** | 1.2 | 0.48 | 0.46 | 0.24 | 0 | 0 | 0 | 0 | 1 | AttributeError |
| med_old_w1 | 0 | 102.5 | 3.00 | 2.96 | 2.60 | 2.63 | 2.38 | 2.63 | 2.35 | 1 | 119/2 |
| med_old_default | 0 | 61.8 | 10.94 | 9.32 | 6.80 | 29.04 | 7.29 | 27.30 | 7.25 | 6 | 119/2 |
| med_old_w4_off | **1** | 23.0 | 7.01 | 5.52 | 4.01 | 14.83 | 3.87 | 19.29 | 4.16 | 6 | **CUDA OOM** |
| med_new_w1 | 0 | 103.6 | 3.44 | 3.15 | 2.84 | 17.02 | 2.11 | 6.57 | 2.04 | 3 | 119/2 |
| med_new_w1_off | 0 | 104.3 | 3.43 | 3.14 | 2.84 | 2.63 | 2.38 | 2.63 | 2.39 | 3 | 119/2 |
| med_new_w4 | 0 | 61.1 | 10.90 | 9.31 | 6.84 | 29.58 | 7.44 | 29.17 | 6.92 | 6 | 119/2 |
| med_new_w4_off | 0 | 60.7 | 10.96 | 9.34 | 6.84 | 11.11 | 7.72 | 10.50 | 7.72 | 6 | 119/2 |
| large_old_w1 | 0 | 351.7 | 3.76 | 3.73 | 2.79 | 2.63 | 2.40 | 2.63 | 2.38 | 1 | 412/29 |
| large_old_default | 0 | 157.9 | 11.11 | 9.47 | 7.93 | 28.40 | 7.87 | 30.05 | 7.90 | 6 | 412/29 |
| large_new_w1 | 0 | 344.6 | 3.55 | 3.27 | 3.05 | 7.18 | 2.11 | 4.20 | 2.08 | 3 | 412/29 |
| large_new_w1_off | 0 | 344.6 | 3.53 | 3.24 | 3.05 | 4.72 | 2.39 | 3.25 | 2.44 | 3 | 412/29 |
| large_new_w4 | 0 | 158.0 | 11.15 | 9.45 | 7.97 | 22.15 | 7.37 | 25.32 | 7.95 | 6 | 412/29 |
| large_new_w4_off | 0 | 155.7 | 11.11 | 9.48 | 8.02 | 11.60 | 8.46 | 18.09 | 9.05 | 6 | 412/29 |

> 「VRAM 平均」是有用量的取樣點的算術平均，會被模型載入期間的低值拉低；§2.3 的「穩態」用中位數，較能代表平常用量。
