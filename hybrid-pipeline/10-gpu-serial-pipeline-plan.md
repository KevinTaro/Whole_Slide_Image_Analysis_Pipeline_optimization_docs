# 10 — GPU 序列瓶頸下一階段方案設計（僅限主題①）

> 承接 [measurement/bottleneck-list.md](./measurement/bottleneck-list.md) 的量測結果。該文件依
> [09-measurement-analysis-plan.md](./09-measurement-analysis-plan.md) 的規格**只做量測與分類、
> 不提解法**；本文件是它明確保留給「下一份文件」的工作——针对 **① GPU under-utilisation from a
> fully serial pipeline（PRIMARY）** 一項，依 [PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)
> 的 Analyze→Plan→Choose 流程，設計方案並定義驗收標準。**只談①**；不對 ②③④⑤⑥⑦ 下解法結論
> （見第 6 節說明為什麼，以及它們會如何被①的方案間接影響）。
>
> 基準：git `ce980d1`（本文件撰寫時的 HEAD；`bottleneck-list.md` 量測於 `96a28ba`，兩者之間
> 無 `hybrid_pipeline.py` / `m1_overlay.py` / `m2_segmentation.py` / `m3_module/` 改動，量測數字
> 對本 HEAD 仍然有效——若之後這些檔案有變更，需重新確認）。錨點：large（441 tiles）= **848.0 s**，
> 此數字**不可覆蓋**，任何方案的效果都要拿它做 ablation 對照。

---

## 1. 問題重述（已知現況，來自 bottleneck-list.md）

- GPU 閒置 **~46–49%** 的 wall clock，即使是唯一的 GPU 工作。B1（三個模型前向：M1 UNet++ →
  M2 Cellpose → M3b DISH Cellpose）本身只佔 wall 的 **45.5%**；真正的槓桿是**閒置**，不是壓縮
  B1 的計算時間（Amdahl 天花板 1.84 是算給 B1 本身的，但這條路線報酬有限）。
- 讀目前 `hybrid_pipeline.py` 原始碼（`run_batch` L575-694、`process_precut_tile` L203-335、
  `_process_one_chunk` L413-478）確認：**每個 tile 完全序列跑完 M1→M2→M3→（PNG/TIFF 寫檔）→
  M3 dot detection→ gc/cache 清理，才開始下一個 tile**。三個模型的 `.eval()`/前向呼叫都是
  「一次一張影像」，沒有任何跨 tile 的 batch 組裝（`batch_size`/`cellpose_batch_size` 只用於
  單張影像內部的 sliding-window 推論批次，不是跨 tile）。
- **關鍵對照**：把 bottleneck-list.md 的子項佔比加起來 —— B2 PNG/TIFF 寫檔 9.05% + B3
  `detect_all_dots` 30.7% + B4 per-tile `gc.collect()` 4.28% ≈ **44.0%**，幾乎精確對上量到的
  GPU 閒置 **46–49%**。換句話說：**GPU 閒置的時間，幾乎就是 B2+B3+B4 這些純 CPU/IO 工作在跑
  的時間**——現在的迴圈結構是「GPU 做完一個 tile 的三個前向 → GPU 完全沒事做 → CPU 花將近一半
  wall clock 寫檔/偵測點位/收記憶體 → 才輪到下一個 tile 的 GPU 前向」。

這個對照是本文件方案設計的核心依據：**①的閒置與②③的 CPU 時間，本質上是同一塊時間的兩種讀法**
（GPU 視角是「閒置」，CPU 視角是「B2/B3/B4 在跑」）。因此解①的閒置問題，不需要先動②③本身的演算法。

---

## 2. 必須尊重的既有限制（讀程式碼確認，非猜測）

`hybrid_pipeline.py` L645-647 的註解與 `backend/algorithms/hybrid/CLAUDE.md` 明確寫死：

> 三個 GPU 模型只在主行程載入一次、共用同一個 CUDA context，**跨 tile 用 `ProcessPoolExecutor`
> 平行是不安全的（fork-under-CUDA）**。

這代表 04-optimization-roadmap.md 的 **M1（`ProcessPoolExecutor(max_workers=4)` 跨 tile 平行）**
提案，在目前架構下**不能直接採用**——是舊 04 文件寫於重構前，尚未對照這條限制重新檢視過的提案
（已在第 7 節回頭更新 04 文件標記此點）。本文件的方案必須繞開 process-level 平行，只在**單一
process 內**想辦法。

同時必須保留：
- `run_batch()` 的 **fail-fast** 語意（任一 tile 真實錯誤即整批 raise 中止，見 L586-591 的
  docstring 說明）——不能因為做了 pipeline/overlap 就讓某個 tile 的錯誤被吞掉或延遲發現。
- 「記憶體有界、不隨 tile 數線性成長」的既有宣稱（bottleneck-list.md 記憶體驗證章節：VRAM
  平坦 5.16GB、RSS 隨細胞數而非 tile 數成長）——per-tile `empty_cache()` + `gc.collect()`
  目前是這個宣稱成立的手段之一，改動迴圈結構後要重新驗證這條還成不成立，不能只看 wall-clock。

---

## 3. 候選方案（依 playbook「便宜優先」排序）

### (a) 平行化瓶頸本身 — 跨 tile GPU batching
把 K 個 tile 的影像疊成一個 batch，各模型一次前向處理 K 張，而非目前一次一張。VRAM 有大量餘裕
（5.16/32GB，用不到 1/6），理論上有空間。
- **風險**：`process_precut_tile`/`_process_one_chunk` 目前的 core-ownership 去重、
  `edge_flags`（哪個 tile 碰到真實 slide 邊）、per-tile 落地檔名，全部是「一次一個 tile」的
  假設寫的；要改成一次 K 個 tile，這些邏輯都要跟著重寫成 batch 版本，改動面大、正確性驗證成本高。
- **前提**：`cellpose_batch_size` 目前是死 config（Class 7，`getattr(config, "cellpose_batch_size", 16)`
  永遠回 16，`Config` dataclass 沒有這個欄位）——要做這個方案，必須先接線這個欄位，否則連量測
  batch size 的效果都做不到。
- **定位**：架構級改動，本輪不做，列入長期（呼應 04 文件 M1/M2 的精神，但换成 batch 而非
  process 平行）。

### (b) 把瓶頸移出關鍵路徑 — 單 process 內的 pipeline/overlap（**推薦，本輪做**）
不改任何模型呼叫方式，只改 `run_batch()` 的迴圈調度：讓 tile N 的 B2（PNG/TIFF 寫檔）+ B3
（`detect_all_dots`，本身已是 joblib `n_jobs=-1` 平行、且用 subprocess，不占 GIL）+ B4
（`empty_cache`/`gc.collect`）在背景執行的同時，主線程/主 CUDA context 已經開始跑 tile N+1
的 B1（M1→M2→M3b 三個前向）。也就是用一個簡單的兩階段生產者-消費者：
GPU 階段（B1，需序列、共用一個 CUDA context）與 CPU/IO 階段（B2+B3+B4，本來就不需要 GPU）
互相重疊，而不是像現在完全首尾相接。
- **為什麼便宜/低風險**：不動任何模型/演算法/輸出格式，不新增 process、不觸碰 CUDA context
  的 fork 限制（背景工作用 thread，不是 process；CUDA 前向仍然只在一個地方發生）。
  `detect_all_dots` 本身用 joblib 走 subprocess（見 bottleneck-list.md ② 的 cProfile 主線程
  `time.sleep`），PNG/TIFF 寫檔走 skimage/tifffile（C 擴充，多半釋放 GIL）——這些工作本來就
  適合丟到背景執行緒/existing subprocess 而不阻塞主線程繼續發下一個 tile 的 GPU 前向。
- **直接對應量到的數字**：第 1 節已算出 B2+B3+B4 ≈ 44%，幾乎等於量到的 GPU 閒置 46–49%。若
  重疊做得好，理論上端到端 wall clock 上限可以逼近 `max(B1, B2+B3+B4)` 而非 `B1+B2+B3+B4`——
  用大規模（441 tiles）的數字概算：`max(45.5%, 44.0%) ≈ 45.5%` vs 目前 `~89.5%`，意味着理論
  上限接近腰斬（但這是上限，不是承諾——實際要看 GPU 前向與下一輪 CPU 工作之間是否真的無等待、
  以及第 7 節的開放風險是否成立）。
- **要解決的技術問題**：確認 PyTorch/Cellpose 的前向呼叫可以在主線程持續發下一個 tile 請求時，
  讓上一個 tile 的 CPU 後處理在背景線程跑而不互搶——這是實作前要先用小腳本驗證的假設（見第 7 節），
  不是本文件斷言一定可行。

### (c) 消除瓶頸 — 架構/硬體級改動
把三次序列前向（M1→M2→M3b）減少成更少次呼叫（例如合併 M2/M3b 兩個 Cellpose 呼叫、或換一顆
一次到位的模型），或者常駐 GPU daemon 避免重載。這對應 04 文件已經記錄的 **M2（換非 SAM
backbone，需重新訓練驗證精度）** 與 **L1（GPU daemon 常駐）**——屬長期、需要精度/架構取捨的
改動，本文件不重複展開，僅確認：這兩項與本文件推薦的 (b) 不衝突，可以在 (b) 驗證完之後再疊加
評估（疊加時仍要各自做 ablation，不能一次改兩件事）。

---

## 4. 推薦下一步

**先做 (b)（單 process pipeline/overlap），理由**：風險最低（不碰 CUDA context 平行限制、不碰
模型呼叫語意）、改動面最小（只動 `run_batch()` 的迴圈調度，`process_precut_tile`/
`_process_one_chunk` 內部不用改）、且直接對應到量測數字算出的閒置來源（B2+B3+B4 ≈ 閒置量）。
**(a) 跨 tile batching 與 (c) 架構替換都先不做**——按 playbook「修一個瓶頸、重新量測、瓶頸會
移動」的原則，(b) 做完重新量測後才決定要不要疊加 (a)/(c)。

---

## 5. 驗收標準（Choose 階段：只看端到端 wall-clock）

1. **負向優化偵測**：用 bottleneck-list.md 完全相同的量測方法（同一組 25/121/441-tile 真實
   WSI 裁切、同硬體 RTX 5090、記錄 git commit + config_hash），重跑一次端到端。若 large（441
   tiles）的總時間**沒有明顯低於 848.0 s**（或持平/更高），停下來重新量測，不能只信任 GPU
   idle_frac 這個局部數字——這是 playbook 明講的紅旗（"optimized" version 與 baseline 打平）。
2. **GPU idle_frac 應可觀下降**：用 `nvidia-smi dmon` 重跑一次資源時間軸取樣（延續 09 文件
   §4.1 的方法），確認 idle_frac 從 0.46–0.49 有實質下降；若下降幅度遠低於第 3.(b) 節理論概算
   （45.5% 附近），要找出哪一段沒重疊成功（例如背景線程被 GIL 卡住、或 joblib subprocess
   啟動本身有延遲），而不是直接宣稱方案失敗或成功。
3. **正確性不可退讓**：輸出的 `report.csv`/cell_id/centroid 需與現有回歸基準比對（`05-dev-testing-guide.md`
   的方法），在 GPU 前向本身非決定性的雜訊範圍內一致（`backend/algorithms/hybrid/CLAUDE.md` 既有
   invariant）。**Fail-fast 語意必須保留**：刻意讓某個 tile 的 `process_precut_tile` 回傳
   `None`，確認整批仍然如預期中止，而非在 pipeline 化之後被背景執行緒吞掉。
4. **記憶體宣稱重新驗證**：pipeline 化後，同時有兩個 tile 的資料在記憶體中（tile N 的 CPU 後
   處理 + tile N+1 的 GPU 前向輸入/輸出），需重新量測 RSS/VRAM 曲線，確認 bottleneck-list.md
   「記憶體有界」的結論在雙緩衝下依然成立（尤其 VRAM 峰值是否因為兩個 tile 的中介張量同時存在
   而上升——目前 5.16/32GB 餘裕很大，預期沒問題，但要實測，不能只憑餘裕大就假設）。
5. **Ablation**：若最終方案疊加了不只一個改動（例如 (b) 之外又順手調了什麼），每個改動都要
   能單獨關閉、單獨量出貢獻，零貢獻的部分要砍掉（playbook Choose 步驟）。

---

## 6. 本輪明確不處理什麼，以及為什麼

- **② `detect_all_dots`（30.7%，Class 1 演算法複雜度）**：本文件**不**建議改它的演算法。
  理由：它目前的 30.7% 已經幾乎完全落在①的「GPU 閒置」時間窗內執行（第 1 節的對照）——(b)
  方案若成功重疊，②的絕對秒數不變，但它從「疊加在端到端時間上」變成「被 B1 遮住看不見」，
  等同免費解決了它在 wall-clock 上的曝光,不用碰演算法本身。**但**：如果 (b) 做完重新量測後，
  ②因為某種原因沒被完全遮住（例如某規模下 B3 比 B1 長，遮不住的部分會變成新的端到端瓶頸），
  就要照 playbook「瓶頸會移動」的原則，把②列為下一輪的 PRIMARY 候選，屆時才需要另立文件討論
  它的演算法方向（Class 1，例如減少 joblib 進程啟動開銷、或改偵測演算法本身）。
- **③ PNG 編碼（9.05%，Amdahl 天花板 1.10，本來就在候選門檻邊緣）**：同理，屬於會被 (b) 順帶
  遮住的 B2 子項，不單獨立案。
- **④ per-tile `gc.collect()`（4.28%）⑤ Phase A/D（2.43%/0.60%）⑥ 模型初始化 ⑦ API 層**：
  全部在 bottleneck-list.md 的 Amdahl 停損線以下（<~10%），依 09 文件 §5.2 的規則本輪不深入，
  本文件延續不處理。**例外**：④ 的 `gc.collect()`/`empty_cache()` 呼叫時機會被 (b) 的迴圈重排
  直接影響（見第 5.4 節），實作 (b) 時必須連動處理，但不是為了優化④本身的耗時,只是為了不破壞
  它原本要保證的記憶體宣稱。
- **`cellpose_batch_size` 死 config（Class 7）**：本文件第 3.(a) 節已指出，只有要做跨 tile
  batching（方案 a）才需要先修；(b) 不需要，本輪不動它。

---

## 7. 實作前必須先解的開放問題

1. **PyTorch/Cellpose 前向能否安全地從「主線程持續發下一個 tile」而「背景線程做上一個 tile 的
   CPU 後處理」這種模式下正常運作**，在本專案的 cu130 / RTX 5090（Blackwell, sm_120）組合上要
   先寫一個獨立小腳本驗證（不要直接改 `hybrid_pipeline.py` 才發現不成立）——CUDA context 是
   process-wide 可以跨 thread 共用沒錯，但要確認 Cellpose 內部（`get_rel_pos`/ViT-SAM attention）
   沒有 thread-unsafe 的全域狀態。
2. **`gc.collect()`/`torch.cuda.empty_cache()` 呼叫時機**要重新設計：目前是「這個 tile 徹底
   結束才呼叫」，pipeline 化後「這個 tile 徹底結束」的定義會變模糊（GPU 前向結束≠CPU 後處理
   結束）——要明確決定呼叫點落在哪個階段的邊界，並在第 5.4 節驗收。
3. **fail-fast 的 timing**：目前一有錯誤立刻 raise 中止整批；pipeline 化後，錯誤可能發生在
   「背景執行緒還在處理 tile N」但「主線程已經在跑 tile N+1 的 GPU 前向」的當下——要設計成
   一旦背景執行緒回報錯誤，主線程要能盡快中止，且不能讓 tile N+1 的部分結果被誤當作已完成寫入。
4. **重疊粒度**：先驗證「兩階段重疊」（GPU(N+1) 與 CPU(N) 重疊）這個最簡單的形式是否已經拿到
   大部分理論收益；除非量出來明顯不夠，否則不要一開始就做更複雜的多階段深度 pipeline
   （對應 karpathy_rule「最小可解決問題的程式碼」）。
