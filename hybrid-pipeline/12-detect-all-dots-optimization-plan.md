# 12 — `detect_all_dots`（② M3 HER2/CEP17 點偵測）優化方案設計

> 承接 [measurement/bottleneck-list.md](./measurement/bottleneck-list.md)（②的原始分類）、
> [measurement/pipeline-overlap-result.md](./measurement/pipeline-overlap-result.md)（①方案 (b)
> 落地後的驗收結果）與 [measurement/gil-contention-diag.md](./measurement/gil-contention-diag.md)
> （①stage 2 的 GIL 診斷）。依 [PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)
> 的 Analyze→Plan→Choose 流程，**只針對 ② `detect_all_dots` 設計方案，不動代碼**（本文件是純規劃文件）。
>
> **範圍界定（呼應使用者指示）**：① GPU under-utilisation 的**本輪**（單 process pipeline/overlap，
> 方案 (b)）已完成並驗收（見 pipeline-overlap-result.md）；① 的**下一階段**（引入 CuPy 等套件深挖
> GPU kernel 本身的算力利用率）是另一個獨立主題，**不在本文件範圍內**。本文件只處理 ②。
>
> 讀碼依據：`backend/algorithms/hybrid/m3_module/m3_dot_detection.py`（`detect_all_dots`
> L97-220、`_detect_one_cell` L223-282）、`m3_dot_kernels.py`（`_detect_red_dots` L46-124、
> `_detect_black_dots` L131-288、`_compute_ring_stats` L302-339）、`m3_elastic_matching.py`
> （`elastic_dish_nucleus_matching` L73-194）。均於本文件撰寫時直接讀取現狀代碼確認，非猜測。

---

## 1. 為什麼現在輪到 ②（架構已經變了，舊分類需要重新解讀）

- **bottleneck-list.md 的 30.7%/260.1s 是在序列 baseline（① 方案 (b) 落地前）量到的**——當時
  doc 10 §6 明確判定「② 的成本幾乎完全落在①的 GPU 閒置視窗內，① 的重疊修好後 ② 會被『免費遮住』，
  不用碰演算法」。
- **這個前提現在被 pipeline-overlap-result.md 自己的觀察推翻了**：① 方案 (b) 落地後（121-tile
  207.9s，idle 從 0.494 降到 0.154），該文件「為何實際省 18.5%、而非理論上限 ~50%」一節寫明——
  重疊率被壓低的原因是**「重疊率是背景 CPU 綁定，`detect_all_dots` 是重的那一極」**。也就是說，
  在新的兩段式 pipeline（GPU 前段 vs CPU 後段，`run_batch` 深度 1）裡，吞吐上限是
  `max(B1_per_tile, (B2+B3+B4)_per_tile)`，而 CPU 後段（B2+B3+B4，`detect_all_dots` 佔其中最大
  份額）比 GPU 前段（B1）**更長**——意味著 `detect_all_dots` 的絕對耗時，現在**直接决定**了整條
  pipeline 的穩態節拍，不再是「反正被 GPU 閒置蓋住、免費」的旁支項目。
- gil-contention-diag.md 的量測（121-tile，py-spy 掛載、wall-clock 有膨脹，只能看比例）也印證了
  這個結構：背景執行緒（`detect_all_dots` + 落地寫檔）總共只佔 wall 的 10%——但那是**在 py-spy
  膨脹過的 866s wall 裡的 10%**，而且是「這 10% 大多已與 GPU 前向重疊」，不是「這 10% 就是它的真實
  成本佔比」。真正該問的問題（本文件要先做的第 0 步）是：在**沒有 py-spy、方案 (b) 已落地**的乾淨
  量測下，`detect_all_dots` 的絕對耗時相對於新的端到端 wall-clock（207.9s@121-tile 這類基準）
  佔多少、Amdahl 天花板多少——這個數字現在還沒有人量過。

**結論**：① 本輪的解法（重疊）沒有讓 ② 變得不重要，反而把 ② 從「隱藏在閒置裡的旁支」升級成「決定
穩態吞吐的那一極」。這正是 pipeline-overlap-result.md 自己指出的下一步方向（"detect_all_dots 改用
process 後端 / 加大 pipeline 深度"）之一半——但本文件只談前半（`detect_all_dots` 本身的優化），
不重複 pipeline 深度（doc 11 已經談過、且與本主題可以獨立疊加，見第 6 節）。

---

## 2. 第 0 步（必須先做）：在目前架構下重新量測 ② 的 Amdahl 天花板

在設計任何解法之前，必須先有一個乾淨的數字，理由見上節——現有的 30.7%（舊序列 baseline）與
10%（py-spy 膨脹、且只看背景執行緒總覽）都不能直接拿來當本文件的決策依據。

**要量什麼**：沿用 bottleneck-list.md 的方法（同一組 25/121/441-tile 真實 WSI 裁切、同硬體），
但改用**目前 HEAD（① 方案 (b) 已落地）的 `run_batch`**，不掛 py-spy（避免像 gil-contention-diag.md
§method 警告的那樣膨脹 wall-clock），對 `detect_all_dots` 這個函式本身加一個計時器（進入/離開時刻),
算出：
1. `detect_all_dots` 自身的絕對耗時（跨 tile 加總），相對於新的端到端 wall-clock 的百分比。
2. 與同一批 tile 的 B1（GPU 前段）耗時比較，確認「CPU 後段是重的那一極」在 25/121/441 三個規模下
   是否穩定成立（pipeline-overlap-result.md 只在 121-tile 驗證過一次）。
3. 由 (1) 算 Amdahl 天花板 `1/(1-p)`。**若天花板 < 加了本文件建議的最便宜方案 (§3(a)) 之後預期能拿到
   的收益量級的門檻（沿用 playbook 的 ~10% 停損慣例），直接停手記錄現況，不執行 §3 的 (b)/(c)。**

這一步本身零風險（只加計時器，不改行為），且是 §3 所有方案該不該做的前提——呼應 karpathy_rule
「先驗證假設」與 doc 10/11 已建立的先例（doc 11 §4(a) 就是同樣的「先診斷、再決定要不要做」模式）。

---

## 3. 候選方案（依 playbook「便宜優先」排序；(a) 與其後選項不互斥，可疊加但需各自 ablation）

### (a) 消除迴圈內的重複常數配置（**最便宜、零風險、與其他方案是否執行無關，直接做**）

讀碼發現兩處明確的重複浪費，兩者都是**同一個 config 衍生的結構元素，在同一次 `detect_all_dots`
呼叫中對每個 cell / 每個候選 blob 重新配置一次**：

1. `m3_dot_kernels.py` `_detect_red_dots`/`_detect_black_dots`（L64/L161-167）：`disk(seed_dilate)`
   每次呼叫（= 每顆 cell）都重新配置一次。`seed_dilate` 全程來自
   `cfg.dot_seed_dilate_radius`/`dot_black_seed_dilate_radius`，同一次 `detect_all_dots` 呼叫內
   對所有 cell 是常數。
2. `_compute_ring_stats`（L328-329）：`disk(ring_gap)` 與 `disk(ring_gap + ring_width)`
   每次呼叫（= 每個**通過面積/圓形度/實心度篩選的候選 blob**，可能每顆 cell 呼叫多次）都重新
   配置一次。`ring_gap`/`ring_width` 同樣全程來自 config，對整次 `detect_all_dots` 呼叫是常數。

**方案**：把這幾個 `disk(...)` 提到 `detect_all_dots` 的呼叫層級算一次（例如作為參數往下傳，或
在 `_detect_one_cell`/`_detect_red_dots`/`_detect_black_dots`/`_compute_ring_stats` 的呼叫鏈上
傳一個已配置好的結構元素而不是半徑），行為完全不變（`disk(r)` 是純函式，同樣的 `r` 永遠回傳同樣
的陣列）——**不需要正確性 ablation**（輸出 bit-exact 不變，只是省掉重複配置的 CPU cycles），
唯一要做的驗證是「跑一次確認輸出真的沒變」而非雜訊地板比對。

**定位**：純粹去掉 Class 1（演算法/常數因子浪費）裡最乾淨的一塊。**收益量級未知**（`disk()`
本身很便宜，這個修正是否能量出可觀差異，取決於 cell 數 × 候選 blob 數的規模——441-tile 有 379
cells，每顆 cell 可能有多個候選 blob 通過篩選才呼叫 `_compute_ring_stats`），**必須量，不能假設
一定有感**，但因為零風險、改動面極小，即使收益不明顯也該做（不會是負向優化）。

### (b) 把 `detect_all_dots` 的 joblib 後端從 `prefer='threads'` 換成 process 後端（**視 §2 診斷結果決定，需先過 doc 11 §7.1 的 spawn-safety 檢查**）

**這不是重新提一個已經被否決的方案**——doc 11 §4(b) 與 gil-contention-diag.md 判定「不做」的理由
是**「換後端無法降低 GPU idle」**（因為 idle 的主因是主執行緒自己的 Python 開銷，不是背景執行緒的
GIL 份額）。但本文件問的是**不同的問題**：`detect_all_dots` 現在是 CPU 後段（B2+B3+B4）裡最重的
一極，**它自己的絕對耗時**能不能靠換後端縮短——這個問題 doc 11 從未評估過（doc 11 §7.1/§7.2 的
spawn-safety 腳本與 memmap 開銷量測**因為 (b) 被否決而從未執行**，不是因為它們過時或不成立）。

- **理由**：gil-contention-diag.md 結果 1 已經量到 `detect.red_black`（12.5%）+
  `regionprops`/per-region Python 迴圈（4.0%）合計 **16.5%** 的 GIL 樣本集中在背景執行緒的
  detect_all_dots 路徑——`prefer='threads'` 讓這些 per-cell 任務在 GIL 序列化下幾乎等於單執行緒跑，
  joblib 平行化的 `n_jobs=-1` 名義平行度沒有真正發揮。換成 process 後端（每個 worker 有自己的
  GIL）理論上能讓這部分真正平行，直接壓縮 `detect_all_dots` 自身的絕對耗時。
- **程式碼裡已經有伏筆**：`m3_dot_detection.py` L192 的註解寫著「joblib 對大陣列自動 memmap，每個
  只 dump 一次供所有 worker 共享唯讀」——這句話**只對 process 後端有意義**（thread 後端本來就共享
  記憶體，不需要 memmap）。這暗示當初設計時就預期過 process 後端這條路，只是後來因為
  `b51ce6a`（見 doc 11 §3）為了避開 memmap 開銷才改成 `prefer='threads'`。
- **必須先解的開放問題（照搬 doc 11 §7.1/§7.2，原封不動，因為前提沒變）**：
  1. **fork-under-CUDA**：三個 GPU 模型在主行程載入、CUDA context 已初始化在先，loky 預設
     `fork` 起 worker pool 與 `CLAUDE.md` 明文排除的「跨 tile ProcessPoolExecutor」是同一類風險。
     **必須顯式 `mp_context=multiprocessing.get_context('spawn')`**，且要先用獨立小腳本驗證
     spawn worker 能正常跑、不卡死 CUDA context。
  2. **memmap 開銷重新量測**：`count_mask`/`L`/`a`/`b` 是 whole-tile 陣列（1024px 量級），process
     後端會觸發 joblib 預設 `max_nbytes=1M` 的 memmap 序列化。要用當前規模（441-tile 379 cells）
     重新量一次這筆開銷，不能沿用 `03-benchmarks-bottlenecks.md` 更早重構前的 4.8% 舊數字。
  3. **worker pool 啟動成本攤提**：loky 預設會重用 executor，第一次呼叫才付 spawn 成本；但
     `detect_all_dots` 是每個 tile 呼叫一次（441 次），要確認 pool 真的跨 tile 重用，而不是每個
     tile 重新 spawn（若是後者，spawn 開銷 × 441 可能直接蓋過 GIL 釋放的收益，變成負向優化）。
- **驗收方式**：與 §2 的第 0 步同一套方法，A/B 比較 `detect_all_dots` 自身耗時（不是端到端，
  先看局部指標），確認真的降了才進到端到端 ablation（同 doc 10/11 的驗收標準）。

### (c) 縮小每個 per-cell 任務本身持有 GIL 的時間（regionprops 開銷）— **與 (b) 互斥的另一條路，若 (b) 因 spawn/memmap 開銷判定不划算，改走這條**

gil-contention-diag.md 明確點名 `regionprops` + per-region Python 迴圈是背景執行緒 GIL 佔用的一部分
（4.0%）。讀碼確認（`m3_dot_kernels.py` L78-79、L181-182）：`_detect_red_dots`/`_detect_black_dots`
用 `regionprops(label_img, intensity_image=...)` 對**所有**候選 blob 建立完整 `RegionProperties`
物件（area/perimeter/solidity/centroid/bbox/coords 等），再逐一用 Python `for p in props` 迴圈套用
面積/圓形度/實心度篩選——多數候選 blob 會在第一道 `area` 篩選就被 `continue` 掉，但**物件建立本身
的開銷已經付了**。

- **方案**：改用 `skimage.measure.regionprops_table`，只要 `area`/`perimeter`/`solidity`/`centroid`/
  `bbox` 幾個欄位（`regionprops_table` 用向量化方式一次算完，不逐一建立 Python 物件），先用 numpy
  布林遮罩一次性篩掉不合格的 blob（取代目前逐一 `continue` 的 Python 迴圈），只對**存活**的少數
  candidate 才用 `label_img == label_id` 取 `coords`（給 `_compute_ring_stats` 用）並算 ring
  stats。這是把「對所有 blob 做 Python 逐一物件化 + 逐一篩選」改成「對所有 blob 做一次向量化篩選、
  只對存活者做 Python 逐一處理」——篩掉的比例越高（大部分候選 blob 面積/形狀不合格），收益越大。
- **風險**：`regionprops_table` 不直接給 `coords`（本函式用 `p.coords`/`rows, cols = p.coords[:, 0],
  p.coords[:, 1]` 取像素座標算 `mean_a_dot`/ring stats 等），需要另外用 `label_img == label_id`
  重建——這是行為等價的改寫，但要仔細核對 `regionprops_table` 的 `area`/`perimeter`/`solidity`算法
  是否與 `regionprops` 物件版本完全一致（理論上底層算法相同，只是介面不同，但需要跑一次 bit-exact
  比對確認，不能只憑理論假設）。改動面比 (a) 大、比 (b) 小，且不涉及 process/spawn 的架構風險。

### (d) 把偵測從「per-cell Python 迴圈」改成「整塊 tile 向量化一次算完」（**最高上限、最高風險，列 backlog，非本輪必做**）

目前的結構是：`detect_all_dots` 先用 `count_mask`（`_build_nucleus_owner_mask`）把整張 tile 切成
「每顆 cell 擁有的 DISH 核區域」，然後用 joblib 對**每顆 cell 分別**呼叫一次 `h_maxima`/`h_minima`/
`binary_dilation`/`label`/`regionprops`（`_detect_one_cell` → `_detect_red_dots`/`_detect_black_dots`，
`m3_dot_kernels.py`）——即使每顆 cell 的 patch 很小，`h_maxima`/`h_minima` 這類形態學重建演算法
每次呼叫都有固定的演算法開銷（不是純線性於像素數），441 顆 cell 就是 441 次獨立呼叫的固定開銷疊加。

**理論上**，既然 `count_mask` 已經把整張 tile 分好區（每個 pixel 最多屬於一個 cell 的核區域），
`h_maxima`/`h_minima`/`binary_dilation`/`label` 這類 skimage 函式本來就是設計成在整張影像上一次呼叫
（它們的效能特性就是為此優化的）——可以整張 tile 的 `L`/`a`/`b`（已用 `count_mask>0` 遮罩過）各只呼叫
一次 `h_maxima`/`h_minima`，再用一次 `label(connectivity=2)` 拿到全 tile 所有候選 blob，最後用
`count_mask` 查表決定每個 blob 屬於哪個 cell——把 O(cells) 次獨立形態學呼叫 + O(cells) 次
`regionprops` 物件化，壓成 O(1) 次（每種操作各一次，紅點黑點各自一輪）。這是把 §3 開頭觀察到的
「per-cell Python 迴圈本身有 GIL/dispatch 稅」連根拔起，而不是像 (b)/(c) 那樣減少稅率。

- **為什麼列 backlog、不是本輪方案**：
  1. **正確性風險最高**：`_compute_ring_stats` 目前用 `cell_roi`（該 cell 的局部核區域）與
     `bg_mask` 限制 ring 統計範圍，確保 ring 不會跨到鄰居 cell 的核區域或背景裡。整塊 tile 算完後，
     要重新設計 ring 統計的邊界邏輯（例如用 `count_mask == cid` 取代目前的 `cell_roi` 局部 patch
     版本），必須逐一比對每個 config 參數（`ring_gap`/`ring_width`/`bg_mask` 排除）在整塊 tile 版本
     下語意完全等價，這比 (c) 的改寫風險高一個量級。
  2. **改動面大**：`_detect_one_cell`/`_detect_red_dots`/`_detect_black_dots`/`_compute_ring_stats`
     四個函式的介面與內部邏輯都要重寫成「整塊 tile 版本」，不是像 (a)/(c) 那樣局部替換。
  3. **收益是否值得，取決於 §2 量到的天花板**：若 §2 量出 `detect_all_dots` 的 Amdahl 天花板已經
     不高（例如做完 (a)(b)/(c) 後剩餘天花板落在 playbook 的 ~10% 停損線附近），不值得為了榨最後
     幾個百分點冒這個正確性風險——這正是 doc 11 §2 對 UNet/Cellpose 19% wall 那塊的處理方式
     （技術可行但天花板/風險比不划算，列 backlog）。
- **前提**：只有在 (a)(b)/(c) 都做完、重新量測後天花板仍然明顯高於停損線，才值得啟動這個選項的
  詳細設計（屆時需要另立文件，比照 doc 10/11 的規格與驗收流程）。

---

## 4. 推薦順序

1. **§2 第 0 步（重新量測，零風險）**——沒有這個數字，後面所有方案的優先順序都是猜的。
2. **(a) 提出重複的 `disk()` 配置（零風險，直接做，與診斷結果無關）**。
3. 依 §2 的診斷結果二選一（不要同時做，逐一 ablation，呼應 playbook 反模式 #6/#7）：
   - 若 `detect_all_dots` 的耗時主要卡在**跨 cell 的 GIL 序列化**（多核心沒有真正平行）→ 走
     **(b) process 後端**，但必須先過 spawn-safety + memmap 開銷驗證（照搬 doc 11 §7.1/§7.2，
     不能省略）。
   - 若診斷顯示 GIL 釋放後收益有限（例如 spawn/memmap 開銷蓋過收益，或本來就有一定程度的真平行）
     → 走 **(c) regionprops_table 改寫**，降低每個 task 自身的常數開銷。
   - (b)/(c) 也可以在各自 ablation 通過、確認都有正貢獻後疊加，但要先分開驗證各自的貢獻量。
4. **(d) 整塊 tile 向量化**——列 backlog，只有 (a)(b)/(c) 做完後天花板仍明顯高於停損線才啟動，
   需要另立文件。

---

## 5. 驗收標準（延續 doc 10/11 的方法論）

1. **基準**：以 §2 第 0 步量到的「① 方案 (b) 已落地、無 py-spy」乾淨數字為新基準（25/121/441-tile
   各自的 `detect_all_dots` 絕對耗時 + 端到端 wall-clock），**不可用 bottleneck-list.md 的舊序列
   baseline（30.7%）或 gil-contention-diag.md 的 py-spy 膨脹數字直接當驗收基準**。
2. **負向優化偵測**：任何改動後，`detect_all_dots` 自身耗時與端到端 wall-clock 都必須低於新基準
   （或至少不劣化）；若打平或更差，立即停手重新量測（playbook 紅旗）。
3. **正確性不可退讓**：(a) 除外（bit-exact，理論上不需要 ablation，但仍建議跑一次確認），(b)/(c)/(d)
   都要用 `report.csv`/centroid 對照雜訊地板（`05-dev-testing-guide.md` 方法），特別是 (c)/(d) 改寫
   了 blob 篩選或 ring 統計的實作路徑，必須逐一比對 `reddot`/`blackdot`/`score` 欄位。
4. **記憶體有界重新驗證（限 (b)）**：process 後端會有 worker pool + memmap 暫存的額外記憶體佔用，
   需要重新量 RSS 曲線，確認 bottleneck-list.md「記憶體有界」的結論不因此破功（同 doc 10 §5.4 的
   驗證方法）。
5. **fork-safety 驗證先於整合（限 (b)）**：doc 11 §7.1 的 spawn-safety 獨立小腳本必須先通過，才能
   改動 `m3_dot_detection.py`。
6. **Ablation**：每個改動獨立驗證貢獻量，零貢獻的部分依 playbook「零貢獻要砍」+ karpathy_rule
   直接還原（呼應 gil-contention-diag.md「方案 (d) gc 重定位」的先例——量出來沒用就承認並還原，
   不要為了「已經做了」而保留）。

---

## 6. 本輪不處理什麼，以及為什麼

- **① 的下一階段（CuPy/GPU kernel 級優化）**：依使用者指示，不在本文件範圍內，屬獨立主題。
- **doc 11 §4(c) 加深 pipeline 深度（depth 2）**：與本文件的方案不衝突、可疊加，但屬於「調整
  pipeline 調度」而非「優化 `detect_all_dots` 本身」，doc 11 已經談過候選設計，本文件不重複展開。
  若 §2 診斷顯示 `detect_all_dots` 即使優化後仍是穩態吞吐的瓶頸（而非氣泡問題），加深 pipeline
  深度也無法降低它自身的耗時（doc 11 §4(c) 已經指出這個侷限），不是本文件方案的替代品。
- **Cellpose/UNet Python mask 重建 19% wall（gil-contention-diag.md「追加深挖」）**：與 ② 是不同的
  瓶頸（主執行緒 GPU 前段內部，不是背景 CPU 後段），已在該文件列為獨立 backlog（第三方套件 patch），
  不在本文件範圍。
- **elastic_dish_nucleus_matching（`m3_elastic_matching.py`）**：讀碼確認已經是向量化實作
  （`cKDTree` 找候選、`find_objects`/`center_of_mass` 算質心與面積、集合運算找重疊配對），沒有
  per-cell Python 迴圈的 O(cells) 重複開銷模式。§2 的計時器仍應把它與 `detect_all_dots` 的其餘部分
  分開量（因為它在 `detect_all_dots` 函式內部被呼叫），但目前讀碼沒有發現需要優化的明顯理由，
  留待 §2 的量測數字證實或推翻這個判斷。
