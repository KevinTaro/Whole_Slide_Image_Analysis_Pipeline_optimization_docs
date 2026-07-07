# 11 — GPU 序列瓶頸 stage 2：detect_all_dots 與主執行緒的 GIL 競爭

> 承接 [measurement/pipeline-overlap-result.md](./measurement/pipeline-overlap-result.md)（方案 (b)
> 兩段式 pipeline/overlap 的實作與量測結果）。方案 (b) 已驗收通過（見該文件），但**沒有拿到理論上限**，
> 且該文件已明確指出剩餘差距的根因與方向（見其「為何實際省 18.5%，而非理論上限 ~50%」一節）。本文件
> 依 [PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md](./PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md) 的
> Analyze→Plan→Choose 流程，針對這個新移動到的瓶頸設計下一步方案。**doc 10 與
> pipeline-overlap-result.md 保持不動**，作為對照規格；本文件不重複展開已解決的①原始問題。
>
> 基準：pipeline-overlap-result.md 量測於 git 工作區基於 `ce980d1`；本文件撰寫時 HEAD 為 `3d7d91c`，
> 兩者之間 `hybrid_pipeline.py`/`m3_dot_detection.py` 無再變動，量測數字對本 HEAD 仍然有效。
> **新錨點（取代 848.0 s，僅限 121-tile 規模比較）**：medium 121 tiles baseline 255.0 s → 方案 (b)
> 後 **207.9 s**，此數字為本文件之後所有 ablation 的比較基準，不可覆蓋。

---

## 1. 現況重述（來自 pipeline-overlap-result.md，已驗收）

| 指標 | baseline | 方案 (b) 後 | 判定 |
|---|---|---|---|
| 121-tile 端到端 | 255.0 s | **207.9 s**（-18.5%） | ✅ |
| GPU idle_frac | 0.494 | **0.154** | ✅ idle 消掉 ~69% |
| 正確性 | — | 0 筆不符（優於雜訊地板） | ✅ |
| peak RSS | 2.95 GB | 3.06 GB（+4%） | ✅ 仍有界 |

理論上限（`max(B1, B2+B3+B4) ≈ 45.5%`，即 wall-clock 應接近腰斬）沒有達到，
pipeline-overlap-result.md 已診斷根因：`detect_all_dots`（`m3_dot_detection.py:194`）用 joblib
`Parallel(..., prefer='threads')`，在背景執行緒裡跑，與主執行緒的 torch CPU 端（tensor 搬運、
`.cpu()`/前處理等非 kernel 純 Python/C 呼叫）搶 **同一個 GIL**，把重疊率壓低。

## 2. Amdahl 停損檢查（先確認這輪還值不值得做）

剩餘 idle **15.4%**（0.154），Amdahl 天花板 `1/(1-0.154) ≈ 1.18`——即使把重疊做到完美，
121-tile 規模最多再省 **~18%**。這剛好卡在 playbook「< ~10% 不值得深入」的停損線**之上**，
所以還在可以嘗試的範圍，但已經接近邊緣：**這輪只做低成本的診斷 + 一個候選修改，不值得為了
最後 15% 投入新架構**。若診斷顯示根因比預期複雜（例如 GIL 競爭發生在多處、難以定位），
應直接停手記錄現況，而不是繼續加碼。

## 3. 根因回顧與一個容易漏掉的兩難

讀 `git log -p` 追 `m3_dot_detection.py` 的 joblib 呼叫，發現一個**前一輪文件沒交叉引用到**的
歷史決策，會直接影響本輪方案的可行性：

- **舊版**（`04-optimization-roadmap.md` §S2，pre-refactor 舊數字）：`detect_all_dots` 原本用
  joblib **process 後端**（未指定 `prefer`，預設 `loky`）。當時量到 process 後端的 memmap
  暫存 `delete_folder` 清理額外佔 **4.8%**（`03-benchmarks-bottlenecks.md` #8），S2 建議
  `backend='loky'` 搭 `prefer='threads'` 或加大 `max_nbytes` 來避免這筆開銷。
- **commit `b51ce6a`（Jun 30）** 實際落地了這個建議：把 `Parallel(n_jobs=n_jobs_eff)` 改成
  `Parallel(n_jobs=n_jobs_eff, prefer='threads')`——**這個改動早於方案 (b)，動機是省掉 process
  後端的 memmap 序列化/清理開銷，跟 GPU 重疊完全無關**（當時 GPU 還是完全序列跑，沒有背景
  執行緒會跟它搶 GIL）。

**兩難**：現在要解的 GIL 競爭問題，解法直覺是「換回 process 後端」——但那正是 `b51ce6a`
當初刻意繞開的方案，換回去等於重新引入已經被量過、被判定不划算的 memmap 開銷（除非
`max_nbytes`/陣列大小的量測前提已經變了）。**不能只看單一維度**（GIL vs memmap）就下結論，
必須重新用當前規模（121/441-tile，細胞數已比 S2 量測時多）量一次，兩個開銷都可能已經變了。

## 4. 候選方案（依 playbook「便宜優先」，且先驗證假設再動代碼）

### (a) 診斷優先 — 定位 GIL 競爭具體發生在哪一段（**推薦，本輪先做這個**）
不直接改 `hybrid_pipeline.py`，先寫一個獨立小腳本（呼應 doc 10 §7 已建立的習慣：「不要直接改
生產代碼才發現假設不成立」），用 `py-spy dump`/`py-spy record` 或 Python 的
`sys.setswitchinterval` + 手動計時，在 121-tile 跑一次時對主執行緒（torch 前向）與背景執行緒
（`detect_all_dots`）做取樣，回答：
1. GIL 競爭是均勻分散在整個 `detect_all_dots` 呼叫期間，還是集中在少數幾個 Python-level 迴圈
   （例如 `per_cell` dict 組裝、`all_dots.extend`）？`_detect_one_cell`/`m3_dot_kernels` 內部
   若以 numpy/scipy 向量化運算為主，C 層迴圈應該會釋放 GIL——若真正卡住的只是少數 Python
   膠水代碼，優化目標應該是**縮小那一小段**，而不是整個換 process 後端。
2. 主執行緒（torch 前向之間）有多少時間其實花在純 Python（tensor 搬運、`.cpu()`、logging）而非
   釋放 GIL 的 CUDA kernel 呼叫上？這部分即使 `detect_all_dots` 完全不碰 GIL，仍然是重疊的
   硬上限。
- **低風險**：只加量測，不改行為。
- **產出**：一個「GIL 競爭時間軸」的量測結果，決定 (b) 還是 (c) 值得做，或者兩者都不值得
  （如果發現真正瓶頸是主執行緒自己的 Python 開銷，那換 `detect_all_dots` 後端不會有幫助）。

### (b) 換 `detect_all_dots` 為 process-safe 後端（**視 (a) 診斷結果決定要不要做**）
若 (a) 確認 GIL 競爭主要來自 `detect_all_dots` 本身，才嘗試把 `prefer='threads'` 換成
`backend='loky'`（或顯式 `mp_context`）。**必須先解決兩個開放問題**（見第 7 節），否則風險
不亞於直接動 `run_batch`：
1. **fork-under-CUDA**：`run_batch()` 在 `detect_all_dots` 第一次被呼叫前，三個 GPU 模型早已
   在主行程載入、CUDA context 已初始化（見 `CLAUDE.md`「三個 GPU 模型只在主行程載入一次…
   跨行程平行是不安全的」）。loky 後端預設在 Linux 上用 `fork` 建立 worker pool 以降低啟動
   成本；若 pool 是在 CUDA 已初始化之後才第一次建立（幾乎確定是），這與 doc 10 §2 明文排除
   的「跨 tile `ProcessPoolExecutor`」屬於**同一類風險**，只是把 fork 對象從整個 tile 迴圈換成
   `detect_all_dots` 內部——**必須顯式指定 `spawn`**（例如 joblib
   `parallel_config(backend='loky', mp_context=multiprocessing.get_context('spawn'))`），
   不能沿用預設 `fork`。
2. **memmap 開銷是否重新出現**：用當前規模（121/441-tile，細胞數已高於 S2 量測時）重新量一次
   `delete_folder`/memmap 佔比，不能假設舊的 4.8% 數字仍然成立；若 process 後端换來的 GIL
   釋放，被 spawn 啟動成本 + memmap 序列化開銷吃掉大半，就是負向優化，要照 playbook 紅旗
   直接停手，不強行採用。
- **定位**：只有在 (a) 明確指向這裡、且第 7 節的 spawn 驗證通過時才做。

### (c) 加深 pipeline 深度（depth 2）
目前是「兩塊在飛」（本塊 GPU 前段 + 前一塊 CPU 後段，見 `run_batch` L772-813）。若 (a) 診斷
顯示問題是**吞吐**而非**氣泡**（即背景執行緒工作量本身沒少，只是穩態下 GIL 讓它變慢，深度
再深也救不了穩態吞吐），這個方案不會有幫助，可直接跳過。若診斷顯示是**氣泡**（例如某些 tile
的 CPU 後段比 GPU 前段長，deeper pipeline 能讓更多 CPU 後段排隊、爭取更多重疊視窗），才值得
把 `ThreadPoolExecutor(max_workers=1)` 加深、`pending` 從單一 slot 改成 FIFO queue。
- **成本**：比 (b) 稍高（要動 `run_batch` 的 pending 佇列邏輯 + 重新驗證記憶體有界，因為同時
  在飛的 tile 數從 2 增加），且第 7 節開放問題 3 的 fail-fast timing 要重新確認（深度越深，
  背景錯誤要多久才被主執行緒發現）。
- **與 (b) 不互斥**，但兩者要分開 ablation，不能一次做兩件事（playbook 反模式 #7）。

---

## 5. 推薦順序

1. **先做 (a)**：寫診斷腳本，量出 GIL 競爭的時間軸分佈。這一步幾乎零風險、成本最低，且是
   (b)/(c) 該不該做的前提——不要在沒有這個數字之前直接改 `run_batch` 或 `m3_dot_detection.py`
   （呼應 karpathy_rule「先驗證假設」與 doc 10 §7 已建立的先例）。
2. 依 (a) 結果二選一（或都不做，見第 2 節停損）：
   - GIL 集中在 `detect_all_dots` 內部 → 做 (b)，且必須先跑通第 7 節第 1 項的 spawn-safety
     小腳本，才能碰 `hybrid_pipeline.py`/`m3_dot_detection.py`。
   - GIL 分散、或主執行緒自身 Python 開銷才是硬上限 → (b) 大概率無效，考慮 (c)，或者直接停手
     記錄「15.4% 的剩餘 idle 主因不是 `detect_all_dots` 後端，本輪判定不划算，停止追這個方向」。
3. **不要同時做 (b) 和 (c)**——即使兩者診斷都顯示有幫助，也要先單獨量一個、ablation 過，
   再疊加第二個，否則出問題不知道歸咎哪個改動（playbook 反模式 #6/#7）。

---

## 6. 驗收標準（延續 doc 10 §5 的方法論，比較基準改為方案 (b) 的 207.9 s / idle 0.154）

1. **負向優化偵測**：任何改動後，121-tile 端到端**必須明顯低於 207.9 s**（或至少不劣化）；
   idle_frac 必須低於 0.154。若打平或更差，立即停下重新量測，不能只信任局部指標。
2. **fork-safety 驗證先於整合**：若做 (b)，第 7 節第 1 項的獨立小腳本必須先通過（確認 spawn
   context 下 loky worker 能正常跑、且不觸發 CUDA context 相關的 crash/deadlock），才能改
   `m3_dot_detection.py`。
3. **正確性不可退讓**：同 doc 10 §5.3 方法（`report.csv`/centroid 對照雜訊地板），(b)/(c)
   任一改動後都要重新比對。
4. **記憶體有界重新驗證**：若做 (c)（加深 pipeline），peak RSS/VRAM 要重新量，確認沒有隨深度
   線性成長（同時在飛的 tile 數增加，理論上 RSS 峰值也會跟著漲，需要量出實際幅度是否可接受）。
5. **Ablation**：(a) 的診斷結果本身要寫進量測記錄（即使最終判定「不值得做 (b)/(c)」，這個
   negative result 也要留下，避免下一輪重複踩同一個坑）。

---

## 7. 實作前必須先解的開放問題

1. **spawn-safety 小腳本**：在改 `m3_dot_detection.py` 前，先寫一個獨立腳本：主行程先初始化
   一次 CUDA（模擬 `run_batch` 載入三個模型後的狀態），再用
   `joblib.parallel_config(backend='loky', mp_context=multiprocessing.get_context('spawn'))`
   跑一個 `Parallel` 呼叫，確認 (i) worker 進程能正常啟動不 crash、(ii) 不會意外複製/卡住主
   行程的 CUDA context、(iii) spawn 的啟動延遲在 121/441-tile 規模下是否可忽略（loky 預設會
   重用 executor，只有第一次呼叫付 spawn 成本，但要實測確認）。
2. **`max_nbytes` 與陣列大小**：目前傳入 `_detect_one_cell` 的 `count_mask`/`L`/`a`/`b` 是
   whole-tile 陣列（1024px 量級）。若走 process 後端，joblib 預設 `max_nbytes=1M` 會觸發
   memmap；要量出這條路徑在當前規模下的實際開銷，不能沿用 S2 時代（更早的重構、可能是不同
   tile 大小/細胞密度）的 4.8% 舊數字。
3. **fail-fast timing（若做 (c) 加深 pipeline）**：深度越深，某個 tile 的背景錯誤要經過越多
   輪 `_collect` 才會被主執行緒發現——需要明確定義「發現即中止」的粒度，不能讓多個 tile 的
   部分結果在錯誤真正被發現前被誤當已完成寫入（同 doc 10 §7.3 的顧慮，深度加深後風險更高）。
4. **這輪之後怎麼辦**：若 (a)+(b)/(c) 做完，idle 仍卡在個位數 % 以上，按第 2 節的停損邏輯，
   記錄現況並停止——不要為了榨最後幾個百分點引入 doc 10 §3(a) 的跨 tile batching 或 §3(c) 的
   架構替換（那兩項的成本/風險評估在 doc 10 已經做過，結論不因這輪而改變，除非之後又有新
   數據）。
