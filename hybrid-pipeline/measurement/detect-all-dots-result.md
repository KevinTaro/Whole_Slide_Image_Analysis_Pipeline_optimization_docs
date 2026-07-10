# ② `detect_all_dots` — 方案落地與量測結果

> 執行 [../12-detect-all-dots-optimization-plan.md](../12-detect-all-dots-optimization-plan.md)
> 的推薦順序：**§2 第 0 步（乾淨重新量測）→ §3(a)（disk() 重複配置消除）→ 依 §2 診斷決定
> (b)/(c)/(d)**。依 [PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md](../PERFORMANCE_BOTTLENECK_PLAYBOOK.quickref.md)
> 的 Measure→Analyze→Choose 紀律。doc 12 保持不動（作為對照規格）。本文件只記錄「做了什麼 + 量到什麼 + 決定什麼」。
>
> 環境：git `00f2c91`（① 方案 (b) 兩段式 overlap pipeline 已在 HEAD）、config_hash `db2b7e6a`、
> RTX 5090 / CUDA 13.0 / torch 2.10.0+cu130、venv `/data/taro_Projects/tsgh/.venv`。
> 量測輸入：`test_picture/_roi_crops/{med,large}_{ihc,dish}.tiff`（medium 8192² = 11×11 = 121 tiles；
> large 16384²級 = 21×21 = 441 tiles）。**無 py-spy**（避免 gil-contention-diag.md §method 警告的 wall 膨脹）。
> 日期：2026-07-11。

---

## 1. 結論（TL;DR）

**doc 12 §1 的核心前提被本文件的量測推翻**。doc 12 §1 假設「① overlap 落地後，`detect_all_dots`
（CPU 後段）比 GPU 前段更長、是決定穩態節拍的『重的那一極』」。**實測相反**：在兩段式 overlap
pipeline 下，**GPU 前段才是關鍵路徑**（441-tile 佔 wall 80.9%），`detect_all_dots` 在背景執行緒上
**完全被 GPU 前段遮住**——每個 scale、每個 CPU 熱狀態下，每-tile `detect_all_dots`（≤ 0.67 s）都
**不到**每-tile GPU 前段（~1.33 s）的一半。

因此：

- **§2 第 0 步**：完成。`detect_all_dots` 對端到端 wall 的**有效** Amdahl 天花板 ≈ **1.0**（不是名目上
  的 1.26–1.49），因為它已被 overlap 遮住 → 觸發 playbook 停損（反模式 #2「優化 <10% 關鍵路徑」）。
- **§3(a)（disk() 重複配置消除）**：**已落地**。bit-exact（report.csv 落在 GPU 雜訊地板內）、微基準
  證明只會更快不會更慢、同熱狀態 apples-to-apples 略快。但因 `detect_all_dots` 被遮住，**端到端收益 = 0**
  ——保留為零風險的元件級去重（呼應 doc 12 §3(a)「即使收益不明顯也該做，不會是負向優化」）。
- **§3 (b) process 後端 / (c) regionprops_table / (d) 整塊 tile 向量化**：**不執行**。依 doc 12 §2 停損
  條款與 playbook——`detect_all_dots` 不在關鍵路徑上，加速它（無論用哪條路）都動不了端到端 wall。
  真正的瓶頸是 GPU 前段（M2+M3b Cellpose 前向，441-tile 佔 80.9% / 571 s），那是 **① 的下一階段
  （CuPy/GPU kernel 級優化），doc 12 §6 明文排除在本輪範圍外**。

---

## 2. §2 第 0 步量測（乾淨基準，① overlap 已落地、無 py-spy）

錨點（每個 scale 一次乾淨跑；`B1 GPU 前段` = UNet core mask + 兩次 Cellpose `segment_windowed` 前向）：

| scale | tiles | end-to-end | `detect_all_dots` 自身 | B1 GPU 前段 | 每-tile detect | 每-tile GPU 前段 | peak RSS |
|---|--:|--:|--:|--:|--:|--:|--:|
| medium | 121 | 184.9 s | 38.1 s（**20.6%**，天花板 1.26） | 143.0 s（77.3%） | 0.320 s | **1.329 s** | 2.87 GB |
| large  | 441 | 724.7 s | 238.6 s（**32.9%**，天花板 1.49） | 586.2 s（80.9%） | 0.579 s | **1.329 s** | 3.91 GB |

**關鍵判讀**（playbook step 2「is it fast」與「is it the bottleneck」要分開）：

1. **`detect_all_dots` 被 overlap 完全遮住。** 兩段式 pipeline（深度 1、單背景執行緒）的穩態節拍是
   `Σ max(GPU_前段_tile, CPU_後段_tile)`。每-tile GPU 前段 ≈ 1.33 s；每-tile CPU 後段（detect ~0.58 s
   + PNG 落地 ~0.18 s + render/crop）≈ 0.9 s < 1.33 s。故 CPU 後段（含 `detect_all_dots`）**整段藏在
   GPU 前段背後**，對關鍵路徑的貢獻 ≈ 0。名目上「detect 佔 wall 20.6%/32.9%」是**自身耗時 ÷ wall**，
   在 overlap 下不代表關鍵路徑佔比。
2. **「detect 佔 wall %」隨密度上升（20.6%→32.9%），但每-tile detect 幾乎不變（0.32→0.58 s，見下方熱
   狀態說明），且始終 < 每-tile GPU 前段。** 密度上升讓 detect 的**總**自身時間變多，但它仍被更長的
   GPU 前段遮住——瓶頸沒有移到它身上。
3. **真正的關鍵路徑是 GPU 前段**（80.9% / 571 s @441，幾乎全是兩次 Cellpose 前向）。這正是 ① 的
   下一階段主題，doc 12 §6 明文排除。

### 2.1 量測衛生：本機 CPU 頻率隨熱狀態變動 ~2×（重要 caveat）

`detect_all_dots` 是純 CPU 工作，對 CPU 頻率敏感。同一份 medium 輸入、**同一份程式碼**，在不同熱狀態下
量到：

| medium 121-tile（同輸入同碼） | `detect_all_dots` 自身 | 每-tile |
|---|--:|--:|
| clean，機器剛開跑（boost） | 38.1 s | 0.320 s |
| clean，緊接 441-tile 長跑之後（throttle） | 79.9 s | 0.671 s |

→ **純 CPU 工作的絕對耗時會因 CPU 頻率差到 2×**。這不影響本文件的結論，因為：**GPU 前段（GPU-bound）
不隨 CPU 熱狀態同步變動，維持 ~1.33 s/tile；而 detect 即使在最壞（throttle）狀態也只有 0.67 s/tile，
仍 < GPU 前段的一半**——detect 被遮住的結論在所有熱狀態下 a fortiori 成立。
**方法學教訓**：跨 stage 比較只在**同一次跑**內做（如上表 large 的 detect 0.579 vs GPU 前段 1.329 皆
取自同一次 441 跑）；跨跑比較純 CPU stage 的絕對秒數不可靠。

---

## 3. §3(a) 已落地：消除迴圈內重複的 `disk()` 結構元素配置

**改動**（只在 `m3_dot_kernels.py`，共 4 個呼叫點 + 1 個 helper）：新增 module 級
`@lru_cache` 的 `_disk_footprint(radius)`，回傳一顆**唯讀**共享 disk footprint；把
`_detect_red_dots` / `_detect_black_dots`（`disk(seed_dilate)`）與 `_compute_ring_stats`
（`disk(ring_gap)`、`disk(ring_gap+ring_width)`）四處的即時 `disk(...)` 換成它。`seed_dilate` /
`ring_gap` / `ring_width` 全程來自 config，對整批只有少數幾個相異半徑，故 cache 極小且跨 tile/批次
只算一次。

> **為何用 lru_cache 而非 doc 12 §3(a) 舉例的「參數往下傳」**：兩者達成相同意圖（「算一次」），
> lru_cache 面極小（不動 4 個函式簽章）、bit-exact（`disk(r)` 是純函式）、且唯讀 footprint 防止任何
> 意外 in-place 寫入。footprint 只被 `binary_dilation` 唯讀使用（已核對本檔全部 disk() 用途），故共享
> 安全。符合 karpathy_rule §3「surgical changes」。

**驗證**：

1. **bit-exact（元件級證明）**：對所有相關半徑 `r∈{0..8}`，`_disk_footprint(r)` 與 `disk(r)`
   `np.array_equal` 為 True、cache 回傳同一物件、footprint 為唯讀。因為這是唯一改動且下游形態學對相同
   footprint 為決定性 → 點偵測輸出依構造不變。
2. **端到端 report.csv 對照（medium）**：clean vs with-(a)，3557 vs 3558 cells，3555 顆 <3px 對上，
   `reddot`/`blackdot` 各 2 筆、`score` 1 筆不符——**恰好落在 pipeline-overlap-result.md 記錄的 GPU
   run-to-run 雜訊地板（2/2/1）內**。差異純為 Cellpose/UNet 前向非決定性，非 (a) 造成。
3. **不是負向優化**：
   - 微基準：`disk(3)`=6.6 µs/call、cache hit=0.04 µs/call；`binary_dilation` 用唯讀 cached footprint
     80.9 µs vs 每次重算 88.1 µs（唯讀 vs 可寫 footprint 差異可忽略 80.9 vs 80.6）。
   - 同熱狀態 apples-to-apples（緊接長跑後）：with-(a) detect 74.8 s **略快於** clean 79.9 s。
4. **端到端收益 = 0**：因 §2 證明 `detect_all_dots` 被 overlap 遮住，即使 detect 略快（0.63 vs
   0.67 s/tile）仍 < GPU 前段 1.33 s/tile → 仍被遮住 → wall 不變。故 (a) 保留為**零風險元件級去重**，
   非端到端優化。（playbook 嚴格讀法會因「零端到端貢獻」建議砍除；但 (a) bit-exact、面極小、只會更快，
   且 doc 12 §3(a) 明文「與診斷結果無關、直接做」，故保留並如實標註。）

---

## 4. §3 (b)/(c)/(d) 不執行——依 doc 12 §2 停損條款

doc 12 §2 明文：「若天花板 < 停損門檻…直接停手記錄現況，不執行 §3 的 (b)/(c)」。本文件 §2 量到的不是
「天花板略高但值得」，而是更強的結論——**`detect_all_dots` 根本不在關鍵路徑上**（被 GPU 前段遮住），
其對端到端 wall 的有效天花板 ≈ 1.0。所以：

- **(b) process 後端**（doc 12 §3(b)）：目標是靠釋放 GIL 縮短 detect 自身絕對耗時。但 detect 已被遮住，
  縮短它動不了 wall；且需先付 doc 11 §7.1/§7.2 的 spawn-safety + memmap 開銷驗證成本。**不做。**
- **(c) regionprops_table 改寫**（doc 12 §3(c)）：同理，降低 detect 自身常數開銷對已遮住的 stage 無端到端
  意義。**不做。**
- **(d) 整塊 tile 向量化**（doc 12 §3(d)）：doc 12 已列 backlog、且前提是「(a)(b)/(c) 做完後天花板仍明顯
  高於停損線」。本文件量到天花板 ≈ 1.0，**前提不成立**，不啟動。

**下一步真正的槓桿**在 GPU 前段（兩次 Cellpose 前向 = 80.9% wall），即 doc 12 §6 明文排除的
**① 下一階段（CuPy/GPU kernel 級優化）**，屬獨立主題，不在本輪範圍。

---

## 5. 附帶修復：量測 harness 對重構後命名的容錯

`scripts/perf_measure.py` 是為舊 git `96a28ba` 寫的，① overlap 重構後 `segment_masked_dish` 這個符號
已不存在（M2 現在也走 `segment_windowed`），舊 harness 會在 `install_wrappers` 直接 `AttributeError`
崩潰。改法：`wrap()` 對缺失符號改為印一行 skip 訊息並略過，而非崩潰。這讓 harness 在命名再變動時仍能
跑完（非侵入式量測工具本就該對受測命名空間的漂移容錯）。

> 副作用（已在 §2 判讀中納入）：M2 與 M3b 現在都是 `segment_windowed`，故 `B1_m3b_cellpose` bucket
> 現在合計了 M2+M3b 兩次 Cellpose 前向；B1 GPU 前段總時 = `B1_unet_coremask` + `B1_m3b_cellpose`。

---

## 6. 重現方式

```bash
cd /data/taro_Projects/tsgh
ROI=backend/algorithms/hybrid/test_picture/_roi_crops
# §2 乾淨基準（medium / large）：
.venv/bin/python scripts/perf_measure.py \
  --ihc "$PWD/$ROI/med_ihc.tiff"   --dish "$PWD/$ROI/med_dish.tiff"   --output <out> --label medium_121tile --workers 8 --gpu-dmon --metrics-dir <m>
.venv/bin/python scripts/perf_measure.py \
  --ihc "$PWD/$ROI/large_ihc.tiff" --dish "$PWD/$ROI/large_dish.tiff" --output <out> --label large_441tile --workers 8 --gpu-dmon --metrics-dir <m>
# 判讀：<m>/<label>_timings.json 內
#   detect 自身 = timings.B3_detect_dots.t（÷ n = 每-tile）
#   GPU 前段    = timings.B1_unet_coremask.t + timings.B1_m3b_cellpose.t（÷ n_tiles = 每-tile）
#   比較每-tile detect vs 每-tile GPU 前段（務必取自同一次跑，見 §2.1 熱狀態 caveat）。
# (a) 正確性：clean 與 with-(a) 各跑一次 medium，report.csv 以最近質心 <3px 配對，
#            比對 reddot/blackdot/score 是否落在雜訊地板（2/2/1）內。
```
