# 09 · Viewer 端 TIFF 相容性：subifd 通則與實測

> **給誰看**：UI（游能舜）＋演算法（謝尚哲）對齊用。
> **一句話結論**：**這條 pipeline 產出的 TIFF，凡是為了餵 VALIS 而生的都用 `subifd=True`；而 `subifd=True` 的金字塔 OpenSlide 讀不到，直接丟給 viewer 會讓概覽層讀爆記憶體。所以「任何要在 OpenSeadragon 顯示的影像，都需要一份 `subifd=False`（最好再 `tile=True`）的 viewer 副本」。**
> 日期：2026-07-09

---

## 1. 為什麼會有這個問題：SubIFD vs 一般 IFD

WSI 是「同一張圖的多解析度金字塔」。金字塔各層在 TIFF 裡可以兩種放法：

| 放法 | `subifd` | 金字塔各層存哪 | 誰讀得到 |
|---|---|---|---|
| 一般 IFD（多頁鏈） | `False` | 主目錄鏈上一頁一層 | ✅ OpenSlide 走主鏈就看到全部層 |
| SubIFD（子目錄） | `True` | 藏在 Level0 的子目錄，主鏈只剩 1 頁 | ✅ VALIS（會鑽 SubIFD）／❌ OpenSlide 只看到 Level0 |

- **VALIS 要 `subifd=True`**：它自己回讀是用 `subifd=level-1`、看 `n-subifds` 欄位（`valis/slide_io.py`）。`subifd=False` 會讓它的 reader 撞 `tiff2vips: page 1 differs from page 0`。
- **OpenSlide 要 `subifd=False`**：它的通用 TIFF driver 只列舉主鏈 IFD、不鑽 SubIFD，所以看到 `subifd=True` 檔會回報 `level_count=1`（金字塔形同不存在）。

我們的 tile server（`backend/io/pyramid.py`）就是用 OpenSlide，因此受 OpenSlide 這條限制。

---

## 2. 全 pipeline 各產出的 subifd 現況（已從程式碼確認）

| 產出檔 | 產生位置 | `subifd` | tiled | OpenSlide 看到金字塔 | 給 OSD 用 |
|---|---|---|---|---|---|
| `*_processed.tiff`（CZI→BigTIFF） | `module1_preprocess.py:312-323` | **True**（明設） | ✅ 1024 | ❌ `level_count=1` | ⚠️ 需 viewer 副本 |
| `*_warped_lvN.ome.tiff`（VALIS 對準圖） | `module4_thumbnail.py:62-84` → `valis/slide_io.py:3685` (`subifd = pyramid`) | **True**（隨 `pyramid=True`） | ✅ 512 | ❌ `level_count=1` | ⚠️ 需 viewer 副本 |
| `Merged_Aligned_lvN.tiff`（疊合成品） | `module4_thumbnail.py:110-116` | **False**（pyvips 預設，未設） | ❌ 無 `tile=True`→strip | ✅ 讀得到 | 🟡 可開，strip 效率待驗 |

> 重點：只有 `Merged_Aligned` 天生對 viewer 友善（subifd=False），但它**沒有 `tile=True`**，是 strip 型金字塔，隨機存取效率不如 tiled，仍建議實測或補存 tiled 版。其餘兩種都是 `subifd=True`，直接顯示會踩雷。

---

## 3. 實測數據（2026-07-09，走 `backend/io/pyramid.py` 真實供給路徑）

以 HER2 40X 為例（`141818 × 114366`，約 16 gigapixel）比較同一張圖的兩種存法：

| 指標 | `HER2_processed`（subifd=True，組員 pipeline 原輸出） | `HER2_40X`（subifd=False，viewer 副本） |
|---|---|---|
| OpenSlide `level_count` | **1** | **11** |
| 冷開 DeepZoom | 1 ms | 7 ms |
| **OSD 首屏 tile（dz L12，整張 fit 螢幕）** | **8,646 ms／張** | **23 ms／張** |
| 再往外縮一級（dz L10） | **OOM**：單張需從原尺寸讀 65538² ≈ **17 GB** | 17 ms |
| 更外層（整張概覽 L8↓） | **OOM**（投影 275 GB → 4400 GB …） | 0.3–1.9 ms |
| 高倍細節（dz L18） | 16 ms（可用） | 6 ms |

**逐層供給時間對照**：

```
subifd=True (餵 VALIS 用)          每 tile 從 os_lvl0 讀        時間
  dz L18 (原尺寸細節)  258px           16 ms   ← 只有放到最大才順
  dz L16               1026px          66 ms
  dz L14               4098px         605 ms
  dz L12 (首屏)        16386px       8299 ms   ← 一打開就卡 8 秒
  dz L10               65538px       OOM(17GB) ← 再縮就爆
  dz L8↓               …             OOM(275GB+)

subifd=False (viewer 副本)        自動挑對應金字塔層
  dz L18~L2  os_lvl0~10  每 tile 恆讀 ~514px   0.3–23 ms   ← 全程平坦、不爆
```

### 3.1 現場事故紀錄
在瀏覽器用 OSD 直接開 `subifd=True` 的 `HER2_processed`，OSD 為填滿概覽視窗**併發**抓一整排概覽 tile，每張要從原尺寸讀 ~1GB → 瞬間吃爆 RAM → tile server `MemoryError`、`GLib-ERROR failed to allocate`、process exit 255，**整台掛掉**（連同時服務的 subifd=False 檔一起停）。
→ 結論不是「subifd=True 比較慢」，而是「**直接把 subifd=True 檔丟給 viewer 會把後端讀爆當掉**」。

### 3.2 viewer 副本產生成本實測（2026-07-09，走真實對齊輸出）

前面 §3 是 sample 圖的顯示效能；這裡補「產一份 viewer 副本要多少時間/容量」的實測。
對象是 7/9 pipeline 真正產出的對齊圖 `HER2_aligned_lv0.ome.tiff`（`155807 × 133474`，
約 20.8 gigapixel），用 pyvips 轉一份 `subifd=False + tile=True` 金字塔
（`compression=jpeg, Q=85, tile=256`）：

| 指標 | 值 |
|---|---|
| 轉檔時間 | **2.09 分鐘 / 張** |
| 來源大小（subifd=True） | 19.32 GB |
| viewer 副本大小（subifd=False） | **2.03 GB**（約來源 1/10；Q85 ＋大片白底 padding 壓縮率高） |
| OpenSlide `level_count`（來源） | **1**（金字塔讀不到，就是 §3 那個會讀爆的狀態） |
| OpenSlide `level_count`（viewer 副本） | **11**（downsamples 1→1026，正常金字塔） |

三張分層全轉約 **6 分鐘、~6 GB**。結論：**產副本是廉價的一次性後處理**（不是每次看圖都做），
而且它是「讓 viewer 能用」的必要步驟、不是效能負擔——真正拖累/當機的是 subifd=True 那條。

---

## 4. 對 UI 的意義與建議

1. **通則**：UI 要在 OSD 顯示的任何影像，其來源若是 `subifd=True`（module1 BigTIFF、VALIS warped OME-TIFF），**都必須先產一份 `subifd=False` + `tile=True` 的 viewer 副本**再餵 tile server。這不是效能微調，是「能不能用／會不會當機」的門檻。
2. **加 OSD 不會改變此結論**：瓶頸在後端讀圖（OpenSlide 從原尺寸 downsample），OSD 只是發出請求的前端；併發抓 tile 反而讓情況更糟。
3. **R14 疊合 overlay 的來源選擇**：
   - 顯示合併結果 → 用 `Merged_Aligned`（subifd=False，方向對；確認/補成 tiled）。
   - 想分層做透明度疊合（dish／her2 各一層）→ 用 `*_warped_*.ome.tiff`，但那是 subifd=True，需 viewer 副本。
4. **不動演算法那條**：`subifd=True` 是 VALIS 必需，保持原樣；viewer 副本是**額外**產物，不可取代原輸出，否則 VALIS 會壞。

---

## 5. 待決策（Open Decisions）

- **viewer 副本在哪一環生？**
  - (a) UI／後端 post-step：不碰演算法模組，UI 這側偵測到 subifd=True 就轉一份 subifd=False 供 viewer。（目前 spike 採此法。）
  - (b) 演算法模組在原輸出**之外**多吐一份 subifd=False viewer 檔（主輸出仍 subifd=True）。
- **替代路（未驗證）**：把後端讀圖器從 OpenSlide 換成會鑽 SubIFD 的 reader（pyvips／tifffile 自己開金字塔），理論上 subifd=True 也能直接給 viewer 用，省掉副本。成本較高、尚未驗證。
- **`Merged_Aligned` 的 strip vs tile**：待實體檔產出後 probe，確認是否需補存 tiled 版。

---

## 6. 參考（file:line）

- `backend/algorithms/thriple_image_layer/module1_preprocess.py:312-323` — BigTIFF `subifd=True`
- `backend/algorithms/thriple_image_layer/module4_thumbnail.py:62-84` — warped OME-TIFF（`pyramid=True`）
- `backend/algorithms/thriple_image_layer/module4_thumbnail.py:110-116` — `Merged_Aligned`（`pyramid=True`、無 subifd／tile）
- `<env>/valis/slide_io.py:3685-3688` — `subifd = pyramid`；VALIS OME-TIFF 寫檔
- `<env>/valis/slide_io.py:1761,1811,1974-1982` — VALIS 以 `subifd=`、`n-subifds` 回讀金字塔
- `backend/io/pyramid.py` — OpenSlide `DeepZoomGenerator` tile server（受 OpenSlide subifd 限制）
