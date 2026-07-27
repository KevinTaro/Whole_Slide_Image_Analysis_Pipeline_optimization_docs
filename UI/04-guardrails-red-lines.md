# 04 · 護欄與紅線

> 上一篇：[03 目錄結構](03-directory-structure.md)　·　下一篇：[05 資料流與 API 合約](05-dataflow-api-contract.md)　·　上層：[README](README.md)

這頁是**動手前必讀**。整個架構的價值＝「UI 與演算法結構性分離」，以下 7 條護欄就是維持這件事的物理界線。踩線＝架構退化成一坨黏合的 code。

---

## 7 條硬性護欄

### 1. `backend/algorithms/` 與 `backend/io/` **禁止 import 任何 web 框架**
不得出現 `import fastapi` / `from fastapi ...` / `import flask` / `starlette` / `pydantic` 的 request model。
**為什麼**：這條是「物理防火牆」的實作。只要演算法層碰不到 web 框架，它就永遠能被 CLI 獨立呼叫、永遠不會被 UI 綁架。這是 [01](01-architecture.md) 整個架構成立的前提。
**適用範圍**：原 `cell_mask/hybrid/` 的 `m0_reader.py`（chunked/precut WSI 讀取）已於 Phase 1 整包搬入 `backend/algorithms/hybrid/`，介面不變（見 [03](03-directory-structure.md)）。這條護欄一樣套用在它們身上——`backend/io/pyramid.py`（實際落地的 tile server 核心，見 [10](10-viewer-ui-implementation.md)）不 import 任何 web 框架，符合本條；若未來新增其他包裝層，包裝層本身也不得 import web 框架。

### 2. `frontend/` **不直接讀本機檔案路徑**
前端拿到的是 `slide_id` / job id / URL，**不是** `D:\cases\xxx.svs`。要看影像一律透過 `/api/tiles/...`、要看結果透過 endpoint。
**為什麼**：安全性 + 可攜性。前端是 Chromium 內嵌視圖，讓它直接摸檔案系統＝打破沙盒、且路徑在不同醫師電腦不通用。

### 3. `backend/api/` **只做轉譯，不放演算法邏輯**
endpoint 的工作＝(a) 驗證參數（Pydantic）→ (b) 呼叫 `algorithms/` 既有函式 → (c) 回傳結果檔路徑 / metadata。**endpoint 內不得出現 numpy 運算、影像處理、業務邏輯**。
**為什麼**：分工清晰。API 層一旦開始「順手算一下」，演算法就漏進 web 層，護欄 1 形同虛設。

### 4. 座標轉換**只在 `backend/api/roi.py`**
viewport 座標 ↔ 原始像素座標的換算，**唯一**允許發生在 `roi.py`。演算法層只認像素座標，前端只認 viewport 座標。
**為什麼**：單一換算點＝單一除錯點。座標轉換散落各處是 off-by-one / 縮放錯位的溫床（見 [05](05-dataflow-api-contract.md)）。

### 5. 一次 PR **盡量只動單一層**
改前端就別同時改演算法；改 API 就別順手重構 io。
**為什麼**：分層 review。單層 PR 才能讓 reviewer（或 AI）用下面的 checklist 逐條掃，跨層 PR 會讓護欄違規藏在雜訊裡。

### 6. Phase 1 **零重寫（lift-and-shift）**
搬檔階段**只改 import path**，函式內容一個字都不動；不改演算法、不換資料結構、不加 abstraction。
**為什麼**：框架綁架風險。搬檔同時重構＝兩個變因混在一起，出錯無法二分定位，且驗收「輸出 bit-identical」會失效（見 [07](07-phase-roadmap.md)）。

### 7. 演算法的輸入輸出邊界：**檔案路徑 + JSON**
- 演算法**輸入**：檔案路徑 + JSON 參數。
- 演算法**輸出**：檔案路徑（結果檔）+ JSON metadata。
- **禁止**：演算法回傳 numpy array 給 API 層、API 層直接操作 numpy array。
**為什麼**：numpy array 跨 API 序列化又大又痛（見 [08](08-pitfalls-open-decisions.md) 陷阱）。以檔案路徑為邊界，大影像留在 filesystem，HTTP 只傳輕量 metadata。

---

## AI code review checklist

給 coding AI（或人）在**每個 UI/API PR** 上逐行掃過，任一項打勾＝擋下：

- [ ] **1.** `algorithms/` 或 `io/` 內有 `import fastapi` / `flask` / `starlette` / web request model？→ 違反護欄 1
- [ ] **2.** `frontend/` 出現本機絕對路徑、或直接讀檔（非透過 `/api/...`）？→ 違反護欄 2
- [ ] **3.** `api/` 的 endpoint 裡有 numpy 運算 / 影像處理 / 業務邏輯（不只是驗證＋呼叫＋回傳）？→ 違反護欄 3
- [ ] **4.** 座標換算（viewport↔pixel）出現在 `roi.py` 以外的檔案？→ 違反護欄 4
- [ ] **5.** 這個 PR 同時動了兩層以上（前端＋演算法 / API＋io）？→ 違反護欄 5，拆 PR
- [ ] **6.**（Phase 1）搬檔 PR 除了 import path 還改了函式內容？→ 違反護欄 6
- [ ] **7.** 有演算法函式回傳 numpy array 給 API、或 endpoint 直接吞 array？→ 違反護欄 7

> 這些規則會寫進 `CLAUDE.md`，coding AI 在任一層工作時只被餵該層檔案，從源頭降低跨層污染。
