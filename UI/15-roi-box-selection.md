# 15 · ROI 改為框選（實作紀錄）

> **改動日**：2026-07-29
> **依據**：[`hybrid_flow_mockup.html`](hybrid_flow_mockup.html) 畫面 1「切片檢視 · 拖曳框出要分析的範圍」
> **動到的檔**：`frontend/src/components/SlideViewer.tsx`、`frontend/src/App.tsx`、
> `frontend/src/components/HybridPanel.tsx`。**後端、API、schema 一律沒動。**

---

## 1. 改了什麼

| | 之前 | 現在 |
|---|---|---|
| 取範圍的方式 | 按「用目前檢視範圍」，把**整個 viewport** 當 ROI | 在切片上直接操作一個**藍色選取框** |
| 調整 | 只能改下方 x/y/w/h 四個數字 | 拖框身移動、拖右下把手縮放；數字輸入仍保留 |
| 視覺回饋 | 無 | 框外壓暗、框上方顯示 `ROI w×h`、即時 tile 估算 |
| 按鈕 | 「用目前檢視範圍」 | 「框選範圍」／已有框時為「重設選取框」 |

送出的內容沒變：仍是 `roi_x / roi_y / roi_w / roi_h`（**影像像素**）打 `POST /api/hybrid/tile`，
四個一起給或都不給。**沒有任何 API 契約變動**，所以 `schema.ts` 不用重生。

---

## 2. 三個關鍵設計決定

### 2.1 框是 OpenSeadragon overlay，不是蓋在畫布上的 div

`SlideViewer` 用 `viewer.addOverlay()` / `updateOverlay()`，位置以
`imageToViewportRectangle()` 換算。**這是整個功能成立的關鍵**：overlay 錨定在影像座標，
使用者縮放平移時框會黏在同一塊組織上。若改用固定定位的 div，框會停在螢幕原處、
與組織脫節，ROI 就失去意義。

mockup 的協作者說明也寫了「OSD 有現成的框選 plugin / overlay，不要手搓」。

### 2.2 `roi` 狀態上提到 `App`

原本 `roi` 是 `HybridPanel` 的 local state，但 OSD 檢視器在 `App` 裡。框選需要兩者共用
**同一個矩形**：檢視器負責畫與拖、面板負責顯示與送出。所以 `roi` 移到 `App`，兩邊都收
`roi` + `onRoiChange`（`viewRect` 保持原樣，用來產生新框的初始位置）。

切換切片或返回疊合檢視時會清掉 `roi` —— 一個 ROI 只屬於它被畫在上面的那張切片。

### 2.3 新框是目前檢視範圍的 80%，不是 100%

`boxFromView()` 內縮到 80% 並置中。理由有二：框線要看得見才知道可以拖；右下角把手若貼在
畫面邊緣會**壓到 OSD 的導覽縮圖**（`navigatorPosition: 'BOTTOM_RIGHT'`）。
因為是從已被夾制在影像內的 `viewRect` 推導，新框不可能一開始就超出切片。

---

## 3. 拖曳的實作細節

- **單位換算**：螢幕像素 → 影像像素走
  `viewport.pointFromPixel()` → `viewportToImageCoordinates()`，所以任何縮放倍率下
  拖曳距離都正確；用原始 pixel delta 會在縮放後失準。
- **拖曳期間關閉 OSD 平移**（`setMouseNavEnabled(false)`），否則切片會在框底下跟著移動，
  放開時再開回來。
- **夾制到影像範圍**：移動時夾 x/y、縮放時夾 w/h，後端會拒絕超出切片的 ROI。
- **`pointermove`/`pointerup` 掛在 `window` 而非框元素上**：縮放時游標經常跑到框外側
  （框還沒長到那裡），掛在元素上會掉事件；掛 window 也保證拖曳一定會結束。
  另外在 `e.buttons === 0` 時自我復原，避免漏掉 `pointerup` 而卡在拖曳狀態。
- **最小尺寸**：`SlideViewer` 只擋到 1px（防止矩形反轉）。真正的
  「寬高都要 ≥ 1024」規則仍由 `HybridPanel` 判斷並顯示紅字警告 —— 那是 tile 尺寸的知識，
  屬於面板，不屬於檢視器。

---

## 4. 驗證紀錄（2026-07-29，真實 WSI `31232×27648`，前端 Windows / 後端 Linux）

`tsc --noEmit` 與 `npm run lint` 皆通過。瀏覽器實測：

| 操作 | 結果 |
|---|---|
| 按「框選範圍」 | `x=3123 y=2765 w=24986 h=22118`，面板估「約 957 個 tile」 |
| 拖曳框身 | → `x=6246 y=5530`，`w/h` 不變 |
| 往左上拖過頭 | → `x=0 y=0` 夾住，未跑出影像 |
| 拖右下把手 | `24986×22118` → `14019×14369`，tile 估算 957 → 342 |
| 放開後連續 hover | 數值不變（拖曳狀態確實解除） |

---

## 5. 沒有做的事

**「在空白處拖曳畫出新框」沒有實作。** mockup 只示範移動與縮放一個既有的框，沒有畫框手勢。
要加的話必須先決定它如何與 OSD 的拖曳平移共存（模式切換按鈕，或 Shift+拖曳之類的修飾鍵），
那是設計決定而非實作細節，所以留給下一輪。目前建立框的唯一入口是「框選範圍」按鈕。

---

## 6. 護欄檢查（[04](04-guardrails-red-lines.md)）

- 護欄 2（前端不碰檔案系統路徑）：仍然只用 `slide_id` 與像素座標。
- 護欄 5（一次只動單層）：只動 `frontend/`，後端與演算法零改動。
- 座標換算集中在 `SlideViewer`（沿用既有 `onViewportChange` 就在做的同一套 OSD 換算），
  演算法層仍只收影像像素。
