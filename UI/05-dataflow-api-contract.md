# 05 · 資料流與 API 合約

> 上一篇：[04 護欄與紅線](04-guardrails-red-lines.md)　·　下一篇：[06 開發環境](06-dev-setup.md)　·　上層：[README](README.md)

## 一次請求的完整路徑

以「醫師按下『跑 pipeline』」為例：

```
① 使用者點按鈕
        │
② frontend  fetch POST /api/pipeline  { slide_id, params:{...} }   ← 只送 JSON，不送檔案
        │  HTTP (127.0.0.1)
        ▼
③ backend/api  endpoint：
        ├─ Pydantic 驗證 params
        ├─ 呼叫 algorithms/ 的既有函式（傳「檔案路徑 + JSON」）   ← 護欄 3/7
        └─ 回 { job_id }（長任務）或 { result_path, metadata }（短任務）
        │
④ algorithms/  純函式跑完 → 把結果**寫成檔案**，回傳「結果檔路徑 + JSON metadata」
        │
⑤ frontend  拿到 result_path / slide_id 後，**不直接讀檔**：              ← 護欄 2
        └─ OpenSeadragon 向 GET /api/tiles/{slide_id}/{level}/{col}_{row}.jpg 逐塊要圖
        │
⑥ backend/api/tiles  用 openslide DeepZoom / pyvips 即時切片回傳 JPEG
```

要點：**大影像永遠留在 backend / filesystem，HTTP 上只跑輕量 JSON 與逐塊 JPEG**。前端從頭到尾沒碰過本機路徑。

---

## 長時間任務（pipeline 一跑數十分鐘）

**不自己刻 WebSocket。** 用 FastAPI 內建能力：

```
POST /api/pipeline        → { job_id }          （立刻回，任務丟 BackgroundTasks）
GET  /api/jobs/{job_id}   → { status, progress, result_path? }   （前端 polling）
```

- **BackgroundTasks + polling** 足以應付「跑很久、偶爾查進度」。TanStack Query 的 `refetchInterval` 直接做輪詢，不手寫 retry。
- **為什麼不用 WebSocket**：要自己管連線生命週期、斷線重連、狀態同步，違反「能不手搓就不搓」。單機單使用者，polling 的成本可忽略。
- **若未來真的要即時進度條** → 升級成 **Server-Sent Events (SSE)**（單向、比 WebSocket 簡單），而不是 WebSocket。

---

## 型別同步流程（backend ↔ frontend 一份真相）

**不手寫兩份型別。** 讓 FastAPI 的 schema 當單一真相，前端型別自動生成：

```
backend/schemas/*.py（Pydantic model）
        │  FastAPI 自動產生
        ▼
/openapi.json（OpenAPI schema）
        │  openapi-typescript 或 orval
        ▼
frontend/src/api/（TypeScript 型別 + typed client）
```

- 改後端 Pydantic model → 重跑 generator → 前端型別立刻跟著變，型別對不上時 `tsc` 直接報錯。
- 這條 pipeline 就是選 **FastAPI** 的最大理由（[02](02-tech-stack-versions.md)）：schema 免費、client 免費。

---

## ROI 座標轉換（唯一換算邊界）

兩套座標系，換算**只在 `backend/api/roi.py`**（護欄 4）：

| 端 | 座標系 | 誰在用 |
|---|---|---|
| 瀏覽器 | **DeepZoom viewport 座標**（0–1 正規化 / tile 相對） | OpenSeadragon + Annotorious |
| 演算法 | **原始影像像素座標**（full-res pixel） | `algorithms/` 只認這個 |

```
① Annotorious 畫 ROI（viewport 座標）
        │ POST /api/roi  { slide_id, viewport_rect }
        ▼
② api/roi.py：viewport → pixel  換算（★ 唯一發生點）
        │ 傳 pixel bbox 給 algorithms
        ▼
③ algorithms 只在該 pixel 區域重跑 → 回結果檔
```

**為什麼死守單一換算點**：viewport↔pixel 牽涉金字塔 level、縮放比、原點對齊，是 off-by-one 與錯位的重災區。集中在 `roi.py`＝出錯只需查一個檔、只需寫一組測試。演算法層永遠不碰 viewport，前端永遠不碰 pixel。
