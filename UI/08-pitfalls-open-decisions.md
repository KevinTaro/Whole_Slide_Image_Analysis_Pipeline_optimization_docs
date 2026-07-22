# 08 · 陷阱與未決事項

> 上一篇：[07 Phase 路線圖](07-phase-roadmap.md)　·　上層：[README](README.md)

## 9 個常見陷阱

| # | 陷阱 | 說明 / 怎麼避 |
|---|---|---|
| 1 | **把 PyQt6 當成我們的框架** | `requirements.txt` 有 `PyQt6 6.10.1`，那是 cellpose / napari 桌面 GUI 相依帶進來的（連 `superqt`/`pyqtgraph`/`QtPy`）。我們做 **pywebview + web**，**不是 Qt app**（[01](01-architecture.md)、[02](02-tech-stack-versions.md)）。 |
| 2 | **讓 numpy array 跨 API 傳** | 演算法回傳 array 給 endpoint、或 endpoint 直接吞 array＝序列化又大又慢。邊界一律「檔案路徑 + JSON」（護欄 7，[04](04-guardrails-red-lines.md)）。 |
| 3 | **記憶體爆掉時亂換 WSI reader** | ~400GB 爆量來自 `m0_stitch` 的**縫合輸出畫布**（輸出端），不是 reader 讀太多。**換 reader 省不了**；解法是 ROI 化 + 停止整片縫合（[06](06-dev-setup.md) 除錯備註）。症狀診斷：先確認是「輸出畫布」還是「讀取」在漲。 |
| 4 | **以為 `config.py` 在 git 裡** | `config.py` 被 `.gitignore` 忽略（root 與 `cell_mask/` 各一條），git 裡只有 `config_example.py`。第一次要 `cp config_example.py config.py`。搬檔 / 對接 Pydantic schema 時記得它不在版控。 |
| 5 | **照 `requirements.txt` 裝套件 / 讀版本** | 它是 drift 的舊快照（`pyvips 3.1.1`、`numpy 2.2.6` 都對不上實際 venv，後者還違反 `numpy<2`）。查版本一律 `uv.lock` / venv（[02](02-tech-stack-versions.md)）。 |
| 6 | **Phase 1 搬檔時「順手」重構** | 搬檔階段只改 import path，函式內容不動（護欄 6）。順手改＝兩個變因混一起，出錯無法二分，且「輸出 bit-identical」驗收失效。 |
| 7 | **viewport 座標與 pixel 座標混用** | 前端只認 DeepZoom viewport 座標，演算法只認 full-res pixel，換算**只在 `api/roi.py`**（護欄 4）。散落各處＝off-by-one / 縮放錯位重災區（[05](05-dataflow-api-contract.md)）。 |
| 8 | **為了方便把本機限制放寬** | 別把 FastAPI 綁 `0.0.0.0`、別讓前端直接讀本機路徑「省事」。影像不得離開本機是**硬性約束**；一放寬就違反臨床資料規範（[01](01-architecture.md) 護欄 2）。 |

---

## 8 個未決事項（需要後續確認）

啟動對應 Phase 前要先跟作者 / 醫師端敲定：

| # | 未決事項 | 影響 |
|---|---|---|
| 1 | 醫師電腦 **OS 分布**（Windows / macOS 各多少） | 決定 PyInstaller / Briefcase target（Phase 5） |
| 2 | 醫師電腦 **有無 GPU** | 決定要不要 CPU-only fallback（Cellpose / torch 目前吃 GPU） |
| 3 | **影像格式範圍**（.svs / .ndpi / .tiff 各多少） | 決定 WSI reader 需支援哪些格式 |
| 4 | 是否需要 **多 case 管理 / 患者清單** | 影響 Phase 3 UI scope |
| 5 | 醫師對 **UI 語言**偏好（繁中 / 英文 / 雙語） | 決定 i18n 要不要進 Phase 3 |
| 6 | **`module3_roi_evaluation.py` 歸類** | 影響 Phase 1 遷移對照（[03](03-directory-structure.md) 未對應清單） |
| 7 | 是否需要 **模型訓練 GUI**（不只推論，還要在 UI 內重訓 / 微調） | 大幅擴張 scope；預設**不做**，需明確要求才排進路線圖 |
| 8 | Python format 工具 **ruff vs black** | 本文檔提議 ruff（lint+format 一把抓）；若團隊已慣用 black 需先統一，避免 format 戰爭 |
