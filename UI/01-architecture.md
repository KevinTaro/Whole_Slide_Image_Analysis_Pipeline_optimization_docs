# 01 · 架構

> 上層：[README](README.md)　·　下一篇：[02 技術棧與版本](02-tech-stack-versions.md)

## 選定方案

**FastAPI（只綁 `127.0.0.1`）+ React Web UI，用 pywebview 打包成桌面 app。**

三層物理分離：**前端（TypeScript）↔ HTTP ↔ 後端（Python）**，中間隔一層 HTTP，演算法完全不知道 web 的存在。

---

## 架構圖

```
doctor-pc（雙擊 .exe）
        │
        ▼
┌─────────────────────────────────────────────┐
│  pywebview native window                     │
│  ┌────────────────────────────────────────┐  │
│  │  Chromium 內嵌視圖                       │  │
│  │  frontend/  (React + OpenSeadragon)     │  │
│  │        ↕  HTTP (127.0.0.1:PORT)         │  │  ← 物理防火牆
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
              │ HTTP
              ▼
┌─────────────────────────────────────────────┐
│  backend/  FastAPI（只綁 127.0.0.1）          │
│  ├─ api/         ← HTTP endpoints（唯一介面） │
│  ├─ algorithms/  ← 你的演算法（純函式，不動）  │
│  └─ io/          ← WSI 讀取、金字塔切片        │
└─────────────────────────────────────────────┘
              │
              ▼
        local filesystem
        （影像 + 結果，不進 git）
```

三層各自的職責：

- **frontend/**：畫面、互動、疊圖、ROI 圈選。純 TypeScript，**看不到 Python**。
- **backend/api/**：HTTP endpoint。**唯一**能同時碰到「HTTP」與「演算法」的地方，只做轉譯（見 [04](04-guardrails-red-lines.md)）。
- **backend/algorithms/ + io/**：既有演算法，純函式。**看不到 HTTP、看不到 fastapi**。

---

## 核心矛盾與解決思路

四條硬性約束彼此拉扯，這個組合是同時滿足它們的解：

| 約束 | 這個組合怎麼滿足 |
|---|---|
| 病理影像**不得離開本機** | FastAPI **只綁 `127.0.0.1`**，不開對外埠；不是 remote server，資料全程在醫師電腦。 |
| 醫師電腦要能**雙擊開啟** | pywebview + PyInstaller 打成單一 `.exe`，不要求裝 Python / node。 |
| **中高互動**（疊 GB 級影像、ROI 圈選、參數微調） | OpenSeadragon 是病理影像業界標準檢視器，處理金字塔影像流暢；ROI 用 Annotorious，不手刻 canvas。 |
| UI 與演算法**結構性分離** | **HTTP 當物理防火牆**：前端是 TS 不可能 `import` 演算法；`algorithms/` 內 `import fastapi` 會被 lint / review 擋下。 |

**關鍵洞見**：前三條要「像 web app 一樣好操作」，第四條要「演算法不被 UI 綁架」。純桌面框架（Qt）靠紀律分離、容易黏合；純 web 違反本機約束。**localhost web + 桌面殼**同時買到 web 的互動生態與桌面的離線部署，而 HTTP 這道邊界讓分離變成物理事實而非靠自律。

---

## 為什麼不選其他方案

| 替代方案 | 為何否決 |
|---|---|
| **PyQt / PySide 純桌面** | UI 與演算法只能靠紀律分離（signal/slot 容易黏合成一坨），AI 訓練資料相對少，金字塔影像瀏覽要自己刻 viewport。 |
| **Electron（Node 後端）** | 要把 Python 演算法重寫成 Node，違反「能不手搓就不搓」。 |
| **Streamlit / Gradio** | ROI 圈選與即時參數互動超出其擅長範圍，客製到中高互動反而變成反向手搓。 |
| **Remote web server** | 直接違反「影像不得離開本機」。 |
| **Tauri（Rust 殼）** | Rust 學習成本高於效益，pywebview 已足夠。 |

> 註：`requirements.txt` 裡出現的 **PyQt6** 是 cellpose / napari 這類桌面 GUI 相依帶進來的，**不是**我們選的框架。別看到 Qt 就以為要做 Qt app。細節見 [02](02-tech-stack-versions.md) 與 [08](08-pitfalls-open-decisions.md)。
