# 06 — 版本與依賴

> **權威優先序**：**venv 實測 ≥ requirements.txt > pyproject.toml**。
> venv 是真正跑起來的環境；requirements/pyproject 已與 venv 漂移，動依賴前以 venv 為準。
> venv 路徑：`/home/sec312/project/tsgh/.venv`（Python **3.11.15**）。

## 精確版本表

| 套件 | **venv 實測（權威）** | requirements.txt | pyproject.toml | 備註 |
| --- | --- | --- | --- | --- |
| torch | **2.11.0+cu130** | 2.10.0+cu130 | （未釘） | cu130 必須，見下 |
| torchvision | **0.26.0+cu130** | 0.25.0+cu130 | （未釘） | 跟 torch 綁 |
| CUDA (torch) | **13.0** | — | — | Blackwell 需要 |
| numpy | **1.26.4** | 2.2.6 | `<2` | ⚠️ 三方衝突 |
| cellpose | **4.0.8** | 4.0.8 | （未釘） | ViT-SAM backbone |
| segment-anything | **1.0** | 1.0 | — | cellpose 依賴 |
| pyvips | **2.2.3** | 3.1.1 | （未釘） | ⚠️ 主版本矛盾 |
| scipy | **1.17.1** | 1.16.3 | （未釘） | — |
| scikit-image | **0.24.0** | 0.25.2 | `>=0.22,<0.25` | ⚠️ 三方衝突 |
| opencv (cv2) | **4.8.1** | 4.9/4.10/4.12 三個 | `<4.9` | ⚠️ requirements 自相矛盾 |
| Pillow | **12.2.0** | 12.0.0 | `>=10` | — |
| segmentation-models-pytorch | **0.5.0** | （未列） | 有列（未釘） | UNet++ |
| joblib | **1.5.3** | 1.5.3 | — | M3 平行 |
| albumentations | （venv 內，unet 前處理） | — | — | — |

> `cellpose` / `segment_anything` import 後無 `__version__` 屬性，版本以 requirements.txt 釘的 `4.0.8` / `1.0` 為準（與前一輪研究一致）。

## GPU / CUDA 相容性（最硬的約束）

- **卡**：NVIDIA **RTX 5090**（32GB）。
- **架構**：**Blackwell，compute capability sm_120**（`torch.cuda.get_device_capability()` → **(12, 0)**）。
- **後果**：
  - Blackwell sm_120 需要**新的 CUDA kernel**，只有 **cu130 build 的 torch** 有；舊 CUDA wheel（cu118/cu121/cu124…）**跑不動這張卡**。
  - 這就是為什麼 venv 是 `torch 2.11.0+cu130`、`torch.version.cuda == 13.0`。
  - **不能為了配合舊套件而降 torch/CUDA** —— 降版 = 這張卡直接不能用 GPU。任何依賴升降都要在「torch 必須 cu130」的框架內解。

## 三方版本衝突清單

以下是 requirements.txt / pyproject.toml **彼此矛盾、或與 venv 不符**之處，動依賴前先認清：

1. **numpy 1 vs 2**：venv `1.26.4`（numpy 1）、pyproject `<2`（要 numpy 1）、但 **requirements 釘 `2.2.6`（numpy 2）**。→ requirements 與現實/pyproject 打架。**跟著 venv 用 numpy 1**（多處 code 假設 numpy 1 行為，且 pyproject 也要 <2）。
2. **scikit-image 0.24 vs 0.25**：venv `0.24.0`、pyproject `>=0.22,<0.25`（上限排除 0.25）、但 **requirements 釘 `0.25.2`**（違反 pyproject 上限）。→ 以 venv 的 `0.24.0` 為準。
3. **pyvips 2.2.3 vs 3.1.1**：venv 是 **2.2.3**，requirements 要 **3.1.1**（主版本差一代）。pyvips 3.x 的 API/binding 行為與 2.x 有差 —— **別照 requirements 升到 3.1.1**，除非驗過 `m0_reader` 的 `new_from_file(access="random")` / `crop` / `gravity` 行為一致。
4. **opencv 自相矛盾**：requirements.txt 同時列 `opencv-contrib-python==4.10.0.84`、`opencv-contrib-python-headless==4.9.0.80`、`opencv-python-headless==4.12.0.88` 三個不同 opencv 發行版與版本 —— 這份 requirements 本身裝不乾淨。venv 實際是單一 `cv2 4.8.1`，pyproject 要 `<4.9`。→ **以 venv 4.8.1 為準**。
5. **torch/torchvision 小版本漂移**：venv `2.11.0/0.26.0` vs requirements `2.10.0/0.25.0`。都 cu130、影響小，但別預期 requirements 能重現 venv。

## 版本升級時的陷阱

- **`torch.cuda.get_device_capability` 相關 workaround 很脆弱**：因為 sm_120 是很新的架構，部分套件（含某些 cellpose/torch 內部路徑）對「未知 compute capability」的處理可能需要 monkeypatch 或特定版本才不報錯。**升 torch/cellpose 時第一件事就是重驗 GPU 真的吃得到**（`torch.cuda.is_available()` + 實跑一次 Cellpose 前向），不要只看 import 過了就當沒事。
- **不要用 requirements.txt 重建環境**：它與 venv 漂移且自相矛盾（numpy 2、pyvips 3、opencv 三胞胎）。要複製環境，**從 venv 反推**（`pip freeze` 現有 venv）比照 requirements.txt 可靠。
- **升任何 GPU 相關套件前**：確認新版有 cu130 / sm_120 支援，否則這張 Blackwell 卡會退回 CPU 或直接崩 —— 這是本專案依賴管理的第一原則。
- **符合專案慣例**：「缺依賴直接報錯、不做 fallback/退化」（專案記憶）—— 別為了相容性偷加 try/except 靜默降級。
