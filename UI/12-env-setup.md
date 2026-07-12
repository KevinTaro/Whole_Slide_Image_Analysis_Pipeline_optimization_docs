# 12 · 環境建置：從零重建 `tsgh311`（Windows / conda）

> **給誰看**：要在自己 Windows 機器上跑 UI（後端 tile server + 前端 + viewer 轉檔腳本）的組員。
> **一句話**：`tsgh311` = **conda-forge 原生層**（Python + openslide/pyvips/nodejs，DLL 由 conda 處理）
> **＋ pip 層**（torch cu121 + cellpose/valis/fastapi 等）。照下面三步可重建。
> 日期：2026-07-12（反推自當時實際運行的環境）
> 跑起來之後怎麼用見 [`11-runbook-teammate.md`](11-runbook-teammate.md)。

> ⚠️ **這是兩套環境中的「Windows conda」那套。** repo 的 `pyproject.toml` / `uv.lock` /
> [`02-tech-stack-versions.md`](02-tech-stack-versions.md) / [`06-dev-setup.md`](06-dev-setup.md)
> 描述的是另一套 **Linux `.venv` + uv**（torch **cu130**、給 RTX 5090）。本篇這套實測是
> **torch 2.5.1+cu121**。**不要在 `tsgh311` 裡 `uv sync`**——它會照 pyproject 拉 cu130，和這裡衝突。

---

## 0. 前置

- 已安裝 **conda / Miniconda**（實測 conda 25.7.0）。
- 顯卡驅動支援 **CUDA 12.1**（torch 走 cu121）。GPU 不同 / 沒 GPU 見 §4。

## 1. 建立 conda 原生層

用 repo 內的 `environment.yml`（一行搞定）：
```powershell
conda env create -f environment.yml
conda activate tsgh311
```

它等同於明確安裝這些（`--from-history` 反推出的原始意圖）：
```powershell
conda create -n tsgh311 -c conda-forge python=3.11 pip `
    openslide=4.0.1 openslide-python=1.4.6 libvips=8.18.4 pyvips=3.1.1 nodejs=26.4.0
```
> **為什麼原生依賴一定走 conda-forge**：openslide / libvips 在 Windows 需要一堆原生 DLL
> （glib、cairo、libtiff、openjpeg、pixman…），conda-forge 會一起裝好；用 pip 在 Windows
> 裝這些會很痛。這是整個 UI 選 conda 而非純 pip 的主因。

## 2. 裝 pip 層（torch cu121 + ML/Web 堆疊）

環境 activate 後：
```powershell
pip install -r requirements-tsgh311.txt
```
`requirements-tsgh311.txt` 是當時環境的精確 freeze，開頭已內建
`--extra-index-url .../whl/cu121`，所以 `torch==2.5.1+cu121` 會正確解出。

## 3. 驗證

```powershell
# 原生層
python -c "import openslide, pyvips; print('openslide', openslide.__version__); print('pyvips', pyvips.__version__)"
# GPU / torch
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"
# 對齊主力
python -c "import valis; print('valis ok')"
# Node（給前端）
node --version    # v26.4.0
npm  --version    # 11.x
```
預期：openslide 4.0.1 / pyvips 3.1.1 / torch 2.5.1+cu121 / cuda 12.1 / avail True。

接著就能照 [`11-runbook-teammate.md`](11-runbook-teammate.md) 起後端與前端。

---

## 4. 陷阱 / 已知事項

- **Node 裝在 env 根目錄、不在 `Scripts\`**：`npm`/`npx` 會報 `'node' is not recognized`，
  要先把 env 根 prepend 進 PATH（runbook §3 有指令）。
- **`VIPS-WARNING ... vips-heif.dll / vips-jxl.dll / vips-poppler.dll 找不到`**：無害。
  那是 pyvips 的 HEIF/JPEG-XL/PDF 選配模組，跟 WSI 的 TIFF 讀寫無關，可忽略。
- **torch 是 cu121，不是 pyproject 的 cu130**：這台機器實測用 cu121。若你的 GPU/驅動需要別的
  CUDA（或要 CPU），改 `requirements-tsgh311.txt` 開頭的 index URL（`cu124` / `cpu`）與 torch pin。
- **別用被移除的 `requirements.txt`**：舊的根 `requirements.txt` 已從版控移除且早已 drift
  （見 [02](02-tech-stack-versions.md) 黑名單）。重建 Windows 環境**只用本篇的
  `environment.yml` + `requirements-tsgh311.txt`**。

---

## 5. 實測組成（反推來源）

| 層 | 來源 | 關鍵套件（實測版本） |
|---|---|---|
| 原生 | conda-forge（99 個） | python 3.11.15、openslide 4.0.1、openslide-python 1.4.6、libvips 8.18.4、pyvips 3.1.1、nodejs 26.4.0 |
| Python | pip（121 個） | torch 2.5.1+cu121、torchvision 0.20.1+cu121、cellpose 4.2.1.1、valis-wsi 1.2.0、fastapi 0.139.0、uvicorn 0.50.2、numpy 1.26.4、scikit-image 0.24.0、pydantic 2.13.4 |

> 反推方法：`conda env export --from-history`（原生層意圖）＋ `conda list`（channel 來源）
> ＋ `pip freeze`（pip 層鎖版）。
