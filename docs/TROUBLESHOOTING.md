# Comprehensive Windows & Machine Learning Troubleshooting Guide

This guide provides practical solutions to technical challenges encountered when developing, training, and deploying RetinaTrust 2.0 on Windows systems.

---

## 1. Windows PowerShell & Virtual Environment Issues

### Symptom: `streamlit : The term 'streamlit' is not recognized as the name of a cmdlet...`
* **Root Cause:** You ran `streamlit run app.py` from a standard PowerShell prompt where the project's dedicated virtual environment (`.venv`) was not active. Windows searched your global system PATH and could not find the executable.
* **Resolution (Method 1 — 100% Reliable):**
  Direct Windows to execute Streamlit explicitly via the virtual environment binary:
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run app.py
  ```
* **Resolution (Method 2 — Activate the Environment):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  streamlit run app.py
  ```

---

### Symptom: `cannot be loaded because running scripts is disabled on this system` (ExecutionPolicy)
* **Root Cause:** By default, Windows PowerShell restricts running unsigned `.ps1` scripts to protect against malicious scripts.
* **Resolution:**
  Temporarily bypass the execution policy for your active terminal session without changing global system security:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

---

### Symptom: `pip install` installs packages into global Python 3.14 instead of `.venv`
* **Root Cause:** Running `pip install <package>` without an active virtual environment invokes the default system Python interpreter registered in Windows environment variables (e.g. Python 3.14). PyTorch with CUDA 12.6 is installed in the project's Python 3.13 `.venv`.
* **Resolution:** Always invoke `pip` using the virtual environment interpreter:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install <package_name>
  ```

---

## 2. Windows Encoding & ONNX Export Issues

### Symptom: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 0: character maps to <undefined>`
* **Root Cause:** On Windows, the default terminal encoding is `cp1252` (Windows Latin-1). When `torch.onnx.export` runs with TorchDynamo enabled (`dynamo=True`), it prints emoji checkmarks (`\u2705`), which `cp1252` cannot encode, causing a crash.
* **Resolution:**
  1. In Python, use TorchScript-based ONNX export (`dynamo=False`):
     ```python
     torch.onnx.export(
         model, dummy_input, export_path,
         dynamo=False,
         input_names=["input"], output_names=["logits"],
         dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
         opset_version=17,
     )
     ```
  2. In PowerShell, enforce UTF-8 output encoding if needed:
     ```powershell
     $OutputEncoding = [System.Text.Encoding]::UTF8
     [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
     ```

---

## 3. PyTorch & CUDA Hardware Optimization

### Symptom: `CUDA out of memory` during training on 6GB VRAM GPUs (e.g. RTX 3050)
* **Root Cause:** Processing high-resolution fundus images with large batch sizes exceeds available GPU memory.
* **Resolution:**
  1. Reduce the batch size in CLI or `configs/default.yaml`:
     ```powershell
     # Use batch size 8 for Swin-T or 16 for DenseNet-121
     .\.venv\Scripts\python.exe -m retinatrust.training.train --arch swin_t --batch-size 8
     ```
  2. Clear cached allocations before starting a run:
     ```python
     import torch
     torch.cuda.empty_cache()
     ```
  3. Ensure input resolution is standardized to $224 \times 224$ (or $512 \times 512$ with batch size 4).

---

### Symptom: Multi-worker DataLoader freezes or hangs on Windows (`num_workers > 0`)
* **Root Cause:** Windows does not support POSIX `fork()`. It spawns fresh Python processes for DataLoader workers using `spawn`. Under CUDA, spawning child processes can deadlock CUDA initialization contexts.
* **Resolution:**
  Keep `num_workers: 0` in `configs/default.yaml` on Windows. All data preprocessing will run synchronously on the main process without deadlocks.

---

## 4. Conformal Prediction & Numerical Clamping

### Symptom: Quantile calculation returns values greater than 1.0 (`1.0000000000000002`)
* **Root Cause:** In finite-sample quantile estimation with Dirichlet sampling or high coverage guarantees ($1 - \alpha = 0.95$), floating-point arithmetic can slightly exceed $1.0$. If non-conformity scores are bounded in $[0, 1]$, this causes mathematical inconsistency.
* **Resolution:**
  Clamp the computed quantile index and target score to $1.0$:
  ```python
  q_val = float(np.quantile(cal_scores, min(1.0, q_level), method="higher"))
  q_val = min(1.0, max(0.0, q_val))
  ```

---

## 5. Web UI & Sample Image Loading

### Symptom: Sample image selection does not load or disappears after upload
* **Root Cause:** Prior to the RetinaTrust 2.0 update, file uploads took strict precedence over sample selection in the condition tree.
* **Resolution:**
  Use the dedicated **`🧪 Curated Validation Samples (Grades 0–4)`** tab in `app.py`. Click the prominent **`[ 📥 Load & Analyze Selected Sample ]`** button. The session state stores the active image, displays the thumbnail preview, and shows the ground-truth concordance card.
