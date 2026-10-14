# RetinaTrust Dataset Setup Guide

This guide explains how to acquire, configure, and verify datasets for the RetinaTrust research prototype.

---

## 1. Primary Dataset: APTOS 2019 Blindness Detection

### Overview
- **Host:** Kaggle Competitions
- **Link:** [https://www.kaggle.com/competitions/aptos2019-blindness-detection/data](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
- **Content:** 3,662 labeled high-resolution retinal fundus photographs captured across clinical sites in rural India.

### Severity Scale (ICDR Standard)
| Grade | Severity Level | Clinical Description |
|:---:|:---|:---|
| **0** | No DR | Healthy retina, no microaneurysms or lesions |
| **1** | Mild DR | Microaneurysms only |
| **2** | Moderate DR | More than just microaneurysms but less than severe NPDR |
| **3** | Severe DR | Extensive intraretinal hemorrhages, venous beading, or IRMA |
| **4** | Proliferative DR | Neovascularization, preretinal or vitreous hemorrhage |

### File Layout Options
You can either place the extracted files in `data/raw/aptos/` or leave them in your download directory:

**Option A: Standard project directory**
```
data/raw/aptos/
|-- train.csv
|-- train_images/
    |-- 000c1434d8d7.png
    |-- ...
```

**Option B: Existing Downloads directory**
RetinaTrust automatically detects datasets placed in:
`C:\Users\khush\Downloads\aptos2019-blindness-detection\`

---

## 2. Independent Dataset: IDRiD (Indian Diabetic Retinopathy Image Dataset)

### Overview
- **Host:** IEEE Dataport / Grand Challenge
- **Link:** [https://idrid.grand-challenge.org/Data/](https://idrid.grand-challenge.org/Data/)
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Role:** Independent out-of-domain evaluation.

### Structure
```
data/raw/idrid/B. Disease Grading/
|-- 1. Original Images/
|   |-- a. Training Set/ (413 images)
|   |-- b. Testing Set/ (103 images)
|-- 2. Groundtruths/
    |-- a. IDRiD_Disease Grading_Training Labels.csv
    |-- b. IDRiD_Disease Grading_Testing Labels.csv
```
RetinaTrust also searches for IDRiD in:
`C:\Users\khush\Downloads\Indian Diabetic Retinopathy Image Dataset\`

---

## 3. Dataset Verification & Manifest Generation
Run the inspection CLI utility using the virtual environment to check columns, image readability, missing files, and class balance:
```powershell
.\.venv\Scripts\python.exe -m retinatrust.data.inspect_dataset --dataset aptos
.\.venv\Scripts\python.exe -m retinatrust.data.inspect_dataset --dataset idrid
```

This utility automatically:
1. Validates image decodability and detects corrupted bytes or unsupported formats.
2. Extracts image resolutions, channels, and aspect ratios.
3. Produces audit summaries:
   - `artifacts/manifests/aptos_inspection.json`
   - `artifacts/manifests/idrid_inspection.json`
   - `reports/figures/aptos_inspection.md`
   - `reports/figures/idrid_inspection.md`
4. Builds mathematically disjoint, stratified splits stored as permanent manifests:
   - `artifacts/manifests/train_manifest.csv` (2,563 images - 70%)
   - `artifacts/manifests/val_manifest.csv` (549 images - 15%)
   - `artifacts/manifests/test_manifest.csv` (550 images - 15%)
   - `artifacts/manifests/idrid_test_manifest.csv` (103 images - Independent OOD)
