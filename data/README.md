# RetinaTrust Data Management

This directory organizes datasets used for training, validation, and independent testing of the RetinaTrust research prototype.

> **CRITICAL DATA ETHICS & PRIVACY POLICY**
> - Medical photographs and patient annotations must **NEVER** be committed to Git or pushed to any public repository.
> - Raw images are ignored by `.gitignore` by default.
> - This software is an educational research prototype and not certified for clinical diagnostics.

---

## Directory Structure
```
data/
|-- README.md
|-- raw/
|   |-- aptos/
|   |   |-- train.csv
|   |   |-- train_images/
|   |       |-- <id_code>.png (or .jpg)
|   |-- idrid/
|       |-- B. Disease Grading/
|           |-- 1. Original Images/
|           |   |-- a. Training Set/
|           |   |-- b. Testing Set/
|           |-- 2. Groundtruths/
|               |-- a. IDRiD_Disease Grading_Training Labels.csv
|               |-- b. IDRiD_Disease Grading_Testing Labels.csv
|-- processed/
```

---

## Primary Dataset: APTOS 2019 Blindness Detection
- **Source:** Kaggle Competition ([APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data))
- **Scale:** 5-point Diabetic Retinopathy (DR) grading:
  - `0`: No DR
  - `1`: Mild DR
  - `2`: Moderate DR
  - `3`: Severe DR
  - `4`: Proliferative DR
- **Default File Layout:** Place `train.csv` and `train_images/` into `data/raw/aptos/` (or update paths in `configs/datasets.yaml`).

---

## Independent External Dataset: IDRiD
- **Source:** [Indian Diabetic Retinopathy Image Dataset](https://idrid.grand-challenge.org/Data/)
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Role:** Strict external generalization testing to evaluate performance under cross-domain distribution shifts (different sensor, population, and lighting conditions).
- **Default File Layout:** Place the unpacked IDRiD folder under `data/raw/idrid/` (or update paths in `configs/datasets.yaml`).
