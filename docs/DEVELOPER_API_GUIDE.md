# Developer & API Integration Guide

This guide covers developer workflows, programmatic Python SDK usage, REST API endpoints, CLI tools, and Docker deployment for RetinaTrust 2.0.

---

## 1. Quick Start: Virtual Environment & Execution

Always ensure commands are executed using the project's dedicated virtual environment (`.venv`) on Windows:

```powershell
# Option A: Run directly via virtual environment python (Recommended)
.\.venv\Scripts\python.exe -m streamlit run app.py

# Option B: Activate PowerShell environment first
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

---

## 2. Python SDK: Programmatic Inference

You can embed the `RetinaPredictor` directly into your Python scripts or clinical pipelines:

```python
from pathlib import Path
from PIL import Image
from retinatrust.inference.predictor import RetinaPredictor

# 1. Initialize predictor with trained checkpoint
predictor = RetinaPredictor(checkpoint_path="artifacts/models/resnet18_best.pt")

# 2. Load retinal fundus photograph
image = Image.open("tests/fixtures/sample_fundus.png").convert("RGB")

# 3. Execute multi-signal inference pipeline
result = predictor.predict(
    image_input=image,
    generate_cam=True,            # Produce Grad-CAM++ saliency heatmap
    cam_threshold=0.15,           # Filter background noise
    apply_ben_graham=True,        # Enable illumination normalization
    return_conformal=True,        # Compute 90% and 95% safety sets
)

# 4. Access structured diagnostic outputs
print(f"Predicted Grade: {result.predicted_grade} ({result.grade_label})")
print(f"Confidence: {result.confidence:.2%}")
print(f"Predictive Entropy: {result.predictive_entropy:.3f} bits")
print(f"Conformal 95% Set: {result.conformal_result.prediction_set_95}")
print(f"Triage Decision: {result.conformal_result.triage_recommendation}")

# 5. Generate and save clinical audit PDF report
pdf_bytes = result.generate_pdf(scan_id="SCAN-2026-001")
with open("screening_audit.pdf", "wb") as f:
    f.write(pdf_bytes)
print("Saved screening_audit.pdf successfully.")
```

---

## 3. FastAPI REST Microservice

RetinaTrust 2.0 provides an asynchronous, typed REST API built with FastAPI and Uvicorn.

### 3.1 Starting the API Service
```powershell
.\.venv\Scripts\python.exe -m uvicorn retinatrust.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is automatically available at:
`http://localhost:8000/docs`

---

### 3.2 Endpoint: `GET /health`
Verifies server health, loaded model architecture, PyTorch device, and calibration status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "architecture": "resnet18",
  "device": "cuda:0",
  "temperature": 1.2007,
  "service_version": "2.0.0"
}
```

---

### 3.3 Endpoint: `POST /predict`
Uploads a fundus photograph and returns the multi-signal clinical screening result.

**cURL Request:**
```bash
curl -X POST "http://localhost:8000/predict?generate_cam=true&cam_threshold=0.15&apply_ben_graham=true" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@retina_scan.jpg;type=image/jpeg"
```

**JSON Response (200 OK):**
```json
{
  "predicted_grade": 2,
  "grade_label": "Moderate DR",
  "grade_description": "More than microaneurysms but less than severe NPDR; cotton wool spots, exudates.",
  "confidence": 0.6842,
  "predictive_entropy": 0.7421,
  "class_probabilities": {
    "0": 0.0812,
    "1": 0.1231,
    "2": 0.6842,
    "3": 0.0914,
    "4": 0.0201
  },
  "is_uncertain": false,
  "uncertainty_reasons": [],
  "quality_warnings": [],
  "conformal_prediction": {
    "prediction_set_90": [2],
    "prediction_set_95": [1, 2],
    "set_size_90": 1,
    "set_size_95": 2,
    "triage_recommendation": "Specialist Review Required (Multi-grade diagnostic ambiguity)",
    "coverage_guarantee_90": 0.90,
    "coverage_guarantee_95": 0.95
  },
  "gradcam_available": true,
  "gradcam_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "ben_graham_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "inference_latency_ms": 38.4
}
```

---

### 3.4 Endpoint: `POST /predict/report`
Uploads a fundus photograph and returns a downloadable, binary Clinical Audit Report in PDF format.

**cURL Request:**
```bash
curl -X POST "http://localhost:8000/predict/report?patient_id=PT-94812" \
     -H "accept: application/pdf" \
     -F "file=@retina_scan.jpg;type=image/jpeg" \
     --output screening_report.pdf
```

---

## 4. Command Line Interface (CLI) Utilities

RetinaTrust 2.0 provides modular CLI utilities for every stage of development and auditing:

### 4.1 Dataset Inspection & Integrity Auditing
Audits image file integrity, verifies labels, checks resolution ranges, and computes class balance:
```powershell
.\.venv\Scripts\python.exe -m retinatrust.data.inspect_dataset --dataset aptos
.\.venv\Scripts\python.exe -m retinatrust.data.inspect_dataset --dataset idrid
```

### 4.2 Model Training
Train any supported architecture with CORAL loss and early stopping:
```powershell
# Fast smoke-test (verifies pipeline end-to-end on GPU in < 60 seconds)
.\.venv\Scripts\python.exe -m retinatrust.training.train --smoke-test

# Train DenseNet-121 with CORAL loss
.\.venv\Scripts\python.exe -m retinatrust.training.train --arch densenet121 --epochs 10 --batch-size 16

# Train Swin-Transformer-Tiny
.\.venv\Scripts\python.exe -m retinatrust.training.train --arch swin_t --epochs 10 --batch-size 8
```

### 4.3 Independent Evaluation & Metric Auditing
Evaluates checkpoints on in-domain APTOS validation or out-of-domain IDRiD test sets:
```powershell
# Evaluate on in-domain APTOS validation set
.\.venv\Scripts\python.exe -m retinatrust.training.evaluate --checkpoint artifacts/models/resnet18_best.pt --split val

# Evaluate on out-of-domain IDRiD external test benchmark
.\.venv\Scripts\python.exe -m retinatrust.training.evaluate --checkpoint artifacts/models/resnet18_best.pt --external-test
```

### 4.4 ONNX Model Export & Benchmark
Exports PyTorch weights to an optimized ONNX computational graph and runs latency benchmarks:
```powershell
.\.venv\Scripts\python.exe -m retinatrust.inference.onnx_export --checkpoint artifacts/models/resnet18_best.pt --output artifacts/models/resnet18.onnx
```

---

## 5. Docker Containerization

RetinaTrust 2.0 is fully containerized for deployment on Linux, Kubernetes, or cloud infrastructure.

### Build and Run with Docker:
```bash
# Build the Docker image
docker build -t retinatrust:2.0 .

# Run the container (maps FastAPI port 8000 and Streamlit port 8501)
docker run -p 8000:8000 -p 8501:8501 --name retinatrust_app retinatrust:2.0
```

### Run with Docker Compose:
```bash
docker compose up -d
```
Access points:
* **Streamlit UI:** `http://localhost:8501`
* **FastAPI Docs:** `http://localhost:8000/docs`

---

## 6. Running the Automated Test Suite

RetinaTrust maintains an exhaustive test suite with $100\%$ pass rate:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
To run specific test categories:
```powershell
# Test CORAL ordinal loss
.\.venv\Scripts\python.exe -m pytest tests/test_coral.py -v

# Test Conformal Prediction coverage guarantees
.\.venv\Scripts\python.exe -m pytest tests/test_conformal.py -v

# Test Grad-CAM++ explainability
.\.venv\Scripts\python.exe -m pytest tests/test_gradcam_pp.py -v

# Test PDF report generation
.\.venv\Scripts\python.exe -m pytest tests/test_report_generator.py -v
```
