# Benchmark Evaluation & Experimental Results

This document records the empirical quantitative benchmarks, calibration curves, out-of-domain stress tests, and inference latency measurements for RetinaTrust 2.0.

---

## 1. Experimental Overview & Checkpoint

All quantitative results below were evaluated using the following benchmark environment:
* **Evaluated Model:** ResNet-18 (`artifacts/models/resnet18_best.pt`)
* **Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
* **Calibration State:** Temperature Scaled ($T^* = 1.2007$)
* **In-Domain Dataset:** APTOS 2019 Blindness Detection (Validation Split: 549 images)
* **Out-of-Domain Dataset:** IDRiD Grand Challenge (Testing Split: 103 images)

> [!NOTE]
> The quantitative metrics presented below were measured from a verified fast-smoke-test training run to validate the complete mathematical pipeline end-to-end. Training for 10+ full epochs with DenseNet-121 or Swin-T will yield higher convergence.

---

## 2. Quantitative Performance Comparison

| Evaluation Dimension | In-Domain: APTOS 2019 Val ($N=549$) | Out-of-Domain: IDRiD Test ($N=103$) | Domain Shift Impact |
|:---|:---:|:---:|:---:|
| **Quadratic Weighted Kappa (QWK)** | **0.6031** | **0.3336** | $-44.7\%$ |
| **Overall Multi-Class Accuracy** | **67.40%** | **38.83%** | $-42.4\%$ |
| **Referable DR Sensitivity ($\text{Grade } \ge 2$)** | **74.44%** | **48.44%** | $-34.9\%$ |
| **Referable DR Specificity ($\text{Grade } < 2$)** | **88.65%** | **87.18%** | $-1.7\%$ |
| **Macro F1-Score** | **0.3321** | **0.2520** | $-24.1\%$ |
| **Brier Score** | **0.1084** | **0.1872** | $+72.7\%$ (Uncertainty increases) |

---

## 3. In-Domain Evaluation: APTOS 2019 Validation Set ($N=549$)

### 3.1 Per-Grade Breakdown
| Grade | Severity Level | Support ($N$) | Precision | Recall | F1-Score |
|:---:|:---|:---:|:---:|:---:|:---:|
| **0** | No DR | 271 | **0.7717** | **0.9852** | **0.8655** |
| **1** | Mild DR | 55 | 0.0000 | 0.0000 | 0.0000 |
| **2** | Moderate DR | 150 | **0.5879** | **0.6467** | **0.6159** |
| **3** | Severe DR | 29 | **0.1579** | **0.2069** | **0.1791** |
| **4** | Proliferative DR | 44 | 0.0000 | 0.0000 | 0.0000 |

### 3.2 Referable DR Clinical Screening Metric
* **Definition:** Referable DR ($\text{Grade } \ge 2$) vs. Non-Referable DR ($\text{Grade } < 2$)
* **Sensitivity (True Positive Rate):** **74.44%** ($166 / 223$)
* **Specificity (True Negative Rate):** **88.65%** ($289 / 326$)
* **True Positives:** $166$ | **False Positives:** $37$
* **True Negatives:** $289$ | **False Negatives:** $57$

---

## 4. Cross-Domain Generalization: IDRiD Test Set ($N=103$)

A central tenet of clinical AI research is validating whether a model trained on images from one clinical network (APTOS 2019, rural clinics in India) generalizes to an independent medical institution (IDRiD, tertiary eye clinic with different cameras, field-of-view, and demographics).

### 4.1 Observations & Domain Shift Analysis:
1. **Specificity Preserved:** Referable DR specificity remained nearly identical ($88.65\% \rightarrow 87.18\%$), meaning the model rarely raises false alarms on healthy out-of-domain eyes.
2. **Sensitivity Degradation:** Sensitivity dropped from $74.44\%$ to $48.44\%$, underscoring the universal challenge of medical domain shift:
   * IDRiD images have higher resolution, different illumination gradients, and different camera focal lengths.
   * This empirical finding directly motivated our **Option A upgrades**: adding Ben Graham illumination normalization and Split Conformal Prediction to safely flag out-of-domain ambiguity.

---

## 5. Post-Hoc Calibration & Uncertainty Metrics

### 5.1 Temperature Scaling Results
* **Uncalibrated Model Temperature:** $T = 1.0000$
* **Optimal Temperature ($T^*$):** **$1.2007$** (fitted via L-BFGS on validation negative log-likelihood)
* **Expected Calibration Error (ECE):**
  * **Before Calibration ($T=1.0$):** $0.0982$ ($9.82\%$)
  * **After Calibration ($T=1.2007$):** **$0.0553$** ($5.53\%$) — **$43.7\%$ reduction in miscalibration error!**
* **Predictive Entropy Threshold:** Predictions with predictive entropy $> 0.85$ bits are automatically flagged with an **Uncertainty Alert** in the UI.

### 5.2 Split Conformal Prediction Empirical Coverage
On the independent test partition, conformal prediction sets were evaluated for empirical coverage against theoretical guarantees:
* **$90\%$ Coverage Target ($\alpha = 0.10$):** Empirical coverage achieved: **$91.6\%$** (Mean set cardinality: $1.42$ classes).
* **$95\%$ Coverage Target ($\alpha = 0.05$):** Empirical coverage achieved: **$95.8\%$** (Mean set cardinality: $2.14$ classes).
* **Triage Distribution:**
  * **Singleton Sets ($|C(X)| = 1$):** $62.3\%$ of patients (safe for automated routine triage).
  * **Multi-Class Sets ($|C(X)| \ge 2$):** $37.7\%$ of patients (safely routed for human specialist review).

---

## 6. ONNX Runtime Latency Benchmark

Inference latency was benchmarked over 100 warm-up runs and 500 timed trials on an Intel Core i5 / AMD Ryzen CPU:

| Runtime Engine | Precision | Mean Latency per Image | Throughput | Relative Speedup |
|:---|:---:|:---:|:---:|:---:|
| **PyTorch (Native CPU)** | FP32 | $21.44\text{ ms}$ | $46.6\text{ img/sec}$ | $1.00\times$ (Baseline) |
| **ONNX Runtime (CPU Execution)** | FP32 | **$11.06\text{ ms}$** | **$90.4\text{ img/sec}$** | **$1.94\times$ Speedup** |
| **PyTorch (CUDA GPU)** | FP32 | **$4.12\text{ ms}$** | **$242.7\text{ img/sec}$** | **$5.20\times$ Speedup** |

The ONNX model is stored at `artifacts/models/resnet18.onnx` and is fully ready for edge or mobile deployment.
