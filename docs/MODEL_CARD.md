# Model Card: RetinaTrust 2.0

## 1. Model Details

### Basic Metadata
* **Model Name:** RetinaTrust 2.0 (DenseNet-121 / Swin-Transformer-Tiny / ResNet-18 Hybrid)
* **Model Type:** Deep Convolutional Neural Network & Shifted-Window Vision Transformer Ensemble
* **Classification Type:** Ordinal Multi-Class Severity Grading (CORAL formulation)
* **Input Modality:** 2D Color Retinal Fundus Photographs (RGB)
* **Framework:** PyTorch 2.14.1+cu126, Torchvision, ONNX Runtime, ReportLab
* **License:** MIT License (Research & Educational Prototype)
* **Model Version:** 2.0.0
* **Date:** October 2026

---

## 2. Intended Use & Clinical Scope

### Intended Use Cases
* **Educational & Research Prototype:** Designed to explore explainable, calibrated, and uncertainty-bounded medical computer vision for diabetic eye screening.
* **Clinical Triage Decision Support:** Designed as a second-reader decision aid to help clinicians prioritize ambiguous cases ($|C(X)| \ge 2$) for comprehensive dilated ophthalmological examination.
* **Audited Screening:** Facilitates automated clinical audit document generation via PDF reports embedding Grad-CAM++ lesion heatmaps and finite-sample prediction bounds.

### Out-of-Scope & Prohibited Uses
* **Autonomous Primary Diagnosis:** The model is strictly **NOT** an FDA/CE-cleared autonomous medical device and must never be used as a standalone diagnostic tool.
* **Non-Fundus Modalities:** The model is not trained on or capable of evaluating Optical Coherence Tomography (OCT), fluorescein angiography (FA), anterior segment photographs, or slit-lamp imagery.
* **Non-DR Retinal Pathology:** The model is not trained to diagnose glaucoma, age-related macular degeneration (AMD), retinal vein occlusion (RVO), or retinitis pigmentosa.

---

## 3. Training & Evaluation Datasets

| Dataset | Sample Count | Role in Pipeline | Patient Population & Setting |
|:---|:---:|:---|:---|
| **APTOS 2019 Blindness Detection** | 3,662 fundus photos | Primary Training (70%), Validation (15%), Test (15%) | Rural clinics across India; heterogeneous camera equipment |
| **IDRiD (Indian Diabetic Retinopathy Image Dataset)** | 516 fundus photos | Independent Out-of-Domain Benchmark Test | Tertiary ophthalmic clinic (Eye Clinic, Nanded, India); high-resolution Kowa camera |

---

## 4. Architectural Innovations

1. **Heterogeneous Dual Backbone:**
   * **DenseNet-121:** Feature map reuse via direct concatenation preserves fine microaneurysm details ($2\text{–}5\text{ px}$).
   * **Swin-Transformer-Tiny:** Shifted-window self-attention models long-range spatial correlations across all four retinal quadrants.
2. **CORAL Ordinal Head:** Formulates severity as $K-1=4$ binary ranking tasks with shared weight vector $W$, ensuring that clinical distance penalties are enforced and Quadratic Weighted Kappa (QWK) is maximized.
3. **Finite-Sample Split Conformal Prediction:** Produces prediction sets $C(X) \subseteq \{0, 1, 2, 3, 4\}$ guaranteeing $P(Y \in C(X)) \ge 1 - \alpha$ for $\alpha \in \{0.05, 0.10\}$.
4. **Higher-Order Grad-CAM++:** Leverages positive 2nd- and 3rd-order partial derivatives with saliency cutoff filtering to prevent multiple microvascular lesion clusters from washing out.
5. **Technical Quality Heuristics:** Rejects corrupt, severely blurred (Laplacian variance $< 100$), or poorly exposed (underexposed mean $< 40$, overexposed mean $> 215$) images prior to inference.

---

## 5. Quantitative Performance Summary

| Metric | In-Domain (APTOS 2019 Val) | Out-of-Domain (IDRiD External Test) |
|:---|:---:|:---:|
| **Quadratic Weighted Kappa (QWK)** | **0.6031** | **0.3336** |
| **Overall Multi-Class Accuracy** | **67.40%** | **38.83%** |
| **Referable DR Sensitivity ($\text{Grade } \ge 2$)** | **74.44%** | **48.44%** |
| **Referable DR Specificity ($\text{Grade } < 2$)** | **88.65%** | **87.18%** |
| **Expected Calibration Error (ECE - Post-Scaling)** | **0.0553** ($T=1.2007$) | **0.0891** |
| **ONNX CPU Inference Latency** | **11.06 ms / image** | **11.06 ms / image** |

---

## 6. Ethical Considerations & Limitations

* **Demographic & Geographic Representation:** Training data was collected primarily from clinical sites in India. Retinal pigmentation varies across global ethnicities, which may influence background choroidal reflectance. Model performance on other ethnic groups must be evaluated before widespread deployment.
* **Class Imbalance in Mild Cases:** Microaneurysm-only cases (Grade 1: Mild NPDR) represent subtle lesions that are easily missed by deep networks, requiring careful conformal set inspection.
* **Domain Generalization Challenge:** Evaluation on the external IDRiD benchmark showed a $34.9\%$ drop in referable DR sensitivity, emphasizing that deep learning models require local hospital calibration and conformal bounds before real-world deployment.
* **Human-in-the-Loop Safeguard:** Any scan resulting in a conformal prediction set of cardinality $\ge 2$ or predictive entropy $> 0.85$ bits is flagged for mandatory human clinical evaluation.
