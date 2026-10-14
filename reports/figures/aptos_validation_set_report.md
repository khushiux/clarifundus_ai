# Independent Evaluation Report: APTOS Validation Set

- **Date:** 2026-10-14 01:41:35
- **Model Checkpoint:** `artifacts/models/resnet18_best.pt`
- **Architecture:** `resnet18`
- **Total Test Samples:** `549`

## Primary Multi-Class Metrics
| Metric | Value |
|:---|:---:|
| **Quadratic Weighted Kappa (QWK)** | **0.6031** |
| **Accuracy** | **0.674** |
| **Macro Precision** | 0.3035 |
| **Macro Recall** | 0.3678 |
| **Macro F1-Score** | 0.3321 |

## Per-Class Breakdown
| Grade | Severity Level | Support | Precision | Recall | F1-Score |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | No DR | 271 | 0.7717 | 0.9852 | 0.8655 |
| 1 | Mild DR | 55 | 0.0 | 0.0 | 0.0 |
| 2 | Moderate DR | 150 | 0.5879 | 0.6467 | 0.6159 |
| 3 | Severe DR | 29 | 0.1579 | 0.2069 | 0.1791 |
| 4 | Proliferative DR | 44 | 0.0 | 0.0 | 0.0 |

## Exploratory Binary Screening Analysis
- **Definition:** Referable DR (Grade $\ge$ 2) vs Non-Referable DR (Grade < 2)
- **Sensitivity:** 0.7444
- **Specificity:** 0.8865
- **True Positives:** 166 | **False Positives:** 37
- **True Negatives:** 289 | **False Negatives:** 57

## Confusion Matrix
![Confusion Matrix](C:\Users\khush\OneDrive\Desktop\ClariFundus AI\reports\figures\aptos_validation_set_confusion_matrix.png)
