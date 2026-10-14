# Independent Evaluation Report: {dataset_name}

- **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Model Checkpoint:** `{checkpoint_path}`
- **Architecture:** `{arch}`
- **Total Test Samples:** `{len(test_dataset)}`

## Primary Multi-Class Metrics
| Metric | Value |
|:---|:---:|
| **Quadratic Weighted Kappa (QWK)** | **{metrics['qwk']}** |
| **Accuracy** | **{metrics['accuracy']}** |
| **Macro Precision** | {metrics['macro_precision']} |
| **Macro Recall** | {metrics['macro_recall']} |
| **Macro F1-Score** | {metrics['macro_f1']} |

## Per-Class Breakdown
| Grade | Severity Level | Support | Precision | Recall | F1-Score |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | No DR | 34 | 0.4478 | 0.8824 | 0.5941 |
| 1 | Mild DR | 5 | 0.0 | 0.0 | 0.0 |
| 2 | Moderate DR | 32 | 0.3077 | 0.125 | 0.1778 |
| 3 | Severe DR | 19 | 0.2609 | 0.3158 | 0.2857 |
| 4 | Proliferative DR | 13 | 0.0 | 0.0 | 0.0 |

## Exploratory Binary Screening Analysis
- **Definition:** Referable DR (Grade $\ge$ 2) vs Non-Referable DR (Grade < 2)
- **Sensitivity:** 0.4844
- **Specificity:** 0.8718
- **True Positives:** 31 | **False Positives:** 5
- **True Negatives:** 34 | **False Negatives:** 33

## Confusion Matrix
![Confusion Matrix](C:\Users\khush\OneDrive\Desktop\ClariFundus AI\reports\figures\idrid_external_test_set_confusion_matrix.png)
