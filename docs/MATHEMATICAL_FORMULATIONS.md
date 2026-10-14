# Mathematical & Algorithmic Formulations

This document provides rigorous mathematical derivations and algorithmic definitions for all methods implemented in RetinaTrust 2.0.

---

## 1. Ophthalmic Preprocessing Pipeline

### 1.1 Ben Graham Illumination Normalization
In clinical datasets, varying lighting and camera vignetting can cause standard CNNs to learn spurious camera artifacts rather than true microvascular pathology.

Let $I(x, y) \in [0, 255]^3$ denote the input 3-channel RGB image. We construct a 2D Gaussian smoothing filter:
$$\mathcal{G}_\sigma(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
where $\sigma = 10$ pixels. The normalized image $I_{\text{norm}}(x, y)$ is computed via local Gaussian background subtraction:
$$I_{\text{norm}}(x, y) = \alpha \cdot I(x, y) - \beta \cdot (\mathcal{G}_\sigma * I)(x, y) + \gamma$$
with standard hyperparameters $\alpha = 4$, $\beta = 4$, and offset $\gamma = 128$.

Finally, to prevent circular border halos from polluting the feature space, a binary circular mask $M(x, y) \in \{0, 1\}$ is applied:
$$I_{\text{final}}(x, y) = I_{\text{norm}}(x, y) \cdot M(x, y) + 128 \cdot (1 - M(x, y))$$
guaranteeing a neutral gray background outside the active retinal perimeter.

---

## 2. CORAL Ordinal Classification Mechanics

### 2.1 The Ordinal Problem Formulation
In nominal classification with $K$ classes (e.g. standard Softmax), the label space $\mathcal{Y} = \{0, 1, \dots, K-1\}$ is treated as an unordered categorical set. In Diabetic Retinopathy, the classes represent a strictly ordered progression:
$$\text{Grade 0} \prec \text{Grade 1} \prec \text{Grade 2} \prec \text{Grade 3} \prec \text{Grade 4}$$

### 2.2 Consistent Rank Logits (CORAL)
Let $\phi(X) \in \mathbb{R}^d$ be the feature embedding produced by the backbone network for input image $X$. We decompose the $K$-class problem into $K-1 = 4$ binary classification tasks:
$$y_i^{(k)} = \mathbb{I}(y_i > k) \quad \text{for } k \in \{0, 1, 2, 3\}$$

The model outputs $K-1$ linear logits using a single shared weight vector $W \in \mathbb{R}^d$ and independent task-specific bias terms $\{b_k\}_{k=0}^{K-2}$:
$$g_k(X) = W^T \phi(X) + b_k \quad \text{for } k \in \{0, 1, 2, 3\}$$

The cumulative binary probability of sample $X$ exceeding threshold $k$ is given by the logistic sigmoid:
$$\hat{P}(Y > k \mid X) = \sigma(g_k(X)) = \frac{1}{1 + \exp\left(-(W^T \phi(X) + b_k)\right)}$$

### 2.3 CORAL Loss Function
The CORAL loss $\mathcal{L}_{\text{CORAL}}$ is the sum of binary cross-entropy losses across all $K-1$ tasks:
$$\mathcal{L}_{\text{CORAL}}(X_i, y_i) = -\sum_{k=0}^{K-2} \left[ y_i^{(k)} \log \sigma(g_k(X_i)) + (1 - y_i^{(k)}) \log (1 - \sigma(g_k(X_i))) \right]$$
where $y_i^{(k)} \in \{0, 1\}$.

### 2.4 Conversion to 5-Class Probability Simplex
To compute class-level probabilities $P(Y = k \mid X)$, we take successive differences between the cumulative probabilities:
$$P(Y = 0 \mid X) = 1 - \hat{P}(Y > 0 \mid X)$$
$$P(Y = 1 \mid X) = \hat{P}(Y > 0 \mid X) - \hat{P}(Y > 1 \mid X)$$
$$P(Y = 2 \mid X) = \hat{P}(Y > 1 \mid X) - \hat{P}(Y > 2 \mid X)$$
$$P(Y = 3 \mid X) = \hat{P}(Y > 2 \mid X) - \hat{P}(Y > 3 \mid X)$$
$$P(Y = 4 \mid X) = \hat{P}(Y > 3 \mid X)$$

Because the weight vector $W$ is shared across all tasks and the model learns strictly decreasing biases $b_0 > b_1 > b_2 > b_3$, the cumulative probabilities are monotonic:
$$\hat{P}(Y > 0) \ge \hat{P}(Y > 1) \ge \hat{P}(Y > 2) \ge \hat{P}(Y > 3)$$
guaranteeing non-negative probabilities $\sum_{k=0}^4 P(Y = k) = 1$.

The predicted grade is the count of binary tasks whose probability exceeds $0.5$:
$$\hat{y} = \sum_{k=0}^{K-2} \mathbb{I}\left(\hat{P}(Y > k \mid X) > 0.5\right)$$

---

## 3. Post-Hoc Temperature Scaling & Calibration

### 3.1 Temperature Scaling
Modern deep neural networks tend to produce overconfident predictions. To calibrate the raw model logits $z_i \in \mathbb{R}^K$, we introduce a single scalar parameter $T > 0$ (the Temperature):
$$\hat{p}_k(X; T) = \frac{\exp(z_k / T)}{\sum_{j=0}^{K-1} \exp(z_j / T)}$$

The optimal temperature $T^*$ is fitted by minimizing the Negative Log-Likelihood (NLL) on a held-out validation dataset $\mathcal{D}_{\text{val}} = \{(X_i, y_i)\}_{i=1}^{N_{\text{val}}}$ while keeping network weights frozen:
$$T^* = \arg\min_{T > 0} \left( -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \hat{p}_{y_i}(X_i; T) \right)$$

### 3.2 Expected Calibration Error (ECE)
To quantify calibration error before and after scaling, the interval $[0, 1]$ is partitioned into $M = 15$ equally spaced confidence bins $B_m = \left( \frac{m-1}{M}, \frac{m}{M} \right]$.

The ECE is computed as the confidence-weighted difference between empirical accuracy and mean confidence:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i), \qquad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_{\hat{y}_i}(X_i)$$

---

## 4. Split Conformal Prediction & Clinical Safety Sets

### 4.1 Non-Conformity Scores
Let $\mathcal{D}_{\text{cal}} = \{(X_i, y_i)\}_{i=1}^n$ be an independent calibration set exchangeable with future test samples. For each sample, we define the non-conformity score $s_i$ as the model's error margin on the true label:
$$s_i = 1 - \hat{P}(Y = y_i \mid X_i)$$
where high $s_i \approx 1$ signifies the model assigned low probability to the true diagnosis.

### 4.2 Finite-Sample Coverage Guarantee
For a target error rate $\alpha \in (0, 1)$ (e.g., $\alpha = 0.05$ for $95\%$ coverage), we compute the empirical quantile $\hat{q}$ of calibration scores:
$$\hat{q} = \text{Quantile}_{1-\alpha}\left(\{s_i\}_{i=1}^n, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

For any new unseen test image $X_{\text{test}}$, the conformal prediction set $C(X_{\text{test}})$ is constructed as:
$$C(X_{\text{test}}) = \left\{ y \in \{0, 1, 2, 3, 4\} : 1 - \hat{P}(Y = y \mid X_{\text{test}}) \le \hat{q} \right\}$$

By the exchangeability of $(X_1, Y_1), \dots, (X_n, Y_n), (X_{\text{test}}, Y_{\text{test}})$, Vovk's Conformal Prediction theorem proves:
$$P\left(Y_{\text{test}} \in C(X_{\text{test}})\right) \ge 1 - \alpha$$
This holds **exactly for any sample size $n$**, without making parametric assumptions on data distribution.

---

## 5. Grad-CAM++ Explainability Formulations

### 5.1 Higher-Order Gradient Formulation
Standard Grad-CAM computes a uniform spatial average of first-order gradients $\frac{\partial Y^c}{\partial A_{i,j}^k}$, which washes out multiple disparate lesion clusters. Grad-CAM++ resolves this by weighting activations using positive higher-order partial derivatives.

Let $A^k$ denote the $k$-th feature map of the target convolutional or transformer layer, and let $Y^c$ denote the class score for target class $c$. The feature importance weight $\alpha_k^c$ is given by:
$$\alpha_k^c = \sum_{i=1}^H \sum_{j=1}^W \omega_{i,j}^{k,c} \cdot \text{ReLU}\left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)$$
where the weighting coefficients $\omega_{i,j}^{k,c}$ are derived via 2nd- and 3rd-order partial derivatives:
$$\omega_{i,j}^{k,c} = \frac{\frac{\partial^2 Y^c}{(\partial A_{i,j}^k)^2}}{2 \frac{\partial^2 Y^c}{(\partial A_{i,j}^k)^2} + \sum_{a=1}^H \sum_{b=1}^W A_{a,b}^k \frac{\partial^3 Y^c}{(\partial A_{a,b}^k)^3}}$$

The final saliency map $L_{\text{Grad-CAM++}}^c$ is computed via linear combination followed by ReLU rectification:
$$L_{\text{Grad-CAM++}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### 5.2 Saliency Thresholding
To filter out low-intensity background noise and focus the clinician's attention strictly on primary lesion clusters, we apply thresholding at cutoff $\tau \in [0, 0.8]$:
$$\tilde{L}^c(x, y) = \begin{cases} L^c(x, y) & \text{if } L^c(x, y) \ge \tau \cdot \max(L^c) \\ 0 & \text{otherwise} \end{cases}$$

---

## 6. Quadratic Weighted Kappa (QWK) Metric

Quadratic Weighted Kappa measures agreement between two raters (the model's prediction $\hat{y}$ vs. ground-truth $y$), penalizing distant misclassifications quadratically.

Let $O_{i,j}$ be the observed $K \times K$ confusion matrix, and let $E_{i,j}$ be the expected confusion matrix under independent assignment:
$$E_{i,j} = \frac{\sum_{k} O_{i,k} \cdot \sum_{k} O_{k,j}}{N}$$

The quadratic penalty weight matrix $w_{i,j}$ is defined as:
$$w_{i,j} = \frac{(i - j)^2}{(K - 1)^2}$$

The Quadratic Weighted Kappa $\kappa_w$ is calculated as:
$$\kappa_w = 1 - \frac{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} w_{i,j} O_{i,j}}{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} w_{i,j} E_{i,j}}$$
* $\kappa_w = 1.0$: Complete agreement.
* $\kappa_w = 0.0$: Agreement equivalent to chance.
* $\kappa_w < 0.0$: Agreement worse than chance.
