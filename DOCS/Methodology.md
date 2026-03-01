# Thesis Methodology & Initial Plan

## 1. Methodology Overview

This project aims to evaluate the diagnostic utility of EEG recordings under two distinct conditions—resting-state and photic stimulation—for neurodegenerative disease classification (Alzheimer’s Disease/AD, Frontotemporal Dementia/FTD, and Healthy Controls/HC).

### 1.1 Data Preparation & Preprocessing

**Datasets**:

- **Dataset A (Resting State)**: OpenNeuro ds004504.
- **Dataset B (Photic Stimulation)**: OpenNeuro ds006036.

**Preprocessing Pipeline**:

1. **Filtering**: Apply a bandpass filter (e.g., 0.5–45 Hz) to remove DC offset, slow drifts, and high-frequency noise/line noise (50/60 Hz).
2. **Resampling**: Downsample to a common sampling rate (e.g., 100 Hz or 128 Hz) to reduce computational load while preserving relevant frequencies.
3. **Artifact Handling**:
   - *Basic*: Amplitude thresholding (reject epochs > ±150 µV).
   - *Advanced*: Independent Component Analysis (ICA) to remove eye blinks/muscle artifacts (optional, depending on compute resources and importance to the topic).
4. **Segmentation**:
   - Instead of a single 30s crop, segment recordings into multiple non-overlapping windows (e.g., 2-5 seconds). This increases the number of training samples and allows for "voting" strategies during inference (e.g., majority vote of all windows for a subject).
5. **Normalization**: Z-score normalization (mean=0, std=1) per trial/window to handle inter-subject variability in amplitude.

### 1.2 Experimental Design (Addressing the 3 Questions)

**RQ1: Which task condition yields better diagnostic performance?**

- Train independent classifiers for **Resting** and **Photic** datasets.
- Compare performance metrics (Accuracy, F1-score, AUC) on hold-out test sets using stratified 5-fold cross-validation.
- *Statistical Test*: Paired t-test or Wilcoxon signed-rank test on fold accuracies.

**RQ2: Does joint training improve generalization?**

- **Experiment**: Combine both datasets into a single "Super-Dataset".
- **Split**: Ensure subject independence (a subject’s resting and photic data must both be in either train OR test, never split across both).
- Compare the "Joint Model" performance on the test set against the single-task models.

**RQ3: Do the two conditions encode complementary information?**

- **Late Fusion (Ensemble)**: Train separate models for Resting and Photic. For a test subject, get probabilities from both models and average them (Soft Voting).
- **Comparison**: Does Fusion outperform the single best modality (Resting or Photic)?

### 1.3 Modeling Strategy

**Primary Model: Detach-ROCKET**

- State-of-the-art for time-series classification.
- Fast training, handles multivariate data well.
- *Hyperparameters*: 10,000 kernels, ridge regression classifier.

**Secondary Model: Deep Learning (DL)**

- **InceptionTime**: A strong baseline for TSP, consisting of ensembles of Inception-like CNN modules.
- **EEGNet**: A compact CNN specifically designed for EEG, good for interpretability (spatial/temporal filters).
- *Training*: Adam optimizer, Cross-Entropy loss, Early Stopping based on validation loss.

### 1.4 Evaluation Metrics

- **Metrics**: Balanced Accuracy (crucial for imbalanced classes), Sensitivity, Specificity, F1-Score, AUROC.
- **Robustness**: Confusion matrices to check if AD is confused with FTD or Controls.

## 2. Alternative Options

### Option A: Preprocessing Strategy

- **Standard**: Bandpass (1-45Hz) + windowing.
- **Alternative**: **Riemannian Geometry**. Instead of raw time-series, compute covariance matrices for each epoch and use Tangent Space Mapping + SVM. This is highly robust for EEG and often outperforms Deep Learning on smaller datasets.

### Option B: Feature Extraction

- **Standard**: Raw time-series input to ROCKET/CNN.
- **Alternative**: **Spectral Features (PSD)**. Compute Power Spectral Density in standard bands (Delta, Theta, Alpha, Beta, Gamma). Train a classic XGBoost/Random Forest on these tabular features. This provides a strong "interpretable" baseline to benchmark the complex models against.

### Option C: Cross-Task Transfer (Zero-Shot)

- Train on **Resting**, Test on **Photic** (and vice-versa).
- This tests whether the "disease signature" is robust to the task context or if the models are overfitting to task-specific signals.
