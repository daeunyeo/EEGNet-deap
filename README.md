# EEGNet : EEG-Based Emotion Classification on DEAP

## 1. Overview

EEGNet (Lawhern et al., 2018) is widely used as a baseline
in BCI deep learning research, frequently cited alongside
more recent architectures.
Implementing EEGNet was chosen as a foundation for understanding
Depthwise and Separable Convolutions before moving on to
these more advanced models.

**Connection to Previous Work**

https://github.com/daeunyeo/sleep-stage-ml-vs-dl
- LSTM lacks the ability to learn spatial relationships between electrodes,
  as it only processes along the time axis
- EEGNet addresses this with Depthwise and Separable Convolutions,
  fundamental building blocks in lightweight deep learning architectures
- Previous experiment was limited by a small dataset (668 epochs),
  preventing deep learning from demonstrating its potential.
  This project applies EEGNet to a larger dataset (DEAP, 32 subjects)
  with raw EEG input to verify whether scale and architecture change outcomes

**Results**
- Best test accuracy: 65.51% on DEAP (32 subjects, 76,800 samples)
- Parameter count reduced from 18,000 (LSTM) to 1,746
- Confirmed generalizability and compactness of the original paper
- Ablation study conducted across epoch count, normalization method,
  and bandpass filter application

---

## 2. Repository Structure


```
eegnet-deap/
├── README.md
├── eegnet_model.ipynb
└── deap_experiment.ipynb     # Preprocessing + training + experiment results
```   
---

## 3. EEGNet Architecture

EEGNet (Lawhern et al., 2018) is a compact CNN architecture that learns
temporal and spatial patterns of EEG signals separately.

Unlike conventional CNNs that learn temporal and spatial patterns simultaneously,
EEGNet uses Depthwise Convolution to process each channel independently,
learning temporal and spatial patterns sequentially without mixing channels.
This approach reduces the parameter count by approximately 100x compared to existing models.

| Component | Role | Output Shape |
|-----------|------|-------------|
| Input | Raw EEG | (B, 1, 32, 128) |
| Block1: Temporal Conv | Frequency pattern learning | (B, 8, 32, 128) |
| Block1: Depthwise Conv | Spatial pattern learning | (B, 16, 1, 128) |
| Block1: AvgPool | Temporal compression | (B, 16, 1, 32) |
| Block2: Separable Conv | Temporal summary + channel mixing | (B, 16, 1, 32) |
| Block2: AvgPool | Temporal compression | (B, 16, 1, 4) |
| Flatten | 1D conversion | (B, 64) |
| Linear | Class output | (B, 2) |

B = batch size

Reference: Lawhern et al. (2018), EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs

---

## 4. EEGNet Model Implementation

Implementation code: [`eegnet_model.ipynb`](eegnet_model.ipynb)

| Item | Result |
|------|--------|
| Input shape | (B, 1, 32, 128) |
| Output shape | (B, 2) |
| Total parameters | 1,746 |
| Environment | Python 3.x / PyTorch |
| Data requirement | Not required (dummy data used for verification) |

---

## 5. Dataset and Preprocessing (DEAP)

Preprocessing code: [`deap_experiment.ipynb`](deap_experiment.ipynb)

### Dataset

DEAP (Database for Emotion Analysis using Physiological signals) is a dataset
collected from 32 subjects while watching 40 music video clips,
containing EEG and physiological signals.

| Item | Details |
|------|---------|
| Subjects | 32 |
| Trials per subject | 40 |
| Channels | 40 (EEG 32 + physiological 8) |
| Sampling rate | 128Hz |
| Labels | Valence, Arousal, Dominance, Liking (1-9) |

This project uses EEG 32 channels and Valence labels only.

### Preprocessing Note

The DEAP preprocessed version already has a 4-45Hz bandpass filter applied.
Applying an additional bandpass filter caused redundant filtering,
resulting in performance degradation (refer to Ablation Study).
Additional bandpass filtering was therefore not applied.

### Preprocessing Steps

| Step | Description | Output Shape |
|------|-------------|-------------|
| 1. EEG channel extraction | Use EEG 32ch only | (40, 32, 8064) |
| 2. Baseline removal | Remove first 3s (384 points) | (40, 32, 7680) |
| 3. Epoch segmentation | 40 trials x 60 seconds | (2400, 32, 128) |
| 4. Subject-wise normalization | z-score per subject | (2400, 32, 128) |
| 5. Dimension expansion | Reshape for model input | (2400, 1, 32, 128) |
| 6. Label binarization | Subject-wise median threshold | (2400,) |

### Label Binarization

Fixed 5-point threshold resulted in class imbalance (77:23).
Subject-wise median threshold was applied instead.

### Final Data

| Item | Value |
|------|-------|
| Total samples | 76,800 |
| Class distribution | 38,280 / 38,520 (approximately 50:50) |
| Train / Test split | 61,440 / 15,360 (8:2) |

---

## 6. Experiment Results

Experiment code: [`deap_experiment.ipynb`](deap_experiment.ipynb)

### 6-1. Final Result

**Training Configuration (final experiment)**

| Item | Value |
|------|-------|
| Experiment condition | Cross-subject, Valence binary classification |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss |
| Batch size | 64 |
| Max epoch | 100 |
| Early Stopping patience | 20 |
| Preprocessing | Subject-wise z-score normalization |

Note: Early experiments used patience=10 or no early stopping (refer to Ablation Study).

**Result**

| Best Test Accuracy | Train Accuracy |
|-------------------|----------------|
| **65.51%** | 63.06% |

### 6-2. Ablation Study

The individual effects of epoch count, normalization method,
and bandpass filter application on performance were measured separately.

| Experiment | Epoch | Preprocessing | Patience | Best Test | Note |
|------------|-------|---------------|----------|-----------|------|
| Baseline | 30 | None | - | 60.53% | - |
| Epoch extension | 100 | None | - | 64.43% | - |
| Sample-wise normalization | 100 | Sample-wise z-score | 10 | 64.49% | - |
| Normalization -> Bandpass | 24 stopped | Redundant filtering | 10 | 56.88% | Failed to converge |
| Bandpass -> Normalization | 55 stopped | Redundant filtering | 10 | 57.46% | No improvement observed |
| Subject-wise normalization | 100 | Subject-wise z-score | 10 | 64.06% | - |
| **Subject-wise normalization** | **100** | **Subject-wise z-score** | **20** | **65.51%** | **Best** |

**Key Findings**

- Epoch extension: most effective improvement (+3.9%p)
- Bandpass filter: performance degradation due to redundant filtering on DEAP preprocessed data
- patience=20 vs patience=10: +1.45%p improvement
- Best combination: subject-wise normalization + patience=20, achieving 65.51%

---

## 7. Limitations and Future Work

### Limitations
- Single seed experiment, reproducibility not verified
- Cross-subject setup is sensitive to EEG variability across subjects
- Limited model capacity with 1,746 parameters (underfitting state, room for improvement)

### Future Work
- k-fold or LOSO validation for improved reliability
- DGCNN to model spatial relationships between electrodes as a graph

---

## 8. References

- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
  EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
  *Journal of Neural Engineering*, 15(5), 056013.
  https://arxiv.org/abs/1611.08024

- Koelstra, S., Muhl, C., Soleymani, M., Lee, J. S., Yazdani, A., Ebrahimi, T., & Patras, I. (2012).
  DEAP: A Database for Emotion Analysis Using Physiological Signals.
  *IEEE Transactions on Affective Computing*, 3(1), 18-31.
  https://doi.org/10.1109/T-AFFC.2011.15
