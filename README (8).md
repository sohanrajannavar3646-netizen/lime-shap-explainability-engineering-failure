# Post-Hoc Explainability of Black-Box Models Using LIME and SHAP
### Engineering Failure Prediction — QUANAD Research Laboratory

> **Status:** 🔬 Active Research | Scientific ML Internship Programme 2025–2026

---

## What This Project Is About

Machine learning models can predict when an engine is about to fail — but they rarely explain *why* they raised an alarm. This is a real problem in engineering: an unexplained warning forces engineers to either shut everything down blindly or ignore it entirely. Neither is acceptable.

This project investigates whether two post-hoc explainability methods — **LIME** and **SHAP** — can reliably and efficiently explain failure predictions made by a neural network trained on real engine sensor data.

The goal is not to build a better predictive model. The goal is to honestly evaluate whether LIME and SHAP are *good enough* to be trusted in an engineering setting — and to find out where they fall short.

---

## The Core Questions

- Does LIME give consistent explanations across repeated runs, or does it change every time?
- Is SHAP fast enough for real-time monitoring as the number of sensors grows?
- Can the output be presented as a visual plot that an engineer can actually read and act on?

---

## How It Works

```
Engine Sensor Data (21 channels)
        ↓
CNN-LSTM Neural Network (PyTorch)
        ↓
Predicts: Remaining Useful Life (RUL)
        ↓
LIME & SHAP applied on top
        ↓
Feature Importance Plot → "Sensor X is driving this prediction"
```

An engineer looks at the output and knows exactly which sensor or subsystem to inspect — instead of getting a vague alarm with no context.

---

## Dataset

**NASA CMAPSS Turbofan Engine Degradation Dataset**
- ~60,000 engine cycle records
- 4 sub-datasets: FD001, FD002, FD003, FD004
- 21 sensor channels per cycle (temperature, pressure, fan speed, fuel flow, etc.)
- Simulated data — ground truth degradation modes are known, making it ideal for validating explainability outputs

Secondary: PRONOSTIA Bearing Degradation Dataset (IEEE PHM 2012 Challenge)

---

## Model Architecture

| Layer | Purpose |
|---|---|
| Input | Sliding window of sensor readings — `[batch, timesteps, 21 features]` |
| 1D Conv (×2) + ReLU | Extract short-range patterns and anomalies |
| MaxPooling | Downsample, reduce noise |
| LSTM (×1–2) | Capture long-range degradation trends |
| Fully Connected | Combine learned features |
| Output (linear) | Predict RUL as a single scalar |

Built with **PyTorch**. Trained using Mean Squared Error loss.

---

## Tech Stack

- **Python** — PyTorch, LIME, SHAP, NumPy, Pandas, Matplotlib, Scikit-learn
- **Model** — CNN-LSTM hybrid for time-series regression
- **Explainability** — `lime` and `shap` libraries applied post-training

---

## Project Structure

```
lime-shap-explainability-engineering-failure/
│
├── data/                   # Dataset loading and preprocessing
├── model/                  # CNN-LSTM architecture and training
├── explainability/         # LIME and SHAP analysis scripts
├── notebooks/              # Exploratory analysis and visualisations
├── results/                # Plots and output figures
└── README.md
```

---

## What's Being Evaluated

**LIME Stability**
Running LIME multiple times on the same input — do the explanations stay consistent or change? Testing whether fixing the random seed and increasing sampled points genuinely helps.

**SHAP Scalability**
Measuring how long SHAP takes to run as sensor count increases. Testing whether approximate SHAP methods are fast enough for realistic monitoring without losing explanation quality.

**Visual Output**
Generating feature importance plots that show which sensors most influenced a prediction — something an engineer can look at and act on in seconds.

---

## Expected Output

A bar chart like this for every prediction:

```
Sensor 9  (EGT)     ████████████  +0.42
Sensor 14 (NF)      ████████      +0.28
Sensor 2  (Ps30)    ██████        +0.19
Sensor 11 (BPR)     ███           -0.11
Sensor 7  (Nf)      ██            -0.08
```

*Which sensors pushed the RUL prediction lower — and by how much.*

---

## Research Context

This project is part of the **QUANAD Research Laboratory Scientific ML Internship Programme 2025–2026**. A research paper documenting findings will be submitted to a journal/conference upon completion.

**Key references:**
- Ribeiro et al. (2016) — *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*
- Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions*

---

## Author

**Sohan MR**
B.Tech Computer Science Engineering, PES University
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

---

*Work in progress — updated regularly throughout the internship.*
