# Physics-Graph Neural Network for Turbofan Engine Failure Prediction
### Explainable AI meets Scientific ML — QUANAD Research Laboratory Internship 2025–2026

> **Status:** 🔬 Active Research | Paper in progress for journal/conference submission

---

## What This Project Is About

Jet engines have 20+ sensors constantly recording temperature, pressure, speed, and fuel flow. Machine learning models watch these sensors and predict when the engine is about to fail — but they never explain *why* they raised an alarm.

Existing tools like SHAP and LIME try to explain these models after training, but they don't understand physics. They find statistical patterns — and those patterns can be physically wrong even when mathematically correct.

This project asks: **what if the neural network itself was shaped like the engine?**

Instead of explaining a black box after the fact, I built a Graph Neural Network where the connections between layers literally follow the physical gas path of a turbofan engine:
Fan → LPC → HPC → HPT → LPT

Information can only flow in the physically correct direction. The explanation comes out automatically from the forward pass — no SHAP needed.

---

## Results

Evaluated on real NASA N-CMAPSS DS02 dataset using the Physics-Consistency (PC) score — a metric measuring whether explanations align with known turbofan thermodynamics.

| Model | RMSE (cycles) | PC Score |
|---|---|---|
| Arm A — CNN-LSTM + SHAP (black box) | 14.32 | -0.016 |
| Arm B — Concept Bottleneck Model | 15.67 | +0.130 |
| Arm C-FC — GNN fully connected (ablation) | 5.53 | +0.536 |
| **Arm C — Physics GNN (ours)** | **5.60** | **+0.274** |

Key findings:
- Physics GNN achieves 2x better physical consistency than CBM
- Dramatically outperforms black-box baseline on both accuracy and explainability
- Ablation reveals concept supervision is a strong driver of physical consistency

---

## The Physics Consistency Score

A novel metric introduced to evaluate whether model explanations are physically meaningful — not just statistically faithful to the model.
PC = ρ_phys × (1 − Δ_cons) × (1 − σ_stab)

- **ρ_phys** — does the explanation highlight the right sensors for the fault?
- **Δ_cons** — does the explanation respect physical cause-and-effect ordering?
- **σ_stab** — is the explanation consistent across repeated runs?

---

## Model Architecture
Engine Sensor Data (14 channels, 30 timesteps)

↓

Sensor → Component Projection (Physics Matrix M)

↓

Graph Convolution Layer 1 (physics edges only)

↓

Graph Convolution Layer 2

↓

Component Health Scores (Fan, LPC, HPC, HPT, LPT)

↓

RUL Prediction (cycles remaining)

Built with PyTorch and PyTorch Geometric.

---

## Dataset

**NASA N-CMAPSS DS02** — Aircraft engine run-to-failure dataset under real flight conditions
- 5.2M training samples, 1.2M test samples
- 14 sensor channels per cycle
- Per-component health parameters for 5 gas-path components
- Real flight trajectory data

---

## Project Structure
lime-shap-explainability-engineering-failure/

│

├── data/                         # N-CMAPSS dataset (not tracked)

├── models/                       # Saved model weights

├── notebooks/

│   ├── physics_xai_turbofan.ipynb    # Baseline: CNN-LSTM + CBM + PC score

│   └── arm_c_physics_gnn.ipynb       # Physics GNN implementation

├── explainability/               # Explainability analysis scripts

├── results/                      # Figures and result files

└── README.md

---

## Tech Stack

- **Python** — PyTorch, PyTorch Geometric, SHAP, NumPy, Pandas, Matplotlib
- **Models** — CNN-LSTM, Concept Bottleneck Model, Graph Neural Network
- **Evaluation** — Physics-Consistency Score, RMSE, MAE
- **Dataset** — NASA N-CMAPSS

---

## Research Context

This project is part of the **QUANAD Research Laboratory Scientific ML Internship Programme 2025–2026**. A research paper documenting methodology and findings is being prepared for submission to a journal or conference upon completion.

**Key references:**
- Ribeiro et al. (2016) — LIME
- Lundberg & Lee (2017) — SHAP
- Naser (2025) — Fundamental flaws of PINNs and XAI in engineering systems
- Forest, Rombach & Fink (2025) — Interpretable prognostics with CBMs

---

## Author

**Sohan MR**
B.Tech Computer Science Engineering, PES University, Bengaluru
Scientific ML Intern — QUANAD Research Laboratory

[LinkedIn](https://www.linkedin.com/in/sohanmr-57b406259) · 
[GitHub](https://github.com/sohanrajannavar3646-netizen)

---

*Active research — updated regularly throughout the internship*
