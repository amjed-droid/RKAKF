# Cubic-Decay Recursive Kurtosis-Aware Kalman Filter (RKAKF)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Signal Processing](https://img.shields.io/badge/Journal-Elsevier%20Signal%20Processing-orange.svg)](https://www.sciencedirect.com/journal/signal-processing)
[![Tests: Passing](https://img.shields.io/badge/Tests-4%2F4%20Passing-brightgreen.svg)](tests/)

Official Python implementation and reproduction suite for the paper:

> **A Cubic-Decay Recursive Kurtosis-Aware Kalman Filter for State Estimation under Structurally Masked Impulsive Disturbances**  
> *Ahmed Sattar Jabbar\* and Haifa Taha Abd Ahmed*  
> Department of Statistics, College of Administration and Economics, Mustansiriyah University, Baghdad, Iraq.  
> Target Journal: **Signal Processing (Elsevier)**, 2026.

---

##  Theoretical Foundations

### 1. The Kurtosis Ratio Paradox
Recursive higher-order moment estimation in Kalman filtering has historically been hindered by an intrinsic transient instability. When estimating the fourth central moment $\mu_{4,k}$ and variance $\mu_{2,k}$ via symmetric Exponentially Weighted Moving Average (EWMA) updates with forgetting factor $\lambda \in (0, 1)$:

$$\mu_{2,k} = \lambda \mu_{2,k-1} + (1 - \lambda)\bar{\nu}_k^2$$
$$\mu_{4,k} = \lambda \mu_{4,k-1} + (1 - \lambda)\bar{\nu}_k^4$$

the recursive sample kurtosis ratio is:

$$\kappa_k = \frac{\mu_{4,k}}{\mu_{2,k}^2 + \epsilon}$$

Following an impulsive shock, the denominator variance dissipates at rate $\lambda^2$, while the numerator fourth moment dissipates at rate $\lambda$. During post-outlier recovery, the ratio behaves as:

$$\kappa_{k_0+t} \sim \frac{\lambda^t}{(\lambda^t)^2} \kappa_{k_0} = \lambda^{-t} \kappa_{k_0} \xrightarrow{t \to \infty} \infty$$

This creates a spurious exponential divergence (**The Kurtosis Ratio Paradox**), causing false alarms to explode long after the disturbance has ceased.

### 2. The Cubic Decay Principle ($\lambda^3$)
We analytically resolve this paradox by introducing a **cubic forgetting factor schedule** ($\lambda^3$) for the recursive fourth moment:

$$\mu_{4,k} = \lambda_{k-1}^3 \mu_{4,k-1} + (1 - \lambda_{k-1}^3)\bar{\nu}_k^4$$

Under this formulation, the homogeneous post-shock decay rate of the kurtosis ratio becomes:

$$\kappa_{k_0+t} \sim \frac{\lambda^{3t}}{(\lambda^t)^2} \kappa_{k_0} = \lambda^t \kappa_{k_0} \xrightarrow{t \to \infty} 0$$

#### Stability Bifurcation Analysis (Corollary 1)
For an arbitrary decay exponent $p \ge 1$ where $\mu_{4,k}$ decays as $\lambda^p$:
* **$p < 2$ (e.g., standard linear EWMA $p=1$):** Negative damping $\lambda^{-(2-p)t} \to \infty$ (spurious divergence / instability).
* **$p = 2$:** Zero damping $\lambda^0 = 1$ (permanent memory lock).
* **$p \ge 3$:** Positive exponential damping $\lambda^{(p-2)t}$.
* **$p = 3$ is the minimal integer exponent** that guarantees asymptotic post-shock stability while exactly harmonizing higher-order forgetting dynamics with first-order linear filter recovery ($\lambda^t$).

---

### 3. Structural Blindness in Cyber-Physical Systems
Standard Kalman filter diagnostic monitors rely on second-order statistics (e.g., Normalized Innovation Squared or moving-window variance). We mathematically prove (Proposition 1) that a **Cauchy-Laplace-Gaussian (CLG) mixture** with dynamic mixing weight $p_3(M) = C/M$ induces **Structural Blindness**:

$$\lim_{M \to \infty} \mathbb{E}[v_k^2] = p_1 \sigma^2 + 2 p_2 b^2 + \frac{2 C \gamma}{\pi} < \infty$$
$$\lim_{M \to \infty} \mathbb{E}[v_k^4] = \mathcal{O}(M^2) \to \infty \implies \lim_{M \to \infty} \kappa(v_k) = \infty$$

Because total innovation variance remains strictly bounded below energy thresholds, conventional second-order detectors fail to trigger, allowing stealth impulsive corruption to directly degrade state estimates.

---

##  Key Features of RKAKF

1. **Fourth-Order Anomaly Detection:** Bypasses structural blindness with zero historical observation buffer overhead.
2. **Cubic Decay Principle:** Guarantees post-shock stability and unbiased asymptotic moments $\mathbb{E}[\mu_4^*] = \mathbb{E}[\nu^4]$.
3. **Automated Gain Revocation:** Inflates $R_k$ exponentially upon kurtosis breach, forcing $K_k \to 0$ and provably bounding single-step estimation errors (Theorem 1).
4. **Asymptotic Snap-back:** Restores nominal Kalman Gain smoothly upon anomaly cessation without step discontinuities.
5. **$\mathcal{O}(d^3)$ Computational Complexity:** Preserves the standard Kalman filter complexity with only $\mathcal{O}(1)$ scalar arithmetic overhead and a four-scalar state footprint.

---

##  Repository Structure

```text
RKAKF/
├── rkakf/                          # Core Python Package
│   ├── __init__.py                 # Public package interface
│   ├── filters.py                  # RKAKF, Standard KF, Huber KF, VB Student-t, PF
│   ├── distributions.py            # CLG mixture, truncated Cauchy, Laplace samplers
│   └── metrics.py                  # RMSE, MAE, Peak Error, TTR, Wilcoxon test
├── tests/                          # Formal Mathematical Verification Tests
│   └── test_rkakf.py               # Unit tests validating Prop 1, Prop 2, Cor 1, Thm 1
├── figures/                        # Output directory for generated publication plots
├── run_experiments.py              # CLI benchmark and reproduction runner
├── reproduce_all_experiments.py    # Complete monolithic reproduction suite
├── requirements.txt                # Python package dependencies
├── pyproject.toml                  # Standard Python packaging (PEP 517/621)
├── LICENSE                         # MIT License
└── README.md                       # Repository documentation
```

---

##  Installation & Quick Start

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/amjed-droid/RKAKF.git
cd RKAKF
pip install -r requirements.txt
```

Or install in editable mode:

```bash
pip install -e .
```

### Python Quickstart

```python
import numpy as np
from rkakf import RKAKF, generate_CLG

# 1. Instantiate RKAKF
filter = RKAKF(F=1.0, H=1.0, Q=0.01, R_nom=0.10, kappa_th=4.0)

# 2. Simulate streaming measurements with an impulsive outlier
measurements = [0.1, -0.05, 0.08, 250.0, 0.02, -0.04]  # outlier at k=3

for k, y_k in enumerate(measurements):
    x_est, kappa, R_k, K_k = filter.step(y_k)
    print(f"Step {k:d} | Meas: {y_k:6.2f} | Est: {x_est:5.3f} | Kurtosis: {kappa:6.2f} | Gain: {K_k:.4f}")
```

---

##  Reproducing Paper Experiments

Run the complete suite or specific experiments via the CLI runner:

### 1. Run All Experiments (Full / Fast Mode)

```bash
# Fast mode (100 Monte Carlo runs - ~5 seconds)
python run_experiments.py --all --fast

# Full mode (1000 Monte Carlo runs)
python run_experiments.py --all
```

### 2. Run Specific Experiments

```bash
# EXP 1: Structural Blindness Verification (Proposition 1)
python run_experiments.py --exp 1

# EXP 2: Localized Stealth Burst Resilience (Standard KF vs RKAKF)
python run_experiments.py --exp 2

# EXP 4: Operational Boundaries (Huber KF vs RKAKF under Continuous CLG Noise)
python run_experiments.py --exp 4

# EXP 5: Large-scale Monte Carlo Analysis (1000 Runs & Wilcoxon Hypothesis Test)
python run_experiments.py --exp 5
```

All generated figures are saved at 300 DPI in the `figures/` directory.

---

##  Summary of Experimental Benchmarks

| Scenario | Evaluated Metric | Standard KF | Huber-Robust KF | Particle Filter | **RKAKF (Proposed)** |
|:---|:---|:---:|:---:|:---:|:---:|
| **Proposition 1** | Empirical Variance ($M \to \infty$) | Diverges (if unconstrained) | Diverges | -- | **Strictly Bounded** ($\le 0.495$) |
| **Proposition 2** | Post-Shock Recovery | Diverges ($p=1$) | Frozen ($p=2$) | -- | **Exponential Decay** ($p=3, \mathcal{O}(\lambda^t)$) |
| **Stealth Burst** | In-Burst RMSE | 1.2144 | 0.8120 | 0.5738 (ESS=1) | **0.5811** (52.1% reduction) |
| **Continuous CLG**| Steady-State RMSE | 1.6320 | **0.6102** | Degenerates | **0.7064** (57.0% reduction) |
| **Peak Outlier**  | Maximum Absolute Error | 12.0048 | 1.7654 | Degenerates | **3.0247** (74.8% reduction) |
| **Drone Telemetry**| Flight Altitude RMSE | 0.4020 | 0.3218 (ripples) | -- | **0.2558** (Smooth tracking) |
| **Complexity**    | Per-Step Scaling | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^3)$ | $\mathcal{O}(N d^2)$ | **$\mathcal{O}(d^3)$** |

> **Operational Boundary Note:** As transparently documented in the paper, memoryless M-estimators (Huber KF) remain optimal for continuous, stationary i.i.d. noise due to the absence of memory accumulation. The proposed RKAKF is uniquely engineered for localized, stealthy cyber-physical bursts where variance-based detectors fail.

---

##  Unit Tests

Run the mathematical invariant unit test suite:

```bash
python -m unittest discover tests
```

Tests verify:
1. `test_proposition1_structural_blindness_bounded_variance`: Bounded variance under growing $M$.
2. `test_proposition2_cubic_decay_resolves_kurtosis_ratio_paradox`: Asymptotic dissipation vs linear divergence.
3. `test_theorem1_gain_revocation_bounds_outlier_error`: Gain suppression and estimation error bounding.
4. `test_rkakf_nominal_performance_comparable_to_kf`: Statistical optimality under pure Gaussian noise.

---

##  Citation

If you use RKAKF in your research or applications, please cite our paper:

```bibtex
@article{jabbar2026rkakf,
  title     = {A Cubic-Decay Recursive Kurtosis-Aware Kalman Filter for State Estimation under Structurally Masked Impulsive Disturbances},
  author    = {Jabbar, Ahmed Sattar and Abd Ahmed, Haifa Taha},
  journal   = {Signal Processing},
  year      = {2026},
  publisher = {Elsevier},
  note      = {Under Review}
}
```

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
