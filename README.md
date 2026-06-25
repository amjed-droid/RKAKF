# Recursive Kurtosis-Aware Kalman Filter (RKAKF): Reference Benchmarking Suite

This repository hosts the official MATLAB simulation suite and reference implementations for the **Recursive Kurtosis-Aware Kalman Filter (RKAKF)**. This software suite is designed to ensure rigorous academic reproducibility and benchmarking of the proposed filter's resilience against stealthy non-Gaussian anomalies and Compound Laplace-Gaussian (CLG) mixture attacks.

To accommodate different reviewer preferences and computational workflows, the codebase is provided in two equivalent architectural layouts:
1. **Monolithic Architecture**: A single, self-contained MATLAB script (`RKAKF_Unified_Publication.m`) containing all experiments and helper functions. This ensures zero-dependency execution and easy reproduction of all figures and tables.
2. **Modular Architecture**: A clean, structured directory layout with dedicated folders for core source code and individual experiments.

---

## 1. Directory & Codebase Structure

The repository is organized as follows:

```
├── RKAKF_Unified_Publication.m   # Unified monolithic script (Run all experiments in one file)
├── load_parameters.m             # Centralized configuration and automated path registration
├── run_all_experiments.m         # Master runner script for the modular codebase
├── generate_latex_tables.m       # LaTeX publication tables generator
├── README.md                     # Documentation
├── src/                          # Core source code
│   ├── filters/                  # State estimators and filters
│   │   ├── run_KF_1D.m           # Baseline Standard Kalman Filter (1D)
│   │   ├── run_KF_2D.m           # Baseline Standard Kalman Filter (2D)
│   │   ├── run_RKAKF_1D.m        # Proposed Recursive Kurtosis-Aware KF (1D)
│   │   ├── run_RKAKF_2D.m        # Proposed Recursive Kurtosis-Aware KF (2D)
│   │   ├── run_Huber_1D.m        # Huber-Robust M-estimator Kalman Filter (1D)
│   │   ├── run_Huber_2D.m        # Huber-Robust M-estimator Kalman Filter (2D)
│   │   └── run_VB_StudentT.m     # Variational Bayes Student-t Robust Filter
│   └── utils/                    # Helper functions and noise generators
│       ├── generate_CLG.m        # Compound Laplace-Gaussian mixture attack generator
│       ├── laplace_rnd.m         # Laplace distribution random sampler
│       ├── cauchy_rnd_trunc.m    # Truncated Cauchy distribution random sampler
│       ├── compute_rmse.m        # Root Mean Squared Error (RMSE) metric evaluator
│       ├── compute_TTR.m         # Time-to-Recovery (TTR) duration evaluator
│       ├── ttr2str.m             # String formatting utility for TTR values
│       └── save_figure_silent.m  # Silent publication-grade figure exporter
└── experiments/                  # Self-contained individual experiment scripts
    ├── exp01_structural_blindness.m  # Energy detector structural blindness verification
    ├── exp02_kf_vs_rkakf.m            # Tracking analysis under localized CLG burst noise
    ├── exp03_pf_vs_rkakf.m            # Particle Filter degeneracy (ESS collapse) demonstration
    ├── exp04_huber_vs_rkakf.m         # Filtering evaluation under continuous CLG noise
    ├── exp05_monte_carlo.m            # 1000-run Monte Carlo statistical analysis
    ├── exp06_drone_trajectory.m       # FPV drone trajectory altitude tracking simulation
    ├── exp07_vb_student_t.m           # Performance comparison with VB-Student-t filter
    ├── exp08_cauchy_vs_clg.m          # Standard Cauchy noise vs. stealthy CLG attack comparison
    ├── exp09_gain_restoration.m       # Kalman gain revocation and exponential recovery demo
    ├── exp10_dashboard.m              # Unified summary error tracking dashboard
    └── exp_m1_2d_system.m             # 2D position-velocity tracking verification (Appendix)
```

---

## 2. Prerequisites & Toolbox Requirements

To run this simulation suite, the following environment and toolboxes are required:
* **MATLAB R2025b or later**: Necessary for modern visualization and analysis functions (such as `xregion`, `movvar`, and `histogram`).
* **Statistics and Machine Learning Toolbox**: Required for statistical tests, sampling, and distribution plotting functions (specifically `signrank`, `randsample`, and `histogram`).
* **Parallel Computing Toolbox**: Required for accelerated execution of Monte Carlo simulation loops via `parfor` in EXP 5 and EXP 7.
  * *Note: If the Parallel Computing Toolbox is not installed, you can simply replace `parfor` with a standard `for` loop in the scripts, and the code will execute successfully (though execution time will increase).*

---

## 3. Benchmark Experiments

The suite executes the following numerical experiments detailed in the manuscript:

1. **`EXP 1` — Structural Blindness Verification**: Demonstrates the energy detector's inability to identify stealthy CLG attacks, while showcasing the divergence of the recursive kurtosis metric.
2. **`EXP 2` — Standard KF vs. RKAKF (Localized Burst)**: Evaluates tracking performance under a localized, high-intensity CLG noise burst.
3. **`EXP 3` — Particle Filter Degeneracy vs. RKAKF**: Illustrates Effective Sample Size (ESS) collapse in Particle Filters during attacks, contrasting with the computational robustness of RKAKF.
4. **`EXP 4` — Huber KF vs. RKAKF (Continuous Noise)**: Analyzes filter tracking characteristics under continuous, non-burst CLG noise.
5. **`EXP 5` — Monte Carlo Analysis (1000 Runs)**: A comprehensive statistical assessment evaluating mean RMSE, mean MAE, maximum error, and Wilcoxon signed-rank statistical significance ($p$-value).
6. **`EXP 6` — UAV Drone Trajectory Tracking**: Benchmarks state estimation performance on a simulated FPV drone trajectory subject to stealthy sensor attacks.
7. **`EXP 7` — Variational Bayes Student-t Comparison**: Compares the proposed filter with the VB-Student-t filter under different attack magnitudes ($M=100$ and $M=1000$).
8. **`EXP 8` — Cauchy vs. CLG Attack Profiles**: Analyzes the stealth properties of CLG attacks versus standard Cauchy noise under energy-detection thresholds.
9. **`EXP 9` — Gain Revocation & Exponential Recovery**: Visualizes the dynamic adaptation of the Kalman gain, recursive kurtosis, and adaptive covariance inflation/restoration during and after an attack.
10. **`EXP 10` — Comparative Performance Dashboard**: Renders a graphical panel summarizing key tracking error comparisons.
11. **`EXP M1` — 2D Kinematic Verification (Appendix)**: Validates the multi-dimensional applicability of RKAKF on a 2D position-velocity tracking system.

---

## 4. Usage Instructions

### Running the Monolithic Suite
In the MATLAB Command Window, execute:
```matlab
% Run the unified monolithic script
RKAKF_Unified_Publication
```

### Running the Modular Suite
In the MATLAB Command Window, execute:
```matlab
% Run the master driver script (automatically registers all paths)
run_all_experiments
```

Alternatively, you can run any individual experiment file directly (e.g., `exp05_monte_carlo` or `exp06_drone_trajectory`). The `load_parameters.m` file called at the start of each script will automatically configure relative path dependencies and add the `src/` filters and utils to your MATLAB session.

All output figures will be written to `RKAKF_Figures/`, and LaTeX source tables will be printed to the terminal console.
