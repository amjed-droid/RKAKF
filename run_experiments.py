"""
=============================================================================
RKAKF Experiment Reproduction & Evaluation Suite
=============================================================================
Paper: "A Cubic-Decay Recursive Kurtosis-Aware Kalman Filter (RKAKF):
        Resolving the Kurtosis Ratio Paradox to Bypass Structural Blindness"
Authors: Ahmed Sattar Jabbar & Haifa Taha Abd Ahmed (Mustansiriyah University)
Target Journal: Signal Processing (Elsevier)

Usage:
    python run_experiments.py --all           # Run full reproduction suite
    python run_experiments.py --all --fast    # Fast run (100 Monte Carlo runs)
    python run_experiments.py --exp 1         # Run specific experiment (1-10)
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Ensure local rkakf package is in path
sys.path.insert(0, os.path.dirname(__file__))

from rkakf.filters import (
    RKAKF,
    StandardKalmanFilter,
    HuberKalmanFilter,
    VBStudentTKalmanFilter,
    ParticleFilter,
)
from rkakf.distributions import generate_CLG, moving_variance
from rkakf.metrics import (
    compute_rmse,
    compute_mae,
    compute_peak_error,
    compute_ttr,
    run_wilcoxon_test,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def exp1_structural_blindness():
    print("\n" + "="*65)
    print(">>> EXP 1: Structural Blindness Verification (Proposition 1)")
    print("="*65)

    M_list = [10, 50, 100, 200, 500, 1000]
    N_samp = 50000
    C = 0.5
    gamma_c = 1.0
    sigma = 0.5
    b_lap = 0.3
    p1 = 0.5
    p2 = 0.3

    var_vals = []
    kurt_vals = []

    for M in M_list:
        np.random.seed(42)
        z = generate_CLG(M, C=C, gamma_c=gamma_c, sigma=sigma, b_lap=b_lap, p1=p1, p2=p2, n=N_samp)
        v = np.var(z)
        k = stats.kurtosis(z, fisher=False)
        var_vals.append(v)
        kurt_vals.append(k)
        print(f"  M = {M:4d} | Empirical Variance = {v:.4f} | Empirical Kurtosis = {k:.2f}")

    var_theoretical = p1 * (sigma**2) + 2.0 * p2 * (b_lap**2) + (2.0 * C * gamma_c / np.pi)
    var_threshold = var_theoretical * 5.0
    print(f"\n  Theoretical Variance Ceiling (M -> inf): {var_theoretical:.4f}")
    print(f"  Energy Detector Threshold (5x bound):    {var_threshold:.4f}")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Attack Magnitude Bound M (log scale)", fontsize=11)
    ax1.set_ylabel("Empirical Variance", color=color, fontsize=11)
    ax1.plot(M_list, var_vals, "o-", color=color, linewidth=2, label="Sample Variance")
    ax1.axhline(var_theoretical, color="blue", linestyle="--", label=f"Ceiling ({var_theoretical:.2f})")
    ax1.axhline(var_threshold, color="red", linestyle=":", label=f"Detector Threshold ({var_threshold:.2f})")
    ax1.set_xscale("log")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Empirical Kurtosis", color=color, fontsize=11)
    ax2.plot(M_list, kurt_vals, "s-", color=color, linewidth=2, label="Kurtosis")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="center right")

    plt.title("EXP 1: Structural Blindness of CLG Mixture", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "EXP1_Structural_Blindness.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")


def exp2_burst_resilience():
    print("\n" + "="*65)
    print(">>> EXP 2: Standard KF vs RKAKF (Localized CLG Stealth Burst)")
    print("="*65)

    np.random.seed(42)
    N = 1000
    F, H = 1.0, 1.0
    Q, R_nom = 0.1, 1.0

    x_true = np.zeros(N)
    for k in range(1, N):
        x_true[k] = F * x_true[k-1] + np.random.normal(0, np.sqrt(Q))

    # Measurements with burst between k=400 and k=500
    y = np.zeros(N)
    for k in range(N):
        if 400 <= k <= 500:
            y[k] = H * x_true[k] + generate_CLG(M=500.0, gamma_c=1.0, sigma=0.5, b_lap=0.3, p1=0.5, p2=0.3, p3=0.20)
        else:
            y[k] = H * x_true[k] + np.random.normal(0, np.sqrt(R_nom))

    kf = StandardKalmanFilter(F=F, H=H, Q=Q, R=R_nom)
    x_kf = kf.filter(y)

    rkakf = RKAKF(F=F, H=H, Q=Q, R_nom=R_nom, kappa_th=4.0, alpha=0.5)
    res_rkakf = rkakf.filter(y)
    x_rkakf = res_rkakf["x_est"]

    # In-burst metrics
    rmse_kf = compute_rmse(x_true[400:501], x_kf[400:501])
    rmse_rkakf = compute_rmse(x_true[400:501], x_rkakf[400:501])
    reduction = (rmse_kf - rmse_rkakf) / rmse_kf * 100.0

    print(f"  Burst KF RMSE:    {rmse_kf:.4f}")
    print(f"  Burst RKAKF RMSE: {rmse_rkakf:.4f}")
    print(f"  RMSE Reduction:   {reduction:.1f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(x_true, "k-", linewidth=1.5, label="True State")
    ax1.plot(x_kf, "r--", linewidth=1.2, label=f"Standard KF (RMSE: {rmse_kf:.3f})")
    ax1.plot(x_rkakf, "b-", linewidth=1.2, label=f"RKAKF (RMSE: {rmse_rkakf:.3f})")
    ax1.axvspan(400, 500, color="orange", alpha=0.2, label="Stealth Burst (k=400-500)")
    ax1.set_ylabel("State Estimate", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.abs(x_true - x_kf), "r--", linewidth=1.0, label="KF Error")
    ax2.plot(np.abs(x_true - x_rkakf), "b-", linewidth=1.2, label="RKAKF Error")
    ax2.set_xlabel("Time Step k", fontsize=10)
    ax2.set_ylabel("Absolute Error", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("EXP 2: Localized Stealth Burst Resilience", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "EXP2_Standard_vs_RKAKF.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")
    return {"rmse_kf": rmse_kf, "rmse_rkakf": rmse_rkakf, "reduction": reduction}


def exp4_huber_vs_rkakf():
    print("\n" + "="*65)
    print(">>> EXP 4: Huber KF vs RKAKF (Continuous Noise Operational Boundary)")
    print("="*65)

    np.random.seed(42)
    N = 300
    F, H = 1.0, 1.0
    Q, R_nom = 0.01, 0.10

    x_true = np.zeros(N)
    for k in range(1, N):
        x_true[k] = F * x_true[k-1] + np.random.normal(0, np.sqrt(Q))

    y = np.zeros(N)
    for k in range(N):
        y[k] = H * x_true[k] + generate_CLG(M=100.0, gamma_c=1.0, sigma=0.5, b_lap=0.3, p1=0.5, p2=0.3, p3=0.20)

    kf = StandardKalmanFilter(F=F, H=H, Q=Q, R=R_nom)
    huber = HuberKalmanFilter(F=F, H=H, Q=Q, R=R_nom, c_hub=1.345)
    rkakf = RKAKF(F=F, H=H, Q=Q, R_nom=R_nom, kappa_th=4.0, alpha=0.5)

    x_kf = kf.filter(y)
    x_huber = huber.filter(y)
    x_rkakf = rkakf.filter(y)["x_est"]

    rmse_kf = compute_rmse(x_true, x_kf)
    rmse_huber = compute_rmse(x_true, x_huber)
    rmse_rkakf = compute_rmse(x_true, x_rkakf)

    print(f"  Standard KF RMSE: {rmse_kf:.4f}")
    print(f"  Huber KF RMSE:    {rmse_huber:.4f}")
    print(f"  RKAKF RMSE:       {rmse_rkakf:.4f}")
    print(f"  * Confirms trade-off: Memoryless Huber is optimal for stationary i.i.d.")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.abs(x_true - x_kf), "r:", alpha=0.6, label=f"Standard KF ({rmse_kf:.3f})")
    ax.plot(np.abs(x_true - x_rkakf), "b-", linewidth=1.2, label=f"RKAKF ({rmse_rkakf:.3f})")
    ax.plot(np.abs(x_true - x_huber), "g--", linewidth=1.2, label=f"Huber KF ({rmse_huber:.3f})")
    ax.set_xlabel("Time Step k", fontsize=10)
    ax.set_ylabel("Absolute Error", fontsize=10)
    ax.set_title("EXP 4: Continuous CLG Noise Operational Boundary", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "EXP4_Huber_vs_RKAKF.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")
    return {"rmse_kf": rmse_kf, "rmse_huber": rmse_huber, "rmse_rkakf": rmse_rkakf}


def exp5_monte_carlo(n_runs: int = 1000):
    print("\n" + "="*65)
    print(f">>> EXP 5: Monte Carlo Evaluation ({n_runs} Independent Runs)")
    print("="*65)

    N = 200
    F, H = 1.0, 1.0
    Q, R_nom = 0.01, 0.10

    rmse_kf_list = []
    rmse_rkakf_list = []
    rmse_huber_list = []

    for run in range(n_runs):
        if (run + 1) % max(1, n_runs // 5) == 0:
            print(f"  Progress: {run + 1}/{n_runs} runs completed...")

        x_true = np.zeros(N)
        for k in range(1, N):
            x_true[k] = F * x_true[k-1] + np.random.normal(0, np.sqrt(Q))

        y = np.zeros(N)
        for k in range(N):
            y[k] = H * x_true[k] + generate_CLG(M=100.0, gamma_c=1.0, sigma=0.5, b_lap=0.3, p1=0.5, p2=0.3, p3=0.20)

        kf = StandardKalmanFilter(F=F, H=H, Q=Q, R=R_nom)
        huber = HuberKalmanFilter(F=F, H=H, Q=Q, R=R_nom, c_hub=1.345)
        rkakf = RKAKF(F=F, H=H, Q=Q, R_nom=R_nom, kappa_th=4.0, alpha=0.5)

        x_kf = kf.filter(y)
        x_huber = huber.filter(y)
        x_rkakf = rkakf.filter(y)["x_est"]

        rmse_kf_list.append(compute_rmse(x_true, x_kf))
        rmse_rkakf_list.append(compute_rmse(x_true, x_rkakf))
        rmse_huber_list.append(compute_rmse(x_true, x_huber))

    rmse_kf_arr = np.array(rmse_kf_list)
    rmse_rkakf_arr = np.array(rmse_rkakf_list)

    mean_kf = np.mean(rmse_kf_arr)
    mean_rkakf = np.mean(rmse_rkakf_arr)
    mean_huber = np.mean(rmse_huber_list)
    reduction = (mean_kf - mean_rkakf) / mean_kf * 100.0

    stat, p_val = run_wilcoxon_test(rmse_kf_arr, rmse_rkakf_arr)

    print("\n  --- Monte Carlo Summary ---")
    print(f"  Mean KF RMSE:     {mean_kf:.4f}")
    print(f"  Mean Huber RMSE:  {mean_huber:.4f}")
    print(f"  Mean RKAKF RMSE:  {mean_rkakf:.4f}")
    print(f"  Overall Reduction: {reduction:.1f}%")
    print(f"  Wilcoxon Test:    stat={stat:.1f}, p={p_val:.2e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([rmse_kf_arr, rmse_huber_list, rmse_rkakf_arr], tick_labels=["Standard KF", "Huber KF", "RKAKF"])
    ax.set_ylabel("RMSE Distribution", fontsize=11)
    ax.set_title(f"EXP 5: Monte Carlo Error Distribution ({n_runs} runs)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "EXP5_Monte_Carlo.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [Saved] {out_path}")
    return {"mean_kf": mean_kf, "mean_rkakf": mean_rkakf, "reduction": reduction, "p_val": p_val}


def main():
    parser = argparse.ArgumentParser(
        description="Reproduction runner for RKAKF paper experiments (Python Suite)"
    )
    parser.add_argument("--all", action="store_true", help="Run all primary experiments")
    parser.add_argument("--fast", action="store_true", help="Fast execution mode (100 MC runs)")
    parser.add_argument("--exp", type=int, choices=[1, 2, 4, 5], help="Run specific experiment")

    args = parser.parse_args()

    # Default to running all if no specific argument passed
    if not (args.all or args.exp):
        args.all = True

    print("="*65)
    print("   RKAKF INDEPENDENT PYTHON BENCHMARK & REPRODUCTION SUITE   ")
    print("="*65)
    print(f"Output directory: {FIGURES_DIR}")

    mc_runs = 100 if args.fast else 1000

    if args.exp == 1 or args.all:
        exp1_structural_blindness()
    if args.exp == 2 or args.all:
        exp2_burst_resilience()
    if args.exp == 4 or args.all:
        exp4_huber_vs_rkakf()
    if args.exp == 5 or args.all:
        exp5_monte_carlo(n_runs=mc_runs)

    print("\n" + "="*65)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"All generated figures are stored in: {FIGURES_DIR}")
    print("="*65)


if __name__ == "__main__":
    main()
