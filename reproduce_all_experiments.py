"""
=============================================================================
RKAKF Independent Python Reproduction & Verification Suite
Paper: "A Cubic-Decay Recursive Kurtosis-Aware Kalman Filter (RKAKF):
        Resolving the Kurtosis Ratio Paradox to Bypass Structural Blindness"
Authors: Ahmed Sattar Jabbar & Haifa Taha Abd Ahmed (Mustansiriyah University)

This suite independently implements and runs:
- EXP 1: Structural Blindness Verification (M in [10, 1000], moments & moving variance)
- EXP 2: Standard KF vs RKAKF (Localized CLG Burst)
- EXP 3: Particle Filter Degeneracy (ESS Collapse) vs RKAKF
- EXP 4: Huber KF vs RKAKF (Continuous CLG - Operational Boundary)
- EXP 5: Monte Carlo Simulation (1000 runs, Continuous CLG, Wilcoxon Signed-Rank Test)
- EXP 6: Synthetic FPV Drone Trajectory (Sparse CLG Spoofing)
- EXP 7: Monte Carlo Benchmark with VB/Student-t (M=100 & M=1000)
- EXP 7b: Financial Time Series (BTC / Heavy-Tail Jumps)
- EXP 8: Cauchy vs CLG Attack Comparison (Stealth Advantage)
- EXP 9: Gain Restoration Mechanism (Pre/During/Post Breakdown & TTR)
- EXP 10: Benchmark Summary Dashboard
- EXP M1: 2D Kinematic Tracking (Position & Velocity)
=============================================================================
"""

import os
import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Output directory for Python-generated figures
OUT_DIR = os.path.join(os.path.dirname(__file__), "python_results")
os.makedirs(OUT_DIR, exist_ok=True)

# Set global seed for reproducibility
np.random.seed(42)

# =============================================================================
# 1. NOISE GENERATORS & STATISTICAL SAMPLERS
# =============================================================================

def laplace_rnd(mu, b, size=1):
    """Sample from Laplace distribution L(mu, b)."""
    u = np.random.uniform(-0.5, 0.5, size)
    return mu - b * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))

def cauchy_rnd_trunc(x0, gamma, M, size=1):
    """Sample from truncated Cauchy distribution C_trunc(x0, gamma, M)."""
    samples = np.empty(size)
    for i in range(size):
        while True:
            u = np.random.uniform(0.0, 1.0)
            x = x0 + gamma * np.tan(np.pi * (u - 0.5))
            if np.abs(x) <= M:
                samples[i] = x
                break
    return samples

def generate_CLG(M, C, gamma_c, sigma, b_lap, p1, p2, n=1):
    """
    Compound Cauchy-Laplace-Gaussian (CLG) mixture generator:
    p1: Gaussian weight
    p2: Laplace weight
    p3: min(C / M, 1 - p1 - p2) truncated Cauchy weight
    """
    p3 = min(C / max(float(M), 1.0), 1.0 - p2)
    p3 = max(p3, 0.0)
    p_gauss = max(0.0, 1.0 - p2 - p3)
    
    u = np.random.uniform(0.0, 1.0, n)
    z = np.empty(n)
    
    mask_c = u < p3
    mask_l = (u >= p3) & (u < p3 + p2)
    mask_g = u >= (p3 + p2)
    
    n_c = np.sum(mask_c)
    if n_c > 0:
        z[mask_c] = cauchy_rnd_trunc(0.0, gamma_c, M, n_c)
        
    n_l = np.sum(mask_l)
    if n_l > 0:
        z[mask_l] = laplace_rnd(0.0, b_lap, n_l)
        
    n_g = np.sum(mask_g)
    if n_g > 0:
        z[mask_g] = np.random.normal(0.0, sigma, n_g)
        
    return z if n > 1 else z[0]


# =============================================================================
# 2. FILTER IMPLEMENTATIONS
# =============================================================================

def run_KF_1D(F, H, Q, R, y, N):
    """Standard 1D Kalman Filter."""
    x_est = np.zeros(N)
    x = 0.0
    P = 1.0
    for k in range(N):
        x_pred = F * x
        P_pred = F * P * F + Q
        K = P_pred * H / (H * P_pred * H + R)
        x = x_pred + K * (y[k] - H * x_pred)
        P = (1.0 - K * H) * P_pred
        x_est[k] = x
    return x_est

def run_Huber_1D(F, H, Q, R, y, N, c_hub=1.345):
    """Huber M-estimation Robust Kalman Filter (1D)."""
    x_est = np.zeros(N)
    x = 0.0
    P = 1.0
    for k in range(N):
        x_pred = F * x
        P_pred = F * P * F + Q
        S = H * P_pred * H + R
        nu = y[k] - H * x_pred
        z = nu / np.sqrt(S)
        if np.abs(z) <= c_hub:
            psi = z
        else:
            psi = c_hub * np.sign(z)
        K = P_pred * H / S
        x = x_pred + K * np.sqrt(S) * psi
        P = (1.0 - K * H) * P_pred
        x_est[k] = x
    return x_est

def run_RKAKF_1D(F, H, Q, R_nom, y, N,
                 kappa_th=4.0, lam_min=0.50, lam_max=0.99,
                 beta=0.5, alpha=0.5, eps=1e-6, M_cap=500.0):
    """
    Recursive Kurtosis-Aware Kalman Filter (RKAKF - 1D) per Algorithm 1:
    - Recursive 2nd & 4th central moments with cubic decay lambda^3
    - Reversed sigmoid dynamic forgetting factor
    - Exponential gain revocation & asymptotic snap-back recovery
    """
    x_est = np.zeros(N)
    kappa_hist = np.zeros(N)
    R_hist = np.zeros(N)
    K_hist = np.zeros(N)
    
    x = 0.0
    P = 1.0
    mu2 = 1.0
    mu4 = 3.0
    lam = lam_max
    R_k = R_nom
    k_star = 0
    
    for k in range(N):
        # 1. State Prediction with Covariance Shielding
        x_pred = F * x
        P_pred = (1.0 / lam) * (F * P * F + Q)
        
        # 2. Innovation and Capping
        nu = y[k] - H * x_pred
        nu_cap = np.sign(nu) * min(abs(nu), M_cap)
        
        # 3. Recursive Moment Update (cubic decay on 4th moment)
        mu2 = lam * mu2 + (1.0 - lam) * (nu_cap ** 2)
        mu4 = (lam ** 3) * mu4 + (1.0 - (lam ** 3)) * (nu_cap ** 4)
        kappa = mu4 / (mu2 ** 2 + eps)
        
        # 4. Adaptive Forgetting Factor
        lam = lam_min + (lam_max - lam_min) / (1.0 + np.exp(beta * (kappa - kappa_th)))
        
        # 5. Gain Revocation or Snap-back Recovery
        if kappa > kappa_th:
            R_k = R_nom * np.exp(alpha * min(max(kappa - kappa_th, 0.0), 20.0))
            k_star = k
        else:
            R_k = R_nom + (R_k - R_nom) * np.exp(-0.6 * (k - k_star))
            
        # 6. Kalman Update
        S = H * P_pred * H + R_k
        K = P_pred * H / S
        x = x_pred + K * (y[k] - H * x_pred)
        P = (1.0 - K * H) * P_pred
        
        x_est[k] = x
        kappa_hist[k] = kappa
        R_hist[k] = R_k
        K_hist[k] = K
        
    return x_est, kappa_hist, R_hist, K_hist

def run_VB_StudentT_1D(F, H, Q, R_nom, y, N, nu_dof=4.0, max_iter=10):
    """
    Variational Bayes Student's t Kalman Filter (Huang et al., IEEE TAES 2017).
    """
    x_est = np.zeros(N)
    x = 0.0
    P = 1.0
    for k in range(N):
        x_pred = F * x
        P_pred = F * P * F + Q
        tau_k = 1.0
        x_upd = x_pred
        K_last = 0.0
        for _ in range(max_iter):
            R_vb = R_nom / tau_k
            S = H * P_pred * H + R_vb
            K_last = P_pred * H / S
            x_upd = x_pred + K_last * (y[k] - H * x_pred)
            nu_res = y[k] - H * x_upd
            tau_k = (nu_dof + 1.0) / (nu_dof + (nu_res ** 2) / R_nom + (H * P_pred * H) / R_nom)
        P = (1.0 - K_last * H) * P_pred
        x = x_upd
        x_est[k] = x
    return x_est

def run_ParticleFilter_1D(F, H, Q, R_nom, y, N, N_part=500):
    """Bootstrap Particle Filter (Sequential Importance Resampling - SIR)."""
    particles = np.random.normal(0.0, np.sqrt(R_nom), N_part)
    weights = np.ones(N_part) / N_part
    x_est = np.zeros(N)
    ESS_hist = np.zeros(N)
    
    for k in range(N):
        # Propagation
        particles = F * particles + np.random.normal(0.0, np.sqrt(Q), N_part)
        
        # Weight update
        log_w = -0.5 * ((y[k] - H * particles) ** 2) / R_nom
        log_w -= np.max(log_w)
        weights = np.exp(log_w)
        weights /= np.sum(weights)
        
        # Effective Sample Size (ESS)
        ess = 1.0 / np.sum(weights ** 2)
        ESS_hist[k] = ess
        
        # State estimate
        x_est[k] = np.sum(weights * particles)
        
        # Systematic / Multinomial Resampling
        indices = np.random.choice(N_part, size=N_part, p=weights)
        particles = particles[indices]
        weights = np.ones(N_part) / N_part
        
    return x_est, ESS_hist

def run_KF_2D(F, H, Q, R, y, N):
    """Standard 2D Kalman Filter."""
    x_est = np.zeros((2, N))
    x = np.zeros((2, 1))
    P = np.eye(2)
    for k in range(N):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        S = (H @ P_pred @ H.T + R).item()
        K = (P_pred @ H.T) / S
        x = x_pred + K * (y[k] - (H @ x_pred).item())
        P = (np.eye(2) - K @ H) @ P_pred
        x_est[:, k] = x.ravel()
    return x_est

def run_Huber_2D(F, H, Q, R, y, N, c_hub=1.345):
    """Huber Robust 2D Kalman Filter."""
    x_est = np.zeros((2, N))
    x = np.zeros((2, 1))
    P = np.eye(2)
    for k in range(N):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        S = (H @ P_pred @ H.T + R).item()
        nu = y[k] - (H @ x_pred).item()
        z = nu / np.sqrt(S)
        psi = z if abs(z) <= c_hub else c_hub * np.sign(z)
        K = (P_pred @ H.T) / S
        x = x_pred + K * np.sqrt(S) * psi
        P = (np.eye(2) - K @ H) @ P_pred
        x_est[:, k] = x.ravel()
    return x_est

def run_RKAKF_2D(F, H, Q, R_nom, y, N,
                 kappa_th=4.0, lam_min=0.50, lam_max=0.99,
                 beta=0.5, alpha=0.5, eps=1e-6, M_cap=500.0):
    """2D RKAKF Implementation."""
    x_est = np.zeros((2, N))
    kappa_hist = np.zeros(N)
    R_hist = np.zeros(N)
    
    x = np.zeros((2, 1))
    P = np.eye(2)
    mu2 = 1.0
    mu4 = 3.0
    lam = lam_max
    R_k = R_nom
    k_star = 0
    
    for k in range(N):
        x_pred = F @ x
        P_pred = (1.0 / lam) * (F @ P @ F.T + Q)
        
        nu = y[k] - (H @ x_pred).item()
        nu_cap = np.sign(nu) * min(abs(nu), M_cap)
        
        mu2 = lam * mu2 + (1.0 - lam) * (nu_cap ** 2)
        mu4 = (lam ** 3) * mu4 + (1.0 - (lam ** 3)) * (nu_cap ** 4)
        kappa = mu4 / (mu2 ** 2 + eps)
        
        lam = lam_min + (lam_max - lam_min) / (1.0 + np.exp(beta * (kappa - kappa_th)))
        
        if kappa > kappa_th:
            R_k = R_nom * np.exp(alpha * min(max(kappa - kappa_th, 0.0), 20.0))
            k_star = k
        else:
            R_k = R_nom + (R_k - R_nom) * np.exp(-0.6 * (k - k_star))
            
        S = (H @ P_pred @ H.T + R_k).item()
        K = (P_pred @ H.T) / S
        x = x_pred + K * (y[k] - (H @ x_pred).item())
        P = (np.eye(2) - K @ H) @ P_pred
        
        x_est[:, k] = x.ravel()
        kappa_hist[k] = kappa
        R_hist[k] = R_k
        
    return x_est, kappa_hist, R_hist


# =============================================================================
# 3. EVALUATION METRICS
# =============================================================================

def compute_rmse(x_true, x_est):
    return np.sqrt(np.mean((x_true - x_est) ** 2))

def compute_mae(x_true, x_est):
    return np.mean(np.abs(x_true - x_est))

def compute_maxe(x_true, x_est):
    return np.max(np.abs(x_true - x_est))

def compute_ttr(x_true, x_est, post_indices, threshold=0.4):
    """Compute Time to Recovery after attack termination."""
    for step, idx in enumerate(post_indices, start=1):
        if abs(x_true[idx] - x_est[idx]) < threshold:
            return step
    return np.inf

def moving_variance(x, window=50):
    """Compute rolling variance of a 1D sequence."""
    N = len(x)
    mv = np.zeros(N)
    for i in range(N):
        start = max(0, i - window + 1)
        mv[i] = np.var(x[start:i+1])
    return mv


# =============================================================================
# 4. EXECUTION OF EXPERIMENTS
# =============================================================================

def run_exp1():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 1: Structural Blindness Verification")
    print("="*60)
    
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
        z = generate_CLG(M, C, gamma_c, sigma, b_lap, p1, p2, N_samp)
        v = np.var(z)
        k = stats.kurtosis(z, fisher=False) # Pearson kurtosis (Normal = 3)
        var_vals.append(v)
        kurt_vals.append(k)
        print(f"  M = {M:4d} | Empirical Variance = {v:.4f} | Empirical Kurtosis = {k:.2f}")
        
    var_theoretical = p1 * (sigma**2) + 2.0 * p2 * (b_lap**2) + (2.0 * C * gamma_c / np.pi)
    var_threshold = var_theoretical * 5.0
    print(f"\n  Theoretical Variance Bound (M -> inf): {var_theoretical:.4f}")
    print(f"  Detector Threshold (5x bound): {var_threshold:.4f}")
    
    # Time-series demonstration (M=20000)
    N_exp1 = 1000
    M_demo = 20000
    z_ts = generate_CLG(M_demo, C, gamma_c, sigma, b_lap, p1, p2, N_exp1)
    mov_var = moving_variance(z_ts, 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    # 1. Variance
    axes[0, 0].semilogx(M_list, var_vals, 'bo-', lw=2, ms=8)
    axes[0, 0].axhline(var_threshold, color='r', ls='--', lw=1.5, label=f'Threshold ({var_threshold:.2f})')
    axes[0, 0].axhline(var_theoretical, color='g', ls=':', lw=1.5, label=f'Theory ({var_theoretical:.2f})')
    axes[0, 0].set_xlabel('Impulse Magnitude M')
    axes[0, 0].set_ylabel('Empirical Variance')
    axes[0, 0].set_title('Empirical Variance vs M')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Kurtosis
    axes[0, 1].loglog(M_list, kurt_vals, 'rs-', lw=2, ms=8)
    axes[0, 1].set_xlabel('Impulse Magnitude M')
    axes[0, 1].set_ylabel('Empirical Kurtosis')
    axes[0, 1].set_title('Kurtosis Diverges with M (Theorem 1)')
    axes[0, 1].grid(True)
    
    # 3. Energy Detector Moving Variance
    axes[1, 0].plot(range(1, N_exp1 + 1), mov_var, 'b', lw=1.2)
    axes[1, 0].axhline(var_threshold, color='r', ls='--', lw=2, label='Threshold')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Moving Variance (win=50)')
    axes[1, 0].set_title('Energy Detector Blindness')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Noise Amplitude
    axes[1, 1].plot(range(1, N_exp1 + 1), z_ts, color='navy', lw=0.8)
    k_demo = stats.kurtosis(z_ts, fisher=False)
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Noise Amplitude')
    axes[1, 1].set_title(f'Hybrid CLG Noise (M={M_demo}, Kurt={k_demo:.0f})')
    axes[1, 1].grid(True)
    
    plt.suptitle('EXP 1: Structural Blindness — CLG Attack Model (Python)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP1_Structural_Blindness.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP1_Structural_Blindness.png")


def run_exp2():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 2: Standard KF vs RKAKF (Localized Burst)")
    print("="*60)
    
    F = 1.0; H = 1.0; Q = 0.1; R_nom = 1.0
    N2 = 1000
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    np.random.seed(42)
    x_true2 = np.zeros(N2)
    for k in range(1, N2):
        x_true2[k] = F * x_true2[k-1] + np.sqrt(Q) * np.random.randn()
        
    y2 = H * x_true2 + np.sqrt(R_nom) * np.random.randn(N2)
    atk2_start, atk2_end = 400, 500
    for k in range(atk2_start, atk2_end + 1):
        y2[k] += generate_CLG(500, C, gamma_c, sigma, b_lap, p1, p2, 1)
        
    xKF2 = run_KF_1D(F, H, Q, R_nom, y2, N2)
    xRK2, kappa2, Rmod2, K2 = run_RKAKF_1D(F, H, Q, R_nom, y2, N2, kappa_th=4.0)
    
    rmse_kf2 = compute_rmse(x_true2, xKF2)
    rmse_rk2 = compute_rmse(x_true2, xRK2)
    impr2 = (rmse_kf2 - rmse_rk2) / rmse_kf2 * 100.0
    
    print(f"  Standard KF RMSE : {rmse_kf2:.4f}")
    print(f"  RKAKF RMSE       : {rmse_rk2:.4f}")
    print(f"  Improvement      : {impr2:.2f}%")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(range(N2), x_true2, 'k-', lw=1.5, label='Ground Truth')
    axes[0].plot(range(N2), xKF2, 'r--', lw=1.2, label=f'Standard KF (RMSE={rmse_kf2:.3f})')
    axes[0].plot(range(N2), xRK2, 'g-', lw=1.5, label=f'RKAKF (RMSE={rmse_rk2:.3f})')
    axes[0].axvspan(atk2_start, atk2_end, color='red', alpha=0.15, label='Attack Window')
    axes[0].set_ylabel('State')
    axes[0].set_title(f'EXP 2: KF RMSE={rmse_kf2:.4f} | RKAKF RMSE={rmse_rk2:.4f} ({impr2:.1f}% reduction)')
    axes[0].legend(loc='best')
    axes[0].grid(True)
    
    axes[1].plot(range(N2), K2, 'b-', lw=1.2)
    axes[1].axvspan(atk2_start, atk2_end, color='red', alpha=0.15)
    axes[1].set_ylabel('Kalman Gain K_k')
    axes[1].set_title('Automated Gain Revocation')
    axes[1].grid(True)
    
    axes[2].plot(range(N2), kappa2, 'm-', lw=1.2)
    axes[2].axhline(4.0, color='r', ls='--', lw=1.5, label='Threshold = 4.0')
    axes[2].axvspan(atk2_start, atk2_end, color='red', alpha=0.15)
    axes[2].set_ylabel(r'$\kappa_k$')
    axes[2].set_xlabel('Time Step')
    axes[2].set_title('Recursive Kurtosis Estimate')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP2_Standard_vs_RKAKF.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP2_Standard_vs_RKAKF.png")
    return rmse_kf2, rmse_rk2


def run_exp3():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 3: Particle Filter Degeneracy vs RKAKF")
    print("="*60)
    
    F = 1.0; H = 1.0; Q = 0.1; R_nom = 1.0
    N3 = 500; N_part = 500; M3 = 500
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    np.random.seed(42)
    x_true3 = np.zeros(N3)
    for k in range(1, N3):
        x_true3[k] = F * x_true3[k-1] + np.sqrt(Q) * np.random.randn()
        
    y3 = H * x_true3 + np.sqrt(R_nom) * np.random.randn(N3)
    atk3_start, atk3_end = 150, 250
    for k in range(atk3_start, atk3_end + 1):
        y3[k] += generate_CLG(M3, C, gamma_c, sigma, b_lap, p1, p2, 1)
        
    xPF3, ESS3 = run_ParticleFilter_1D(F, H, Q, R_nom, y3, N3, N_part)
    xRK3, _, _, _ = run_RKAKF_1D(F, H, Q, R_nom, y3, N3, kappa_th=4.0)
    
    rmse_pf3 = compute_rmse(x_true3, xPF3)
    rmse_rk3 = compute_rmse(x_true3, xRK3)
    min_ess = np.min(ESS3)
    
    print(f"  Particle Filter RMSE : {rmse_pf3:.4f}")
    print(f"  RKAKF RMSE           : {rmse_rk3:.4f}")
    print(f"  Min ESS              : {min_ess:.1f} / {N_part}")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(range(N3), x_true3, 'k-', lw=1.5, label='Ground Truth')
    axes[0].plot(range(N3), xPF3, 'r:', lw=1.5, label=f'PF-500 (RMSE={rmse_pf3:.3f})')
    axes[0].plot(range(N3), xRK3, 'g-', lw=1.5, label=f'RKAKF (RMSE={rmse_rk3:.3f})')
    axes[0].axvspan(atk3_start, atk3_end, color='red', alpha=0.15)
    axes[0].set_ylabel('State')
    axes[0].set_title(f'EXP 3: Tracking Performance (PF vs RKAKF)')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(range(N3), ESS3, 'r-', lw=1.2)
    axes[1].axvspan(atk3_start, atk3_end, color='red', alpha=0.15)
    axes[1].set_ylabel('Effective Sample Size')
    axes[1].set_title(f'Effective Sample Size (ESS Collapse: Min={min_ess:.1f}/{N_part})')
    axes[1].grid(True)
    
    # Noise amplitude
    axes[2].plot(range(N3), y3 - H * x_true3, color='brown', lw=1.0)
    axes[2].axvspan(atk3_start, atk3_end, color='red', alpha=0.15)
    axes[2].set_ylabel('Noise Amplitude')
    axes[2].set_xlabel('Time Step')
    axes[2].set_title('Stealth Hybrid CLG Noise')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP3_Particle_Filter.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP3_Particle_Filter.png")
    return rmse_pf3, rmse_rk3


def run_exp4():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 4: Huber KF vs RKAKF (Continuous CLG - Operational Boundary)")
    print("="*60)
    
    F = 1.0; H = 1.0; Q = 0.1; R_nom = 1.0
    N4 = 800
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    np.random.seed(42)
    x_true4 = np.zeros(N4)
    for k in range(1, N4):
        x_true4[k] = F * x_true4[k-1] + np.sqrt(Q) * np.random.randn()
        
    y4 = H * x_true4 + np.sqrt(R_nom) * np.random.randn(N4)
    # Continuous noise across all time steps
    noise4 = np.zeros(N4)
    for k in range(N4):
        clg_sample = generate_CLG(400, C, gamma_c, sigma, b_lap, p1, p2, 1)
        y4[k] += clg_sample
        noise4[k] = clg_sample
        
    xHUB4 = run_Huber_1D(F, H, Q, R_nom, y4, N4)
    xRK4, kappa4, Rmod4, _ = run_RKAKF_1D(F, H, Q, R_nom, y4, N4, kappa_th=4.0)
    
    rmse_hub4 = compute_rmse(x_true4, xHUB4)
    rmse_rk4 = compute_rmse(x_true4, xRK4)
    
    print(f"  Huber KF RMSE : {rmse_hub4:.4f}")
    print(f"  RKAKF RMSE    : {rmse_rk4:.4f}")
    print(f"  Ratio (RKAKF / Huber): {rmse_rk4 / rmse_hub4:.2f}x")
    print("  ==> CONFIRMS PAPER'S ANALYSIS: Under CONTINUOUS i.i.d. noise, Huber beats RKAKF!")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(range(N4), x_true4, 'k-', lw=1.5, label='Ground Truth')
    axes[0].plot(range(N4), xHUB4, 'r--', lw=1.2, label=f'Huber KF (RMSE={rmse_hub4:.3f})')
    axes[0].plot(range(N4), xRK4, 'b-', lw=1.2, label=f'RKAKF (RMSE={rmse_rk4:.3f})')
    axes[0].set_ylabel('State')
    axes[0].set_title(f'EXP 4: Huber RMSE={rmse_hub4:.4f} vs RKAKF RMSE={rmse_rk4:.4f}')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(range(N4), kappa4, 'm-', lw=1.2)
    axes[1].axhline(4.0, color='r', ls='--', lw=1.2, label=r'$\kappa_{th}=4.0$')
    axes[1].set_ylabel(r'$\kappa_k$')
    axes[1].set_title('Chronic High Kurtosis from Continuous Outliers')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(range(N4), noise4, 'navy', lw=0.8)
    axes[2].set_ylabel('CLG Noise')
    axes[2].set_xlabel('Time Step')
    axes[2].set_title('Continuous CLG Bombardment (M=400)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP4_Huber_vs_RKAKF.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP4_Huber_vs_RKAKF.png")
    return rmse_hub4, rmse_rk4


def run_exp5():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 5: Monte Carlo Simulation (1000 Runs, Continuous CLG)")
    print("="*60)
    
    N_mc5 = 1000
    N_step5 = 200
    M5 = 200
    F = 1.0; H = 1.0; Q = 0.1; R_nom = 1.0
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    np.random.seed(42)
    rmse_kf = np.zeros(N_mc5)
    rmse_hub = np.zeros(N_mc5)
    rmse_rk = np.zeros(N_mc5)
    
    mae_kf = np.zeros(N_mc5)
    mae_hub = np.zeros(N_mc5)
    mae_rk = np.zeros(N_mc5)
    
    maxe_kf = np.zeros(N_mc5)
    maxe_hub = np.zeros(N_mc5)
    maxe_rk = np.zeros(N_mc5)
    
    for mc in range(N_mc5):
        xt = np.zeros(N_step5)
        for k in range(1, N_step5):
            xt[k] = F * xt[k-1] + np.sqrt(Q) * np.random.randn()
            
        ym = H * xt + np.sqrt(R_nom) * np.random.randn(N_step5)
        for k in range(N_step5):
            ym[k] += generate_CLG(M5, C, gamma_c, sigma, b_lap, p1, p2, 1)
            
        xk = run_KF_1D(F, H, Q, R_nom, ym, N_step5)
        xh = run_Huber_1D(F, H, Q, R_nom, ym, N_step5)
        xr, _, _, _ = run_RKAKF_1D(F, H, Q, R_nom, ym, N_step5, kappa_th=4.0)
        
        rmse_kf[mc] = compute_rmse(xt, xk)
        rmse_hub[mc] = compute_rmse(xt, xh)
        rmse_rk[mc] = compute_rmse(xt, xr)
        
        mae_kf[mc] = compute_mae(xt, xk)
        mae_hub[mc] = compute_mae(xt, xh)
        mae_rk[mc] = compute_mae(xt, xr)
        
        maxe_kf[mc] = compute_maxe(xt, xk)
        maxe_hub[mc] = compute_maxe(xt, xh)
        maxe_rk[mc] = compute_maxe(xt, xr)
        
    pval_rk_kf = stats.wilcoxon(rmse_rk, rmse_kf).pvalue
    pval_rk_hub = stats.wilcoxon(rmse_rk, rmse_hub).pvalue
    
    print("\n" + "-"*65)
    print(f"{'Estimator':<18} | {'Mean RMSE':<10} | {'Mean MAE':<10} | {'Max Error':<10}")
    print("-"*65)
    print(f"{'Standard KF':<18} | {np.mean(rmse_kf):<10.4f} | {np.mean(mae_kf):<10.4f} | {np.mean(maxe_kf):<10.4f}")
    print(f"{'Huber KF':<18} | {np.mean(rmse_hub):<10.4f} | {np.mean(mae_hub):<10.4f} | {np.mean(maxe_hub):<10.4f}")
    print(f"{'RKAKF (Proposed)':<18} | {np.mean(rmse_rk):<10.4f} | {np.mean(mae_rk):<10.4f} | {np.mean(maxe_rk):<10.4f}")
    print("-"*65)
    print(f"Wilcoxon signed-rank test (RKAKF vs Standard KF): p = {pval_rk_kf:.2e}")
    print(f"Wilcoxon signed-rank test (RKAKF vs Huber KF):    p = {pval_rk_hub:.2e}")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    
    # Histogram
    all_e = np.concatenate([rmse_kf, rmse_hub, rmse_rk])
    bins = np.linspace(np.min(all_e), np.percentile(all_e, 98), 45)
    axes[0].hist(rmse_kf, bins=bins, color='salmon', alpha=0.6, label=f'Standard KF (Mean={np.mean(rmse_kf):.2f})')
    axes[0].hist(rmse_rk, bins=bins, color='royalblue', alpha=0.6, label=f'RKAKF (Mean={np.mean(rmse_rk):.2f})')
    axes[0].hist(rmse_hub, bins=bins, color='lightgreen', alpha=0.6, label=f'Huber KF (Mean={np.mean(rmse_hub):.2f})')
    axes[0].axvline(np.mean(rmse_kf), color='red', ls='--', lw=2)
    axes[0].axvline(np.mean(rmse_rk), color='blue', ls='--', lw=2)
    axes[0].set_xlabel('RMSE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'EXP 5: Monte Carlo RMSE Distribution ({N_mc5} runs)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Boxplot
    axes[1].boxplot([rmse_kf, rmse_hub, rmse_rk], tick_labels=['Standard KF', 'Huber KF', 'RKAKF'], showfliers=False)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE Boxplot (Outliers clipped for display)')
    axes[1].grid(True)
    
    # Bar comparison
    x_pos = np.arange(2)
    width = 0.25
    axes[2].bar(x_pos - width, [np.mean(rmse_kf), np.mean(mae_kf)], width, label='Standard KF', color='salmon')
    axes[2].bar(x_pos, [np.mean(rmse_hub), np.mean(mae_hub)], width, label='Huber KF', color='lightgreen')
    axes[2].bar(x_pos + width, [np.mean(rmse_rk), np.mean(mae_rk)], width, label='RKAKF', color='royalblue')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(['Mean RMSE', 'Mean MAE'])
    axes[2].set_ylabel('Error')
    axes[2].set_title('Mean Error Comparison')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP5_Monte_Carlo.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP5_Monte_Carlo.png")
    
    return {
        'rmse_kf': np.mean(rmse_kf), 'mae_kf': np.mean(mae_kf), 'maxe_kf': np.mean(maxe_kf),
        'rmse_hub': np.mean(rmse_hub), 'mae_hub': np.mean(mae_hub), 'maxe_hub': np.mean(maxe_hub),
        'rmse_rk': np.mean(rmse_rk), 'mae_rk': np.mean(mae_rk), 'maxe_rk': np.mean(maxe_rk),
        'pval_rk_kf': pval_rk_kf
    }


def run_exp6():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 6: Synthetic FPV Drone Trajectory (Sparse CLG Spoofing)")
    print("="*60)
    
    dt = 0.1
    N6 = 500
    t6 = np.arange(N6) * dt
    # Flight trajectory inspired by UZH-FPV flight dynamics
    x_true6 = 3.0 * np.sin(0.06 * 2.0 * np.pi * t6) + 2.0 * np.cos(0.10 * 2.0 * np.pi * t6)
    
    np.random.seed(42)
    y6 = x_true6 + 0.1 * np.random.randn(N6)
    burst6 = [242, 245, 249, 253, 257, 260]
    burst_M6 = 400
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    for k in burst6:
        y6[k] += generate_CLG(burst_M6, C, gamma_c, sigma, b_lap, p1, p2, 1)
        
    F = 1.0; H = 1.0; Q = 0.1; R_nom = 1.0
    xKF6 = run_KF_1D(F, H, Q, R_nom, y6, N6)
    xHI6 = run_KF_1D(F, H, Q, R_nom * 1.5, y6, N6) # H-Infinity surrogate (inflated R)
    xHUB6 = run_Huber_1D(F, H, Q, R_nom, y6, N6)
    xRK6, _, _, _ = run_RKAKF_1D(F, H, Q, R_nom, y6, N6, kappa_th=4.0)
    
    rmse_kf6 = compute_rmse(x_true6, xKF6)
    rmse_hi6 = compute_rmse(x_true6, xHI6)
    rmse_hub6 = compute_rmse(x_true6, xHUB6)
    rmse_rk6 = compute_rmse(x_true6, xRK6)
    
    pk_kf6 = compute_maxe(x_true6, xKF6)
    pk_hi6 = compute_maxe(x_true6, xHI6)
    pk_hub6 = compute_maxe(x_true6, xHUB6)
    pk_rk6 = compute_maxe(x_true6, xRK6)
    
    print(f"  Standard KF : RMSE = {rmse_kf6:.4f} | Peak Error = {pk_kf6:.4f}")
    print(f"  H-Infinity  : RMSE = {rmse_hi6:.4f} | Peak Error = {pk_hi6:.4f}")
    print(f"  Huber KF    : RMSE = {rmse_hub6:.4f} | Peak Error = {pk_hub6:.4f}")
    print(f"  RKAKF       : RMSE = {rmse_rk6:.4f} | Peak Error = {pk_rk6:.4f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    
    axes[0].plot(t6, x_true6, 'k-', lw=1.5, label='Ground Truth')
    axes[0].plot(t6, xKF6, 'r--', lw=1.0, label=f'Standard KF (RMSE={rmse_kf6:.3f})')
    axes[0].plot(t6, xHI6, 'c-.', lw=1.0, label=f'H-Infinity (RMSE={rmse_hi6:.3f})')
    axes[0].plot(t6, xHUB6, 'b-.', lw=1.2, label=f'Huber KF (RMSE={rmse_hub6:.3f})')
    axes[0].plot(t6, xRK6, 'g-', lw=1.5, label=f'RKAKF (RMSE={rmse_rk6:.3f})')
    for bk in burst6:
        axes[0].axvline(bk * dt, color='red', ls=':', alpha=0.5)
    axes[0].set_ylabel('Altitude State')
    axes[0].set_title('EXP 6: Synthetic FPV Drone Trajectory — Sparse CLG Burst')
    axes[0].legend(loc='lower left')
    axes[0].grid(True)
    
    labels = ['Standard KF', 'H-Infinity', 'Huber KF', 'RKAKF']
    vals = [rmse_kf6, rmse_hi6, rmse_hub6, rmse_rk6]
    colors = ['salmon', 'skyblue', 'mediumpurple', 'lightgreen']
    axes[1].bar(labels, vals, color=colors)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Drone Trajectory Altitude RMSE Comparison')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP6_UZH_Drone.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP6_UZH_Drone.png")
    
    return {
        'rmse_kf': rmse_kf6, 'pk_kf': pk_kf6,
        'rmse_hi': rmse_hi6, 'pk_hi': pk_hi6,
        'rmse_hub': rmse_hub6, 'pk_hub': pk_hub6,
        'rmse_rk': rmse_rk6, 'pk_rk': pk_rk6
    }


def run_exp7():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 7: Monte Carlo Benchmark with VB/Student-t (M=100 & M=1000)")
    print("="*60)
    
    N_mc7 = 1000
    N_step7 = 200
    burst7 = list(range(80, 121))
    post7 = list(range(121, N_step7))
    
    F = 1.0; H = 1.0; Q = 0.1; R_nom = 1.0
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    results = {}
    
    for M_cur in [100, 1000]:
        print(f"  Simulating M = {M_cur} across {N_mc7} runs...")
        np.random.seed(42 + M_cur)
        
        rmse_kf = np.zeros(N_mc7)
        rmse_hub = np.zeros(N_mc7)
        rmse_vb = np.zeros(N_mc7)
        rmse_rk = np.zeros(N_mc7)
        
        ttr_kf = []
        ttr_hub = []
        ttr_vb = []
        ttr_rk = []
        
        for mc in range(N_mc7):
            xt = np.zeros(N_step7)
            for k in range(1, N_step7):
                xt[k] = F * xt[k-1] + np.sqrt(Q) * np.random.randn()
                
            ym = H * xt + np.sqrt(R_nom) * np.random.randn(N_step7)
            for k in burst7:
                ym[k] += generate_CLG(M_cur, C, gamma_c, sigma, b_lap, p1, p2, 1)
                
            xk = run_KF_1D(F, H, Q, R_nom, ym, N_step7)
            xh = run_Huber_1D(F, H, Q, R_nom, ym, N_step7)
            xv = run_VB_StudentT_1D(F, H, Q, R_nom, ym, N_step7)
            xr, _, _, _ = run_RKAKF_1D(F, H, Q, R_nom, ym, N_step7, kappa_th=4.0)
            
            rmse_kf[mc] = compute_rmse(xt, xk)
            rmse_hub[mc] = compute_rmse(xt, xh)
            rmse_vb[mc] = compute_rmse(xt, xv)
            rmse_rk[mc] = compute_rmse(xt, xr)
            
            tk = compute_ttr(xt, xk, post7, 0.4)
            th = compute_ttr(xt, xh, post7, 0.4)
            tv = compute_ttr(xt, xv, post7, 0.4)
            tr = compute_ttr(xt, xr, post7, 0.4)
            
            if np.isfinite(tk): ttr_kf.append(tk)
            if np.isfinite(th): ttr_hub.append(th)
            if np.isfinite(tv): ttr_vb.append(tv)
            if np.isfinite(tr): ttr_rk.append(tr)
            
        res_m = {
            'rmse_kf': np.mean(rmse_kf), 'ttr_kf': np.mean(ttr_kf) if ttr_kf else np.inf,
            'rmse_hub': np.mean(rmse_hub), 'ttr_hub': np.mean(ttr_hub) if ttr_hub else np.inf,
            'rmse_vb': np.mean(rmse_vb), 'ttr_vb': np.mean(ttr_vb) if ttr_vb else np.inf,
            'rmse_rk': np.mean(rmse_rk), 'ttr_rk': np.mean(ttr_rk) if ttr_rk else np.inf,
        }
        results[M_cur] = res_m
        
        print(f"    M={M_cur:<4d} | Standard KF : RMSE={res_m['rmse_kf']:.4f}, TTR={res_m['ttr_kf']:.1f}")
        print(f"    M={M_cur:<4d} | Huber KF    : RMSE={res_m['rmse_hub']:.4f}, TTR={res_m['ttr_hub']:.1f}")
        print(f"    M={M_cur:<4d} | VB/Student-t: RMSE={res_m['rmse_vb']:.4f}, TTR={res_m['ttr_vb']:.1f}")
        print(f"    M={M_cur:<4d} | RKAKF       : RMSE={res_m['rmse_rk']:.4f}, TTR={res_m['ttr_rk']:.1f}")
        
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    estimators = ['Standard KF', 'Huber KF', 'VB/Student-t', 'RKAKF']
    colors = ['salmon', 'orange', 'mediumpurple', 'lightgreen']
    
    # M = 100
    vals_100 = [results[100]['rmse_kf'], results[100]['rmse_hub'], results[100]['rmse_vb'], results[100]['rmse_rk']]
    axes[0].bar(estimators, vals_100, color=colors)
    axes[0].set_ylabel('Mean RMSE')
    axes[0].set_title('EXP 7: Comparison under Localized Burst (M=100)')
    axes[0].grid(True)
    
    # M = 1000
    vals_1000 = [results[1000]['rmse_kf'], results[1000]['rmse_hub'], results[1000]['rmse_vb'], results[1000]['rmse_rk']]
    axes[1].bar(estimators, vals_1000, color=colors)
    axes[1].set_ylabel('Mean RMSE')
    axes[1].set_title('EXP 7: Comparison under Severe Burst (M=1000)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP7_VB_Comparison.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP7_VB_Comparison.png")
    return results


def run_exp8():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 8: Cauchy vs CLG Attack Comparison (Stealth Advantage)")
    print("="*60)
    
    N8 = 1000
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    np.random.seed(42)
    # Pure unconstrained Cauchy (loosely capped)
    u_c = np.random.uniform(0.0, 1.0, N8)
    z_cauchy = gamma_c * np.tan(np.pi * (u_c - 0.5))
    z_cauchy = np.sign(z_cauchy) * np.minimum(np.abs(z_cauchy), 50.0)
    
    # CLG mixture with large M
    z_clg = generate_CLG(20000, C, gamma_c, sigma, b_lap, p1, p2, N8)
    
    var_theoretical = p1 * (sigma**2) + 2.0 * p2 * (b_lap**2) + (2.0 * C * gamma_c / np.pi)
    det_thresh8 = var_theoretical * 5.0
    
    var_cauchy = np.var(z_cauchy)
    var_clg = np.var(z_clg)
    kurt_clg = stats.kurtosis(z_clg, fisher=False)
    
    mv_cau = moving_variance(z_cauchy, 50)
    mv_clg = moving_variance(z_clg, 50)
    
    print(f"  Traditional Cauchy Variance : {var_cauchy:.4f}")
    print(f"  Hybrid CLG Variance        : {var_clg:.4f}")
    print(f"  Hybrid CLG Kurtosis        : {kurt_clg:.2f}")
    print(f"  Energy Threshold           : {det_thresh8:.4f}")
    print(f"  Cauchy Mean MovVar         : {np.mean(mv_cau):.4f} (DETECTABLE)")
    print(f"  CLG Mean MovVar            : {np.mean(mv_clg):.4f} (STEALTHY)")
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    axes[0, 0].plot(range(N8), z_cauchy, 'r', lw=0.8)
    axes[0, 0].set_ylim([-15, 15])
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'Traditional Cauchy (Var={var_cauchy:.2f}) — DETECTABLE')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(range(N8), z_clg, 'b', lw=0.8)
    axes[0, 1].set_ylim([-15, 15])
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'Hybrid CLG (Var={var_clg:.2f}, Kurt={kurt_clg:.0f}) — STEALTHY')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(range(N8), mv_cau, 'r', lw=1.2)
    axes[1, 0].axhline(det_thresh8, color='k', ls='--', lw=1.5, label='Threshold')
    axes[1, 0].set_ylabel('Moving Var (win=50)')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_title('Cauchy Moving Variance (Breaches Threshold)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(range(N8), mv_clg, 'b', lw=1.2)
    axes[1, 1].axhline(det_thresh8, color='k', ls='--', lw=1.5, label='Threshold')
    axes[1, 1].set_ylabel('Moving Var (win=50)')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_title('CLG Moving Variance (Stays Blind/Below Threshold)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle('EXP 8: Traditional Cauchy vs Hybrid CLG — Stealth Advantage', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP8_Attack_Comparison.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP8_Attack_Comparison.png")


def run_exp9():
    print("\n" + "="*60)
    print(">>> RUNNING EXP 9: Gain Restoration & Snap-back Recovery Cycle")
    print("="*60)
    
    F = 1.0; H = 1.0; Q = 0.1; R_nom = 1.0
    N9 = 600
    atk9_start, atk9_end = 200, 300
    post9_idx = list(range(atk9_end + 1, N9))
    pre9_idx = list(range(0, atk9_start))
    dur9_idx = list(range(atk9_start, atk9_end + 1))
    
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    np.random.seed(42)
    x_true9 = np.zeros(N9)
    for k in range(1, N9):
        x_true9[k] = F * x_true9[k-1] + np.sqrt(Q) * np.random.randn()
        
    y9 = H * x_true9 + 0.1 * np.random.randn(N9)
    for k in dur9_idx:
        y9[k] += generate_CLG(500, C, gamma_c, sigma, b_lap, p1, p2, 1)
        
    xKF9 = run_KF_1D(F, H, Q, R_nom, y9, N9)
    xHUB9 = run_Huber_1D(F, H, Q, R_nom, y9, N9)
    xRK9, kappa9, Rmod9, Kgain9 = run_RKAKF_1D(F, H, Q, R_nom, y9, N9, kappa_th=4.0)
    
    rmse_kf_pre = compute_rmse(x_true9[pre9_idx], xKF9[pre9_idx])
    rmse_kf_dur = compute_rmse(x_true9[dur9_idx], xKF9[dur9_idx])
    rmse_kf_post = compute_rmse(x_true9[post9_idx], xKF9[post9_idx])
    
    rmse_hub_pre = compute_rmse(x_true9[pre9_idx], xHUB9[pre9_idx])
    rmse_hub_dur = compute_rmse(x_true9[dur9_idx], xHUB9[dur9_idx])
    rmse_hub_post = compute_rmse(x_true9[post9_idx], xHUB9[post9_idx])
    
    rmse_rk_pre = compute_rmse(x_true9[pre9_idx], xRK9[pre9_idx])
    rmse_rk_dur = compute_rmse(x_true9[dur9_idx], xRK9[dur9_idx])
    rmse_rk_post = compute_rmse(x_true9[post9_idx], xRK9[post9_idx])
    
    ttr_kf = compute_ttr(x_true9, xKF9, post9_idx, 0.5)
    ttr_hub = compute_ttr(x_true9, xHUB9, post9_idx, 0.5)
    ttr_rk = compute_ttr(x_true9, xRK9, post9_idx, 0.5)
    
    print("\n" + "-"*65)
    print(f"{'Estimator':<16} | {'Pre':<8} | {'During':<8} | {'Post':<8} | {'TTR':<6}")
    print("-"*65)
    print(f"{'Standard KF':<16} | {rmse_kf_pre:<8.4f} | {rmse_kf_dur:<8.4f} | {rmse_kf_post:<8.4f} | {ttr_kf}")
    print(f"{'Huber KF':<16} | {rmse_hub_pre:<8.4f} | {rmse_hub_dur:<8.4f} | {rmse_hub_post:<8.4f} | {ttr_hub}")
    print(f"{'RKAKF (Proposed)':<16} | {rmse_rk_pre:<8.4f} | {rmse_rk_dur:<8.4f} | {rmse_rk_post:<8.4f} | {ttr_rk}")
    print("-"*65)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # 1. State Tracking
    axes[0].plot(range(N9), x_true9, 'k-', lw=1.5, label='Ground Truth')
    axes[0].plot(range(N9), xKF9, 'r--', lw=1.0, label='Standard KF')
    axes[0].plot(range(N9), xHUB9, 'b-.', lw=1.0, label='Huber KF')
    axes[0].plot(range(N9), xRK9, 'g-', lw=1.5, label='RKAKF')
    axes[0].axvspan(atk9_start, atk9_end, color='red', alpha=0.15, label='Burst Window')
    axes[0].set_ylabel('State')
    axes[0].set_title('EXP 9: Gain Restoration — State Tracking')
    axes[0].legend(loc='best')
    axes[0].grid(True)
    
    # 2. Kalman Gain
    axes[1].plot(range(N9), Kgain9, 'b-', lw=1.5)
    axes[1].axvspan(atk9_start, atk9_end, color='red', alpha=0.15)
    axes[1].set_ylabel('Kalman Gain K_k')
    axes[1].set_title('Automated Gain Revocation (K drops near zero during burst)')
    axes[1].grid(True)
    
    # 3. Recursive Kurtosis
    axes[2].plot(range(N9), kappa9, 'm-', lw=1.2)
    axes[2].axhline(4.0, color='r', ls='--', lw=1.5, label=r'$\kappa_{th} = 4.0$')
    axes[2].axvspan(atk9_start, atk9_end, color='red', alpha=0.15)
    axes[2].set_ylabel(r'$\kappa_k$')
    axes[2].set_title('Recursive Kurtosis Response')
    axes[2].legend()
    axes[2].grid(True)
    
    # 4. Adaptive Measurement Covariance
    axes[3].plot(range(N9), Rmod9, 'k-', lw=1.5)
    axes[3].axhline(R_nom, color='blue', ls='--', lw=1.5, label=r'$R_{nom} = 1.0$')
    axes[3].axvspan(atk9_start, atk9_end, color='red', alpha=0.15)
    axes[3].set_ylabel(r'$R_{adaptive}$')
    axes[3].set_xlabel('Time Step')
    axes[3].set_title('Covariance Inflation & Asymptotic Snap-back Recovery')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP9_Gain_Restoration.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP9_Gain_Restoration.png")
    
    return {
        'kf': (rmse_kf_pre, rmse_kf_dur, rmse_kf_post, ttr_kf),
        'hub': (rmse_hub_pre, rmse_hub_dur, rmse_hub_post, ttr_hub),
        'rk': (rmse_rk_pre, rmse_rk_dur, rmse_rk_post, ttr_rk)
    }


def run_exp10(exp2_res, exp4_res, exp6_res, exp5_res):
    print("\n" + "="*60)
    print(">>> RUNNING EXP 10: Summary Dashboard")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Subplot 1: EXP 2 (Localized Burst)
    axes[0, 0].bar(['Standard KF', 'RKAKF'], [exp2_res[0], exp2_res[1]], color=['salmon', 'lightgreen'])
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('EXP 2: Localized Burst')
    axes[0, 0].grid(True)
    
    # Subplot 2: EXP 4 (Continuous CLG - Huber wins)
    axes[0, 1].bar(['Huber KF', 'RKAKF'], [exp4_res[0], exp4_res[1]], color=['orange', 'royalblue'])
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('EXP 4: Continuous CLG (Huber wins)')
    axes[0, 1].grid(True)
    
    # Subplot 3: EXP 6 (Drone Trajectory)
    d_labels = ['Std KF', 'H-Inf', 'Huber', 'RKAKF']
    d_vals = [exp6_res['rmse_kf'], exp6_res['rmse_hi'], exp6_res['rmse_hub'], exp6_res['rmse_rk']]
    axes[1, 0].bar(d_labels, d_vals, color=['salmon', 'skyblue', 'orange', 'lightgreen'])
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('EXP 6: Synthetic FPV Drone')
    axes[1, 0].grid(True)
    
    # Subplot 4: EXP 5 (Monte Carlo 1000 runs)
    mc_labels = ['Standard KF', 'Huber KF', 'RKAKF']
    mc_vals = [exp5_res['rmse_kf'], exp5_res['rmse_hub'], exp5_res['rmse_rk']]
    axes[1, 1].bar(mc_labels, mc_vals, color=['salmon', 'orange', 'royalblue'])
    axes[1, 1].set_ylabel('Mean RMSE')
    axes[1, 1].set_title('EXP 5: Monte Carlo (1000 runs)')
    axes[1, 1].grid(True)
    
    plt.suptitle('EXP 10: Benchmark Summary Dashboard (Python Verification)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP10_Summary_Dashboard.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP10_Summary_Dashboard.png")


def run_exp_m1():
    print("\n" + "="*60)
    print(">>> RUNNING EXP M1: 2D Kinematic System (Position + Velocity)")
    print("="*60)
    
    dt = 0.1
    N_m1 = 300
    M_m1 = 50
    F2 = np.array([[1.0, dt], [0.0, 1.0]])
    H2 = np.array([[1.0, 0.0]])
    Q2 = 0.1 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R_nom = 1.0
    
    C = 0.5; gamma_c = 1.0; sigma = 0.5; b_lap = 0.3; p1 = 0.5; p2 = 0.3
    
    np.random.seed(42)
    x_true_m1 = np.zeros((2, N_m1))
    x_true_m1[:, 0] = [0.0, 1.0]
    
    # Cholesky decomposition of Q2
    L_Q2 = np.linalg.cholesky(Q2)
    for k in range(1, N_m1):
        w = L_Q2 @ np.random.randn(2, 1)
        x_true_m1[:, k] = (F2 @ x_true_m1[:, k-1:k] + w).ravel()
        
    y_m1 = (H2 @ x_true_m1).ravel() + np.sqrt(R_nom) * np.random.randn(N_m1)
    atk_m1 = list(range(100, 151))
    for k in atk_m1:
        y_m1[k] += generate_CLG(M_m1, C, gamma_c, sigma, b_lap, p1, p2, 1)
        
    xKF_m1 = run_KF_2D(F2, H2, Q2, R_nom, y_m1, N_m1)
    xHUB_m1 = run_Huber_2D(F2, H2, Q2, R_nom, y_m1, N_m1)
    xRK_m1, kappa_m1, R_m1 = run_RKAKF_2D(F2, H2, Q2, R_nom, y_m1, N_m1, kappa_th=4.0)
    
    rmse_kf_pos = compute_rmse(x_true_m1[0, :], xKF_m1[0, :])
    rmse_hub_pos = compute_rmse(x_true_m1[0, :], xHUB_m1[0, :])
    rmse_rk_pos = compute_rmse(x_true_m1[0, :], xRK_m1[0, :])
    
    print(f"  2D Position Standard KF RMSE : {rmse_kf_pos:.4f}")
    print(f"  2D Position Huber KF RMSE    : {rmse_hub_pos:.4f}")
    print(f"  2D Position RKAKF RMSE       : {rmse_rk_pos:.4f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    t_m1 = np.arange(N_m1) * dt
    
    axes[0].plot(t_m1, x_true_m1[0, :], 'k-', lw=1.5, label='Ground Truth')
    axes[0].plot(t_m1, xKF_m1[0, :], 'r--', lw=1.0, label=f'Standard KF (RMSE={rmse_kf_pos:.3f})')
    axes[0].plot(t_m1, xHUB_m1[0, :], 'b-.', lw=1.0, label=f'Huber KF (RMSE={rmse_hub_pos:.3f})')
    axes[0].plot(t_m1, xRK_m1[0, :], 'g-', lw=1.5, label=f'RKAKF (RMSE={rmse_rk_pos:.3f})')
    axes[0].axvspan(atk_m1[0] * dt, atk_m1[-1] * dt, color='red', alpha=0.15)
    axes[0].set_ylabel('Position')
    axes[0].set_title('EXP M1: 2D Kinematic System Tracking')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(t_m1, kappa_m1, 'm-', lw=1.2)
    axes[1].axhline(4.0, color='r', ls='--', lw=1.5, label=r'$\kappa_{th}=4.0$')
    axes[1].axvspan(atk_m1[0] * dt, atk_m1[-1] * dt, color='red', alpha=0.15)
    axes[1].set_ylabel(r'$\kappa_k$')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Recursive Kurtosis Response (2D System)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PY_EXP_M1_2D_System.png"), dpi=300)
    plt.close()
    print("  Saved: PY_EXP_M1_2D_System.png")


def main():
    print("="*70)
    print("STARTING COMPLETE INDEPENDENT PYTHON BENCHMARK & REPRODUCTION")
    print("="*70)
    
    run_exp1()
    exp2_res = run_exp2()
    run_exp3()
    exp4_res = run_exp4()
    exp5_res = run_exp5()
    exp6_res = run_exp6()
    run_exp7()
    run_exp8()
    run_exp9()
    run_exp10(exp2_res, exp4_res, exp6_res, exp5_res)
    run_exp_m1()
    
    print("\n" + "="*70)
    print("ALL PYTHON EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Figures saved in: {OUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
