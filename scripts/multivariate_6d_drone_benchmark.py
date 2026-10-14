"""
Test script: Multivariate 6D and Nonlinear Drone Tracking under CLG Spoofing Attacks.
Compares:
1. Standard Kalman Filter (KF / EKF)
2. H-Infinity Filter (H-Inf / E-H-Inf)
3. Huber Robust Filter (Huber / E-Huber)
4. RKAKF Channel-Decoupled (Proposed)
5. RKAKF Mardia Multivariate (Proposed)
"""

import numpy as np
import scipy.linalg as la

# =============================================================================
# 1. NOISE GENERATOR
# =============================================================================
def cauchy_rnd_trunc(x0, gamma, M, size=1):
    samples = np.empty(size)
    for i in range(size):
        while True:
            u = np.random.uniform(0.0, 1.0)
            x = x0 + gamma * np.tan(np.pi * (u - 0.5))
            if np.abs(x) <= M:
                samples[i] = x
                break
    return samples

def laplace_rnd(mu, b, size=1):
    u = np.random.uniform(-0.5, 0.5, size)
    return mu - b * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))

def generate_CLG(M, C=0.5, gamma_c=1.0, sigma=0.5, b_lap=0.3, p1=0.5, p2=0.3, n=1):
    p3 = min(C / max(float(M), 1.0), 1.0 - p2)
    p3 = max(p3, 0.0)
    
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
    return z

def compute_rmse(x_true, x_est):
    return np.sqrt(np.mean((x_true - x_est)**2))

def compute_maxe(x_true, x_est):
    return np.max(np.abs(x_true - x_est))

# =============================================================================
# 2. MULTIVARIATE 6D DRONE (3D Position + 3D Velocity)
# =============================================================================
def run_multivariate_6d_drone():
    np.random.seed(42)
    dt = 0.1
    N = 500
    t = np.arange(N) * dt
    
    # 3D Agile UZH-FPV flight trajectory
    px = 5.0 * np.sin(0.04 * 2 * np.pi * t) + 2.0 * np.cos(0.08 * 2 * np.pi * t)
    py = 4.0 * np.cos(0.04 * 2 * np.pi * t) + 3.0 * np.sin(0.06 * 2 * np.pi * t)
    pz = 3.0 * np.sin(0.06 * 2 * np.pi * t) + 2.0 * np.cos(0.10 * 2 * np.pi * t)
    
    vx = np.gradient(px, dt)
    vy = np.gradient(py, dt)
    vz = np.gradient(pz, dt)
    
    x_true = np.vstack([px, py, pz, vx, vy, vz]) # (6, N)
    
    # System matrices
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    F = np.block([[I3, dt * I3], [Z3, I3]]) # (6, 6)
    H = np.block([[I3, Z3]])                # (3, 6)
    
    q_scale = 0.10
    Q_block = np.block([
        [(dt**3 / 3.0) * I3, (dt**2 / 2.0) * I3],
        [(dt**2 / 2.0) * I3, dt * I3]
    ]) * q_scale
    
    sigma_r = 0.5
    R_nom = (sigma_r ** 2) * np.eye(3)
    
    # Clean observations + nominal Gaussian noise
    y = np.zeros((3, N))
    for k in range(N):
        y[:, k] = (H @ x_true[:, k:k+1]).ravel() + np.random.normal(0.0, sigma_r, 3)
        
    # Inject localized stealth CLG burst on z-channel (altitude sensor spoofing)
    # matching the agile burst window in EXP 6
    burst_idx = [242, 245, 249, 253, 257, 260]
    burst_M = 400.0
    for k in burst_idx:
        y[2, k] += generate_CLG(burst_M, C=0.5, gamma_c=1.0, sigma=0.5, b_lap=0.3, p1=0.5, p2=0.3, n=1)[0]
        
    # --- 1. Standard 6D KF ---
    x_kf = np.zeros((6, N))
    x_curr = x_true[:, 0:1].copy()
    P_curr = np.eye(6) * 1.0
    for k in range(N):
        x_pred = F @ x_curr
        P_pred = F @ P_curr @ F.T + Q_block
        S = H @ P_pred @ H.T + R_nom
        K = P_pred @ H.T @ la.inv(S)
        nu = y[:, k:k+1] - H @ x_pred
        x_curr = x_pred + K @ nu
        P_curr = (np.eye(6) - K @ H) @ P_pred
        x_kf[:, k] = x_curr.ravel()
        
    # --- 2. 6D H-Infinity Filter (R_Hinf = 1.5 R_nom) ---
    x_hinf = np.zeros((6, N))
    x_curr = x_true[:, 0:1].copy()
    P_curr = np.eye(6) * 1.0
    R_hinf = 1.5 * R_nom
    for k in range(N):
        x_pred = F @ x_curr
        P_pred = F @ P_curr @ F.T + Q_block
        S = H @ P_pred @ H.T + R_hinf
        K = P_pred @ H.T @ la.inv(S)
        nu = y[:, k:k+1] - H @ x_pred
        x_curr = x_pred + K @ nu
        P_curr = (np.eye(6) - K @ H) @ P_pred
        x_hinf[:, k] = x_curr.ravel()

    # --- 3. 6D Huber Robust KF ---
    x_hub = np.zeros((6, N))
    x_curr = x_true[:, 0:1].copy()
    P_curr = np.eye(6) * 1.0
    c_hub = 1.345
    for k in range(N):
        x_pred = F @ x_curr
        P_pred = F @ P_curr @ F.T + Q_block
        S = H @ P_pred @ H.T + R_nom
        S_inv = la.inv(S)
        nu = y[:, k:k+1] - H @ x_pred
        
        # Pointwise Huber score weighting on standardized residual
        d_m = np.sqrt(np.maximum(0.0, float((nu.T @ S_inv @ nu).item())))
        if d_m > c_hub:
            psi_weight = c_hub / d_m
        else:
            psi_weight = 1.0
            
        K = P_pred @ H.T @ S_inv
        x_curr = x_pred + K @ nu * psi_weight
        P_curr = (np.eye(6) - K @ H) @ P_pred
        x_hub[:, k] = x_curr.ravel()

    # --- 4. 6D RKAKF (Channel-Decoupled - Section 4.5) ---
    # Independent recursive moments for each sensor channel j in {0, 1, 2}
    x_rk_dec = np.zeros((6, N))
    x_curr = x_true[:, 0:1].copy()
    P_curr = np.eye(6) * 1.0
    
    m_dim = 3
    mu2 = np.ones(m_dim) * (sigma_r**2)
    mu4 = 3.0 * (mu2**2)
    lam = np.ones(m_dim) * 0.98
    R_diag = np.ones(m_dim) * (sigma_r**2)
    k_star = np.zeros(m_dim)
    
    kappa_th = 4.0
    lam_min = 0.55
    lam_max = 0.98
    beta = 2.0
    alpha = 1.0
    gamma_r = 0.60
    eps = 1e-6
    M_cap = 50.0
    
    for k in range(N):
        # Time update with historical minimum forgetting factor across channels
        lam_eff = float(np.min(lam))
        x_pred = F @ x_curr
        P_pred = (1.0 / lam_eff) * (F @ P_curr @ F.T + Q_block)
        
        nu = (y[:, k:k+1] - H @ x_pred).ravel()
        
        # Channel-wise moment updates
        for j in range(m_dim):
            nu_j_cap = np.sign(nu[j]) * min(abs(nu[j]), M_cap)
            mu2[j] = lam[j] * mu2[j] + (1.0 - lam[j]) * (nu_j_cap**2)
            mu4[j] = (lam[j]**3) * mu4[j] + (1.0 - (lam[j]**3)) * (nu_j_cap**4)
            kappa_j = mu4[j] / (mu2[j]**2 + eps)
            
            lam[j] = lam_min + (lam_max - lam_min) / (1.0 + np.exp(beta * (kappa_j - kappa_th)))
            
            if kappa_j > kappa_th:
                R_diag[j] = (sigma_r**2) * np.exp(alpha * min(kappa_j - kappa_th, 20.0))
                k_star[j] = k
            else:
                R_diag[j] = (sigma_r**2) + (R_diag[j] - (sigma_r**2)) * np.exp(-gamma_r * (k - k_star[j]))
                
        R_k = np.diag(R_diag)
        S = H @ P_pred @ H.T + R_k
        K = P_pred @ H.T @ la.inv(S)
        x_curr = x_pred + K @ (y[:, k:k+1] - H @ x_pred)
        # Joseph stabilized covariance update (Remark 3)
        IKH = np.eye(6) - K @ H
        P_curr = IKH @ P_pred @ IKH.T + K @ R_k @ K.T
        x_rk_dec[:, k] = x_curr.ravel()

    # --- 5. 6D RKAKF (Mardia Multivariate - Section 4.5) ---
    x_rk_mardia = np.zeros((6, N))
    x_curr = x_true[:, 0:1].copy()
    P_curr = np.eye(6) * 1.0
    
    mu2_m = float(m_dim)
    mu4_m = float(m_dim * (m_dim + 2))
    lam_m = 0.98
    R_scale = 1.0
    k_star_m = 0
    kappa_th_mardia = 1.33 # Mardia threshold normalized for m=3
    
    for k in range(N):
        x_pred = F @ x_curr
        P_pred = (1.0 / lam_m) * (F @ P_curr @ F.T + Q_block)
        
        nu = y[:, k:k+1] - H @ x_pred
        S_nom = H @ P_pred @ H.T + R_nom
        # Mahalanobis squared norm r_k^2
        r2 = float((nu.T @ la.inv(S_nom) @ nu).item())
        r2_cap = min(r2, M_cap**2)
        
        mu2_m = lam_m * mu2_m + (1.0 - lam_m) * r2_cap
        mu4_m = (lam_m**3) * mu4_m + (1.0 - (lam_m**3)) * (r2_cap**2)
        kappa_mardia = (m_dim * mu4_m) / ((m_dim + 2) * (mu2_m**2) + eps)
        
        lam_m = lam_min + (lam_max - lam_min) / (1.0 + np.exp(beta * (kappa_mardia - kappa_th_mardia)))
        
        if kappa_mardia > kappa_th_mardia:
            R_scale = np.exp(alpha * min(kappa_mardia - kappa_th_mardia, 20.0))
            k_star_m = k
        else:
            R_scale = 1.0 + (R_scale - 1.0) * np.exp(-gamma_r * (k - k_star_m))
            
        R_k = R_scale * R_nom
        S = H @ P_pred @ H.T + R_k
        K = P_pred @ H.T @ la.inv(S)
        x_curr = x_pred + K @ nu
        IKH = np.eye(6) - K @ H
        P_curr = IKH @ P_pred @ IKH.T + K @ R_k @ K.T
        x_rk_mardia[:, k] = x_curr.ravel()

    # --- Compute Metrics ---
    # 3D Position RMSE & Peak Error:
    def get_pos_metrics(x_est):
        pos_err_3d = np.linalg.norm(x_true[:3, :] - x_est[:3, :], axis=0) # Euclidean distance in 3D
        rmse_3d = np.sqrt(np.mean(pos_err_3d**2))
        peak_3d = np.max(pos_err_3d)
        
        z_rmse = compute_rmse(x_true[2, :], x_est[2, :])
        z_peak = compute_maxe(x_true[2, :], x_est[2, :])
        return rmse_3d, peak_3d, z_rmse, z_peak

    rmse_kf_3d, pk_kf_3d, r_z_kf, p_z_kf = get_pos_metrics(x_kf)
    rmse_hi_3d, pk_hi_3d, r_z_hi, p_z_hi = get_pos_metrics(x_hinf)
    rmse_hub_3d, pk_hub_3d, r_z_hub, p_z_hub = get_pos_metrics(x_hub)
    rmse_dec_3d, pk_dec_3d, r_z_dec, p_z_dec = get_pos_metrics(x_rk_dec)
    rmse_mar_3d, pk_mar_3d, r_z_mar, p_z_mar = get_pos_metrics(x_rk_mardia)
    
    print("\n" + "="*70)
    print(">>> RESULTS: FULLY MULTIVARIATE 6D DRONE (3D Position + 3D Velocity)")
    print("="*70)
    print(f"{'Estimator':<25} | {'3D Pos RMSE':<12} | {'3D Peak Err':<12} | {'Z-RMSE':<10} | {'Z-Peak':<10}")
    print("-"*70)
    print(f"{'Standard KF (6D)':<25} | {rmse_kf_3d:<12.4f} | {pk_kf_3d:<12.4f} | {r_z_kf:<10.4f} | {p_z_kf:<10.4f}")
    print(f"{'H-Infinity (6D)':<25} | {rmse_hi_3d:<12.4f} | {pk_hi_3d:<12.4f} | {r_z_hi:<10.4f} | {p_z_hi:<10.4f}")
    print(f"{'Huber-Robust (6D)':<25} | {rmse_hub_3d:<12.4f} | {pk_hub_3d:<12.4f} | {r_z_hub:<10.4f} | {p_z_hub:<10.4f}")
    print(f"{'RKAKF Decoupled (6D)':<25} | {rmse_dec_3d:<12.4f} | {pk_dec_3d:<12.4f} | {r_z_dec:<10.4f} | {p_z_dec:<10.4f}")
    print(f"{'RKAKF Mardia (6D)':<25} | {rmse_mar_3d:<12.4f} | {pk_mar_3d:<12.4f} | {r_z_mar:<10.4f} | {p_z_mar:<10.4f}")
    print("="*70)
    
    red_kf = (rmse_kf_3d - rmse_dec_3d) / rmse_kf_3d * 100.0
    red_hub = (rmse_hub_3d - rmse_dec_3d) / rmse_hub_3d * 100.0
    red_hi = (rmse_hi_3d - rmse_dec_3d) / rmse_hi_3d * 100.0
    print(f"RKAKF Decoupled Improvement over Standard KF : {red_kf:.2f}%")
    print(f"RKAKF Decoupled Improvement over Huber KF    : {red_hub:.2f}%")
    print(f"RKAKF Decoupled Improvement over H-Infinity  : {red_hi:.2f}%")

# =============================================================================
# 3. NONLINEAR DRONE MODEL (Nonlinear Aerodynamic Drag + Spherical Radar / Range-Bearing)
# =============================================================================
def run_nonlinear_drone():
    np.random.seed(42)
    dt = 0.1
    N = 500
    t = np.arange(N) * dt
    
    # 3D True trajectory
    px = 5.0 * np.sin(0.04 * 2 * np.pi * t) + 2.0 * np.cos(0.08 * 2 * np.pi * t)
    py = 4.0 * np.cos(0.04 * 2 * np.pi * t) + 3.0 * np.sin(0.06 * 2 * np.pi * t)
    pz = 3.0 * np.sin(0.06 * 2 * np.pi * t) + 2.0 * np.cos(0.10 * 2 * np.pi * t)
    
    vx = np.gradient(px, dt)
    vy = np.gradient(py, dt)
    vz = np.gradient(pz, dt)
    x_true = np.vstack([px, py, pz, vx, vy, vz])
    
    # Nonlinear dynamics: Quadrotor velocity with quadratic aerodynamic drag:
    # dot(v) = a_cmd - c_drag * ||v|| * v
    c_drag = 0.05
    
    def f_nonlin(x_vec):
        p = x_vec[:3]
        v = x_vec[3:]
        v_norm = np.linalg.norm(v)
        # continuous acceleration approx
        p_next = p + v * dt
        v_next = v - c_drag * v_norm * v * dt
        return np.vstack([p_next, v_next])
        
    def F_jacobian(x_vec):
        v = x_vec[3:].ravel()
        v_norm = max(np.linalg.norm(v), 1e-6)
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        # df/dv = I - c_drag * dt * (v_norm * I + (v * v^T)/v_norm)
        df_dv = I3 - c_drag * dt * (v_norm * I3 + np.outer(v, v) / v_norm)
        return np.block([[I3, dt * I3], [Z3, df_dv]])
        
    # Nonlinear observation: Spherical Radar tracking [range, azimuth, elevation]
    # r = sqrt(px^2 + py^2 + pz^2)
    # theta = arctan2(py, px)
    # phi = arcsin(pz / r)
    def h_nonlin(x_vec):
        px, py, pz = x_vec[0, 0], x_vec[1, 0], x_vec[2, 0]
        r = np.sqrt(px**2 + py**2 + pz**2)
        theta = np.arctan2(py, px)
        phi = np.arcsin(np.clip(pz / max(r, 1e-6), -1.0, 1.0))
        return np.array([[r], [theta], [phi]])
        
    def H_jacobian(x_vec):
        px, py, pz = x_vec[0, 0], x_vec[1, 0], x_vec[2, 0]
        r2 = px**2 + py**2 + pz**2
        r = np.sqrt(r2)
        rho2 = px**2 + py**2
        rho = np.sqrt(max(rho2, 1e-6))
        
        # dr/dx
        dr = np.array([px/r, py/r, pz/r, 0, 0, 0])
        # dtheta/dx
        dth = np.array([-py/rho2, px/rho2, 0, 0, 0, 0])
        # dphi/dx
        dphi = np.array([-px*pz/(r2*rho), -py*pz/(r2*rho), rho/r2, 0, 0, 0])
        return np.vstack([dr, dth, dphi])

    # Measurement noise: range error 0.2m, angle error 0.02 rad (~1.1 deg)
    R_nom = np.diag([0.2**2, 0.02**2, 0.02**2])
    Q_block = np.eye(6) * 0.01
    
    # Generate noisy nonlinear measurements
    y_nl = np.zeros((3, N))
    for k in range(N):
        h_true = h_nonlin(x_true[:, k:k+1])
        noise = np.random.multivariate_normal([0, 0, 0], R_nom).reshape(3, 1)
        y_nl[:, k] = (h_true + noise).ravel()
        
    # Inject localized stealth CLG burst on the elevation channel (phi / altimetric angle)
    burst_idx = [242, 245, 249, 253, 257, 260]
    for k in burst_idx:
        # 1.5 rad spike (~85 degrees spoofing)
        y_nl[2, k] += generate_CLG(M=1.5, C=0.2, gamma_c=0.05, sigma=0.02, b_lap=0.01, n=1)[0]
        
    # --- 1. Standard Extended Kalman Filter (EKF) ---
    x_ekf = np.zeros((6, N))
    x_curr = x_true[:, 0:1].copy()
    P_curr = np.eye(6) * 0.5
    for k in range(N):
        x_pred = f_nonlin(x_curr)
        F_k = F_jacobian(x_curr)
        P_pred = F_k @ P_curr @ F_k.T + Q_block
        
        H_k = H_jacobian(x_pred)
        h_pred = h_nonlin(x_pred)
        nu = y_nl[:, k:k+1] - h_pred
        # wrap angle residuals
        nu[1, 0] = (nu[1, 0] + np.pi) % (2 * np.pi) - np.pi
        nu[2, 0] = (nu[2, 0] + np.pi) % (2 * np.pi) - np.pi
        
        S = H_k @ P_pred @ H_k.T + R_nom
        K = P_pred @ H_k.T @ la.inv(S)
        x_curr = x_pred + K @ nu
        P_curr = (np.eye(6) - K @ H_k) @ P_pred
        x_ekf[:, k] = x_curr.ravel()

    # --- 2. Extended Huber Filter (E-Huber) ---
    x_ehub = np.zeros((6, N))
    x_curr = x_true[:, 0:1].copy()
    P_curr = np.eye(6) * 0.5
    c_hub = 1.345
    for k in range(N):
        x_pred = f_nonlin(x_curr)
        F_k = F_jacobian(x_curr)
        P_pred = F_k @ P_curr @ F_k.T + Q_block
        
        H_k = H_jacobian(x_pred)
        h_pred = h_nonlin(x_pred)
        nu = y_nl[:, k:k+1] - h_pred
        nu[1, 0] = (nu[1, 0] + np.pi) % (2 * np.pi) - np.pi
        nu[2, 0] = (nu[2, 0] + np.pi) % (2 * np.pi) - np.pi
        
        S = H_k @ P_pred @ H_k.T + R_nom
        S_inv = la.inv(S)
        d_m = np.sqrt(np.maximum(0.0, float((nu.T @ S_inv @ nu).item())))
        psi_w = c_hub / d_m if d_m > c_hub else 1.0
        
        K = P_pred @ H_k.T @ S_inv
        x_curr = x_pred + K @ nu * psi_w
        P_curr = (np.eye(6) - K @ H_k) @ P_pred
        x_ehub[:, k] = x_curr.ravel()

    # --- 3. Extended RKAKF (ERK-AKF - Proposed) ---
    x_erk = np.zeros((6, N))
    x_curr = x_true[:, 0:1].copy()
    P_curr = np.eye(6) * 0.5
    
    m_dim = 3
    sigmas_nom = np.array([0.2, 0.02, 0.02])
    mu2 = sigmas_nom**2
    mu4 = 3.0 * (mu2**2)
    lam = np.ones(3) * 0.98
    R_diag = sigmas_nom**2
    k_star = np.zeros(3)
    
    kappa_th = 4.0
    lam_min = 0.55
    lam_max = 0.98
    beta = 2.0
    alpha = 1.0
    gamma_r = 0.60
    eps = 1e-6
    M_cap_arr = np.array([5.0, 0.5, 0.5])
    
    for k in range(N):
        lam_eff = float(np.min(lam))
        x_pred = f_nonlin(x_curr)
        F_k = F_jacobian(x_curr)
        P_pred = (1.0 / lam_eff) * (F_k @ P_curr @ F_k.T + Q_block)
        
        H_k = H_jacobian(x_pred)
        h_pred = h_nonlin(x_pred)
        nu = (y_nl[:, k:k+1] - h_pred).ravel()
        nu[1] = (nu[1] + np.pi) % (2 * np.pi) - np.pi
        nu[2] = (nu[2] + np.pi) % (2 * np.pi) - np.pi
        
        for j in range(m_dim):
            nu_j_cap = np.sign(nu[j]) * min(abs(nu[j]), M_cap_arr[j])
            mu2[j] = lam[j] * mu2[j] + (1.0 - lam[j]) * (nu_j_cap**2)
            mu4[j] = (lam[j]**3) * mu4[j] + (1.0 - (lam[j]**3)) * (nu_j_cap**4)
            kappa_j = mu4[j] / (mu2[j]**2 + eps)
            
            lam[j] = lam_min + (lam_max - lam_min) / (1.0 + np.exp(beta * (kappa_j - kappa_th)))
            
            if kappa_j > kappa_th:
                R_diag[j] = (sigmas_nom[j]**2) * np.exp(alpha * min(kappa_j - kappa_th, 20.0))
                k_star[j] = k
            else:
                R_diag[j] = (sigmas_nom[j]**2) + (R_diag[j] - (sigmas_nom[j]**2)) * np.exp(-gamma_r * (k - k_star[j]))
                
        R_k = np.diag(R_diag)
        S = H_k @ P_pred @ H_k.T + R_k
        K = P_pred @ H_k.T @ la.inv(S)
        x_curr = x_pred + K @ nu.reshape(3, 1)
        IKH = np.eye(6) - K @ H_k
        P_curr = IKH @ P_pred @ IKH.T + K @ R_k @ K.T
        x_erk[:, k] = x_curr.ravel()

    # Metrics
    pos_err_ekf = np.linalg.norm(x_true[:3, :] - x_ekf[:3, :], axis=0)
    pos_err_ehub = np.linalg.norm(x_true[:3, :] - x_ehub[:3, :], axis=0)
    pos_err_erk = np.linalg.norm(x_true[:3, :] - x_erk[:3, :], axis=0)
    
    rmse_ekf = np.sqrt(np.mean(pos_err_ekf**2))
    rmse_ehub = np.sqrt(np.mean(pos_err_ehub**2))
    rmse_erk = np.sqrt(np.mean(pos_err_erk**2))
    
    pk_ekf = np.max(pos_err_ekf)
    pk_ehub = np.max(pos_err_ehub)
    pk_erk = np.max(pos_err_erk)
    
    print("\n" + "="*70)
    print(">>> RESULTS: NONLINEAR DRONE TRACKING (Aerodynamic Drag + Spherical Radar)")
    print("="*70)
    print(f"{'Estimator':<25} | {'3D Pos RMSE':<15} | {'3D Peak Err':<15}")
    print("-"*70)
    print(f"{'Standard EKF':<25} | {rmse_ekf:<15.4f} | {pk_ekf:<15.4f}")
    print(f"{'Huber-Robust EKF':<25} | {rmse_ehub:<15.4f} | {pk_ehub:<15.4f}")
    print(f"{'Extended RKAKF (ERK-AKF)':<25} | {rmse_erk:<15.4f} | {pk_erk:<15.4f}")
    print("="*70)
    print(f"ERK-AKF Improvement over Standard EKF : {(rmse_ekf - rmse_erk)/rmse_ekf * 100:.2f}%")
    print(f"ERK-AKF Improvement over Huber EKF    : {(rmse_ehub - rmse_erk)/rmse_ehub * 100:.2f}%")

if __name__ == '__main__':
    run_multivariate_6d_drone()
    run_nonlinear_drone()
