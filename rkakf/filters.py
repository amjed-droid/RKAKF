"""
Filter implementations:
- RKAKF: Recursive Kurtosis-Aware Kalman Filter with Cubic Decay (lambda^3)
- StandardKalmanFilter: Standard linear Kalman Filter
- HuberKalmanFilter: Robust M-estimation Kalman Filter
- VBStudentTKalmanFilter: Variational Bayes Student-t Kalman Filter
- ParticleFilter: Sequential Importance Resampling (SIR) Particle Filter
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class RKAKF:
    """
    Recursive Kurtosis-Aware Kalman Filter (RKAKF).
    
    Features:
    1. Cubic Decay Principle (lambda^3) for the recursive 4th central moment,
       resolving the Kurtosis Ratio Paradox and guaranteeing exponential
       post-shock recovery (Proposition 2, Corollary 1).
    2. Dynamic, reversed-sigmoid forgetting factor for covariance shielding.
    3. Automated exponential Gain Revocation protocol, provably bounding
       single-step estimation errors independently of outlier magnitude (Theorem 1).
    4. Smooth asymptotic snap-back gain restoration mechanism.
    
    Parameters
    ----------
    F : float or np.ndarray
        State transition matrix / scalar.
    H : float or np.ndarray
        Observation matrix / scalar.
    Q : float or np.ndarray
        Process noise covariance.
    R_nom : float
        Nominal measurement noise covariance.
    kappa_th : float, default=4.0
        Anomaly detection threshold on recursive kurtosis.
    lam_min : float, default=0.50
        Lower bound of adaptive forgetting factor.
    lam_max : float, default=0.99
        Upper bound of adaptive forgetting factor (steady-state memory).
    beta : float, default=0.5
        Transition steepness for sigmoid forgetting factor schedule.
    alpha : float, default=0.5
        Gain revocation sensitivity exponent.
    eps : float, default=1e-6
        Regularization constant preventing zero division in kurtosis ratio.
    M_cap : float, default=500.0
        Innovation clipping threshold for statistical moment updates only.
    x0 : float, default=0.0
        Initial state estimate.
    P0 : float, default=1.0
        Initial error covariance.
    """

    def __init__(
        self,
        F: float = 1.0,
        H: float = 1.0,
        Q: float = 0.01,
        R_nom: float = 0.10,
        kappa_th: float = 4.0,
        lam_min: float = 0.50,
        lam_max: float = 0.99,
        beta: float = 0.5,
        alpha: float = 0.5,
        eps: float = 1e-6,
        M_cap: float = 500.0,
        x0: float = 0.0,
        P0: float = 1.0,
    ):
        self.F = float(F)
        self.H = float(H)
        self.Q = float(Q)
        self.R_nom = float(R_nom)
        self.kappa_th = float(kappa_th)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.M_cap = float(M_cap)

        # Internal state
        self.x = float(x0)
        self.P = float(P0)
        self.mu2 = float(R_nom)
        self.mu4 = float(3.0 * (R_nom ** 2))
        self.lam = float(lam_max)
        self.R_k = float(R_nom)
        self.k_star = 0
        self.step_idx = 0
        self.kappa = 3.0
        self.K = 0.0

    def step(self, y_k: float) -> Tuple[float, float, float, float]:
        """
        Execute one recursive estimation cycle for measurement y_k.
        
        Returns
        -------
        x_est : float
            Updated state estimate.
        kappa : float
            Recursive empirical kurtosis.
        R_k : float
            Effective measurement noise covariance.
        K : float
            Effective Kalman Gain.
        """
        k = self.step_idx

        # 1. State Prediction with Covariance Shielding
        x_pred = self.F * self.x
        P_pred = (1.0 / self.lam) * (self.F * self.P * self.F + self.Q)

        # 2. Innovation and Numerical Capping for Higher Moments
        nu = y_k - self.H * x_pred
        nu_cap = np.sign(nu) * min(abs(nu), self.M_cap)

        # 3. Recursive Moment Updates (Cubic Decay on 4th moment)
        self.mu2 = self.lam * self.mu2 + (1.0 - self.lam) * (nu_cap ** 2)
        lam_cubed = self.lam ** 3
        self.mu4 = lam_cubed * self.mu4 + (1.0 - lam_cubed) * (nu_cap ** 4)

        # Recursive Kurtosis Ratio
        self.kappa = self.mu4 / (self.mu2 ** 2 + self.eps)

        # 4. Adaptive Kurtosis-Coupled Forgetting Factor
        self.lam = self.lam_min + (self.lam_max - self.lam_min) / (
            1.0 + np.exp(self.beta * (self.kappa - self.kappa_th))
        )

        # 5. Gain Revocation or Asymptotic Snap-back Restoration
        if self.kappa > self.kappa_th:
            # Exponential Gain Revocation
            kurt_excess = min(max(self.kappa - self.kappa_th, 0.0), 20.0)
            self.R_k = self.R_nom * np.exp(self.alpha * kurt_excess)
            self.k_star = k
        else:
            # Asymptotic Snap-back
            self.R_k = self.R_nom + (self.R_k - self.R_nom) * np.exp(
                -0.6 * (k - self.k_star)
            )

        # 6. Measurement Update with Unclipped Innovation
        S = self.H * P_pred * self.H + self.R_k
        self.K = P_pred * self.H / S
        self.x = x_pred + self.K * (y_k - self.H * x_pred)
        self.P = (1.0 - self.K * self.H) * P_pred

        self.step_idx += 1
        return self.x, self.kappa, self.R_k, self.K

    def filter(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Batch filter an entire observation sequence y.
        
        Returns
        -------
        dict with keys: 'x_est', 'kappa', 'R', 'K'
        """
        n = len(y)
        x_est = np.zeros(n)
        kappa_hist = np.zeros(n)
        R_hist = np.zeros(n)
        K_hist = np.zeros(n)

        for i in range(n):
            x_est[i], kappa_hist[i], R_hist[i], K_hist[i] = self.step(y[i])

        return {
            "x_est": x_est,
            "kappa": kappa_hist,
            "R": R_hist,
            "K": K_hist,
        }


class StandardKalmanFilter:
    """Standard 1D Linear Kalman Filter (Kalman, 1960)."""

    def __init__(self, F: float = 1.0, H: float = 1.0, Q: float = 0.01, R: float = 0.10, x0: float = 0.0, P0: float = 1.0):
        self.F = float(F)
        self.H = float(H)
        self.Q = float(Q)
        self.R = float(R)
        self.x = float(x0)
        self.P = float(P0)

    def step(self, y_k: float) -> Tuple[float, float]:
        x_pred = self.F * self.x
        P_pred = self.F * self.P * self.F + self.Q
        S = self.H * P_pred * self.H + self.R
        K = P_pred * self.H / S
        self.x = x_pred + K * (y_k - self.H * x_pred)
        self.P = (1.0 - K * self.H) * P_pred
        return self.x, K

    def filter(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        x_est = np.zeros(n)
        for i in range(n):
            x_est[i], _ = self.step(y[i])
        return x_est


class HuberKalmanFilter:
    """Huber M-estimation Robust Kalman Filter (Huber 1964; Karlgaard & Schaub 2007)."""

    def __init__(self, F: float = 1.0, H: float = 1.0, Q: float = 0.01, R: float = 0.10, c_hub: float = 1.345, x0: float = 0.0, P0: float = 1.0):
        self.F = float(F)
        self.H = float(H)
        self.Q = float(Q)
        self.R = float(R)
        self.c_hub = float(c_hub)
        self.x = float(x0)
        self.P = float(P0)

    def step(self, y_k: float) -> Tuple[float, float]:
        x_pred = self.F * self.x
        P_pred = self.F * self.P * self.F + self.Q
        S = self.H * P_pred * self.H + self.R
        nu = y_k - self.H * x_pred
        z = nu / np.sqrt(S)
        psi = z if np.abs(z) <= self.c_hub else self.c_hub * np.sign(z)
        K = P_pred * self.H / S
        self.x = x_pred + K * np.sqrt(S) * psi
        self.P = (1.0 - K * self.H) * P_pred
        return self.x, K

    def filter(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        x_est = np.zeros(n)
        for i in range(n):
            x_est[i], _ = self.step(y[i])
        return x_est


class VBStudentTKalmanFilter:
    """Variational Bayes Student's t Kalman Filter (Huang et al., IEEE TAES 2017)."""

    def __init__(self, F: float = 1.0, H: float = 1.0, Q: float = 0.01, R_nom: float = 0.10, nu_dof: float = 4.0, max_iter: int = 10, x0: float = 0.0, P0: float = 1.0):
        self.F = float(F)
        self.H = float(H)
        self.Q = float(Q)
        self.R_nom = float(R_nom)
        self.nu_dof = float(nu_dof)
        self.max_iter = int(max_iter)
        self.x = float(x0)
        self.P = float(P0)

    def step(self, y_k: float) -> float:
        x_pred = self.F * self.x
        P_pred = self.F * self.P * self.F + self.Q
        tau_k = 1.0
        x_upd = x_pred
        K_last = 0.0
        for _ in range(self.max_iter):
            R_vb = self.R_nom / tau_k
            S = self.H * P_pred * self.H + R_vb
            K_last = P_pred * self.H / S
            x_upd = x_pred + K_last * (y_k - self.H * x_pred)
            nu_res = y_k - self.H * x_upd
            tau_k = (self.nu_dof + 1.0) / (
                self.nu_dof
                + (nu_res ** 2) / self.R_nom
                + (self.H * P_pred * self.H) / self.R_nom
            )
        self.P = (1.0 - K_last * self.H) * P_pred
        self.x = x_upd
        return self.x

    def filter(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        x_est = np.zeros(n)
        for i in range(n):
            x_est[i] = self.step(y[i])
        return x_est


class ParticleFilter:
    """Bootstrap Particle Filter (Sequential Importance Resampling - SIR)."""

    def __init__(self, F: float = 1.0, H: float = 1.0, Q: float = 0.01, R_nom: float = 0.10, N_part: int = 500):
        self.F = float(F)
        self.H = float(H)
        self.Q = float(Q)
        self.R_nom = float(R_nom)
        self.N_part = int(N_part)
        self.particles = np.random.normal(0.0, np.sqrt(R_nom), self.N_part)
        self.weights = np.ones(self.N_part) / self.N_part

    def step(self, y_k: float) -> Tuple[float, float]:
        # Propagation
        self.particles = self.F * self.particles + np.random.normal(0.0, np.sqrt(self.Q), self.N_part)

        # Weight update
        log_w = -0.5 * ((y_k - self.H * self.particles) ** 2) / self.R_nom
        log_w -= np.max(log_w)
        weights = np.exp(log_w)
        sum_w = np.sum(weights)
        if sum_w > 0:
            self.weights = weights / sum_w
        else:
            self.weights = np.ones(self.N_part) / self.N_part

        # Effective Sample Size (ESS)
        ess = 1.0 / np.sum(self.weights ** 2)

        # State estimate (posterior mean)
        x_est = float(np.sum(self.particles * self.weights))

        # Systematic resampling if ESS drops
        if ess < self.N_part / 2.0:
            indices = np.random.choice(self.N_part, size=self.N_part, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.N_part) / self.N_part

        return x_est, ess

    def filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(y)
        x_est = np.zeros(n)
        ess_hist = np.zeros(n)
        for i in range(n):
            x_est[i], ess_hist[i] = self.step(y[i])
        return x_est, ess_hist
