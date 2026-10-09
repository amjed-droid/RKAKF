"""
Statistical noise generators and heavy-tailed mixture samplers.
Includes the dynamically constrained Cauchy-Laplace-Gaussian (CLG) mixture
formalized in Proposition 1 (Structural Blindness).
"""

import numpy as np


def laplace_rnd(mu: float = 0.0, b: float = 1.0, size: int = 1) -> np.ndarray:
    """
    Sample from a zero-mean Laplace distribution L(mu, b).
    
    Parameters
    ----------
    mu : float
        Location parameter (mean).
    b : float
        Scale parameter (diversity).
    size : int
        Number of samples to draw.
        
    Returns
    -------
    np.ndarray
        Array of drawn Laplace samples.
    """
    u = np.random.uniform(-0.5, 0.5, size)
    return mu - b * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))


def cauchy_rnd_trunc(
    x0: float = 0.0, gamma: float = 1.0, M: float = 1000.0, size: int = 1
) -> np.ndarray:
    """
    Sample from a symmetric truncated Cauchy distribution C_trunc(x0, gamma, M)
    on the compact interval [x0 - M, x0 + M].
    
    Parameters
    ----------
    x0 : float
        Location parameter (median).
    gamma : float
        Scale parameter (half-width at half-maximum).
    M : float
        Truncation amplitude threshold.
    size : int
        Number of samples to draw.
        
    Returns
    -------
    np.ndarray
        Array of drawn truncated Cauchy samples.
    """
    samples = np.empty(size)
    for i in range(size):
        while True:
            u = np.random.uniform(0.0, 1.0)
            x = x0 + gamma * np.tan(np.pi * (u - 0.5))
            if np.abs(x - x0) <= M:
                samples[i] = x
                break
    return samples


def generate_CLG(
    M: float,
    C: Optional[float] = 0.5,
    gamma_c: float = 1.0,
    sigma: float = 0.5,
    b_lap: float = 0.3,
    p1: float = 0.5,
    p2: float = 0.3,
    p3: Optional[float] = None,
    n: int = 1,
) -> np.ndarray:
    """
    Compound Cauchy-Laplace-Gaussian (CLG) mixture generator.
    
    If p3 is provided, uses fixed mixture weights (p1, p2, p3).
    Otherwise, uses the dynamic scaling p3(M) = min(C / M, 1 - p1 - p2) from Proposition 1.
    """
    if p3 is None:
        p3_eff = min(float(C) / max(float(M), 1.0), max(0.0, 1.0 - p1 - p2))
    else:
        p3_eff = float(p3)

    p3_eff = max(0.0, min(1.0, p3_eff))
    p1_eff = max(0.0, 1.0 - p2 - p3_eff)

    u = np.random.uniform(0.0, 1.0, n)
    z = np.empty(n)

    mask_c = u < p3_eff
    mask_l = (u >= p3_eff) & (u < p3_eff + p2)
    mask_g = u >= (p3_eff + p2)

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


def moving_variance(x: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Compute moving-window sample variance for energy-based anomaly detection.
    
    Parameters
    ----------
    x : np.ndarray
        Input sequence.
    window_size : int
        Sliding window length.
        
    Returns
    -------
    np.ndarray
        Array of local sample variances.
    """
    n = len(x)
    var_seq = np.zeros(n)
    for i in range(n):
        idx_start = max(0, i - window_size + 1)
        w = x[idx_start : i + 1]
        var_seq[i] = np.var(w) if len(w) > 1 else 0.0
    return var_seq
