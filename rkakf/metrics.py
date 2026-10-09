"""
Statistical performance evaluation metrics for state estimation.
Includes RMSE, MAE, Peak Estimation Error, Time-to-Recovery (TTR),
and non-parametric hypothesis testing (Wilcoxon Signed-Rank Test).
"""

from typing import Tuple, Optional
import numpy as np
import scipy.stats as stats


def compute_rmse(y_true: np.ndarray, y_est: np.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_est) ** 2)))


def compute_mae(y_true: np.ndarray, y_est: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_est)))


def compute_peak_error(y_true: np.ndarray, y_est: np.ndarray) -> float:
    """Compute Peak (Maximum) Absolute Estimation Error."""
    return float(np.max(np.abs(y_true - y_est)))


def compute_ttr(
    errors: np.ndarray,
    baseline_threshold: float,
    shock_index: int,
    consecutive_steps: int = 3,
) -> int:
    """
    Compute Time to Recovery (TTR) following an impulsive attack.
    Defined as the number of time steps post-shock until the absolute error
    returns and stays beneath baseline_threshold for consecutive_steps.

    Parameters
    ----------
    errors : np.ndarray
        Array of absolute estimation errors |x - x_est|.
    baseline_threshold : float
        Error threshold defining recovery to nominal tracking.
    shock_index : int
        Index at which the shock terminated.
    consecutive_steps : int, default=3
        Required consecutive steps below threshold.

    Returns
    -------
    int
        Time-to-recovery in steps, or len(errors) - shock_index if unrecovered.
    """
    n = len(errors)
    for t in range(shock_index, n - consecutive_steps):
        if np.all(errors[t : t + consecutive_steps] <= baseline_threshold):
            return t - shock_index
    return n - shock_index


def run_wilcoxon_test(
    errors_baseline: np.ndarray, errors_proposed: np.ndarray
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test to verify statistical significance
    between the baseline (KF) and proposed (RKAKF) error distributions.

    Returns
    -------
    stat : float
        Test statistic.
    p_value : float
        Two-sided asymptotic p-value.
    """
    diff = errors_baseline - errors_proposed
    # Exclude zero differences
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0, 1.0
    stat, p_value = stats.wilcoxon(diff, alternative="greater")
    return float(stat), float(p_value)
