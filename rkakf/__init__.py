"""
RKAKF: Recursive Kurtosis-Aware Kalman Filter
=============================================
A Python library for higher-order statistical Kalman filtering, featuring the
Cubic Decay Principle (lambda^3) to resolve the Kurtosis Ratio Paradox and
bypass Structural Blindness in cyber-physical systems.

Authors:
    Ahmed Sattar Jabbar (ahmed.state.me@gmail.com)
    Haifa Taha Abd Ahmed (haefaa_adm@uomustansiriyah.edu.iq)
Department of Statistics, Mustansiriyah University, Baghdad, Iraq.

Reference:
    "A Cubic-Decay Recursive Kurtosis-Aware Kalman Filter (RKAKF):
     Resolving the Kurtosis Ratio Paradox to Bypass Structural Blindness",
    Submitted to Signal Processing (Elsevier), 2026.
"""

__version__ = "1.0.0"
__author__ = "Ahmed Sattar Jabbar & Haifa Taha Abd Ahmed"
__license__ = "MIT"

from .filters import (
    RKAKF,
    StandardKalmanFilter,
    HuberKalmanFilter,
    VBStudentTKalmanFilter,
    ParticleFilter,
)

from .distributions import (
    generate_CLG,
    cauchy_rnd_trunc,
    laplace_rnd,
)

from .metrics import (
    compute_rmse,
    compute_mae,
    compute_peak_error,
    compute_ttr,
    run_wilcoxon_test,
)

__all__ = [
    "RKAKF",
    "StandardKalmanFilter",
    "HuberKalmanFilter",
    "VBStudentTKalmanFilter",
    "ParticleFilter",
    "generate_CLG",
    "cauchy_rnd_trunc",
    "laplace_rnd",
    "compute_rmse",
    "compute_mae",
    "compute_peak_error",
    "compute_ttr",
    "run_wilcoxon_test",
]
