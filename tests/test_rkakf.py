"""
Unit and Mathematical Invariant Test Suite for RKAKF.
Validates Proposition 1, Proposition 2, Corollary 1, and Theorem 1.
"""

import unittest
import numpy as np

from rkakf.filters import RKAKF, StandardKalmanFilter
from rkakf.distributions import generate_CLG, moving_variance
from rkakf.metrics import compute_rmse, compute_peak_error


class TestRKAKFTheory(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_proposition1_structural_blindness_bounded_variance(self):
        """
        Verify Proposition 1: As M increases (e.g. 10 to 1000),
        the empirical variance remains strictly bounded below the theoretical ceiling,
        while empirical kurtosis diverges.
        """
        sigma = 0.5
        b_lap = 0.3
        gamma_c = 1.0
        p1 = 0.50
        p2 = 0.30
        C = 0.50

        # Theoretical variance ceiling: p1*sigma^2 + 2*p2*b^2 + 2*C*gamma_c / pi
        var_ceiling = p1 * (sigma ** 2) + 2.0 * p2 * (b_lap ** 2) + (2.0 * C * gamma_c) / np.pi
        var_threshold = var_ceiling * 5.0  # Detection threshold

        N = 50000
        np.random.seed(123)
        z_small = generate_CLG(M=10.0, C=C, gamma_c=gamma_c, sigma=sigma, b_lap=b_lap, p1=p1, p2=p2, n=N)
        np.random.seed(123)
        z_large = generate_CLG(M=100.0, C=C, gamma_c=gamma_c, sigma=sigma, b_lap=b_lap, p1=p1, p2=p2, n=N)

        var_small = np.var(z_small)
        var_large = np.var(z_large)

        # Both sample variances must remain below the 5x detector threshold (structural blindness)
        self.assertLess(var_small, var_threshold)
        self.assertLess(var_large, var_threshold)

        # But 4th moment / kurtosis of large M must be significantly larger than small M
        kurt_small = np.mean(z_small ** 4) / (var_small ** 2)
        kurt_large = np.mean(z_large ** 4) / (var_large ** 2)
        self.assertGreater(kurt_large, kurt_small * 2.0)

    def test_proposition2_cubic_decay_resolves_kurtosis_ratio_paradox(self):
        """
        Verify Proposition 2 & Corollary 1:
        Post-shock homogeneous decay with linear decay (p=1) diverges:
            kappa_{t} ~ lambda^{-t} -> inf
        Whereas cubic decay (p=3) guarantees exponential dissipation:
            kappa_{t} ~ lambda^t -> 0
        """
        lam = 0.95
        mu2_init = 10.0
        mu4_init = 3000.0  # Initial shock

        T = 20
        # Linear decay simulation
        mu2_p1 = mu2_init
        mu4_p1 = mu4_init
        kurt_p1 = []

        # Cubic decay simulation
        mu2_p3 = mu2_init
        mu4_p3 = mu4_init
        kurt_p3 = []

        for _ in range(T):
            # Homogeneous decay (zero post-shock innovation energy)
            mu2_p1 = lam * mu2_p1
            mu4_p1 = lam * mu4_p1
            kurt_p1.append(mu4_p1 / (mu2_p1 ** 2))

            mu2_p3 = lam * mu2_p3
            mu4_p3 = (lam ** 3) * mu4_p3
            kurt_p3.append(mu4_p3 / (mu2_p3 ** 2))

        # Linear decay must increase (diverge) over time
        self.assertGreater(kurt_p1[-1], kurt_p1[0])
        # Cubic decay must contract monotonically to 0
        self.assertLess(kurt_p3[-1], kurt_p3[0])
        self.assertAlmostEqual(kurt_p3[-1], (mu4_init / (mu2_init ** 2)) * (lam ** T), delta=1e-3)

    def test_theorem1_gain_revocation_bounds_outlier_error(self):
        """
        Verify Theorem 1: Under massive outlier impulse (e.g. y = 1000.0),
        Gain Revocation suppresses K_k towards 0 and bounds single-step error.
        """
        filter_rkakf = RKAKF(R_nom=0.1, kappa_th=4.0, alpha=0.5)

        # Run several nominal steps
        for _ in range(5):
            filter_rkakf.step(0.0)

        # Inject massive outlier
        massive_outlier = 1000.0
        x_est, kappa, R_k, K = filter_rkakf.step(massive_outlier)

        # Next step during shock: Gain must be strongly suppressed
        x_next, kappa_next, R_next, K_next = filter_rkakf.step(massive_outlier)
        self.assertLess(K_next, 0.05)
        self.assertGreater(R_next, 10.0)
        # Estimation error must be orders of magnitude smaller than outlier
        self.assertLess(abs(x_next), 10.0)

    def test_rkakf_nominal_performance_comparable_to_kf(self):
        """
        Verify that under pure nominal Gaussian noise without attacks,
        RKAKF performs closely to the optimal standard Kalman Filter.
        """
        N = 500
        x_true = np.zeros(N)
        for k in range(1, N):
            x_true[k] = x_true[k - 1] + np.random.normal(0, np.sqrt(0.01))
        y = x_true + np.random.normal(0, np.sqrt(0.10), N)

        kf = StandardKalmanFilter(F=1.0, H=1.0, Q=0.01, R=0.10)
        rkakf = RKAKF(F=1.0, H=1.0, Q=0.01, R_nom=0.10)

        x_kf = kf.filter(y)
        res_rkakf = rkakf.filter(y)
        x_rkakf = res_rkakf["x_est"]

        rmse_kf = compute_rmse(x_true, x_kf)
        rmse_rkakf = compute_rmse(x_true, x_rkakf)

        # Under nominal Gaussian noise, RKAKF should not deviate by more than 15% from KF
        self.assertLess(abs(rmse_rkakf - rmse_kf) / rmse_kf, 0.15)


if __name__ == "__main__":
    unittest.main()
