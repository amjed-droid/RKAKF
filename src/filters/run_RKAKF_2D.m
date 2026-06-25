function [x_est, kappa_hist, R_hist] = run_RKAKF_2D(...
    F, H, Q, R_nom, y, N, ...
    kappa_th, lam_min, lam_max, beta, alpha, eps, M_cap)
    % Recursive Kurtosis-Aware Kalman Filter (RKAKF) - 2D Version

    x_est      = zeros(2, N);
    kappa_hist = zeros(1, N);
    R_hist     = zeros(1, N);

    x = zeros(2,1); P = eye(2);
    mu2 = 1.0; mu4 = 3.0;
    lam = lam_max; R_k = R_nom;
    k_star = 0;

    for k = 1:N
        x_pred = F * x;
        P_pred = (1/lam) * (F * P * F' + Q);

        nu     = y(k) - H * x_pred;   % scalar
        nu_cap = sign(nu) * min(abs(nu), M_cap);

        mu2 = lam   * mu2 + (1 - lam)   * nu_cap^2;
        mu4 = lam^3 * mu4 + (1 - lam^3) * nu_cap^4;

        kappa = mu4 / (mu2^2 + eps);

        lam = lam_min + (lam_max - lam_min) / ...
            (1 + exp(beta * (kappa - kappa_th)));

        if kappa > kappa_th
            R_k    = R_nom * exp(alpha * min(max(kappa - kappa_th, 0), 20));
            k_star = k;
        else
            R_k = R_nom + (R_k - R_nom) * exp(-0.6 * (k - k_star));
        end

        S = H * P_pred * H' + R_k;
        K = P_pred * H' / S;
        x = x_pred + K * (y(k) - H * x_pred);
        P = (1 - K*H) * P_pred;

        x_est(:,k)    = x;
        kappa_hist(k) = kappa;
        R_hist(k)     = R_k;
    end
end
