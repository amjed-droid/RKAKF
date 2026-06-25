function [x_est, kappa_hist, R_hist] = run_RKAKF_1D(...
    F, H, Q, R_nom, y, N, ...
    kappa_th, lam_min, lam_max, beta, alpha, eps, M_cap)
    % Recursive Kurtosis-Aware Kalman Filter (RKAKF) - 1D Version

    x_est      = zeros(1, N);
    kappa_hist = zeros(1, N);
    R_hist     = zeros(1, N);

    x = 0; P = 1.0;
    mu2 = 1.0; mu4 = 3.0;
    lam = lam_max; R_k = R_nom;
    k_star = 0;

    for k = 1:N
        % Prediction (adaptive forgetting factor)
        x_pred = F * x;
        P_pred = (1/lam) * (F * P * F' + Q);

        % Innovation with capping
        nu     = y(k) - H * x_pred;
        nu_cap = sign(nu) * min(abs(nu), M_cap);

        % Recursive moment update
        mu2 = lam   * mu2 + (1 - lam)   * nu_cap^2;
        mu4 = lam^3 * mu4 + (1 - lam^3) * nu_cap^4;

        % Recursive kurtosis
        kappa = mu4 / (mu2^2 + eps);

        % Adaptive forgetting factor (sigmoid)
        lam = lam_min + (lam_max - lam_min) / ...
            (1 + exp(beta * (kappa - kappa_th)));

        % Gain revocation or exponential recovery
        if kappa > kappa_th
            R_k    = R_nom * exp(alpha * min(max(kappa - kappa_th, 0), 20));
            k_star = k;
        else
            R_k = R_nom + (R_k - R_nom) * exp(-0.6 * (k - k_star));
        end

        % Kalman update
        S = H * P_pred * H' + R_k;
        K = P_pred * H' / S;
        x = x_pred + K * (y(k) - H * x_pred);
        P = (1 - K*H) * P_pred;

        x_est(k)      = x;
        kappa_hist(k) = kappa;
        R_hist(k)     = R_k;
    end
end
