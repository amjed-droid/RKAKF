function x_est = run_KF_1D(F, H, Q, R, y, N)
    % Standard 1D Kalman Filter
    x_est = zeros(1, N);
    x = 0; P = 1.0;
    for k = 1:N
        x_pred = F * x;
        P_pred = F * P * F' + Q;
        K = P_pred * H' / (H * P_pred * H' + R);
        x = x_pred + K * (y(k) - H * x_pred);
        P = (1 - K*H) * P_pred;
        x_est(k) = x;
    end
end
