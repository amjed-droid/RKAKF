function x_est = run_Huber_1D(F, H, Q, R, y, N)
    % Huber Robust Kalman Filter - 1D Version
    c_hub = 1.345;
    x_est = zeros(1, N);
    x = 0; P = 1.0;
    for k = 1:N
        x_pred = F * x;
        P_pred = F * P * F' + Q;
        S  = H * P_pred * H' + R;
        nu = y(k) - H * x_pred;
        z  = nu / sqrt(S);
        if abs(z) <= c_hub
            psi = z;
        else
            psi = c_hub * sign(z);
        end
        K = P_pred * H' / S;
        x = x_pred + K * sqrt(S) * psi;
        P = (1 - K*H) * P_pred;
        x_est(k) = x;
    end
end
