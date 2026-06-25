function [x_est, P_hist] = run_KF_2D(F, H, Q, R, y, N)
    % Standard 2D Kalman Filter
    x_est  = zeros(2, N);
    P_hist = zeros(2, 2, N);
    x = zeros(2,1); P = eye(2);
    for k = 1:N
        x_pred = F * x;
        P_pred = F * P * F' + Q;
        K = P_pred * H' / (H * P_pred * H' + R);
        x = x_pred + K * (y(k) - H * x_pred);
        P = (1 - K*H) * P_pred;
        x_est(:,k)    = x;
        P_hist(:,:,k) = P;
    end
end
