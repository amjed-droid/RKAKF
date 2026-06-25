function x_est = run_VB_StudentT(F, H, Q, R_nom, y, N)
    % Variational Bayes Student's t-Based Robust Kalman Filter (Huang et al. 2017)
    nu_dof   = 4;     % degrees of freedom
    max_iter = 10;    % VB iterations per step

    x_est = zeros(1, N);
    x = 0; P = 1.0;

    for k = 1:N
        x_pred = F * x;
        P_pred = F * P * F' + Q;

        % VB iterations
        tau_k = 1.0;
        x_upd = x_pred; K_last = 0;
        for iter = 1:max_iter
            R_vb  = R_nom / tau_k;
            S     = H * P_pred * H' + R_vb;
            K_last = P_pred * H' / S;
            x_upd = x_pred + K_last * (y(k) - H * x_pred);
            nu_res = y(k) - H * x_upd;
            tau_k  = (nu_dof + 1) / ...
                (nu_dof + nu_res^2/R_nom + H*P_pred*H'/R_nom);
        end

        P = (1 - K_last*H) * P_pred;
        x = x_upd;
        x_est(k) = x;
    end
end
