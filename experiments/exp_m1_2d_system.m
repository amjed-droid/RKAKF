% exp_m1_2d_system.m
% EXP M1 — 2D Kinematic System (Appendix)

fprintf('\n[EXP M1] 2D System (Position + Velocity)...\n');
load_parameters;
rng(42);

N_m1 = 300;
M_m1 = 50;
dt_m1 = dt;
F2  = [1, dt_m1; 0, 1];
H2  = [1, 0];
Q2  = 0.1 * [dt_m1^3/3, dt_m1^2/2; dt_m1^2/2, dt_m1];   % positive definite

x_true_m1 = zeros(2, N_m1);
x_true_m1(:,1) = [0; 1];
for k = 2:N_m1
    x_true_m1(:,k) = F2*x_true_m1(:,k-1) + chol(Q2)'*randn(2,1);
end

y_m1 = H2*x_true_m1 + sqrt(R_nom)*randn(1,N_m1);
atk_m1 = 100:150;
for k = atk_m1
    y_m1(k) = y_m1(k) + generate_CLG(M_m1, C, gamma_c, sigma, b_lap, p1, p2, 1);
end

[xKF_m1, ~]   = run_KF_2D(F2, H2, Q2, R_nom, y_m1, N_m1);
xHUB_m1       = run_Huber_2D(F2, H2, Q2, R_nom, y_m1, N_m1);
[xRK_m1, kappa_m1, ~] = run_RKAKF_2D(F2, H2, Q2, R_nom, y_m1, N_m1, ...
    kappa_th, lambda_min, lambda_max, beta_sig, alpha_gain, epsilon, M_cap);

rmse_kf_m1  = compute_rmse(x_true_m1(1,:), xKF_m1(1,:));
rmse_hub_m1 = compute_rmse(x_true_m1(1,:), xHUB_m1(1,:));
rmse_rk_m1  = compute_rmse(x_true_m1(1,:), xRK_m1(1,:));

impr_hub = (rmse_kf_m1 - rmse_hub_m1)/rmse_kf_m1 * 100;
impr_rk  = (rmse_kf_m1 - rmse_rk_m1) /rmse_kf_m1 * 100;

fprintf('  Standard KF : RMSE=%.4f\n', rmse_kf_m1);
fprintf('  Huber KF    : RMSE=%.4f (%.1f%% improvement)\n', rmse_hub_m1, impr_hub);
fprintf('  RKAKF       : RMSE=%.4f (%.1f%% improvement)\n', rmse_rk_m1, impr_rk);

t_m1 = (1:N_m1)*dt_m1;
fig_m1 = figure('Name','EXP_M1','Position',[100 100 900 500]);
subplot(2,1,1);
plot(t_m1, x_true_m1(1,:),'k-','LineWidth',1.5); hold on;
plot(t_m1, xKF_m1(1,:),'r--','LineWidth',1.0);
plot(t_m1, xHUB_m1(1,:),'b-.','LineWidth',1.0);
plot(t_m1, xRK_m1(1,:),'g-','LineWidth',1.5);
xregion(atk_m1(1)*dt_m1, atk_m1(end)*dt_m1, 'FaceColor',[1 0.8 0.8],'FaceAlpha',0.3);
legend('True','Standard KF','Huber KF','RKAKF'); ylabel('Position');
title('EXP M1: 2D Kinematic System — Position'); grid on;
subplot(2,1,2);
plot(t_m1, kappa_m1,'m-','LineWidth',1.2); hold on;
yline(kappa_th,'r--','\kappa_{th}','LineWidth',1.5);
xregion(atk_m1(1)*dt_m1, atk_m1(end)*dt_m1, 'FaceColor',[1 0.8 0.8],'FaceAlpha',0.3);
ylabel('\kappa_k'); xlabel('Time (s)'); title('Recursive Kurtosis — 2D System'); grid on;
save_figure_silent(fig_m1, fullfile(out_dir,'EXP_M1_2D_System.png'));

save('results_expm1.mat', 'rmse_kf_m1', 'rmse_hub_m1', 'rmse_rk_m1', 'impr_hub', 'impr_rk', ...
    'x_true_m1', 'xKF_m1', 'xHUB_m1', 'xRK_m1', 'kappa_m1', 't_m1', 'atk_m1', 'dt_m1');
