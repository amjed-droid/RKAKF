% exp06_drone_trajectory.m
% EXP 6 — Synthetic FPV Drone Trajectory

fprintf('\n[EXP 6] Synthetic FPV Drone Trajectory...\n');
load_parameters;
rng(42);

N6   = 500;
t6   = (0:N6-1)*dt;
x_true6 = 3*sin(0.06*2*pi*t6) + 2*cos(0.10*2*pi*t6);
y6 = x_true6 + 0.1*randn(1,N6);
% Sparse CLG burst at specified steps
burst6 = [242,245,249,253,257,260];
burst_M6 = 400;
for k = burst6
    y6(k) = y6(k) + generate_CLG(burst_M6, C, gamma_c, sigma, b_lap, p1, p2, 1);
end

xKF6    = run_KF_1D(F, H, Q, R_nom, y6, N6);
xHI6    = run_KF_1D(F, H, Q, R_nom*1.5, y6, N6);   % H-infinity via inflated R
xHUB6   = run_Huber_1D(F, H, Q, R_nom, y6, N6);
[xRK6, ~, ~] = run_RKAKF_1D(F, H, Q, R_nom, y6, N6, ...
    kappa_th, lambda_min, lambda_max, beta_sig, alpha_gain, epsilon, M_cap);

rmse_kf6  = compute_rmse(x_true6, xKF6);
rmse_hi6  = compute_rmse(x_true6, xHI6);
rmse_hub6 = compute_rmse(x_true6, xHUB6);
rmse_rk6  = compute_rmse(x_true6, xRK6);
pk_kf6    = max(abs(x_true6 - xKF6));
pk_hi6    = max(abs(x_true6 - xHI6));
pk_hub6   = max(abs(x_true6 - xHUB6));
pk_rk6    = max(abs(x_true6 - xRK6));

fprintf('  Standard KF  : RMSE=%.4f | Peak Error=%.4f\n', rmse_kf6,  pk_kf6);
fprintf('  H-Infinity   : RMSE=%.4f | Peak Error=%.4f\n', rmse_hi6,  pk_hi6);
fprintf('  Huber KF     : RMSE=%.4f | Peak Error=%.4f\n', rmse_hub6, pk_hub6);
fprintf('  RKAKF        : RMSE=%.4f | Peak Error=%.4f\n', rmse_rk6,  pk_rk6);

fig6 = figure('Name','EXP6','Position',[100 100 1000 800]);
subplot(2,1,1);
plot(t6, x_true6,'k-','LineWidth',1.5); hold on;
plot(t6, xKF6,'r--','LineWidth',1.0);
plot(t6, xHI6,'c-.','LineWidth',1.0);
plot(t6, xHUB6,'b-.','LineWidth',1.2);
plot(t6, xRK6,'g-','LineWidth',1.5);
for bk = burst6, xline(bk*dt,'r:','LineWidth',0.8); end
legend('True','Standard KF','H-Infinity','Huber KF','RKAKF','Location','best');
ylabel('State (Altitude)');
title('EXP 6: Synthetic FPV Drone Trajectory — Sparse CLG Burst'); grid on;
subplot(2,1,2);
bar([rmse_kf6, rmse_hi6, rmse_hub6, rmse_rk6]);
set(gca,'XTickLabel',{'Standard KF','H-Infinity','Huber KF','RKAKF'});
ylabel('RMSE'); title('RMSE Comparison (Drone)'); grid on;
save_figure_silent(fig6, fullfile(out_dir,'EXP6_UZH_Drone.png'));

save('results_exp6.mat', 'rmse_kf6', 'rmse_hi6', 'rmse_hub6', 'rmse_rk6', ...
    'pk_kf6', 'pk_hi6', 'pk_hub6', 'pk_rk6', 't6', 'x_true6', 'xKF6', 'xHI6', 'xHUB6', 'xRK6', 'burst6');
