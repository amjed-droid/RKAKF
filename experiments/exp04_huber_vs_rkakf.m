% exp04_huber_vs_rkakf.m
% EXP 4 — Huber KF vs RKAKF (Continuous CLG)

fprintf('\n[EXP 4] Huber KF vs RKAKF (Continuous CLG)...\n');
load_parameters;
rng(42);

N4 = 800;
x_true4 = zeros(1, N4);
for k = 2:N4, x_true4(k) = F*x_true4(k-1) + sqrt(Q)*randn(); end
% Continuous CLG — not localized burst
y4 = H*x_true4 + sqrt(R_nom)*randn(1,N4);
for k = 1:N4
    y4(k) = y4(k) + generate_CLG(400, C, gamma_c, sigma, b_lap, p1, p2, 1);
end

xHUB4 = run_Huber_1D(F, H, Q, R_nom, y4, N4);
[xRK4, kappa4, ~] = run_RKAKF_1D(F, H, Q, R_nom, y4, N4, ...
    kappa_th, lambda_min, lambda_max, beta_sig, alpha_gain, epsilon, M_cap);

rmse_hub4 = compute_rmse(x_true4, xHUB4);
rmse_rk4  = compute_rmse(x_true4, xRK4);
fprintf('  RMSE Huber KF : %.4f\n', rmse_hub4);
fprintf('  RMSE RKAKF    : %.4f\n', rmse_rk4);
fprintf('  (Expected: Huber wins under continuous CLG noise)\n');

% Quality check: under continuous CLG, Huber should win (or be very close)
if rmse_hub4 >= rmse_rk4
    warning('RKAKF beat Huber under continuous CLG — check parameters');
end

fig4 = figure('Name','EXP4','Position',[100 100 900 500]);
subplot(2,1,1);
plot(1:N4, x_true4,'k-','LineWidth',1.5); hold on;
plot(1:N4, xHUB4,'b-.','LineWidth',1.2);
plot(1:N4, xRK4,'g-','LineWidth',1.5);
legend('True',sprintf('Huber KF (RMSE=%.3f)',rmse_hub4), ...
    sprintf('RKAKF (RMSE=%.3f)',rmse_rk4),'Location','best');
ylabel('State'); title('EXP 4: Huber vs RKAKF — Continuous CLG'); grid on;
subplot(2,1,2);
plot(1:N4, kappa4,'m-','LineWidth',1.2); hold on;
yline(kappa_th,'r--','\kappa_{th}','LineWidth',1.5);
ylabel('\kappa_k'); xlabel('Time Step'); title('Recursive Kurtosis'); grid on;
save_figure_silent(fig4, fullfile(out_dir,'EXP4_Huber_vs_RKAKF.png'));

save('results_exp4.mat', 'rmse_hub4', 'rmse_rk4', 'x_true4', 'xHUB4', 'xRK4', 'kappa4');
