% exp02_kf_vs_rkakf.m
% EXP 2 — Standard KF vs RKAKF (Localized Burst)

fprintf('\n[EXP 2] KF vs RKAKF...\n');
load_parameters;
rng(42);

N2 = 1000;
x_true2 = zeros(1, N2);
for k = 2:N2
    x_true2(k) = F*x_true2(k-1) + sqrt(Q)*randn();
end
y2 = H*x_true2 + sqrt(R_nom)*randn(1,N2);
atk2 = 400:500;
for k = atk2
    y2(k) = y2(k) + generate_CLG(500, C, gamma_c, sigma, b_lap, p1, p2, 1);
end

xKF2   = run_KF_1D(F, H, Q, R_nom, y2, N2);
[xRK2, kappa2, Rmod2] = run_RKAKF_1D(F, H, Q, R_nom, y2, N2, ...
    kappa_th, lambda_min, lambda_max, beta_sig, alpha_gain, epsilon, M_cap);

rmse_kf2 = compute_rmse(x_true2, xKF2);
rmse_rk2 = compute_rmse(x_true2, xRK2);
fprintf('  RMSE Standard KF : %.4f\n', rmse_kf2);
fprintf('  RMSE RKAKF       : %.4f\n', rmse_rk2);

fig2 = figure('Name','EXP2','Position',[100 100 900 500]);
subplot(2,1,1);
plot(1:N2, x_true2,'k-','LineWidth',1.5); hold on;
plot(1:N2, xKF2,'r--','LineWidth',1.2);
plot(1:N2, xRK2,'g-','LineWidth',1.5);
xregion(atk2(1), atk2(end),'FaceColor',[1 0.8 0.8],'FaceAlpha',0.3);
legend('True','Standard KF','RKAKF','Attack'); ylabel('State');
title(sprintf('EXP 2: KF RMSE=%.4f | RKAKF RMSE=%.4f', rmse_kf2, rmse_rk2)); grid on;
subplot(2,1,2);
plot(1:N2, kappa2,'m-','LineWidth',1.2); hold on;
yline(kappa_th,'r--','\kappa_{th}','LineWidth',1.5);
xregion(atk2(1), atk2(end),'FaceColor',[1 0.8 0.8],'FaceAlpha',0.3);
ylabel('\kappa_k'); xlabel('Time Step'); title('Recursive Kurtosis'); grid on;
save_figure_silent(fig2, fullfile(out_dir,'EXP2_Standard_vs_RKAKF.png'));

save('results_exp2.mat', 'rmse_kf2', 'rmse_rk2', 'x_true2', 'xKF2', 'xRK2', 'kappa2', 'atk2');
