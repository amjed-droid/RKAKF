% exp05_monte_carlo.m
% EXP 5 — Monte Carlo (1000 runs, Continuous CLG)

fprintf('\n[EXP 5] Monte Carlo (N_mc=1000, Continuous CLG)...\n');
load_parameters;
rng(42);

N_mc5   = 1000;
N_step5 = 200;
M5      = 200;

rmse_kf5  = zeros(1,N_mc5); rmse_hub5 = zeros(1,N_mc5); rmse_rk5 = zeros(1,N_mc5);
mae_kf5   = zeros(1,N_mc5); mae_hub5  = zeros(1,N_mc5); mae_rk5  = zeros(1,N_mc5);
maxe_kf5  = zeros(1,N_mc5); maxe_hub5 = zeros(1,N_mc5); maxe_rk5 = zeros(1,N_mc5);

parfor mc = 1:N_mc5
    xt = zeros(1,N_step5); xt(1) = 0;
    for k = 2:N_step5, xt(k) = F*xt(k-1) + sqrt(Q)*randn(); end
    ym = H*xt + sqrt(R_nom)*randn(1,N_step5);
    for k = 1:N_step5
        ym(k) = ym(k) + generate_CLG(M5, C, gamma_c, sigma, b_lap, p1, p2, 1);
    end
    xk = run_KF_1D(F, H, Q, R_nom, ym, N_step5);
    xh = run_Huber_1D(F, H, Q, R_nom, ym, N_step5);
    xr = run_RKAKF_1D(F, H, Q, R_nom, ym, N_step5, ...
        kappa_th, lambda_min, lambda_max, beta_sig, alpha_gain, epsilon, M_cap);
    rmse_kf5(mc)  = compute_rmse(xt, xk);
    rmse_hub5(mc) = compute_rmse(xt, xh);
    rmse_rk5(mc)  = compute_rmse(xt, xr);
    mae_kf5(mc)   = mean(abs(xt - xk));
    mae_hub5(mc)  = mean(abs(xt - xh));
    mae_rk5(mc)   = mean(abs(xt - xr));
    maxe_kf5(mc)  = max(abs(xt - xk));
    maxe_hub5(mc) = max(abs(xt - xh));
    maxe_rk5(mc)  = max(abs(xt - xr));
end

pval_rk_kf  = signrank(rmse_rk5, rmse_kf5);
pval_rk_hub = signrank(rmse_rk5, rmse_hub5);

fprintf('  Standard KF : Mean RMSE=%.4f | MAE=%.4f | MaxErr=%.4f\n', ...
    mean(rmse_kf5), mean(mae_kf5), mean(maxe_kf5));
fprintf('  Huber KF    : Mean RMSE=%.4f | MAE=%.4f | MaxErr=%.4f\n', ...
    mean(rmse_hub5), mean(mae_hub5), mean(maxe_hub5));
fprintf('  RKAKF       : Mean RMSE=%.4f | MAE=%.4f | MaxErr=%.4f\n', ...
    mean(rmse_rk5), mean(mae_rk5), mean(maxe_rk5));
fprintf('  p-value (RKAKF vs KF)  : %.2e\n', pval_rk_kf);
fprintf('  p-value (RKAKF vs Hub) : %.2e\n', pval_rk_hub);

fig5 = figure('Name','EXP5','Position',[100 100 1000 800]);
subplot(3,1,1);
all_e = [rmse_kf5, rmse_hub5, rmse_rk5];
edges5 = linspace(min(all_e)*0.95, max(all_e)*1.05, 40);
histogram(rmse_kf5, edges5,'FaceColor',[1 0.5 0.5],'FaceAlpha',0.7); hold on;
histogram(rmse_hub5,edges5,'FaceColor',[0.5 1 0.5],'FaceAlpha',0.7);
histogram(rmse_rk5, edges5,'FaceColor',[0.5 0.5 1],'FaceAlpha',0.7);
legend('Standard KF','Huber KF','RKAKF'); xlabel('RMSE'); ylabel('Count');
title(sprintf('EXP 5 Monte Carlo RMSE Distribution (%d runs)', N_mc5)); grid on;
subplot(3,1,2);
boxplot([rmse_kf5', rmse_hub5', rmse_rk5'], 'Labels',{'Standard KF','Huber KF','RKAKF'});
ylabel('RMSE'); title('RMSE Box Plot'); grid on;
subplot(3,1,3);
bar([mean(rmse_kf5), mean(rmse_hub5), mean(rmse_rk5); ...
     mean(mae_kf5),  mean(mae_hub5),  mean(mae_rk5)]);
set(gca,'XTickLabel',{'RMSE','MAE'});
legend('Standard KF','Huber KF','RKAKF'); ylabel('Error');
title('Mean Error Metrics'); grid on;
save_figure_silent(fig5, fullfile(out_dir,'EXP5_Monte_Carlo.png'));

save('results_exp5.mat', 'rmse_kf5', 'rmse_hub5', 'rmse_rk5', ...
    'mae_kf5', 'mae_hub5', 'mae_rk5', ...
    'maxe_kf5', 'maxe_hub5', 'maxe_rk5', ...
    'pval_rk_kf', 'pval_rk_hub', 'N_mc5', 'M5');
