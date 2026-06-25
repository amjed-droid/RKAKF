% exp03_pf_vs_rkakf.m
% EXP 3 — Particle Filter Degeneracy vs RKAKF

fprintf('\n[EXP 3] Particle Filter vs RKAKF...\n');
load_parameters;
rng(42);

N3 = 500; N_part = 500; M3 = 500;
x_true3 = zeros(1, N3);
for k = 2:N3, x_true3(k) = F*x_true3(k-1) + sqrt(Q)*randn(); end
y3 = H*x_true3 + sqrt(R_nom)*randn(1,N3);
% Localized burst — PF degenerates, RKAKF isolates
atk3 = 150:250;
for k = atk3
    y3(k) = y3(k) + generate_CLG(M3, C, gamma_c, sigma, b_lap, p1, p2, 1);
end

% Particle Filter
particles = randn(N_part,1)*sqrt(R_nom);
weights   = ones(N_part,1)/N_part;
xPF3      = zeros(1,N3);
ESS_hist  = zeros(1,N3);
for k = 1:N3
    particles = F*particles + sqrt(Q)*randn(N_part,1);
    log_w = -0.5*(y3(k) - H*particles).^2/R_nom;
    log_w = log_w - max(log_w);
    weights = exp(log_w); weights = weights/sum(weights);
    ESS_hist(k) = 1/sum(weights.^2);
    xPF3(k) = sum(weights.*particles);
    idx = randsample(N_part, N_part, true, weights);
    particles = particles(idx); weights = ones(N_part,1)/N_part;
end

[xRK3, kappa3, ~] = run_RKAKF_1D(F, H, Q, R_nom, y3, N3, ...
    kappa_th, lambda_min, lambda_max, beta_sig, alpha_gain, epsilon, M_cap);

rmse_pf3 = compute_rmse(x_true3, xPF3);
rmse_rk3 = compute_rmse(x_true3, xRK3);
fprintf('  RMSE Particle Filter : %.4f\n', rmse_pf3);
fprintf('  RMSE RKAKF           : %.4f\n', rmse_rk3);
fprintf('  ESS Collapse: min ESS = %.1f / %d\n', min(ESS_hist), N_part);

fig3 = figure('Name','EXP3','Position',[100 100 900 500]);
subplot(2,1,1);
plot(1:N3, x_true3,'k-','LineWidth',1.5); hold on;
plot(1:N3, xPF3,'r:','LineWidth',1.5);
plot(1:N3, xRK3,'g-','LineWidth',1.5);
legend('True',sprintf('PF-%d (RMSE=%.3f)',N_part,rmse_pf3), ...
    sprintf('RKAKF (RMSE=%.3f)',rmse_rk3),'Location','best');
ylabel('State'); title('EXP 3: Particle Filter vs RKAKF'); grid on;
subplot(2,1,2);
plot(1:N3, ESS_hist,'r','LineWidth',1);
ylabel('ESS'); xlabel('Time Step'); title('ESS Collapse'); grid on;
save_figure_silent(fig3, fullfile(out_dir,'EXP3_Particle_Filter.png'));

save('results_exp3.mat', 'rmse_pf3', 'rmse_rk3', 'x_true3', 'xPF3', 'xRK3', 'ESS_hist', 'N_part');
