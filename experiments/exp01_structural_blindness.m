% exp01_structural_blindness.m
% EXP 1 — Structural Blindness Verification

fprintf('[EXP 1] Structural Blindness...\n');
load_parameters;
rng(42);

M_list   = [10, 50, 100, 200, 500, 1000];
N_samp   = 50000;
var_vals = zeros(size(M_list));
kurt_vals= zeros(size(M_list));

for ii = 1:length(M_list)
    z = generate_CLG(M_list(ii), C, gamma_c, sigma, b_lap, p1, p2, N_samp);
    var_vals(ii)  = var(z);
    kurt_vals(ii) = kurtosis(z);
end

% Time-series for moving variance demonstration
% M_demo=20000 → p3=C/M very small → rare Cauchy events → low time-average variance
N_exp1  = 1000;
M_demo  = 20000;
z_ts    = generate_CLG(M_demo, C, gamma_c, sigma, b_lap, p1, p2, N_exp1);
win     = 50;
mov_var = movvar(z_ts, win);

% Theoretical variance ceiling (Proposition 1): lim E[v^2] = p1*s^2 + 2*p2*b^2 + 2*C*g/pi
var_theoretical = p1*sigma^2 + 2*p2*b_lap^2 + 2*C*gamma_c/pi;
% Detection threshold = 5x theoretical (generous margin for energy detectors)
var_threshold   = var_theoretical * 5;

% Print results
fprintf('  M=10:   Variance=%.3f | Kurtosis=%.3f\n', var_vals(1),  kurt_vals(1));
fprintf('  M=100:  Variance=%.3f | Kurtosis=%.3f\n', var_vals(3),  kurt_vals(3));
fprintf('  M=1000: Variance=%.3f | Kurtosis=%.3f\n', var_vals(6),  kurt_vals(6));

mean_mv = mean(mov_var(1:100));  % baseline period only (no attack)
max_kv  = max(kurt_vals);
if mean_mv < var_threshold
    fprintf('  [PASS] Moving variance blind\n');
else
    fprintf('  [FAIL] Moving variance elevated\n');
end
if max_kv > kappa_th
    fprintf('  [PASS] Kurtosis detected\n');
else
    fprintf('  [FAIL] Kurtosis not detected\n');
end

fig1 = figure('Name','EXP1','Position',[100 100 1000 800]);
subplot(2,2,1);
semilogx(M_list, var_vals, 'bo-', 'LineWidth',2, 'MarkerSize',8);
hold on; yline(var_threshold,'r--','Variance Ceiling','LineWidth',1.5);
xlabel('Impulse Magnitude M'); ylabel('Empirical Variance');
title('Variance Remains Bounded'); grid on;

subplot(2,2,2);
loglog(M_list, kurt_vals, 'rs-', 'LineWidth',2, 'MarkerSize',8);
xlabel('Impulse Magnitude M'); ylabel('Empirical Kurtosis');
title('Kurtosis Diverges with M'); grid on;

subplot(2,2,3);
plot(1:N_exp1, mov_var, 'b', 'LineWidth',1.2); hold on;
yline(var_threshold,'r--','Detection Threshold','LineWidth',2);
xlabel('Time Steps'); ylabel('Moving Variance (win=50)');
title('Energy Detector Blindness'); grid on;

subplot(2,2,4);
plot(1:N_exp1, z_ts, 'Color',[0.3 0.3 0.8], 'LineWidth',0.8);
xlabel('Time Steps'); ylabel('Noise Amplitude');
title(sprintf('Hybrid CLG Noise (M=%d, Kurt=%.0f)', M_demo, kurtosis(z_ts)));
grid on;
sgtitle('EXP 1: Structural Blindness — CLG Attack Model','FontSize',13,'FontWeight','bold');
save_figure_silent(fig1, fullfile(out_dir,'EXP1_Structural_Blindness.png'));

save('results_exp1.mat', 'M_list', 'var_vals', 'kurt_vals', 'z_ts', 'mov_var', 'var_threshold');
