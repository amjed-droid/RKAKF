% exp08_cauchy_vs_clg.m
% EXP 8 — Cauchy vs CLG Attack Comparison

fprintf('\n[EXP 8] Cauchy vs CLG Attack Comparison...\n');
load_parameters;
rng(42);

N8 = 1000;
% Pure Cauchy (gamma=1.0, uncontrolled) — wide variance, DETECTABLE
z_cauchy = gamma_c * tan(pi*(rand(1,N8)-0.5));
z_cauchy = sign(z_cauchy) .* min(abs(z_cauchy), 50);   % loose cap

% CLG with large M → p3=C/M tiny → Cauchy events rare → variance stays bounded
z_clg    = generate_CLG(20000, C, gamma_c, sigma, b_lap, p1, p2, N8)';

% Detection threshold from Proposition 1
var_theoretical = p1*sigma^2 + 2*p2*b_lap^2 + 2*C*gamma_c/pi;
det_thresh8   = var_theoretical * 5;

var_cauchy   = var(z_cauchy);
var_clg      = var(z_clg);
kurt_clg     = kurtosis(z_clg);

% Moving variance of CLG sequence
mv_clg = movvar(z_clg, 50);
mv_cau = movvar(z_cauchy, 50);

fprintf('  Cauchy variance   : %.4f\n', var_cauchy);
fprintf('  CLG variance      : %.4f\n', var_clg);
fprintf('  CLG kurtosis      : %.4f\n', kurt_clg);
if mean(mv_cau) > det_thresh8
    fprintf('  [DETECTED] Cauchy variance exceeds threshold\n');
end
if mean(mv_clg) < det_thresh8
    fprintf('  [STEALTHY] CLG variance stays below threshold\n');
end

fig8 = figure('Name','EXP8','Position',[100 100 1000 800]);
subplot(2,2,1);
plot(1:N8, z_cauchy,'r','LineWidth',0.8); hold on;
yline(det_thresh8,'k--','Threshold','LineWidth',1.5);
ylabel('Amplitude'); title(sprintf('Cauchy (Var=%.2f) — DETECTABLE',var_cauchy));
ylim([-15 15]); grid on;
subplot(2,2,2);
plot(1:N8, z_clg,'b','LineWidth',0.8); hold on;
yline(det_thresh8,'k--','Threshold','LineWidth',1.5);
ylabel('Amplitude'); title(sprintf('CLG (Var=%.2f, Kurt=%.0f) — STEALTHY',var_clg,kurt_clg));
ylim([-15 15]); grid on;
subplot(2,2,3);
plot(1:N8, mv_cau,'r','LineWidth',1.2); hold on;
yline(det_thresh8,'k--','Threshold','LineWidth',1.5);
ylabel('Moving Var'); xlabel('Time Step'); title('Cauchy Moving Variance'); grid on;
subplot(2,2,4);
plot(1:N8, mv_clg,'b','LineWidth',1.2); hold on;
yline(det_thresh8,'k--','Threshold','LineWidth',1.5);
ylabel('Moving Var'); xlabel('Time Step'); title('CLG Moving Variance (Blind)'); grid on;
sgtitle('EXP 8: Traditional Cauchy vs Hybrid CLG — Stealth Advantage','FontSize',12,'FontWeight','bold');
save_figure_silent(fig8, fullfile(out_dir,'EXP8_Attack_Comparison.png'));

save('results_exp8.mat', 'var_cauchy', 'var_clg', 'kurt_clg', 'z_cauchy', 'z_clg', 'mv_cau', 'mv_clg', 'det_thresh8');
