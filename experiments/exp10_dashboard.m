% exp10_dashboard.m
% EXP 10 — Summary Dashboard

fprintf('\n[EXP 10] Summary Dashboard...\n');
load_parameters;

% Ensure required data is available
if ~exist('results_exp2.mat', 'file')
    fprintf('  results_exp2.mat not found. Running exp02_kf_vs_rkakf...\n');
    exp02_kf_vs_rkakf;
end
load('results_exp2.mat', 'rmse_kf2', 'rmse_rk2');

if ~exist('results_exp4.mat', 'file')
    fprintf('  results_exp4.mat not found. Running exp04_huber_vs_rkakf...\n');
    exp04_huber_vs_rkakf;
end
load('results_exp4.mat', 'rmse_hub4', 'rmse_rk4');

if ~exist('results_exp6.mat', 'file')
    fprintf('  results_exp6.mat not found. Running exp06_drone_trajectory...\n');
    exp06_drone_trajectory;
end
load('results_exp6.mat', 'rmse_kf6', 'rmse_hi6', 'rmse_hub6', 'rmse_rk6');

if ~exist('results_exp5.mat', 'file')
    fprintf('  results_exp5.mat not found. Running exp05_monte_carlo...\n');
    exp05_monte_carlo;
end
load('results_exp5.mat', 'rmse_kf5', 'rmse_hub5', 'rmse_rk5', 'N_mc5');

fig10 = figure('Name','EXP10','Position',[100 100 1000 800]);
subplot(2,2,1);
bar([rmse_kf2, rmse_rk2],'FaceColor','flat');
set(gca,'XTickLabel',{'Standard KF','RKAKF'});
ylabel('RMSE'); title('EXP 2: Localized Burst'); grid on;

subplot(2,2,2);
bar([rmse_hub4, rmse_rk4],'FaceColor','flat');
set(gca,'XTickLabel',{'Huber KF','RKAKF'});
ylabel('RMSE'); title('EXP 4: Continuous CLG (Huber wins)'); grid on;

subplot(2,2,3);
bar([rmse_kf6, rmse_hi6, rmse_hub6, rmse_rk6],'FaceColor','flat');
set(gca,'XTickLabel',{'Std KF','H-Inf','Huber','RKAKF'},'XTickLabelRotation',15);
ylabel('RMSE'); title('EXP 6: Drone Trajectory'); grid on;

subplot(2,2,4);
bar([mean(rmse_kf5), mean(rmse_hub5), mean(rmse_rk5)],'FaceColor','flat');
set(gca,'XTickLabel',{'Standard KF','Huber KF','RKAKF'});
ylabel('Mean RMSE'); title(sprintf('EXP 5: Monte Carlo (%d runs)', N_mc5)); grid on;

sgtitle('EXP 10: Benchmark Summary Dashboard','FontSize',13,'FontWeight','bold');
save_figure_silent(fig10, fullfile(out_dir,'EXP10_Summary_Dashboard.png'));
