% generate_latex_tables.m
% Generates all publication-ready LaTeX tables by loading experiment results.

% Initialize paths using parameters configuration
load_parameters;

fprintf('\n==============================================\n');
fprintf('  LATEX TABLES FOR PUBLICATION\n');
fprintf('==============================================\n\n');

% Ensure results exist or run the experiments
if ~exist('results_exp5.mat', 'file')
    fprintf('results_exp5.mat not found. Running exp05_monte_carlo...\n');
    exp05_monte_carlo;
end
load('results_exp5.mat');

if ~exist('results_exp7.mat', 'file')
    fprintf('results_exp7.mat not found. Running exp07_vb_student_t...\n');
    exp07_vb_student_t;
end
load('results_exp7.mat');

if ~exist('results_exp9.mat', 'file')
    fprintf('results_exp9.mat not found. Running exp09_gain_restoration...\n');
    exp09_gain_restoration;
end
load('results_exp9.mat');

if ~exist('results_exp6.mat', 'file')
    fprintf('results_exp6.mat not found. Running exp06_drone_trajectory...\n');
    exp06_drone_trajectory;
end
load('results_exp6.mat');

if ~exist('results_expm1.mat', 'file')
    fprintf('results_expm1.mat not found. Running exp_m1_2d_system...\n');
    exp_m1_2d_system;
end
load('results_expm1.mat');

%% --- TABLE 2 (Monte Carlo, Continuous CLG) ---
fprintf('--- TABLE 2 (Monte Carlo, Continuous CLG, N_mc=%d) ---\n', N_mc5);
fprintf('\\begin{table}[H]\n');
fprintf('\\caption{Monte Carlo Performance under Continuous CLG Noise (%d runs, $M=%d$).}\n', N_mc5, M5);
fprintf('\\label{tab:mc_continuous}\n');
fprintf('\\centering\n');
fprintf('\\begin{tabular}{@{}lcccc@{}}\n');
fprintf('\\toprule\n');
fprintf('\\textbf{Estimator} & \\textbf{Mean RMSE} & \\textbf{Mean MAE} & \\textbf{Mean Max Error} & \\textbf{$p$-value} \\\\\n');
fprintf('\\midrule\n');
fprintf('Standard KF & %.4f & %.4f & %.4f & -- \\\\\n', mean(rmse_kf5), mean(mae_kf5), mean(maxe_kf5));
fprintf('Huber-Robust KF & %.4f & %.4f & %.4f & -- \\\\\n', mean(rmse_hub5), mean(mae_hub5), mean(maxe_hub5));
fprintf('\\textbf{RKAKF (Proposed)} & \\textbf{%.4f} & \\textbf{%.4f} & \\textbf{%.4f} & $%.2e$ \\\\\n', ...
    mean(rmse_rk5), mean(mae_rk5), mean(maxe_rk5), pval_rk_kf);
fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');

%% --- TABLE 3 (Monte Carlo, Localized Burst, M=100 and M=1000) ---
fprintf('--- TABLE 3 (Monte Carlo, Localized Burst, M=100 and M=1000) ---\n');
fprintf('\\begin{table}[H]\n');
fprintf('\\caption{Monte Carlo Performance under Localized CLG Burst (%d runs). VB/Student-$t$ included.}\n', N_mc7);
fprintf('\\label{tab:mc_burst_vb}\n');
fprintf('\\centering\n');
fprintf('\\begin{tabular}{@{}lcccc@{}}\n');
fprintf('\\toprule\n');
fprintf('& \\multicolumn{2}{c}{$M=100$} & \\multicolumn{2}{c}{$M=1000$} \\\\\n');
fprintf('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n');
fprintf('\\textbf{Estimator} & \\textbf{RMSE} & \\textbf{TTR} & \\textbf{RMSE} & \\textbf{TTR} \\\\\n');
fprintf('\\midrule\n');
fprintf('Standard KF & %.4f & %.1f & %.4f & %.1f \\\\\n', ...
    mn7_kf_100, ttr7_kf_100, mn7_kf_1k, ttr7_kf_1k);
fprintf('Huber-Robust KF & %.4f & %.1f & %.4f & %.1f \\\\\n', ...
    mn7_hub_100, ttr7_hub_100, mn7_hub_1k, ttr7_hub_1k);
fprintf('VB/Student-$t$ & %.4f & %.1f & %.4f & %.1f \\\\\n', ...
    mn7_vb_100, ttr7_vb_100, mn7_vb_1k, ttr7_vb_1k);
fprintf('\\textbf{RKAKF (Proposed)} & \\textbf{%.4f} & \\textbf{%.1f} & \\textbf{%.4f} & \\textbf{%.1f} \\\\\n', ...
    mn7_rk_100, ttr7_rk_100, mn7_rk_1k, ttr7_rk_1k);
fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');

%% --- TABLE 4 (Temporal RMSE + TTR) ---
fprintf('--- TABLE 4 (Temporal RMSE + TTR, EXP 9) ---\n');
fprintf('\\begin{table}[H]\n');
fprintf('\\caption{Temporal RMSE and Time-to-Recovery (TTR) under Localized CLG Burst (EXP 9).}\n');
fprintf('\\label{tab:temporal_rmse_ttr}\n');
fprintf('\\centering\n');
fprintf('\\begin{tabular}{@{}lcccc@{}}\n');
fprintf('\\toprule\n');
fprintf('\\textbf{Estimator} & \\textbf{Pre-Attack RMSE} & \\textbf{During RMSE} & \\textbf{Post RMSE} & \\textbf{TTR (steps)} \\\\\n');
fprintf('\\midrule\n');
fprintf('Standard KF & %.4f & %.4f & %.4f & %s \\\\\n', ...
    rmse9_kf_pre, rmse9_kf_dur, rmse9_kf_post, ttr2str(ttr9_kf));
fprintf('Huber-Robust KF & %.4f & %.4f & %.4f & %s \\\\\n', ...
    rmse9_hub_pre, rmse9_hub_dur, rmse9_hub_post, ttr2str(ttr9_hub));
fprintf('\\textbf{RKAKF (Proposed)} & \\textbf{%.4f} & \\textbf{%.4f} & \\textbf{%.4f} & \\textbf{%s} \\\\\n', ...
    rmse9_rk_pre, rmse9_rk_dur, rmse9_rk_post, ttr2str(ttr9_rk));
fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');

%% --- TABLE 5 (Drone Trajectory) ---
fprintf('--- TABLE 5 (Drone Trajectory, EXP 6) ---\n');
fprintf('\\begin{table}[H]\n');
fprintf('\\caption{Filter Performance on Synthetic FPV Drone Trajectory (EXP 6).}\n');
fprintf('\\label{tab:drone}\n');
fprintf('\\centering\n');
fprintf('\\begin{tabular}{@{}lcc@{}}\n');
fprintf('\\toprule\n');
fprintf('\\textbf{Estimator} & \\textbf{RMSE} & \\textbf{Peak Error} \\\\\n');
fprintf('\\midrule\n');
fprintf('Standard KF & %.4f & %.4f \\\\\n', rmse_kf6,  pk_kf6);
fprintf('H-Infinity KF & %.4f & %.4f \\\\\n', rmse_hi6, pk_hi6);
fprintf('Huber-Robust KF & %.4f & %.4f \\\\\n', rmse_hub6, pk_hub6);
fprintf('\\textbf{RKAKF (Proposed)} & \\textbf{%.4f} & \\textbf{%.4f} \\\\\n', rmse_rk6, pk_rk6);
fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');

%% --- APPENDIX TABLE (2D System) ---
fprintf('--- APPENDIX TABLE (2D System, EXP M1) ---\n');
fprintf('\\begin{table}[H]\n');
fprintf('\\caption{Position RMSE on 2D Kinematic System (Appendix, EXP M1).}\n');
fprintf('\\label{tab:2d_system}\n');
fprintf('\\centering\n');
fprintf('\\begin{tabular}{@{}lcc@{}}\n');
fprintf('\\toprule\n');
fprintf('\\textbf{Estimator} & \\textbf{RMSE} & \\textbf{Improvement vs KF (\\%%)} \\\\\n');
fprintf('\\midrule\n');
fprintf('Standard KF & %.4f & -- \\\\\n', rmse_kf_m1);
fprintf('Huber-Robust KF & %.4f & %.1f\\%% \\\\\n', rmse_hub_m1, impr_hub);
fprintf('\\textbf{RKAKF (Proposed)} & \\textbf{%.4f} & \\textbf{%.1f\\%%} \\\\\n', rmse_rk_m1, impr_rk);
fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\end{table}\n\n');
