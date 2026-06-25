% exp07_vb_student_t.m
% EXP 7 — Monte Carlo with VB/Student-t (M=100 and M=1000)

fprintf('\n[EXP 7] Monte Carlo with VB/Student-t (M=100 and M=1000)...\n');
load_parameters;
rng(42);

N_mc7   = 1000;
N_step7 = 200;
burst7  = 80:120;
post7   = 121:N_step7;

for mag_idx = 1:2
    M7 = [100, 1000];
    M_cur = M7(mag_idx);
    fprintf('  Running M=%d...\n', M_cur);

    rmse7_kf  = zeros(1,N_mc7); rmse7_hub = zeros(1,N_mc7);
    rmse7_vb  = zeros(1,N_mc7); rmse7_rk  = zeros(1,N_mc7);
    ttr7_kf   = zeros(1,N_mc7); ttr7_hub  = zeros(1,N_mc7);
    ttr7_vb   = zeros(1,N_mc7); ttr7_rk   = zeros(1,N_mc7);

    parfor mc = 1:N_mc7
        xt7 = zeros(1,N_step7); xt7(1) = 0;
        for k = 2:N_step7, xt7(k) = F*xt7(k-1) + sqrt(Q)*randn(); end
        ym7 = H*xt7 + sqrt(R_nom)*randn(1,N_step7);
        for k = burst7
            ym7(k) = ym7(k) + generate_CLG(M_cur, C, gamma_c, sigma, b_lap, p1, p2, 1);
        end
        xk7 = run_KF_1D(F, H, Q, R_nom, ym7, N_step7);
        xh7 = run_Huber_1D(F, H, Q, R_nom, ym7, N_step7);
        xv7 = run_VB_StudentT(F, H, Q, R_nom, ym7, N_step7);
        xr7 = run_RKAKF_1D(F, H, Q, R_nom, ym7, N_step7, ...
            kappa_th, lambda_min, lambda_max, beta_sig, alpha_gain, epsilon, M_cap);
        rmse7_kf(mc)  = compute_rmse(xt7, xk7);
        rmse7_hub(mc) = compute_rmse(xt7, xh7);
        rmse7_vb(mc)  = compute_rmse(xt7, xv7);
        rmse7_rk(mc)  = compute_rmse(xt7, xr7);
        ttr7_kf(mc)   = compute_TTR(xt7, xk7, post7, 0.4);
        ttr7_hub(mc)  = compute_TTR(xt7, xh7, post7, 0.4);
        ttr7_vb(mc)   = compute_TTR(xt7, xv7, post7, 0.4);
        ttr7_rk(mc)   = compute_TTR(xt7, xr7, post7, 0.4);
    end

    mn_rmse7_kf  = mean(rmse7_kf);  mn_rmse7_hub = mean(rmse7_hub);
    mn_rmse7_vb  = mean(rmse7_vb);  mn_rmse7_rk  = mean(rmse7_rk);
    mn_ttr7_kf   = mean(ttr7_kf(isfinite(ttr7_kf)));
    mn_ttr7_hub  = mean(ttr7_hub(isfinite(ttr7_hub)));
    mn_ttr7_vb   = mean(ttr7_vb(isfinite(ttr7_vb)));
    mn_ttr7_rk   = mean(ttr7_rk(isfinite(ttr7_rk)));

    fprintf('  M=%d | RKAKF RMSE=%.4f TTR=%.1f | Hub RMSE=%.4f TTR=%.1f\n', ...
        M_cur, mn_rmse7_rk, mn_ttr7_rk, mn_rmse7_hub, mn_ttr7_hub);

    % Quality check - only assert if both are finite
    if isfinite(mn_ttr7_rk) && isfinite(mn_ttr7_hub)
        assert(mn_ttr7_rk <= mn_ttr7_hub, ...
            sprintf('WARNING: RKAKF TTR should be <= Huber TTR (M=%d)', M_cur));
    end

    if mag_idx == 1
        mn7_kf_100=mn_rmse7_kf; mn7_hub_100=mn_rmse7_hub;
        mn7_vb_100=mn_rmse7_vb; mn7_rk_100=mn_rmse7_rk;
        ttr7_kf_100=mn_ttr7_kf; ttr7_hub_100=mn_ttr7_hub;
        ttr7_vb_100=mn_ttr7_vb; ttr7_rk_100=mn_ttr7_rk;

        fig7a = figure('Name','EXP7_M100','Position',[100 100 900 500]);
        bar([mn_rmse7_kf, mn_rmse7_hub, mn_rmse7_vb, mn_rmse7_rk]);
        set(gca,'XTickLabel',{'Standard KF','Huber KF','VB/Student-t','RKAKF'});
        ylabel('Mean RMSE'); title(sprintf('EXP 7: VB Comparison M=100 (%d runs)',N_mc7));
        grid on;
        save_figure_silent(fig7a, fullfile(out_dir,'EXP7_VB_Comparison_M100.png'));
    else
        mn7_kf_1k=mn_rmse7_kf; mn7_hub_1k=mn_rmse7_hub;
        mn7_vb_1k=mn_rmse7_vb; mn7_rk_1k=mn_rmse7_rk;
        ttr7_kf_1k=mn_ttr7_kf; ttr7_hub_1k=mn_ttr7_hub;
        ttr7_vb_1k=mn_ttr7_vb; ttr7_rk_1k=mn_ttr7_rk;

        fig7b = figure('Name','EXP7_M1000','Position',[100 100 900 500]);
        bar([mn_rmse7_kf, mn_rmse7_hub, mn_rmse7_vb, mn_rmse7_rk]);
        set(gca,'XTickLabel',{'Standard KF','Huber KF','VB/Student-t','RKAKF'});
        ylabel('Mean RMSE'); title(sprintf('EXP 7: VB Comparison M=1000 (%d runs)',N_mc7));
        grid on;
        save_figure_silent(fig7b, fullfile(out_dir,'EXP7_VB_Comparison_M1000.png'));
    end
end

save('results_exp7.mat', 'mn7_kf_100', 'mn7_hub_100', 'mn7_vb_100', 'mn7_rk_100', ...
    'ttr7_kf_100', 'ttr7_hub_100', 'ttr7_vb_100', 'ttr7_rk_100', ...
    'mn7_kf_1k', 'mn7_hub_1k', 'mn7_vb_1k', 'mn7_rk_1k', ...
    'ttr7_kf_1k', 'ttr7_hub_1k', 'ttr7_vb_1k', 'ttr7_rk_1k', 'N_mc7');
