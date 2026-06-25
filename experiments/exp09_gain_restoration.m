% exp09_gain_restoration.m
% EXP 9 — Gain Restoration Mechanism

fprintf('\n[EXP 9] Gain Restoration Mechanism...\n');
load_parameters;
rng(42);

N9   = 600;
atk9 = 200:300;
post9_idx = 301:N9;

x_true9 = zeros(1,N9);
for k = 2:N9, x_true9(k) = F*x_true9(k-1) + sqrt(Q)*randn(); end
y9 = H*x_true9 + 0.1*randn(1,N9);
for k = atk9
    y9(k) = y9(k) + generate_CLG(500, C, gamma_c, sigma, b_lap, p1, p2, 1);
end

xKF9  = run_KF_1D(F, H, Q, R_nom, y9, N9);
xHUB9 = run_Huber_1D(F, H, Q, R_nom, y9, N9);
[xRK9, kappa9, Rmod9] = run_RKAKF_1D(F, H, Q, R_nom, y9, N9, ...
    kappa_th, lambda_min, lambda_max, beta_sig, alpha_gain, epsilon, M_cap);

% Kalman gain history for RKAKF (recompute via state trajectory)
Kgain9 = zeros(1,N9);
P9 = 1.0; x9 = 0; lam9 = lambda_max; mu2_9 = 1; mu4_9 = 3; R9k = R_nom; kstar9 = 0;
for k = 1:N9
    xp9 = F*x9; Pp9 = (1/lam9)*(F*P9*F' + Q);
    nu9  = y9(k) - H*xp9;
    nuc9 = sign(nu9)*min(abs(nu9), M_cap);
    mu2_9 = lam9*mu2_9 + (1-lam9)*nuc9^2;
    mu4_9 = lam9^3*mu4_9 + (1-lam9^3)*nuc9^4;
    kap9  = mu4_9/(mu2_9^2 + epsilon);
    lam9  = lambda_min + (lambda_max-lambda_min)/(1+exp(beta_sig*(kap9-kappa_th)));
    if kap9 > kappa_th
        R9k = R_nom*exp(alpha_gain*min(max(kap9-kappa_th,0),20));
        kstar9 = k;
    else
        R9k = R_nom + (R9k-R_nom)*exp(-0.6*(k-kstar9));
    end
    Kgain9(k) = Pp9*H'/(H*Pp9*H' + R9k);
    x9 = xp9 + Kgain9(k)*(y9(k)-H*xp9);
    P9 = (1-Kgain9(k)*H)*Pp9;
end

% Temporal RMSE
pre9  = 1:199;
dur9  = 200:300;

rmse9_kf_pre  = compute_rmse(x_true9(pre9),  xKF9(pre9));
rmse9_kf_dur  = compute_rmse(x_true9(dur9),  xKF9(dur9));
rmse9_kf_post = compute_rmse(x_true9(post9_idx), xKF9(post9_idx));
rmse9_hub_pre  = compute_rmse(x_true9(pre9),  xHUB9(pre9));
rmse9_hub_dur  = compute_rmse(x_true9(dur9),  xHUB9(dur9));
rmse9_hub_post = compute_rmse(x_true9(post9_idx), xHUB9(post9_idx));
rmse9_rk_pre   = compute_rmse(x_true9(pre9),  xRK9(pre9));
rmse9_rk_dur   = compute_rmse(x_true9(dur9),  xRK9(dur9));
rmse9_rk_post  = compute_rmse(x_true9(post9_idx), xRK9(post9_idx));

ttr9_kf  = compute_TTR(x_true9, xKF9,  post9_idx, 0.5);
ttr9_hub = compute_TTR(x_true9, xHUB9, post9_idx, 0.5);
ttr9_rk  = compute_TTR(x_true9, xRK9,  post9_idx, 0.5);

% Quality check: kurtosis must exceed threshold during attack
assert(max(kappa9(atk9(1):atk9(end))) > kappa_th, ...
    'WARNING: Kurtosis did not exceed threshold during attack window');

fprintf('  Standard KF : Pre=%.4f | Dur=%.4f | Post=%.4f | TTR=%s\n', ...
    rmse9_kf_pre, rmse9_kf_dur, rmse9_kf_post, ttr2str(ttr9_kf));
fprintf('  Huber KF    : Pre=%.4f | Dur=%.4f | Post=%.4f | TTR=%s\n', ...
    rmse9_hub_pre, rmse9_hub_dur, rmse9_hub_post, ttr2str(ttr9_hub));
fprintf('  RKAKF       : Pre=%.4f | Dur=%.4f | Post=%.4f | TTR=%s\n', ...
    rmse9_rk_pre, rmse9_rk_dur, rmse9_rk_post, ttr2str(ttr9_rk));

fig9 = figure('Name','EXP9','Position',[100 100 1000 1000]);
subplot(4,1,1);
plot(1:N9, x_true9,'k-','LineWidth',1.5); hold on;
plot(1:N9, xKF9,'r--','LineWidth',1.0);
plot(1:N9, xHUB9,'b-.','LineWidth',1.0);
plot(1:N9, xRK9,'g-','LineWidth',1.5);
xregion(atk9(1),atk9(end),'FaceColor',[1 0.8 0.8],'FaceAlpha',0.3);
legend('True','Standard KF','Huber KF','RKAKF'); ylabel('State');
title('EXP 9: Gain Restoration — State Tracking'); grid on;

subplot(4,1,2);
plot(1:N9, Kgain9,'b-','LineWidth',1.5);
xregion(atk9(1),atk9(end),'FaceColor',[1 0.8 0.8],'FaceAlpha',0.3);
ylabel('Kalman Gain'); title('Gain Revocation During Attack'); grid on;

subplot(4,1,3);
plot(1:N9, kappa9,'m-','LineWidth',1.2); hold on;
yline(kappa_th,'r--','\kappa_{th}','LineWidth',1.5);
xregion(atk9(1),atk9(end),'FaceColor',[1 0.8 0.8],'FaceAlpha',0.3);
ylabel('\kappa_k'); title('Recursive Kurtosis'); grid on;

subplot(4,1,4);
plot(1:N9, Rmod9,'k-','LineWidth',1.5); hold on;
yline(R_nom,'b--','R_{nom}','LineWidth',1.5);
xregion(atk9(1),atk9(end),'FaceColor',[1 0.8 0.8],'FaceAlpha',0.3);
ylabel('R_{adaptive}'); xlabel('Time Step');
title('Covariance Inflation & Restoration'); grid on;
save_figure_silent(fig9, fullfile(out_dir,'EXP9_Gain_Restoration.png'));

save('results_exp9.mat', 'rmse9_kf_pre', 'rmse9_kf_dur', 'rmse9_kf_post', 'ttr9_kf', ...
    'rmse9_hub_pre', 'rmse9_hub_dur', 'rmse9_hub_post', 'ttr9_hub', ...
    'rmse9_rk_pre', 'rmse9_rk_dur', 'rmse9_rk_post', 'ttr9_rk', ...
    'x_true9', 'xKF9', 'xHUB9', 'xRK9', 'Kgain9', 'kappa9', 'Rmod9', 'atk9', 'N9', 'post9_idx');
