% load_parameters.m
% Load all parameters for the RKAKF experiments

% Path Resolution: Add src/ (with its subfolders) and experiments/ to the MATLAB search path
if exist('src', 'dir')
    addpath(genpath('src'));
    addpath('experiments');
elseif exist('../src', 'dir')
    addpath(genpath('../src'));
    addpath('../experiments');
end

% Output directory
out_dir = 'RKAKF_Figures';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% Silent export mode by default
set(0, 'DefaultFigureVisible', 'off'); 

% Global parameters
dt     = 0.1;
F      = 1;  H = 1;  Q = 0.1;  R_nom = 1.0;

% CLG Attack parameters
p1      = 0.5;    % Gaussian weight
p2      = 0.3;    % Laplace weight
C       = 0.5;    % Cauchy coupling constant
sigma   = 0.5;    % Gaussian std
b_lap   = 0.3;    % Laplace scale
gamma_c = 1.0;    % Cauchy scale

% RKAKF Hyperparameters
kappa_th   = 4.0;
lambda_max = 0.99;
lambda_min = 0.50;
beta_sig   = 0.5;
alpha_gain = 0.5;
epsilon    = 1e-6;
M_cap      = 500;
