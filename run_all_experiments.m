% run_all_experiments.m
% Master script to run all RKAKF experiments and generate all outputs.

clc; clear; close all;

% Run parameter configuration to initialize the MATLAB search paths
load_parameters;

fprintf('=====================================================\n');
fprintf('  RKAKF Unified Experiment Suite - Master Runner\n');
fprintf('=====================================================\n\n');

% Run individual experiments
exp01_structural_blindness;
exp02_kf_vs_rkakf;
exp03_pf_vs_rkakf;
exp04_huber_vs_rkakf;
exp05_monte_carlo;
exp06_drone_trajectory;
exp07_vb_student_t;
exp08_cauchy_vs_clg;
exp09_gain_restoration;
exp10_dashboard;
exp_m1_2d_system;

fprintf('=====================================================\n');
fprintf('  All experiments complete.\n');
fprintf('  Figures saved to: RKAKF_Figures/\n');
fprintf('=====================================================\n');
