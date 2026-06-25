function ttr = compute_TTR(x_true, x_est, post_idx, threshold)
    % Compute Time-to-Recovery (TTR)
    ttr = Inf;
    for i = 1:length(post_idx)
        if abs(x_true(post_idx(i)) - x_est(post_idx(i))) < threshold
            ttr = i;
            return;
        end
    end
end
