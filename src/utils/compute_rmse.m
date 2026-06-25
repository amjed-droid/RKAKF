function r = compute_rmse(x_true, x_est)
    % Compute Root Mean Squared Error (RMSE)
    r = sqrt(mean((x_true - x_est).^2));
end
