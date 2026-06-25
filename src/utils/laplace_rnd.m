function x = laplace_rnd(mu, b_lap)
    % Laplace random sample generator
    u = rand() - 0.5;
    x = mu - b_lap * sign(u) * log(1 - 2*abs(u));
end
