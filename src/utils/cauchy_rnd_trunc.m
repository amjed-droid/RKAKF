function x = cauchy_rnd_trunc(x0, gamma, M)
    % Truncated Cauchy random sample generator
    while true
        u = rand();
        x = x0 + gamma * tan(pi*(u - 0.5));
        if abs(x) <= M
            return;
        end
    end
end
