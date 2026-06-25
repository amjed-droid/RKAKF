function z = generate_CLG(M, C, gamma_c, sigma, b_lap, p1, p2, n)
    % Compound Laplace-Gaussian mixture with truncated Cauchy
    p3 = min(C / max(M, 1), 1 - p1 - p2);
    p3 = max(p3, 0);
    z  = zeros(n, 1);
    for i = 1:n
        u = rand();
        if u < p1
            z(i) = sigma * randn();
        elseif u < p1 + p2
            z(i) = laplace_rnd(0, b_lap);
        else
            z(i) = cauchy_rnd_trunc(0, gamma_c, M);
        end
    end
end
