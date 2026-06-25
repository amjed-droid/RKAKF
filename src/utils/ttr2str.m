function s = ttr2str(ttr)
    % Convert TTR value to LaTeX math string or normal text
    if isinf(ttr)
        s = '$\infty$';
    else
        s = num2str(ttr);
    end
end
