function S = generate_recurrence(s, m, tau)
% Generates a recurrence plot from time-series data
    y = s';

    N = length(y);
    N2 = N - tau * (m - 1);

    for mi = 1:m
        xe(:, mi) = y([1:N2] + tau * (mi-1));
    end

    x1 = repmat(xe, N2, 1);
    x2 = reshape(repmat(xe(:), 1, N2)', N2 * N2, m);

    S = sqrt(sum( (x1 - x2) .^ 2, 2 ));
    S = reshape(S, N2, N2);
end