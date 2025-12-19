function y = addcp(x, Ncp)
% ADDCP Add cyclic prefix to OFDM symbols
% y = addcp(x, Ncp)
% 
% Inputs:
%   x   - Input signal (can be vector or matrix)
%         If matrix, each column is treated as one OFDM symbol
%   Ncp - Length of cyclic prefix (number of samples)
%
% Output:
%   y   - Signal with cyclic prefix added

% Handle row or column vectors
if size(x, 1) == 1
    % Row vector
    N = length(x);
    cp = x(end-Ncp+1:end);
    y = [cp, x];
else
    % Column vector or matrix
    [N, num_symbols] = size(x);
    y = zeros(N + Ncp, num_symbols);
    
    for k = 1:num_symbols
        % Take last Ncp samples and prepend them
        cp = x(end-Ncp+1:end, k);
        y(:, k) = [cp; x(:, k)];
    end
end
end