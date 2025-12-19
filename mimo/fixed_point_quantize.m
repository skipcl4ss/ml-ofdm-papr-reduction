function xq = fixed_point_quantize(x, TWL, FWL)
% Fixed-point quantization without Fixed-Point Designer toolbox
% Mimics: fi(x, 1, TWL, FWL, 'roundmode', 'round', 'overflowmode', 'saturate')
%
% Inputs:
%   x   - Input signal (real or complex)
%   TWL - Total Word Length (total bits including sign bit)
%   FWL - Fractional Word Length (bits for fractional part)
%
% Output:
%   xq  - Quantized signal

% Calculate integer word length
IWL = TWL - FWL - 1; % -1 for sign bit

% Calculate quantization step
delta = 2^(-FWL);

% Calculate saturation limits (for signed numbers)
max_val = (2^(TWL-1) - 1) * delta;  % Maximum positive value
min_val = -2^(TWL-1) * delta;        % Maximum negative value

% Handle complex numbers
if isreal(x)
    xq = quantize_real(x, delta, min_val, max_val);
else
    xq_real = quantize_real(real(x), delta, min_val, max_val);
    xq_imag = quantize_real(imag(x), delta, min_val, max_val);
    xq = xq_real + 1i * xq_imag;
end

end

function y = quantize_real(x, delta, min_val, max_val)
% Quantize real-valued signal
%
% Step 1: Saturate (clip to range)
x_sat = max(min(x, max_val), min_val);

% Step 2: Round to nearest quantization level
y = round(x_sat / delta) * delta;

% Step 3: Saturate again (in case rounding pushed it over)
y = max(min(y, max_val), min_val);
end