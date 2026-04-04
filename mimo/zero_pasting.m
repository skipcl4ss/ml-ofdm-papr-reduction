function y=zero_pasting(x)
% Paste zeros at the center half of the input sequence x
N=length(x); M=ceil(N/4); y = [x(1:M) zeros(1,N/2) x(N-M+1:N)];
end