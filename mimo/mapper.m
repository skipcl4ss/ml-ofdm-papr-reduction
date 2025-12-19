function [modulated_symbols, Mod] = mapper(b, N)
%If N is given, makes a block of N random 2^b-PSK/QAM modulated symbols
% Otherwise, a block of 2^b-PSK/QAM modulated symbols for [0:2^b-1].
M = 2^b; % Modulation order or Alphabet (Symbol) size

if b == 1
    Mod = 'BPSK'; 
    A = 1;
    if nargin == 2 % A block of N random 2^b-PSK/QAM modulated symbols
        data = randi([0 M-1], 1, N);
        modulated_symbols = A * pskmod(data, M);
    else
        modulated_symbols = A * pskmod(0:M-1, M);
    end
    
elseif b == 2
    Mod = 'QPSK'; 
    A = 1;
    if nargin == 2
        data = randi([0 M-1], 1, N);
        modulated_symbols = A * pskmod(data, M, pi/4);
    else
        modulated_symbols = A * pskmod(0:M-1, M, pi/4);
    end
    
else
    Mod = [num2str(2^b) 'QAM']; 
    Es = 1; 
    A = sqrt(3/2/(M-1)*Es);
    if nargin == 2
        data = randi([0 M-1], 1, N);
        modulated_symbols = A * qammod(data, M, 'gray');
    else
        modulated_symbols = A * qammod(0:M-1, M, 'gray');
    end
end
end