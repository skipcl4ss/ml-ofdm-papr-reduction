function [modulated_symbols,Mod] = mapper(b,N)
%If N is given, makes a block of N random 2^b-PSK/QAM modulated symbols
% Otherwise, a block of 2^b-PSK/QAM modulated symbols for [0:2^b-1].
M=2^b; % Modulation order or Alphabet (Symbol) size
if b==1, Mod='BPSK'; A=1; mod_object=modem.pskmod('M',M);
    elseif b==2, Mod='QPSK'; A=1;
        mod_object=modem.pskmod('M',M, 'PhaseOffset',pi/4);
    else Mod=[num2str(2^b) 'QAM']; Es=1; A=sqrt(3/2/(M-1)*Es);
        mod_object=modem.qammod('M',M,'SymbolOrder','gray');
end
if nargin==2 % A block of N random 2^b-PSK/QAM modulated symbols
    modulated_symbols = A*modulate(mod_object,randint(1,N,M));
else
    modulated_symbols = A*modulate(mod_object,[0:M-1]);
end