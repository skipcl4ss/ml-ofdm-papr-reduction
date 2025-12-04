function [x_clipped,sigma]=clipping(x,CR,sigma)
% CR: Clipping Ratio, sigma: sqrt(variance of x)
if nargin<3
    x_mean=mean(x); x_dev=x-x_mean;
    sigma=sqrt(x_dev*x_dev'/length(x));
end
x_clipped = x; CL = CR*sigma; % Clipping level
ind = find(abs(x)>CL); % Indices to clip
x_clipped(ind)=x(ind)./abs(x(ind))*CL; % Eq.(7.18b) limitation to CL