function obj = rf_sample(obj, Napp)
% rf_sample samples from the distribution 
%
% copyright (c) 2010 
% Fuxin Li - fuxin.li@ins.uni-bonn.de
% Catalin Ionescu - catalin.ionescu@ins.uni-bonn.de
% Cristian Sminchisescu - cristian.sminchisescu@ins.uni-bonn.de

switch obj.distribution
  case 'gaussian'
    S = sqrt(obj.kernel_param) * sqrt(2) * randn(obj.dim, Napp);
  case 'sech'
    % for chi2 kernel
    % here we use the inverse method for sampling
    snet = rand(obj.dim, Napp);
    S = 2/pi*log(tan(pi/2 * snet));
  case 'cauchy'
    % for intersection kernel
    S = obj.kernel_param * tan(pi * (rand(obj.dim, Napp) - 0.5));
  case 'period'
    S = [2 : 2: 2*floor(Napp/2)]/2;
    S = [-S S];
    % Change Napp to the correct form
    obj.Napp = obj.dim * (length(S) + 1);
    obj.Nperdim = length(S)+1;
    S = obj.period * repmat(S, [obj.dim 1]);
  otherwise
    error('Unknown sampling distribution');
end
obj.omega = S;
end