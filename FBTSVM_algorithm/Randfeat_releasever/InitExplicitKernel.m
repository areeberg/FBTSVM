function obj = InitExplicitKernel( kernel, alpha, D, Napp, options )
%INITEXPLICITKERNEL compute kernel based on explicit linear features
%
% kernel - the name of the kernel. Supported options are: 
%             'rbf': Gaussian, 
%             'laplace': Laplacian, 
%             'chi2': Chi-square, 
%             'chi2_skewed': Skewed Chi-square,
%             'intersection', Histogram Intersection, 
%             'intersection_skewed', Skewed Histogram Intersection
% alpha  - the parameter of the kernel, e.g., the gamma in \exp(-gamma ||x-y||) 
%        for the Gaussian.
% D      - the number of dimensions
% Napp 	 - the number of random points you want to sample
% options: options. Now including only: 
%         options.method: 'sampling' or 'signals', signals for [Vedaldi 
%                         and Zisserman 2010] type of fixed interval sampling. 
%                         'sampling' for [Rahimi and Recht 2007] type of 
%                         Monte Carlo sampling.
%
% copyright (c) 2010 
% Fuxin Li - fuxin.li@ins.uni-bonn.de
% Catalin Ionescu - catalin.ionescu@ins.uni-bonn.de
% Cristian Sminchisescu - cristian.sminchisescu@ins.uni-bonn.de

% number of explicit features with which to approximate
if nargin < 4
  Napp = 10; 
end

switch kernel
  case 'rbf'
    % check
    obj = rf_init('gaussian', alpha, D, Napp, options);
    
  case 'laplace'
    % not verified
    obj = rf_init('laplace', alpha, D, Napp, options);
    
  case 'chi2'
    options.method = 'signals';
    options.period = 6e-1;
    obj = rf_init('chi2', alpha, D, Napp, options);
    
  case 'chi2_skewed'
    obj = rf_init('chi2', alpha, D, Napp, options);
    obj.name = 'chi2_skewed';
    
  case 'intersection'
    obj = rf_init('intersection', alpha, D, Napp, options);
  
  case 'intersection_skewed'
    obj = rf_init('intersection', alpha, D, Napp, options);
    obj.name = 'intersection_skewed';
    % Linear: no approximation, Napp is ignored
    case 'linear'
        obj.name = 'linear';
        obj.Napp = D;
        obj.dim = D;
        obj.final_dim = D;
  otherwise
    error('Unknown kernel');
end

end

