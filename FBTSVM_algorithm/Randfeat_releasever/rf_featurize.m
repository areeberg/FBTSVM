function F = rf_featurize(obj, X, Napp)
%RF_FEATURIZE returns the features corresponding to the inputs X
%
% obj   - random feature object initialized by rf_init.
% Napp  - specifies the number of features to be extracted. If obj.method is
%       signals then it is specified per dimension as 2*floor(Napp/2)+1 
%       otherwise it is the number of terms in the MC approximation.
%
% copyright (c) 2010 
% Fuxin Li - fuxin.li@ins.uni-bonn.de
% Catalin Ionescu - catalin.ionescu@ins.uni-bonn.de
% Cristian Sminchisescu - cristian.sminchisescu@ins.uni-bonn.de

[N D] = size(X);

if ~exist('Napp','var')
    if ~isfield(obj,'period') && isfield(obj,'omega')
        Napp = size(obj.omega,2);
    else
        Napp = obj.Napp;
    end
end
if isfield(obj,'omega') && Napp > size(obj.omega,2) && ~isfield(obj,'period')
    disp(['Warning: selected number of random features ' num2str(Napp) 'more than built-in number of random features ' num2str(size(obj.omega,2)) '.']);
    disp(['Changing the number of random features to ' num2str(size(obj.omega,2)) '.']);
    disp('You can increase the built-in number in rf_init()');
    Napp = size(obj.omega,2);
end

if D ~= obj.dim
  error('Dimension mismatch!');
end

switch obj.name
    case 'gaussian'
        F = sqrt(2) * (cos( X * obj.omega(:,1:Napp) + repmat(obj.beta'*2*pi,N,1)));
       
        
    case 'chi2'
        % only this fourier analytic treatment no mc estimation yet for chi2
        if strcmp(obj.distribution, 'period')
            F = zeros(N, D * obj.Nperdim);
            for i = 1: D
                F(:,((i-1)*obj.Nperdim+1):(i*obj.Nperdim)) = sqrt(obj.period) * [sech(0)*sqrt(X(:,i)), ...
                    cos(log(X(:,i))*obj.omega(i,1:(obj.Nperdim-1))) .* sqrt(X(:,i) * sech(pi * obj.omega(i,1:(obj.Nperdim-1))))];
            end
            F(isinf(F)) = 0;
            F(isnan(F)) = 0;
        end
    case 'chi2_skewed'
        % skewed multiplicative chi-square kernel
        F = sqrt(2) * cos( log(X+obj.kernel_param) * 0.5 * obj.omega(:,1:Napp) + repmat(obj.beta'*2*pi,N,1));
    case 'intersection_skewed'
        F = sqrt(2) * cos( log(X+obj.kernel_param) * obj.omega(:,1:Napp) + repmat(obj.beta'*2*pi,N,1));
    case 'laplace'
        F = sqrt(2) * (cos( X * obj.omega(:,1:Napp) + repmat(obj.beta'*2*pi,N,1)));
        % Linear is just replicate
    case 'linear'
        F = X;
        
    case 'intersection'
    F = [];
    for i = 1: D
      cterm = cos(log(X(:,i))*obj.omega(i,:)) .* sqrt(2/pi* X(:,i) * (1./(1 + 4 * obj.omega(i,:).^2)));
%       cterm(isnan(cterm)) = 0;
      sterm = sin(log(X(:,i))*obj.omega(i,:)) .* sqrt(2/pi* X(:,i) * (1./(1 + 4 * obj.omega(i,:).^2)));
%       sterm(isnan(sterm)) = 0;
      
      F = [F sqrt(obj.period) * [sqrt(2/pi)*sqrt(X(:,i)), cterm, sterm ]];
    end
    % this is a clean up of the rf. If X is 0 then log(X) becomes infinity
    % and cos(log(X)) is NaN. We correct this in the end by putting 0
    F(isnan(F)) = 0; 
    
    otherwise
        error('Unknown kernel approximation scheme');
end
end
