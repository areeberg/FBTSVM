function data = approx_kernel(trdata,Parameter)
addpath('/Randfeat_releasever');

%% Inputs
% data - data input matrix (NxD)
% Parameters
%% Output
%data with approximate kernel

if isfield(Parameter,'kernel_name')==1
kobj = InitExplicitKernel(Parameter.kernel_name,Parameter.kernel_param, Parameter.feat_dimensionality, Parameter.Napp,Parameter.options);
data= rf_featurize(kobj, double(trdata));
    
end


end