function ftsvm_struct = create_fbtsvm_model(trdata,trlabel,Parameter,kobj)
addpath('/Randfeat_releasever');

%% Inputs
% data - data input matrix (NxD)
% label - label input array (Nx1, +1 and -1)
% ini_size - initialization size (Data percentual from the begin)
%% Output
%model 

data=[]; %Pre-initialize for efficiency
label=[]; %Pre-initialize for efficiency
if isfield(Parameter,'kernel_name')==1
    %kobj = InitExplicitKernel(Parameter.kernel_name,Parameter.kernel_param, Parameter.feat_dimensionality, Parameter.Napp,Parameter.options);
      %kobj = InitExplicitKernel('rbf',0.4, 2, 150,[]);
     trdata= rf_featurize(kobj, double(trdata));
     
end
%% Create model

[ftsvm_struct,data,label] = offline_DAGtraining(trdata,trlabel,Parameter);
%Clear initial traindata from memory



end