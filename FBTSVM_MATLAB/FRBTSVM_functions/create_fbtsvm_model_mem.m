function ftsvm_struct = create_fbtsvm_model_mem(trdata,trlabel,Parameter,kobj)
%addpath('/Randfeat_releasever');

%% Inputs
% data - data input matrix (NxD)
% label - label input array (Nx1, +1 and -1)
% ini_size - initialization size (Data percentual from the begin)
%% Output
%model 
%ftsvm_struct(1,1) - original data and label

data=[]; %Pre-initialize for efficiency
label=[]; %Pre-initialize for efficiency
oridata=trdata;
orilabel=trlabel;
if isfield(Parameter,'kernel_name')==1
    %kobj = InitExplicitKernel(Parameter.kernel_name,Parameter.kernel_param, Parameter.feat_dimensionality, Parameter.Napp,Parameter.options);
      %kobj = InitExplicitKernel('rbf',0.4, 2, 150,[]);
     trdata= rf_featurize(kobj, double(trdata));
     
end
%% Create model

[ftsvm_struct,data,label] = offline_DAGtraining(trdata,trlabel,Parameter);
%Clear initial traindata from memory

ftsvm_struct(1,1).oridata=oridata;
ftsvm_struct(1,1).orilabel=orilabel;

end