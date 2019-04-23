function ftsvm_struct = create_bin_model_mem(trdata,trlabel,Parameter,kobj)
%addpath('/Randfeat_releasever');

%% Inputs
% data - data input matrix (NxD)
% label - label input array (Nx1, +1 and -1)
% ini_size - initialization size (Data percentual from the begin)
%% Output
%model 

data=[]; %Pre-initialize for efficiency
label=[]; %Pre-initialize for efficiency
oridata=trdata;
orilabel=trlabel;
if isfield(Parameter,'kernel_name')==1
     trdata= rf_featurize(kobj, double(trdata));
     
end
fprintf('Optimising ...\n');
%Initial training
[ftsvm_struct,data,label] = offline_bintrain_mem(trdata,trlabel,Parameter);

ftsvm_struct.oridata=oridata;
ftsvm_struct.orilabel=orilabel;

%Clear initial traindata from memory



end