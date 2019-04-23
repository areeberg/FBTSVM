% copyright (c) 2010 
% Fuxin Li - fuxin.li@ins.uni-bonn.de
% Catalin Ionescu - catalin.ionescu@ins.uni-bonn.de
% Cristian Sminchisescu - cristian.sminchisescu@ins.uni-bonn.de

% DEMO for approximation to the kernel matrix

load bow_features
test_kernel(Feats,Feats,'rbf',10,200);

load phog_features
test_kernel(Feats,Feats,'chi2_skewed',0.03,200);
load bow_features
test_kernel(Feats,Feats,'chi2_skewed',0.03,200);

load bow_features
test_kernel(Feats,Feats,'laplace',1,200);
load bow_features
test_kernel(Feats,Feats,'chi2',[],3);
