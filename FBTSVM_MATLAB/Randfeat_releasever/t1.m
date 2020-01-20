%% Falkon test
addpath('/media/alexandre/Data/Datasets/iftsvm/MNIST');
addpath('/home/alexandre/MATLAB/Randfeat_releasever');

%% Read datasets

%Training dataset
load('traindata.mat');
load('trainlabel.mat');

%Testing dataset
load('testlabel.mat');
load('testdata.mat');

tic
kobj = InitExplicitKernel('rbf',10, 784, 1500,[]);
z_omega = rf_featurize(kobj, double(traindata));
toc

load bow_features
test_kernel(Feats,Feats,'chi2_skewed',0.03,200);


