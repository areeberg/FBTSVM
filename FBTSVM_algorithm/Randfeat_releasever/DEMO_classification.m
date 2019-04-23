% copyright (c) 2010 
% Fuxin Li - fuxin.li@ins.uni-bonn.de
% Catalin Ionescu - catalin.ionescu@ins.uni-bonn.de
% Cristian Sminchisescu - cristian.sminchisescu@ins.uni-bonn.de

% Classification DEMO with linear approximation to different kernels
addpath('/media/alexandre/Data/Datasets/iftsvm/MNIST');
addpath('/home/alexandre/MATLAB/Randfeat_releasever');

%% Read datasets

%Training dataset
load('traindata.mat');
load('trainlabel.mat');

%Testing dataset
load('testlabel.mat');
load('testdata.mat');

addpath('./liblinear')

disp('Training model with random Fourier features on Gaussian kernel.')
kobj = InitExplicitKernel('rbf',10, 784, 1500,[]);
t1 = rf_featurize(kobj, double(traindata));
svmmodel = svmlin_train(double(trainlabel), sparse(t1));
testdata= rf_featurize(kobj, double(testdata));
[pred, accuracy] = svmlin_predict(double(testlabel), sparse(testdata), svmmodel);

load bow_features;
disp('Training model with linear kernel.');
svmmodel = svmlin_train(double(Labels(1:1601)'), sparse(double(Feats(1:1601,:))));
[pred, accuracy] = svmlin_predict(double(Labels(1602:3202)'), sparse(double(Feats(1602:3202,:))), svmmodel);

disp('Training model with random Fourier features on additive chi-square kernel.');
kobj = InitExplicitKernel('chi2',[], 300, 5,[]);
z_omega = rf_featurize(kobj, double(Feats));

svmmodel = svmlin_train(double(Labels(1:1601)'), sparse(z_omega(1:1601,:)));
[pred, accuracy] = svmlin_predict(double(Labels(1602:3202)'), sparse(z_omega(1602:3202,:)), svmmodel);

disp('Training model with random Fourier features on Gaussian kernel.')
kobj = InitExplicitKernel('rbf',10, 300, 1500,[]);
z_omega = rf_featurize(kobj, double(Feats));
svmmodel = svmlin_train(double(Labels(1:1601)'), sparse(z_omega(1:1601,:)));
[pred, accuracy] = svmlin_predict(double(Labels(1602:3202)'), sparse(z_omega(1602:3202,:)), svmmodel);

disp('Training model with random Fourier features on Laplacian kernel.')
kobj = InitExplicitKernel('laplace',1, 300, 1500,[]);
z_omega = rf_featurize(kobj, double(Feats));
svmmodel = svmlin_train(double(Labels(1:1601)'), sparse(z_omega(1:1601,:)));
[pred, accuracy] = svmlin_predict(double(Labels(1602:3202)'), sparse(z_omega(1602:3202,:)), svmmodel);

disp('Training model with random Fourier features on skewed chi-square kernel.')
kobj = InitExplicitKernel('chi2_skewed',0.03, 300, 1500,[]);
z_omega = rf_featurize(kobj, double(Feats));
svmmodel = svmlin_train(double(Labels(1:1601)'), sparse(z_omega(1:1601,:)));
[pred, accuracy] = svmlin_predict(double(Labels(1602:3202)'), sparse(z_omega(1602:3202,:)), svmmodel);
