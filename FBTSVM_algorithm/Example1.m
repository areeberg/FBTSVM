%% Gisette dataset example - Single training

%% Load dataset
%Gisette dataset
addpath(genpath(pwd))
load('/gisette.mat');

% %Train dataset
traindata=full(Data.train.X);
trainlabel=Data.train.Y;
clearvars Data


Parameter.CC = 8; %C1=C3
Parameter.CC2=8;
Parameter.CR = 2; %C2=C4
Parameter.CR2=2;
Parameter.eps=0.001; %epsilon to avoid inverse matrix calculation error
Parameter.maxeva=500; %maximum of function evaluations to each train/update the model
Parameter.u=0.01; %fuzzy parameter
Parameter.epsilon=1e-10; %fuzzy epsilon
Parameter.repetitions=3; %number of repetitions to forget
Parameter.phi=0;
%Kernel approximation parameters
%if you want to use linear kernel, do not create the parameters in the
%structure (comment the Paramater. lines below).
% Parameter.kernel_name='rbf';
% Parameter.kernel_param=0.2;
% Parameter.feat_dimensionality=5000;
% Parameter.Napp=6000;
% Parameter.options=[];
%kobj = InitExplicitKernel(Parameter.kernel_name, Parameter.kernel_param, Parameter.feat_dimensionality,Parameter.Napp, Parameter.options);
kobj=[];
ini_size=0.005; %percentage of the data
batch_size=1000; %int or percentage

%Train model (initial training + incremental training)
model=bin_iFBTSVM(traindata,trainlabel,ini_size,batch_size,Parameter,kobj)


%% Classify model
 % %Test dataset

load('testdata.mat');
load('testlabel.mat');

[acc2,outclass2,time2, fp2, fn2]= bin_classify(model,testdata,testlabel,kobj);