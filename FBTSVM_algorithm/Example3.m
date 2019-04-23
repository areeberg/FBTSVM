%% Fold 3 breakhis PFTAS dataset example - Single training
%86
addpath(genpath(pwd))
%% Load dataset

load('breakhis_traindata')
load('breakhis_trainlabel')
% %Train dataset
traindata=breakhis_traindata;
trainlabel=breakhis_trainlabel;
clearvars breakhis_traindata breakhis_trainlabel



Parameter.CC = 8; %C1=C3
Parameter.CC2=8;
Parameter.CR = 2; %C2=C4
Parameter.CR2=2;
Parameter.eps=0.001; %epsilon to avoid inverse matrix calculation error
Parameter.maxeva=500; %maximum of function evaluations to each train/update the model
%fuzzy
Parameter.u=0.01; %fuzzy parameter
Parameter.epsilon=1e-10; %fuzzy epsilon
%forgetting
Parameter.repetitions=3;
Parameter.phi=0.00001;
%Kernel approximation parameters
%if you want to use linear kernel, do not create the parameters in the
%structure (comment the Paramater. lines below).
Parameter.kernel_name='rbf';
Parameter.kernel_param=0.2;
Parameter.feat_dimensionality=163;
Parameter.Napp=300;
Parameter.options=[];
kobj = InitExplicitKernel(Parameter.kernel_name, Parameter.kernel_param, Parameter.feat_dimensionality,Parameter.Napp, Parameter.options);


ini_size=0.1; %Initial training size
batch_size=200; %int or percentage

%Train model (initial training + incremental training)
model=bin_iFBTSVM(traindata,trainlabel,ini_size,batch_size,Parameter,kobj)


%% Classify model
 % %Test dataset
load('breakhis_testlabel');
 load('breakhis_testdata');
[acc2,outclass2,time2, fp2, fn2]= bin_classify(model,breakhis_testdata,breakhis_testlabel,kobj);
