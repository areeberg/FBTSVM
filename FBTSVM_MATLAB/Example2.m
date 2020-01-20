%% Gisette dataset example - Create and update model

%% Load dataset
%Gisette dataset
addpath(genpath(pwd))
load('/gisette.mat');

%Train dataset
traindata=full(Data.train.X);
trainlabel=Data.train.Y;
clearvars Data

%% Set the parameters
Parameter.CC = 8; %C1=C3
Parameter.CC2=8;
Parameter.CR = 2; %C2=C4
Parameter.CR2=2;
Parameter.eps=0.001; %epsilon to avoid inverse matrix calculation error
Parameter.maxeva=500; %maximum of function evaluations to each train/update the model
Parameter.u=0.01; %fuzzy parameter
Parameter.epsilon=1e-10; %fuzzy epsilon
Parameter.repetitions=3;
Parameter.phi=0;

% To use the Kernel approximation parameters uncomment the lines below
% Parameter.kernel_name='rbf';
% Parameter.kernel_param=0.2;
% Parameter.feat_dimensionality=5000;
% Parameter.Napp=6000;
% Parameter.options=[];
kobj=[];
ini_size=0.005; %percentage of the data
batch_size=1000; %int or percentage

%% Initial training

%If using kernel approximation
%traindata=approx_kernel(traindata,Parameter);

traindata_ini=traindata(1:ceil(length(traindata)*ini_size),:);
trainlabel_ini=trainlabel(1:ceil(length(trainlabel)*ini_size));


model=create_bin_model(traindata_ini,trainlabel_ini,Parameter,kobj)



%% Update model
%get the rest of the data
traindata_up=traindata(ceil(length(traindata)*ini_size)+1:end,:);
trainlabel_up=trainlabel(ceil(length(trainlabel)*ini_size)+1:end);

model=update_bin_model(model,traindata_up,trainlabel_up,batch_size,Parameter,kobj)


%% Classify model
 % %Test dataset
load('testdata.mat');
load('testlabel.mat');

[acc2,outclass2,time2, fp2, fn2]= bin_classify(model,testdata,testlabel);








