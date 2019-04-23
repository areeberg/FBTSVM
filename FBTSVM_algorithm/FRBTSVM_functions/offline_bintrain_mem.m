function  [ftsvm_struct,data,label] = offline_bintrain_mem(Traindata,Trainlabel,Parameter)
%% Function: train offline binary
% Input:      
% Traindata         -  the train data where the feature are stored
% Trainlabel        -  the  lable of train data  
% Parameter         -  the parameters for the model
%
% Output:    
% ftsvm_struct      -  ftsvm model


%Check if there is at least two different classes
A=unique(Trainlabel);

if size(A,1)<=1
    disp('Two classes are required for the initial training.');
    %exit
end


data=Traindata; %Data used for the initial training
label=Trainlabel; %Label used for the initial training
 

CC=Parameter.CC; %Usually C1=C3, so CC=CC2
CC2=Parameter.CC2;
CR=Parameter.CR; %Usually C2=C4, so CR=CR2
CR2=Parameter.CR2;
eps=Parameter.eps; 
max_eva=Parameter.maxeva; %maximum of function evaluations to each train/update the model

st1 = cputime;
[groupIndex, groupString] = grp2idx(Trainlabel); %get groups
groupIndex = 1 - (2* (groupIndex-1));



Xp=data(groupIndex==1,:);
Lp=Trainlabel(groupIndex==1);
Xn=data(groupIndex==-1,:);
Ln=Trainlabel(groupIndex==-1);
X=[Xp;Xn];
L=[Lp;Ln];
Xpi=find(label==1);
Lpi=find(label==-1);
% compute fuzzy membership
[sp,sn,NXpv,NXnv]=fuzzy(Xp,Xn,Parameter);


lp=sum(groupIndex==1);
ln=sum(groupIndex==-1);

S=[Xp ones(lp,1)];
R=[Xn ones(ln,1)];

CCp=CC*sn;
CCn=CC2*sp;

        tic
        [alpha ,vp,iter,pgp] =  calc(S,R,CR,CCp,eps,max_eva);%positive

        [beta , vn,iter2,pgn] =  calc(R,S,CR2,CCn,eps,max_eva);%negative
       
        vn=-vn;
        ttrain=toc;

ExpendTime=cputime - st1; 

ftsvm_struct.X = X;
ftsvm_struct.L = L;
ftsvm_struct.sp = sp;
ftsvm_struct.sn = sn;
ftsvm_struct.alpha = alpha;
ftsvm_struct.beta  = beta;
ftsvm_struct.vp = vp;
ftsvm_struct.vn = vn;
ftsvm_struct.Parameter = Parameter;
ftsvm_struct.groupString=groupString;
ftsvm_struct.time=ExpendTime;
ftsvm_struct.NXpv=NXpv;
ftsvm_struct.NXnv=NXnv;
ftsvm_struct.nv=length(NXpv)+length(NXnv);
ftsvm_struct.Xp=Xp; %input positive values
ftsvm_struct.Xn=Xn; %input negative values
ftsvm_struct.Xpi=Xpi;
ftsvm_struct.Lpi=Lpi;
ftsvm_struct.pgp=pgp;
ftsvm_struct.pgn=pgn;



end