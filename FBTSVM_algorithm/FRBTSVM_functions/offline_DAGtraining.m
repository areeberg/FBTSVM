function  [ftsvm_struct,trdata,trlabel] = offline_DAGtraining(Traindata,Trainlabel,Parameter)

%% Function:  train iFBTSVM multi DAG approach
% Input:      
% Traindata         -  the train data where the feature are stored
% Trainlabel        -  the  lable of train data  
% Parameter         -  the parameters for ftsvm
%
% Output:    
% ftsvm_struct      -  ftsvm model

trdata=Traindata;
trlabel=Trainlabel;

%new version
CC=Parameter.CC; %Usually C1=C3, so CC=CC2
CC2=Parameter.CC2;
CR=Parameter.CR; %Usually C2=C4, so CR=CR2
CR2=Parameter.CR2;
eps=Parameter.eps; 
max_eva=Parameter.maxeva; %maximum of function evaluations to each train/update the model

 st1 = cputime;

numclasses=size(unique(Trainlabel),1);
classes=unique(Trainlabel);

for cl=1:numclasses
Xp=Traindata(Trainlabel==classes(cl),:);
Xpi=find(Trainlabel==(classes(cl)));
lp=size(Trainlabel(Trainlabel==classes(cl)),1);
Lp=ones(lp,1);

    for scl=cl+1:numclasses
     
    Lpi=find(Trainlabel==(classes(scl)));
    Xn=Traindata(Trainlabel==classes(scl),:);
    ln=size(Trainlabel(Trainlabel==classes(scl)),1);
    Ln=-ones(ln,1);

    X=[Xp;Xn];
    L=[Lp;Ln];
    % compute fuzzy membership
    [sp,sn,NXpv,NXnv]=fuzzy(Xp,Xn,Parameter);


S=[Xp ones(lp,1)];
R=[Xn ones(ln,1)];

%TODO - update to calc and CCp and CCn
% CC1=CC*sn;
% CC2=CC*sp;
% 
% fprintf('Optimising ...\n');
% 
%         tic
%         [alpha ,vp,iter,pgp] =  L1CDex(R,S,CR,CC2);%positive
% 
%         [beta , vn,iter2,pgn] =  L1CDex(S,R,CR,CC1);%negative
%         
%         
CCp=CC*sn;
CCn=CC2*sp;

%fprintf('Optimising ...\n');

        tic
        [alpha ,vp,iter,pgp] =  calc(R,S,CR2,CCn,eps,max_eva);%positive

        [beta , vn,iter2,pgn] =  calc(S,R,CR,CCp,eps,max_eva);%negative
        
        
        
        vn=-vn;
        ttrain=toc;
  
ExpendTime=cputime - st1; 

%ftsvm_struct.scaleData=scaleData;

%ftsvm_struct(cl).X = X;
%ftsvm_struct(cl).L = L;
ftsvm_struct(cl,scl).sp = sp;
ftsvm_struct(cl,scl).sn = sn;

ftsvm_struct(cl,scl).alpha = alpha;
ftsvm_struct(cl,scl).beta  = beta;
ftsvm_struct(cl,scl).vp = vp;
ftsvm_struct(cl,scl).vn = vn;

ftsvm_struct(cl,scl).Parameter = Parameter;
ftsvm_struct(cl,scl).time=ExpendTime;

ftsvm_struct(cl,scl).NXpv=NXpv;
ftsvm_struct(cl,scl).NXnv=NXnv;
ftsvm_struct(cl,scl).nv=length(NXpv)+length(NXnv);
%ftsvm_struct(cl).Xp=Xp; %input positive values
%ftsvm_struct(cl).Xn=Xn; %input negative values

ftsvm_struct(cl,scl).pgp=unique(pgp);
ftsvm_struct(cl,scl).pgn=unique(pgn);
ftsvm_struct(cl,scl).classe=classes(cl);
ftsvm_struct(cl,scl).Xpi=Xpi;
ftsvm_struct(cl,scl).Lpi=Lpi;




    end
end
clearvars vp vn alpha beta CC1 CC2 Xp Xn sp sn pgp pgn NXpv NXnv lp ln 
end