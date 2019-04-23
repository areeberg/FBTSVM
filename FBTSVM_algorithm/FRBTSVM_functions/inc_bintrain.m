function  [ftsvm_struct,trdata,trlabel] = inc_bintrain(Traindata,Trainlabel,Parameter,ftsvm_struct,data,label)
%addpath('/home/alexandre/MATLAB/Randfeat_releasever');
%% Function: train incremental binary iFBTSVM
pos=false;
neg=false;
if ( nargin>6||nargin<6) 
    %help  ftsvmtrain
end

% CC=Parameter.CC; %Get C1=C3
% CR=Parameter.CR; %Get C2=C4

CC=Parameter.CC; %Usually C1=C3, so CC=CC2
CC2=Parameter.CC2;
CR=Parameter.CR; %Usually C2=C4, so CR=CR2
CR2=Parameter.CR2;
eps=Parameter.eps; 
max_eva=Parameter.maxeva; %maximum of function evaluations to each train/update the model
st1 = cputime;


[groupIndex, groupString] = grp2idx(Trainlabel);
groupIndex = 1 - (2* (groupIndex-1));
scaleData = [];
type=[0 0];
%% +1 class
trainp=Traindata((Trainlabel==1),:);

if isempty(trainp)==1
  warning('No +1 data');
else
  S=createlinearSR(trainp,ftsvm_struct,'positive'); 
  if  isempty(S)==1
    warning('After filtered there is no +1 data left');   
  else
      type(1,1)=1;
      
  end%S if    
end %trainp if

trainn=Traindata((Trainlabel==-1),:);
if isempty(trainn)==1
  warning('No -1 data');
else
  R=createlinearSR(trainn,ftsvm_struct,'positive');  
  if  isempty(R)==1
  warning('After filtered there is no -1 data left');  
  else
     type(1,2)=1;
  end%S if    
end %trainp if

%% Update the model

if isequal(type,[0 0])==1
   disp('No update needed'); 
    trdata=data; 
    trlabel=label; 
%ftsvm_struct=ftsvm_struct;
end

if isequal(type,[1 0])==1
   disp('Update +1 only');   
   
    trdata=[data;S(:,1:end-1)]; 
    trlabel=[label;ones(size(S,1),1)]; 
     
Xpi=find(label==1);
Lpi=find(label==-1);
Xp=data(Xpi,:);
Xn=data(Lpi,:);
Xp=[Xp;S(:,1:end-1)];

%Xn=[Xn;R(:,1:end-1)];
Xpi=find(trlabel==1);
Lpi=find(trlabel==-1);
Lp=ones(size(Xp,1),1);
Ln=-ones(size(Xn,1),1); 
X=[Xp;Xn];
L=[Lp;Ln];
% compute fuzzy membership
[sp,sn,NXpv,NXnv]=fuzzy(Xp,Xn,Parameter);
 
 S=[Xp ones(size(Xp,1),1)];
 R=[Xn ones(size(Xn,1),1)];
%  CC1=CC*sn;
%  CC2=CC*sp;
%  [alpha ,vp,iter,pgp] =  L1CDex(R,S,CR,CC2);%positive
%  [beta , vn,iter2,pgn] =  L1CDex(S,R,CR,CC1);%negative
  CCp=CC*sn;
CCn=CC2*sp;

fprintf('Optimising ...\n');

        tic
        [alpha ,vp,iter,pgp] =  calc(R,S,CR2,CCp,eps,max_eva);%positive

        [beta , vn,iter2,pgn] =  calc(S,R,CR,CCn,eps,max_eva);%negative
        
 ExpendTime=cputime - st1; 
 
 
end

if isequal(type,[0 1])==1  %Updated [1,0] to [0,1]
   disp('Update -1 only');   
    trdata=[data;R(:,1:end-1)]; 
    trlabel=[label;-1*ones(size(R,1),1)]; 
        
Xpi=find(label==1);
Lpi=find(label==-1);
Xp=data(Xpi,:);
Xn=data(Lpi,:);
%Xp=[Xp;S(:,1:end-1)];
Xn=[Xn;R(:,1:end-1)];
Xpi=find(trlabel==1);
Lpi=find(trlabel==-1);
Lp=ones(size(Xp,1),1);
Ln=-ones(size(Xn,1),1); 
X=[Xp;Xn];
L=[Lp;Ln];
% compute fuzzy membership
[sp,sn,NXpv,NXnv]=fuzzy(Xp,Xn,Parameter);
 
 S=[Xp ones(size(Xp,1),1)];
 R=[Xn ones(size(Xn,1),1)];
%  CC1=CC*sn;
%  CC2=CC*sp;
%  [alpha ,vp,iter,pgp] =  L1CDex(R,S,CR,CC2);%positive
%  [beta , vn,iter2,pgn] =  L1CDex(S,R,CR,CC1);%negative
 
  CCp=CC*sn;
CCn=CC2*sp;

fprintf('Optimising ...\n');

        tic
        [alpha ,vp,iter,pgp] =  calc(R,S,CR,CCp,eps,max_eva);%positive

        [beta , vn,iter2,pgn] =  calc(S,R,CR2,CCn,eps,max_eva);%negative
        
 ExpendTime=cputime - st1;     
    
    
end

if isequal(type,[1 1])==1
   disp('Update both classes');   
   %Add new filtered data to current data
trdata=[data;S(:,1:end-1);R(:,1:end-1)]; 
trlabel=[label;ones(size(S,1),1);-1*ones(size(R,1),1)]; 

Xpi=find(label==1);
Lpi=find(label==-1);
Xp=data(Xpi,:);
Xn=data(Lpi,:);
Xp=[Xp;S(:,1:end-1)];
Xn=[Xn;R(:,1:end-1)];
Xpi=find(trlabel==1);
Lpi=find(trlabel==-1);
Lp=ones(size(Xp,1),1);
Ln=-ones(size(Xn,1),1); 
X=[Xp;Xn];
L=[Lp;Ln];
% compute fuzzy membership
[sp,sn,NXpv,NXnv]=fuzzy(Xp,Xn,Parameter);
 
 S=[Xp ones(size(Xp,1),1)];
 R=[Xn ones(size(Xn,1),1)];
 %old version
%  CC1=CC*sn;
%  CC2=CC*sp;
%  [alpha ,vp,iter,pgp] =  L1CDex(R,S,CR,CC2);%positive
%  [beta , vn,iter2,pgn] =  L1CDex(S,R,CR,CC1);%negative
 
 CCp=CC*sn;
CCn=CC2*sp;

fprintf('Optimising ...\n');

        tic
        [alpha ,vp,iter,pgp] =  calc(R,S,CR,CCn,eps,max_eva);%positive

        [beta , vn,iter2,pgn] =  calc(S,R,CR2,CCp,eps,max_eva);%negative
 ExpendTime=cputime - st1; 
end






%% Updating the model (ftsvm_struct) 

if isequal(type,[0 0])==1

ftsvm_struct=ftsvm_struct;
else
%ftsvm_struct=ftsvm_struct;
%NEW
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


end
