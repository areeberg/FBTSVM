function  [ftsvm_struct,trdata,trlabel] = inc_ifbtsvm_mem(Traindata,Trainlabel,Parameter,ftsvm_struct,data,label,Traindata_ori)
%addpath('/home/alexandre/MATLAB/Randfeat_releasever');
%% Function:  train cdftsvm multi DAG approach
% Input:      
% Traindata         -  the train data where the feature are stored
% Trainlabel        -  the  lable of train data  
% Parameter         -  the parameters for ftsvm
%
% Output:    
% ftsvm_struct      -  ftsvm model


%New version
CC=ftsvm_struct(1,2).Parameter.CC; %Usually C1=C3, so CC=CC2
CC2=ftsvm_struct(1,2).Parameter.CC2;
CR=ftsvm_struct(1,2).Parameter.CR; %Usually C2=C4, so CR=CR2
CR2=ftsvm_struct(1,2).Parameter.CR2;
eps=ftsvm_struct(1,2).Parameter.eps; 
max_eva=ftsvm_struct(1,2).Parameter.maxeva; %maximum of function evaluations to each train/update the model
st1 = cputime;

%% Activate the slice-variable multithread
if ftsvm_struct(1,2).Parameter.sliv==true
  parforArg = Inf;
else
  parforArg = 0;
end

%% SEPARATE DATA INTO CLASSES 1
numclasses=size(unique(Trainlabel),1);
classes=unique(Trainlabel);
data_ori=ftsvm_struct(1,1).oridata;
for j=1:numclasses
   tdata{j}=Traindata(Trainlabel==classes(j),:);
   tlabel{j}=classes(j)*ones(size(tdata{j},1),1);
   tdata_ori{j}=Traindata_ori(Trainlabel==classes(j),:);
end

%% FILTER DATA----------------------------------------------------------- 
tlabel_ori=tlabel;
for cl=1:numclasses
    trainp=(tdata{cl});

v=[];  
vori=[];
voril=[];

a=find(classes~=classes(cl));
 for scl=1:size(a,1)   
    
ve=sort([cl a(scl)]);        

[S{ve(1),ve(2)},dele]=createlinearSR(trainp,ftsvm_struct(ve(1),ve(2)),'positive');
%Sn{cl,nonempty(scl)}=createlinearSR(trainn,ftsvm_struct(cl,nonempty(scl)),'negative');
v=[v;S{ve(1),ve(2)}];
tdata_orit=tdata_ori{1,cl};

tdata_orit(dele,:)=[];


vori=[vori; tdata_orit]; 

end
filtdata{cl}=unique(v,'rows');
filtdata_ori{cl}=unique(vori,'rows');
data=[data;(filtdata{cl}(:,1:end-1))];
label=[label;(classes(cl)*ones(size(filtdata{cl},1),1))];
filtlabel{cl}=(classes(cl)*ones*find(~cellfun('isempty',filtdata(cl))));
data_ori=[data_ori;(filtdata_ori{cl}(:,:))];
end
trdata=data;
trlabel=label;

ftsvm_struct(1,1).oridata=data_ori;
ftsvm_struct(1,1).orilabel=label;

clearvars filtdata_ori data_ori vori tdata_orit

clearvars cl scl
%--------------------------------------------------------------------


%% SEPARATE DATA INTO CLASSES 2 (after filtering)
% numclasses=size(unique(Trainlabel),1);
% classes=unique(Trainlabel);
% 
% for j=1:numclasses
%    tdata{j}=Traindata(Trainlabel==classes(j),:);
%    tlabel{j}=classes(j)*ones(size(tdata{j},1),1);
%    
% end
%% 
%CR=ftsvm_struct(1,2).Parameter.CR;

%start paralel pool
nfiltclass=size(find(~cellfun('isempty',filtlabel(:))),1);
nempty = find(~cellfun('isempty',filtdata()));
classes=unique(label);
for cl=1:nfiltclass-1
nonempty = find(arrayfun(@(ftsvm_struct) ~isempty(ftsvm_struct.vp),ftsvm_struct(cl,:)));


parfor (scl=1:size(nonempty,2),parforArg) %parfor
%for (scl=1:size(nonempty,2)) %parfor  
Xpi=find(label==(classes(cl)));


if size(classes,1)==2
Lpi=find(label==find(unique(classes~=classes(cl))));

else
Lpi=find(label==(classes(nonempty(scl))));
%Lpi=find(label==((nonempty(scl))));

end

Xp=data(Xpi,:);
Xn=data(Lpi,:);


 Lp=ones(size(Xp,1),1);
 Ln=-ones(size(Xn,1),1);
 
 X=[Xp;Xn];
 L=[Lp;Ln];
% compute fuzzy membership
[sp,sn,NXpv,NXnv]=fuzzy(Xp,Xn,Parameter);
 

 S=[Xp ones(size(Xp,1),1)];
 R=[Xn ones(size(Xn,1),1)];  
        
CCp=CC*sn;
CCn=CC2*sp;

%fprintf('Optimising ...\n');

        tic
        %maybe S and R are swapped
        [alpha ,vp,iter,pgp] =  calc(R,S,CR,CCn,eps,max_eva);%positive

        [beta , vn,iter2,pgn] =  calc(S,R,CR2,CCp,eps,max_eva);%negative
        
        
        %vars to update
        alpha1{scl}=alpha;
        beta1{scl}=beta;
        sp1{scl}=sp;
        sn1{scl}=sn;
        xpi1{scl}=Xpi;
        lpi1{scl}=Lpi;
        xp1{scl}=Xp;
        xn1{scl}=Xn;
        vp1{scl}=vp;
        vn1{scl}=vn;
        pgp1{scl}=pgp;
        pgn1{scl}=pgn;
        eTime{scl}=cputime - st1; 
        
end
ftsvm_struct=fillstruct(ftsvm_struct,cl,alpha1,beta1,sp1,sn1,xpi1,lpi1,vp1,vn1,pgp1,pgn1,eTime);

clearvars beta1 alpha1 sp1 sn1 xpi1 lpi1 xp1 xn1 vp1 vn1 pgp1 pgn1 eTime

%tabela2(cl)=tabela;
end



end
%end