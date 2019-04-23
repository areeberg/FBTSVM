function [ftsvm_struct,ndata,nlabel,score] = forget_bin(ftsvm_struct,data,label,score)
%phi - threshold to delete point
%score - 
%Alpha and beta <= phi. Forget a point when alpha and beta are <= phi "score" times to all
%classes.

n=ftsvm_struct.Parameter.repetitions;
phi=ftsvm_struct.Parameter.phi;
%get classes
classes=unique(label);
numclasses=size(classes,1);

[a,b]= find(arrayfun(@(ftsvm_struct) ~isempty(ftsvm_struct.vp),ftsvm_struct));

Calpha=ftsvm_struct.Xpi(ftsvm_struct.alpha<=phi);
Cbeta=ftsvm_struct.Lpi(ftsvm_struct.beta<=phi);
D=[Calpha;Cbeta];

%caso nao exista score ainda
if isempty(score)==1 
score=D;
score(:,2)=ones(size(D,1),1);

 else
[inte,idx,idx2]=intersect(D,score(:,1),'stable');
score(idx2,2)=score(idx2,2)+1;
%get different data
diff=setdiff(D,score(:,1));
diff(:,2)=ones(size(diff,1),1);
%add different data to score
score=[score;diff];
     
 end
%verificar caso Calpha ou Cbeta==0 


%% Removal section
exdata=find(score(:,2)>=n);
if isempty(exdata)==1
ndata=data;
nlabel=label;
end


if isempty(exdata)==0
ndata=data;
nlabel=label;

ndata(score(exdata),:)=[];
nlabel(score(exdata))=[];


v1=ftsvm_struct.Xpi;
v2=ftsvm_struct.Lpi;

v11=ftsvm_struct.alpha;
v12=ftsvm_struct.beta;

v21=ftsvm_struct.sp;
v22=ftsvm_struct.sn;


ftsvm_struct.alpha=v11;
ftsvm_struct.beta=v12;
ftsvm_struct.Xpi=v1;
ftsvm_struct.Lpi=v2;
ftsvm_struct.sp=v21;
ftsvm_struct.sn=v22;
score(exdata,:)=[];


end

end
