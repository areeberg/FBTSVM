function [ftsvm_struct,ndata,nlabel,score] = forgn(ftsvm_struct,data,label,score)
%Alpha and beta == 0. Forget a point when alpha and beta are 0 "n" times to all
%classes.

n=ftsvm_struct(1,2).Parameter.repetitions;
phi=ftsvm_struct(1,2).Parameter.phi;
%get classes
classes=unique(label);
numclasses=size(classes,1);

[a,b]= find(arrayfun(@(ftsvm_struct) ~isempty(ftsvm_struct.vp),ftsvm_struct));

for i=1:size(a,1)
C{a(i),i}=ftsvm_struct(a(i),b(i)).Xpi(ftsvm_struct(a(i),b(i)).alpha<=phi);
C{b(i),i}=ftsvm_struct(a(i),b(i)).Lpi(ftsvm_struct(a(i),b(i)).beta<=phi);
end

%map of classifiers into ftsvm_struct
idx=not(cellfun(@isempty,C));
X=[];

for i=1:size(C,1)
    val=find(idx(i,:)~=0);
    
    %add an empty check here!
%     if isempty(C(i,val(1)))==1
%     continue
%     end
    if isempty(cell2mat(C(i,val(1))))==1
       disp('o') 
    end
    X=cell2mat(C(i,val(1))); 
    
    
    
    for j=2:size(val,2)
   A=cell2mat(C(i,val(j)));
    X=intersect(X,A);       
    end
    Y{i}=X;
    
end

%data with alpha and beta are 0
D = cell2mat(cellfun(@(Y)vertcat(Y{:}),num2cell(Y,2),'UniformOutput',false));
%% Counting alpha and beta ==0
%if exist('score','var')==0
 if isempty(score)==1   
%copy D
score=D;
%count 1 to each occurence of beta and alpha==0
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


%% Points removal section
%remove points that alpha and beta are equal to "n"
exdata=find(score(:,2)>=n);
if isempty(exdata)==1
ndata=data;
nlabel=label;
end

%get score bigger then n

if isempty(exdata)==0

%score(exdata)=[];

%map of classifiers into ftsvm_struct
%Y are the points with alpha and beta equal to 0

for i=1:size(a,1)
    %Get all possible points
%C{a(i),i}=ftsvm_struct(a(i),b(i)).Xpi;
%C{b(i),i}=ftsvm_struct(a(i),b(i)).Lpi;

C{a(i),i}=ftsvm_struct(a(i),b(i)).Xpi(ftsvm_struct(a(i),b(i)).alpha<=phi);
C{b(i),i}=ftsvm_struct(a(i),b(i)).Lpi(ftsvm_struct(a(i),b(i)).beta<=phi);
end

idx=not(cellfun(@isempty,C));
X=[];
for i=1:size(C,1)
    val=find(idx(i,:)~=0);
    
    %add an empty check here!
    X=cell2mat(C(i,val(1))); 
    
    for j=2:size(val,2)
   %A=cell2mat(C(i,val(j)));
    X=intersect(X,exdata);       
    end
    Y{i}=X;
end

%D are the points where alpha AND beta are 0
D = cell2mat(cellfun(@(Y)vertcat(Y{:}),num2cell(Y,2),'UniformOutput',false));
ndata=data;
nlabel=label;
%remove the points that are 0
if max(D)<=size(ndata,1)
% if size(D,1)~=0
%     disp('a')
% end
    
ndata(D,:)=[];
nlabel(D)=[];
score(exdata,:)=[];
end
% ndata(D,:)=[];
% nlabel(D)=[];
% score(D,:)=[];


    

for cl=1:numclasses-1
nonempty = find(arrayfun(@(ftsvm_struct) ~isempty(ftsvm_struct.vp),ftsvm_struct(cl,:)));

    for scl=1:size(nonempty,2)
         bb=cell2mat(Y(cl)); %first value x - (x,y)
         cc=cell2mat(Y(nonempty(scl))); %second value y - (x,y)
    
         
         v1=ftsvm_struct(cl,nonempty(scl)).Xpi;
         v2=ftsvm_struct(cl,nonempty(scl)).Lpi;
         bb1=find((ismember(v1,bb))==1);
         cc1=find((ismember(v2,cc))==1);
         %bb1=find((ismember(v1,score(:,1)))==1);
         %cc1=find((ismember(v2,score(:,1)))==1);
         
         
         v1(bb1)=[];
         v2(cc1)=[];
         
         
         %converter o bb em posicao de vetor
         v12=ftsvm_struct(cl,nonempty(scl)).alpha;
         v12(bb1)=[];
         v22=ftsvm_struct(cl,nonempty(scl)).beta;
         v22(cc1)=[];
         
         v13=ftsvm_struct(cl,nonempty(scl)).sp;
         v13(bb1)=[];
         v23=ftsvm_struct(cl,nonempty(scl)).sn;
         v23(cc1)=[];
         
         
    ftsvm_struct(cl,nonempty(scl)).Xpi=v1;
    ftsvm_struct(cl,nonempty(scl)).Lpi=v2;
    
    ftsvm_struct(cl,nonempty(scl)).alpha=v12;
    ftsvm_struct(cl,nonempty(scl)).beta=v22;
        
    ftsvm_struct(cl,nonempty(scl)).sp=v13;
    ftsvm_struct(cl,nonempty(scl)).sn=v23;
    
    end

end
end
end