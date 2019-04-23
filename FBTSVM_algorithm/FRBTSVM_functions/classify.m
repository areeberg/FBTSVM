function [acc,outclass,time, fp, fn,dist]= classify(ftsvm_struct,Testdata,Testlabel,kobj)
%addpath('/home/alexandre/MATLAB/Randfeat_releasever');
% Function:  testing ftsvm on test data DAG version
% Input:
% ftsvm_struct         - the trained  ftsvm model
% Testdata             - test data
% Testlabel            - test label
%
% Output:
% acc                    - accuracy
% outclass               - predict label


    
    
    if isfield(ftsvm_struct(1,2).Parameter,'kernel_name')==1
    %kobj = InitExplicitKernel(ftsvm_struct(1,2).Parameter.kernel_name,ftsvm_struct(1,2).Parameter.kernel_param, ftsvm_struct(1,2).Parameter.feat_dimensionality, ftsvm_struct(1,2).Parameter.Napp,ftsvm_struct(1,2).Parameter.options);
      %kobj = InitExplicitKernel('rbf',0.4, 2, 150,[]);

    Testdata= rf_featurize(kobj, double(Testdata));
    end

    [rt,ct]=size(Testdata);
    
    tic;
    
    fprintf('Testing ...\n');
        
    
%Get all possible classes
classes = unique(Testlabel);

% Get number of classes
n_classes = size(ftsvm_struct,2);

A=zeros(n_classes,n_classes);

%number of classifiers    
numclassifiers = size(find(arrayfun(@(ftsvm_struct) ~isempty(ftsvm_struct.vp),ftsvm_struct)),1);


remain = classes;
i=1;
j=2;
resp=0;
fans=[];
dist=[];
for k=1:size(Testdata,1)
%while length(remain) > 1
for x=1:n_classes-1
       %remain=remain(1:end-1);
       vp=ftsvm_struct(i,j).vp;
       vn=ftsvm_struct(i,j).vn;
       fp=(Testdata(k,:)*vp(1:(length(vp)-1))+vp(length(vp)))./norm(vp(1:(length(vp)-1)));
       fn=(Testdata(k,:)*vn(1:(length(vn)-1))+vn(length(vn)))./norm(vn(1:(length(vn)-1)));
       pp=i;
       jj=j;
       if abs(fp)<abs(fn)==1
           i=j;
           j=j+1;
           
           %remain(i)=[];
       else
           
       j=j+1;
      %remain(j)=[];
       end
       
 
end

if abs(fp)<abs(fn)==1
    dist(k)=fp;
    resp=jj;
    
else
    dist(k)=fn;
    resp=pp;
    
end
%disp(resp)
fans=[fans (resp-1)];
remain=classes;
i=1;
j=2;
resp=0;
end
fans=fans';
dist=dist';
correctn=sum(fans==Testlabel);
        acc=100*correctn/length(Testlabel);
        fprintf('Accuracy : %3.4f (%d/%d)\n',acc,correctn,length(Testlabel));
outclass=fans;
time=toc;
clearvars Testdata Testlabel
end
