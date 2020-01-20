function [acc,outclass,time, fp, fn]= bin_classify_mem(ftsvm_struct,Testdata,Testlabel,kobj)
% Function:  testing ftsvm on test data
% Input:
% Model         - the trained model
% Testdata             - test data
% Testlabel            - test label
%
% Output:
% acc                    - accuracy
% outclass               - predict label
% time                   - processing time
% time                   - distance to positive plane
% time                   - distance to negative plane



    
    if isfield(ftsvm_struct.Parameter,'kernel_name')==1    
        %Testdata= rf_featurize(kobj, double(Testdata));

        Testdata= rf_featurize(kobj, double([ftsvm_struct.oridata;Testdata]));
        Testdata=Testdata(size(ftsvm_struct.oridata,1)+1:end,:);
    
    end
    
    

    [rt,ct]=size(Testdata);
    
    tic;
    
    %if ~isempty(ftsvm_struct.scaleData)
      %  scaleData=ftsvm_struct.scaleData;
%         for k = 1:size(Testdata, 2)
%             Testdata(:,k) = scaleData.scaleFactor(k) * ...
%                 (Testdata(:,k) +  scaleData.shift(k));
%         end
   % end
    
    
    %groupString=ftsvm_struct.groupString;
    vp=ftsvm_struct.vp;
    vn=ftsvm_struct.vn;
    
    X=ftsvm_struct.X;
    
    
    fprintf('Testing ...\n');
 
        
            fp=(Testdata*vp(1:(length(vp)-1))+vp(length(vp)))./norm(vp(1:(length(vp)-1)));
            fn=(Testdata*vn(1:(length(vn)-1))+vn(length(vn)))./norm(vn(1:(length(vn)-1)));

    
    f=fp+fn;
    
    classified=ones(rt,1);
    classified(abs(fp)>abs(fn)) = -1;
     classified(classified == -1) = 2;
    
    
    outclass = classified;
    unClassified = isnan(outclass);
    [~,groupString,glevels] = grp2idx(ftsvm_struct(1).L);
    
    outclass = glevels(outclass(~unClassified),:);
    
    if nargin>=3
        correct=sum(outclass==Testlabel);
        acc=100*correct/length(Testlabel);
        fprintf('Accuracy : %3.4f (%d/%d)\n',acc,correct,length(Testlabel));
    else
        acc=[];
        fprintf('the accuracy can not be calculated, because of lack of the labels of testing data\n');
    end
    time= toc;
end
