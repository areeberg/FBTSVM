function ftsvm_struct = update_bin_model_cd(ftsvm_struct,trdata,trlabel,batch_size,Parameter,kobj)
%addpath('/Randfeat_releasever');

%% Inputs
% data - data input matrix (NxD)
% label - label input array (Nx1, +1 and -1)
% ini_size - initialization size (Data percentual from the begin)
%% Output
%ftsvm_struct - model updated

data=[]; %Pre-initialize for efficiency
label=[]; %Pre-initialize for efficiency
if isfield(Parameter,'kernel_name')==1
     trdata= rf_featurize(kobj, double(trdata));
     
end

 data=trdata; %Pre-initialize for efficiency
 label=trlabel; %Pre-initialize for efficiency
    
 %data=[ftsvm_struct.X;trdata];
 %label=[ftsvm_struct.L;trlabel];
%% Incremental training
%Get the rest of the data
%traindata11=data;
%trainlabel11=label;

traindata11=trdata;
trainlabel11=trlabel;


integerTest=(batch_size==floor(batch_size));
inc_dsize=size(trainlabel11,1);

if integerTest==1
    bats=batch_size;
    %warning('-> The last batch size fits the data size.');
else
    bats=batch_size*inc_dsize;
    bats=round(bats);
    %warning('-> The last batch size fits the data size.');
    
end

score=[];
p=1;

while p<inc_dsize
    
    
  [ftsvm_struct,data,label]=inc_bintrain(traindata11(p:p+bats,:),trainlabel11(p:p+bats),Parameter,ftsvm_struct,data,label);

  [ftsvm_struct,data,label,score]=forget_bin(ftsvm_struct,data,label,score);
   p=p+bats;

  
  if (p+bats)>inc_dsize && p<inc_dsize
     resto=rem(inc_dsize,bats)-1;
     
     if resto==-1
         bats=bats-1; 
     else
     bats=resto;
     end
  end
  


end

if inc_dsize==1
      [ftsvm_struct,data,label]=inc_bintrain_cd(traindata11,trainlabel11,Parameter,ftsvm_struct,data,label);

  [ftsvm_struct,data,label,score]=forget_bin(ftsvm_struct,data,label,score);
end

end