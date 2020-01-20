function ftsvm_struct = update_bin_model_mem(ftsvm_struct,trdata,trlabel,batch_size,Parameter,kobj)
%% TODO - CREATE A VERSION THAT DO NOT EXCLUDE POINTS DURING UPDATE (ONLY @DECREMENTAL PHASE)

%% Inputs
% data - data input matrix (NxD)
% label - label input array (Nx1, +1 and -1)
% ini_size - initialization size (Data percentual from the begin)
%% Output
%ftsvm_struct - model updated

data=[]; %Pre-initialize for efficiency
label=[]; %Pre-initialize for efficiency
trdataori=trdata;
trlabelori=trlabel;
trdata=[ftsvm_struct.oridata;trdata];

if isfield(Parameter,'kernel_name')==1
     trdata= rf_featurize(kobj, double(trdata));
     
end

 data=trdata(1:size(ftsvm_struct.oridata,1),:); %Pre-initialize for efficiency
 trori=ftsvm_struct.oridata;
 label=ftsvm_struct.orilabel; %Pre-initialize for efficiency
trlabel=[label;trlabel];
%% Incremental training
%Get the rest of the data
traindata11=trdata(size(ftsvm_struct.oridata,1)+1:end,:);
trainlabel11=trlabel(size(ftsvm_struct.oridata,1)+1:end);


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
    
    
[ftsvm_struct,data,label]=inc_bintrain_mem(traindata11(p:p+bats,:),trainlabel11(p:p+bats),Parameter,ftsvm_struct,data,label,trdataori(p:p+bats,:));

  
[ftsvm_struct,data,label,score]=forget_bin_mem(ftsvm_struct,data,label,score);
 

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
%disp('a');
%get the not excluded data
end