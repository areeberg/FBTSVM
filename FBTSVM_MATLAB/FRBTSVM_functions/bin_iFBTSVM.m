function ftsvm_struct = bin_iFBTSVM(trdata,trlabel,ini_size,batch_size,Parameter,kobj)
addpath('/Randfeat_releasever');


if isfield(Parameter,'kernel_name')==1    
trdata= rf_featurize(kobj, double(trdata));
    
end

%% Initial training
%Split the data into first training and the rest
traindata1=trdata(1:ceil(length(trdata)*ini_size),:);
trainlabel1=trlabel(1:ceil(length(trlabel)*ini_size));


data=[]; %Pre-initialize for efficiency
label=[]; %Pre-initialize for efficiency

%Initial training
[ftsvm_struct,data,label] = offline_bintrain(traindata1,trainlabel1,Parameter);
%Clear initial traindata from memory
clearvars traindata1 trainlabel1  


%% Incremental training
%Get the rest of the data
traindata11=trdata(ceil(length(trdata)*ini_size)+1:end,:);
trainlabel11=trlabel(ceil(length(trlabel)*ini_size)+1:end);



integerTest=(batch_size==floor(batch_size));
inc_dsize=size(trainlabel11,1);

if integerTest==1
    bats=batch_size;
    warning('-> The last batch size fits the data size.');
else
    bats=batch_size*inc_dsize;
    bats=round(bats);
    warning('-> The last batch size fits the data size.');
    
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