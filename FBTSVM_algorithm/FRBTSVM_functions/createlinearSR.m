function [SR,dele] = createlinearSR(Kx,ftsvm_struct,class)
%length(Kx)

if class=='positive'
SR=[Kx ones(size(Kx,1),1)];
gradp=-SR(:,:)*ftsvm_struct.vp-1;
bgradp=gradp>=max(max(ftsvm_struct.pgp)); %regarding +1
sgradp=gradp<=min(min(ftsvm_struct.pgp));
fgradp=bgradp+sgradp;
dele=find(fgradp>=1);
SR(find(fgradp>=1),:)=[];

end

if class=='negative'
   SR=[Kx ones(size(Kx,1),1)];
gradp=-SR(:,:)*ftsvm_struct.vn-1;
bgradp=gradp>=max(max(ftsvm_struct.pgn)); %regarding -1
sgradp=gradp<=min(min(ftsvm_struct.pgn));
fgradp=bgradp+sgradp; 
  dele=find(fgradp>=1);
    SR(find(fgradp>=1),:)=[];
  
end

end
