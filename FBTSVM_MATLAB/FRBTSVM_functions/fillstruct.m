function [ftsvm_struct]= fillstruct(ftsvm_struct,cl,alpha,beta,sp,sn,xpi,lpi,vp,vn,pgp,pgn,eTime)

sa=size(alpha,2);
nonempty = find(arrayfun(@(ftsvm_struct) ~isempty(ftsvm_struct.vp),ftsvm_struct(cl,:)));

for i=1:sa
ftsvm_struct(cl,nonempty(i)).alpha = cell2mat(alpha(i));
ftsvm_struct(cl,nonempty(i)).beta = cell2mat(beta(i));
ftsvm_struct(cl,nonempty(i)).sp=cell2mat(sp(i));
ftsvm_struct(cl,nonempty(i)).sn=cell2mat(sn(i));
ftsvm_struct(cl,nonempty(i)).Xpi=cell2mat(xpi(i));
ftsvm_struct(cl,nonempty(i)).Lpi=cell2mat(lpi(i));
ftsvm_struct(cl,nonempty(i)).vp=cell2mat(vp(i));
ftsvm_struct(cl,nonempty(i)).vn=cell2mat(vn(i));
ftsvm_struct(cl,nonempty(i)).pgp=cell2mat(pgp(i));
ftsvm_struct(cl,nonempty(i)).pgn=cell2mat(pgn(i));
ftsvm_struct(cl,nonempty(i)).time=cell2mat(eTime(i));
%ftsvm_struct(cl,nonempty(i)).Parameter=Paremeter;

end
end