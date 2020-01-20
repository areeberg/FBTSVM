function [sp,sn,XPnoise,XNnoise,time]=fuzzy(Xp,Xn,Parameter)
%% Function:  compute fuzzy membership
% Input:      
% Xp                        -  the positive samples
% Xn                        -  the  negative samples 
% Parameter         -  the parameters 
% Parameter.u=0.01;        %fuzzy parameter
% Parameter.epsilon=1e-10; %fuzzy epsilon
%
% Output:    
% sp                         - the fuzzy mebership vlaue for Xp
% sn                         - the fuzzy mebership vlaue for Xn
% XPnoise - positive class noise
% Xnnoise - negative class noise
% time - execution time


if ( nargin>3||nargin<3) % check correct number of arguments
    disp('Wrong number of arguments');
    return
else

     %u=0.1;
     u=Parameter.u;
     
     %eplison=1e-10;
     eplison=Parameter.epsilon;
     
     tic;
     
    [rxp,cxp]=size(Xp);
    [rxn,cxn]=size(Xn);
 
            
            Xp_cen=mean(Xp);
            Xn_cen=mean(Xn);
            radiusxp=sum((repmat(Xp_cen,rxp,1)-Xp).^2,2);%||Xi+-Xcen+||^2
            radiusxpxn=sum((repmat(Xn_cen,rxp,1)-Xp).^2,2);%||xi+-Xcen-||^2
            radiusmaxxp=max(radiusxp);
            
            radiusxn=sum((repmat(Xn_cen,rxn,1)-Xn).^2,2);%||Xi--Xcen-||^2
            radiusxnxp=sum((repmat(Xp_cen,rxn,1)-Xn).^2,2);%||xi--Xcen+||^2
      
            radiusmaxxn=max(radiusxn);
            sp=zeros(rxp,1);
            XPnoise=find(radiusxp>=radiusxpxn);
            XPnormal=find(radiusxp<radiusxpxn);
            sp(XPnormal,1)=(1-u).*(1-sqrt(abs(radiusxp(XPnormal,1))./(radiusmaxxp+eplison)));
            sp(XPnoise,1)=u.*(1-sqrt(abs(radiusxp(XPnoise,1))./(radiusmaxxp+eplison)));
            
            sn=zeros(rxn,1);
            XNnoise=find(radiusxn>=radiusxnxp);
            XNnormal=find(radiusxn<radiusxnxp);
            sn(XNnormal,1)=(1-u)*(1-sqrt(abs(radiusxn(XNnormal,1))./(radiusmaxxn+eplison)));
            sn(XNnoise,1)=u.*(1-sqrt(abs(radiusxn(XNnoise,1))./(radiusmaxxn+eplison)));
            
            
 
      sp=mapminmax(sp',eps,1)';
      sn=mapminmax(sn',eps,1)';
      
      time=toc;
end

