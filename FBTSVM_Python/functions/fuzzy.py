## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
import numpy as np
import pandas as pd
from sklearn import preprocessing

def fuzzy_membership(Xp,Xn,parameters):
    print("Approx kernel")
    u=parameters.iloc[0].loc['u']
    epsilon=parameters.iloc[0].loc['eps']
    sizeXp=len(Xp)
    sizeXn=len(Xn)

    Xpcenter=np.mean(Xp,axis=0)
    Xncenter=np.mean(Xn,axis=0)


    radiusxp=np.sum(np.square(np.tile(Xpcenter,(sizeXp,1))-Xp),axis=1) #||Xi+-Xcen+||^2
    radiusxpxn=np.sum(np.square(np.tile(Xncenter,(sizeXp,1))-Xp),axis=1) #||xi--Xcen+||^2
    radiusmaxxp=radiusxp.max() #max value

    radiusxn=np.sum(np.square(np.tile(Xncenter,(sizeXn,1))-Xn),axis=1) #||Xi--Xcen-||^2
    radiusxnxp=np.sum(np.square(np.tile(Xpcenter,(sizeXn,1))-Xn),axis=1) #||xi--Xcen+||^2
    radiusmaxxn=radiusxn.max()

    XPnoise=np.greater_equal(radiusxp,radiusxpxn)
    XPnoise_index=np.where(XPnoise)[0]
    XPnormal=np.greater(radiusxpxn,radiusxp)
    XPnormal_index=np.where(XPnormal)[0]
    sp=np.zeros(sizeXp)
    sp[XPnormal_index]=(1-u)*(1-np.square(np.absolute(radiusxp[XPnormal_index])/(radiusmaxxp+epsilon))) #sp at normal index
    sp[XPnoise_index]=u*(1-np.square(np.absolute(radiusxp[XPnoise_index])/(radiusmaxxp+epsilon))) #sp at noise index
    sp=np.expand_dims(sp,axis=1)


    XNnoise=np.greater_equal(radiusxn,radiusxnxp)
    XNnoise_index=np.where(XNnoise)[0]
    XNnormal=np.greater(radiusxnxp,radiusxn)
    XNnormal_index=np.where(XNnormal)[0]
    sn=np.zeros(sizeXn)
    sn[XNnormal_index]=(1-u)*(1-np.square(np.absolute(radiusxn[XNnormal_index])/(radiusmaxxn+epsilon))) #sp at normal index
    sn[XNnoise_index]=u*(1-np.square(np.absolute(radiusxn[XNnoise_index])/(radiusmaxxn+epsilon))) #sp at noise index
    scaler = preprocessing.MinMaxScaler(feature_range=(epsilon, 1))
    sn=np.expand_dims(sn,axis=1)


    #The fit transform can be upgraded to consider the min and max from all dataset instead of each incremental iteration - idk if this action improve the classifier, however, worth the try
    sp=scaler.fit_transform(sp)
    sn=scaler.fit_transform(sn)


    return sp,sn,XPnoise,XNnoise
