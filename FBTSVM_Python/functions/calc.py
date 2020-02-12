## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
import numpy as np
import pandas as pd
from sklearn import preprocessing

def calc_train(H,G,constantC,CCx,parameters):
    epsilon=parameters.iloc[0].loc['eps']
    maxeva=parameters.iloc[0].loc['maxeva']
    num_colH=np.size(H,1)
    lenG=np.size(G,0)
    E=np.eye(num_colH, dtype=int)
    pdb.set_trace()
    E[-1][-1]=0

    #We could optimize this operation
    Q_bar=np.linalg.solve((np.dot(np.transpose(H),H)+constantC*E),np.transpose(G))
    Q=[]
    for i in range(lenG):
        Qv=np.dot(G[i,:],Q_bar[:,i])
        Q.append(Qv)

    #Continue from X_new = 1:l (lenG)


    pdb.set_trace()


    return sp,sn,XPnoise,XNnoise
