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
    pg1=[]
    E=np.eye(num_colH, dtype=int)
    E[-1][-1]=0

    #We could optimize this operation
    Q_bar=np.linalg.solve((np.dot(np.transpose(H),H)+constantC*E),np.transpose(G))
    Q=[]
    for i in range(lenG):
        Qv=np.dot(G[i,:],Q_bar[:,i])
        Q.append(Qv)

    #Continue from X_new = 1:l (lenG)
    X_new=np.arange(lenG)
    X_old=np.arange(lenG)
    alpha=np.zeros(lenG)
    alpha_old=np.zeros(lenG)
    v=np.zeros((num_colH,1))
    PGmax_old=float("inf") #M_bar
    PGmin_old=float("-inf") #m_bar
    num_iteration=0


    for i in range(maxeva):
        PGmax_new=float("-inf") #M
        PGmin_new=float("inf") #m
        R=len(X_old)
        np.random.shuffle(X_old) #randomize the X_old
        for j in range(R):
            pos=X_old[j]
            pg=-np.matmul(G[pos,:],v)-1
            PG=0
            if alpha[pos]==0: #-->begin if alpha
                if pg>PGmax_old:
                    np.delete(X_new,pos)
                    continue
                elif pg<0:
                    PG=pg
            elif alpha[pos]==CCx[pos]:
                if pg<PGmin_old:
                    np.delete(X_new,pos)
                    continue
                elif pg>0:
                    PG=pg
            else:
                PG=pg
            PGmax_new=np.maximum(PGmax_new,PG)
            PGmin_new=np.minimum(PGmin_new,PG)
            if np.absolute(PG)>1e-12:
                alpha_old[pos]=alpha[pos]
                alpha[pos]=np.minimum(np.maximum(alpha[pos]-pg/Q[pos],0),CCx[pos])

                v_aux=Q_bar[:,pos]*(alpha[pos]-alpha_old[pos])
                v_aux=np.expand_dims(v_aux,axis=1)
                v=np.subtract(v,v_aux)
                if PG != 0:
                    pg1.append(PG)

        X_old=X_new
        #increment the maxeva iterator
        num_iteration=num_iteration+1
        if PGmax_new-PGmin_new <=epsilon:
            if len(X_old)==lenG:
                break
            else:
                X_old=np.arange(lenG)
                X_new=np.arange(lenG)
                PGmax_old=float("inf")
                PGmin_old=float("-inf")

        if PGmax_new<=0:
            PGmax_old=float("inf")
        else:
            PGmin_old=PGmax_new
        if PGmin_old >=0:
            PGmin_old= float("-inf")
        else:
            PGmin_old=PGmin_new
        #pdb.set_trace()
    #pdb.set_trace()

    return alpha,v,num_iteration,pg1
    
