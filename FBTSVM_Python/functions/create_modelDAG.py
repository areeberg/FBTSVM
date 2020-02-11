## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
import numpy as np
import pandas as pd
import sys
sys.path.append("/media/alexandre/57268F1949DB0319/MATLAB/FBTSVM/FBTSVM_Python/functions/")
from approx_k import approx_kernel
from fuzzy import fuzzy_membership

def create_model(parameters,data_x,data_y):
    print("Creating initial model")
    pdb.set_trace()
    #get the classes
    classes=np.unique(data_y)
    #get the number of classes
    num_classes=len(classes)

#Select a class
    for currentclass in classes:
        print("loop over two classes only for the DAG algorithm")
        #Xp - all training data from one class
        Xpi=np.where(data_y==currentclass) # indexes from data which are equal to the respective class
        Xp=data_x[Xpi] # The data from the indexes above
        lp=len(Xpi[0]) # the number of instances
        Lp=np.ones(lp) #array of ones
        #pdb.set_trace()
        otherclasses=np.delete(classes,currentclass)
        #pdb.set_trace()
        for ocl in otherclasses:
            print("s")
            Lpi=np.where(data_y==ocl) # indexes from data which are equal another class (negative from currentclass)
            Xn=data_x[Lpi] # The data from the indexes above
            ln=len(Lpi[0]) # the number of instances
            Ln=-1*np.ones(ln) #array of ones
            #pdb.set_trace()
            X=np.concatenate((Xp,Xn))
            L=np.concatenate((Lp,Ln))
            #The fuzzy function can be improved
            sp,sn,NXpv,NXnv=fuzzy_membership(Xp,Xn,parameters);
            #XP_one and XN_one variables
            XP_one=np.append(Xp, np.ones((len(Xn), 1)), axis=1)
            XN_one=np.append(Xn, np.ones((len(Xn),1)),axis=1)
            pdb.set_trace()


            CCp=parameters.iloc[0].loc['CC']*sn
            CCn=parameters.iloc[0].loc['CC2']*sp
            pdb.set_trace()

            #Implement here the main function (calc)

            pdb.set_trace()




    return X_features
