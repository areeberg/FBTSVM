## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import Nystroem
from numpy import linalg as LA
import numpy as np
import pandas as pd


def classify(models,data_x,data_y,parameters):
    #if using approx kernel, calculate here as well

    classes=np.unique(data_y) #the labels from the data_y
    num_classes=len(classes) #the num of classes
    #get the number of models

    num_models=len(models)
    pdb.set_trace()

#model 0 -> 0 - 1
#model 1 -> 0 - 2
#model 2 -> 1 - 2

    for row in data_x:
        for mod in models:
            pdb.set_trace()
            currentclass=mod.currentclass
            ocl=mod.ocl
            vp=mod.vp
            vn=mod.vn
            fp=(np.matmul(row,vp[:-1])+vp[-1])/(LA.norm(vp[:-1]))
            fn=(np.matmul(row,vn[:-1])+vn[-1])/(LA.norm(vn[:-1]))

            if abs(fp)<abs(fn):
                i=j
                j=j+1
            else:
                j=j+1


        #pdb.set_trace()




    #todo implement the other methods
    return X_features
