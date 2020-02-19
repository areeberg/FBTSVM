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
    AA=np.empty((len(models),len(models)))
    AA[:]=np.nan
    distances=[]
    answers=[]
    i=0
    for mod in models:
        currentclass=mod.currentclass
        ocl=mod.ocl
        #pdb.set_trace()
        AA[currentclass][ocl]=i
        i=i+1
        #pdb.set_trace()


#model 0 -> 0 - 1
#model 1 -> 0 - 2
#model 2 -> 1 - 2
    i=0
    j=1
    pp=[]
    jj=[]
    loop_len=int(len(models)-1)
    for row in data_x:
        for x in range(loop_len):
            mod_pos=AA[i][j]
            #pdb.set_trace()
            mod=models[int(mod_pos)]
            vp=mod.vp
            vn=mod.vn
            fp=(np.matmul(row,vp[:-1])+vp[-1])/(LA.norm(vp[:-1]))
            fn=(np.matmul(row,vn[:-1])+vn[-1])/(LA.norm(vn[:-1]))
            pp=i
            jj=j

            if abs(fp)<abs(fn):
                i=j
                j=j+1
            else:
                j=j+1
        if abs(fp)<abs(fn):
            distances.append(fp)
            answers.append(jj)
        else:
            distances.append(fn)
            answers.append(pp)
        i=0
        j=1
    #pdb.set_trace()

    answers=np.asarray(answers)

    correct=np.where(answers==data_y)
    correct_num=len(correct[0])
    accuracy=100*correct_num/len(data_y)
    print("Accuracy-> ", accuracy)

    return accuracy, distances,fp,fn,answers
