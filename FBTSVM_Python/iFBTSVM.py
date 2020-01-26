import pdb
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import sys

sys.path.append("/media/alexandre/57268F1949DB0319/MATLAB/FBTSVM/FBTSVM_Python/functions/")
from approx_k import approx_kernel


data_iris = load_iris()
data_X = data_iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
data_Y = data_iris.target

## new set of parameters
CC = 8 #C1=C3
CC2=8
CR = 2 #C2=C4
CR2=2
eps=0.0000001 #epsilon to avoid inverse matrix calculation error
maxeva=500 #maximum of function evaluations to each train/update the model
u=0.01 #fuzzy parameter
epsilon=1e-10# fuzzy epsilon
repetitions=5 #Must be an int
phi=0.00001
sliv=True #True or False

kernel_type='ACHI2' #AdditiveChi2Sampler
kernel_type='Nystroem' #Nystroem
kernel_type='SCHI2'#SkewedChi2Sampler


## Define the approximate Kernel specs----------------
# RBF - https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler
kernel_type='RBF' #Random Kitchen Sinks
n_components=20
gamma=1
random_state=1
kernel_structure={'kernel_type':[kernel_type],'gamma':[gamma],'random_state':[random_state]}
#-----------------------------------------------------
kernel_type='ACHI2' #Random Kitchen Sinks
n_components=20
gamma=1
random_state=1
kernel_structure={'kernel_type':[kernel_type],'gamma':[gamma],'random_state':[random_state]}


##  Create a Pandas structure to store the paremeters
#dataframe test
data = {'CC':[CC],'CC2':[CC2],'CR':[CR],'CR2':[CR2],'eps':[eps],'maxeva':[maxeva],'u':[u],'repetitions':[repetitions],'phi':[phi],'sliv':[sliv]}

dataframe = pd.DataFrame(data)
dataframe=dataframe.append(kernel_structure,ignore_index=True)
pdb.set_trace()

#access elements from df
#print(dataframe.iloc[0].loc['CC'])
#print(dataframe.iloc[1].loc['kernel_type'])


## TODO:  Test the kernel approximation with RBFSamples
pdb.set_trace()
data_xk=approx_kernel(dataframe,data_X,data_Y)
pdb.set_trace()
