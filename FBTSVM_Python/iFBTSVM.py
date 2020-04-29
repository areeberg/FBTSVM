from sklearn.kernel_approximation import RBFSampler
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import sys
import pdb
sys.path.append("/media/alexandre/57268F1949DB0319/MATLAB/FBTSVM/FBTSVM_Python/functions/")
from approx_k import approx_kernel
from create_modelDAG import create_model
from classify import classify
from update_model import update_model
from sklearn.model_selection import train_test_split
#DATA MUST BE A NUMPY ARRAY
data_iris = load_iris()
# data_X = data_iris.data[:, :2]  # we only take the first two features. We could
#                       # avoid this ugly slicing by using a two-dim dataset
# data_Y = data_iris.target

#Shuffle data
data_X,data_Y=shuffle(data_iris.data,data_iris.target)
data1_X=data_X[1:int(len(data_X)/2),:]
data1_Y=data_Y[1:int(len(data_Y)/2)]

data2_X=data_X[int(len(data_X)/2)+1:int(len(data_X)),:]
data2_Y=data_Y[int(len(data_Y)/2)+1:int(len(data_Y))]

## new set of parameters
CC = 8 #C1=C3
CC2=8
CR = 2 #C2=C4
CR2=2
eps=0.0000001 #epsilon to avoid inverse matrix calculation error
maxeva=500 #maximum of function evaluations to each train/update the model
u=0.01 #fuzzy parameter
epsilon=1e-10# fuzzy epsilon
repetitions=3 #Must be an int
phi=0.00001
sliv=True #True or False

kernel_type='ACHI2' #AdditiveChi2Sampler
kernel_type='Nystroem' #Nystroem
kernel_type='SCHI2'#SkewedChi2Sampler
kernel_type='RBF' #Random Kitchen Sinks

## Define the approximate Kernel specs----------------
# RBF - https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler
kernel_type='RBF' #Random Kitchen Sinks
n_components=20
gamma=1
random_state=1
kernel_structure={'kernel_type':[kernel_type],'gamma':[gamma],'random_state':[random_state]}
# #-----------------------------------------------------
# kernel_type='ACHI2' #Additive Chi Squared Kernel - https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler.html#sklearn.kernel_approximation.AdditiveChi2Sampler
# sample_steps=2
# sample_interval=None
# random_state=1
# kernel_structure={'kernel_type':[kernel_type],'sample_steps':[sample_steps],'sample_interval':[sample_interval]}
#
# #-----------------------------------------------------
# kernel_type='SCHI2' #Skewed chi-squared Kernel - https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler.html#sklearn.kernel_approximation.SkewedChi2Sampler
# skewdness=0.5
# n_components=0
# random_state=1
# kernel_structure={'kernel_type':[kernel_type],'skewdness':[skewdness],'n_components':[n_components],'random_state':[random_state]}
#
#
# #-----------------------------------------------------
# kernel_type='Nystroem' #Additive Chi Squared Kernel - https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler.html#sklearn.kernel_approximation.AdditiveChi2Sampler
# n_components=20
# gamma=1
# random_state=1
# coef0=None
# degree=None
# kernel_params=None
# n_components=10
# random_state=None
# #when execute the Nystroem, use the rbf at the kernel
#kernel_structure={'kernel_type':[kernel_type],'sample_steps':[sample_steps],'sample_interval':[sample_interval]}
kernel_structure=pd.DataFrame(kernel_structure)


#Calculate the approximate kernel, RBF as example in this case
data_xk=approx_kernel(kernel_structure,data_X,data_Y)

##  Create a Pandas structure to store the paremeters
#dataframe test
parameters = {'CC':[CC],'CC2':[CC2],'CR':[CR],'CR2':[CR2],'eps':[eps],'maxeva':[maxeva],'u':[u],'repetitions':[repetitions],'phi':[phi],'sliv':[sliv]}
parameters = pd.DataFrame(parameters)
model=create_model(parameters,data1_X,data1_Y)
#pdb.set_trace()
acc,outclass,fp,fn,answers=classify(model,data2_X,data2_Y,parameters)

batch_size=20
model_updated=update_model(parameters,data2_X,data2_Y,batch_size,model,data1_X,data1_Y)
acc,outclass,fp,fn,answers=classify(model,data2_X,data2_Y,parameters)

pdb.set_trace()
#dataframe=dataframe.append(kernel_structure,ignore_index=True)

#access elements from df
#print(dataframe.iloc[0].loc['CC'])
#print(dataframe.iloc[1].loc['kernel_type'])
