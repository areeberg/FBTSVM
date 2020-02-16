## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import Nystroem
import numpy as np
import pandas as pd

def approx_kernel(kernel_structure,data_x,data_y):
    #print("Approx kernel")
    #pdb.set_trace()
    if kernel_structure.iloc[0].loc['kernel_type']=='RBF':
        #pdb.set_trace()
        rbf_feature = RBFSampler(gamma=1,n_components=10,random_state=1)
        X_features = rbf_feature.fit_transform(data_x)
    if kernel_structure.iloc[0].loc['kernel_type']=='ACHI2':
        chi2sampler = AdditiveChi2Sampler(sample_steps=10,sample_interval=1)
        X_features = chi2sampler.fit_transform(X, y)
    #todo implement the other methods
    return X_features
