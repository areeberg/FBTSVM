## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import Nystroem
import numpy as np
import pandas as pd

def approx_kernel(dataframe,data_x,data_y):
    if dataframe.iloc[1].loc['kernel_type']=='RBF':
        rbf_feature = RBFSampler(gamma=1, n_components=1, random_state=1)
        X_features = rbf_feature.fit_transform(data_x)
    if dataframe.iloc[1].loc['kernel_type']=='CHI2':
        chi2sampler = AdditiveChi2Sampler(sample_steps=2)
        X_features = chi2sampler.fit_transform(data_x, data_y)
    #print(X_features)

    return (X_features)
