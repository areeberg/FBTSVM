import pdb
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import AdditiveChi2Sampler
import numpy as np
import pandas as pd

def approx_kernel(kernel_structure,data_x,data_y):

    print("A")
    pdb.set_trace()
    if kernel_structure.iloc[0].loc['kernel_type']=='RBF':
        pdb.set_trace()
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X_features = rbf_feature.fit_transform(data_x)
    if kernel_structure.iloc[0].loc['kernel_type']=='ACHI2':
        chi2sampler = AdditiveChi2Sampler(sample_steps=10,sample_interval=1)
        X_features = chi2sampler.fit_transform(X, y)
    print(X_features)

    return X_features
