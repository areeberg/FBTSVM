import pdb
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import pandas as pd

def approx_kernel(dataframe,data_x):
    if dataframe.iloc[0].loc['kernel_type']=='RBF':
        rbf_feature = RBFSampler(gamma=1, random_state=1)
        X_features = rbf_feature.fit_transform(data_x)
    print(X_features)

    return (data_kernel)
