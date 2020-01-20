import pdb
from sklearn.kernel_approximation import RBFSampler
import numpy as np


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

kernel_type='RBF'
gamma=1
random_state=1


## TODO:  Create a Structured Datatype Creation to store the paremeters
## TODO:  Test the kernel approximation with RBFSamples
