## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
import sys
sys.path.append("/media/alexandre/57268F1949DB0319/MATLAB/FBTSVM/FBTSVM_Python/functions/")
from approx_k import approx_kernel
from fuzzy import fuzzy_membership
from calc import calc_train

class data_structure:
    def __init__(self,sp,sn,alpha,beta,vp,vn,NXpv,NXnv,pgp,pgn,currentclass,ocl,Xpi,Lpi):
        self.sp=sp
        self.sn=sn
        self.alpha=alpha
        self.beta=beta
        self.vp=vp
        self.vn=vn
        self.NXpv=NXpv
        self.NXnv=NXnv
        self.pgp=pgp
        self.pgn=pgn
        self.currentclass=currentclass #+1
        self.ocl=ocl #-1
        self.Xpi=Xpi
        self.Lpi=Lpi
