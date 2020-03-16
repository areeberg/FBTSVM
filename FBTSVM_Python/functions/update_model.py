## APPROXIMATE KERNEL FUNCTION
## Currently there are only the sklearn functions available

import pdb
import numpy as np
import pandas as pd
import sys
sys.path.append("/media/alexandre/57268F1949DB0319/MATLAB/FBTSVM/FBTSVM_Python/functions/")
from approx_k import approx_kernel
from fuzzy import fuzzy_membership
from calc import calc_train
from aux_functions import data_structure
import math

#model 0 -> 0 - 1
#model 1 -> 0 - 2
#model 2 -> 1 - 2


#Initial data filter - For this case consider only the positive values (currentclass)
def createlinearSR(data,currentmodel,parameters):
    print("SX")
    ones=np.ones((len(data),1))
    data=np.append(data,ones,axis=1)
    grad=np.matmul(-data,currentmodel.vp) -1
    bigger_grad=grad>=max(currentmodel.pgp)
    smaller_grad=grad<=min(currentmodel.pgp)
    bs_grad=np.any([bigger_grad,smaller_grad],axis=0)
    index=np.argwhere(bs_grad==True)
    index=np.delete(index,1,1)
    data=np.delete(data,index,axis=0)

    return data



def inc_model(parameters,traindata,trainlabel,model):
    print("D")
    #separate the data into classes (data and label)
    classes=np.unique(trainlabel)
    num_classes=len(classes)
    num_models=len(model)
    AA=np.empty((len(model),len(model)))
    AA[:]=np.nan
    i=0
    for mod in model:
        currentclass=mod.currentclass
        ocl=mod.ocl
        #pdb.set_trace()
        AA[currentclass][ocl]=i
        i=i+1
    pdb.set_trace()

    for currentclass in classes:
        #print("loop over two classes only for the DAG algorithm")
        #Xp - all training data from one class
        Xpi=np.where(trainlabel==currentclass) # indexes from data which are equal to the respective class
        Xp=traindata[Xpi] # The data from the indexes above
        lp=len(Xpi[0]) # the number of instances
        Lp=np.ones(lp) #array of ones
        #pdb.set_trace()
        otherclasses=np.delete(classes,np.where(classes==currentclass))
        current_data=[]

        # if len(model)!=0:
        #     for mod in model:
        #         cl_pos=mod.currentclass
        #         #pdb.set_trace()
        #         if mod.ocl==currentclass & currentclass>mod.currentclass:
        #             otherclasses=np.delete(otherclasses,np.where(otherclasses==mod.currentclass))
                    #pdb.set_trace()


        #check if the structure already exists
        #print(currentclass)
        #pdb.set_trace()
        if len(otherclasses)>0:

            for ocl in otherclasses:
                #send the data Xp and the respective models (currentclass and ocl), and the positive or negatives
                #S{ve(1),ve(2)}=createlinearSR(trainp,ftsvm_struct(ve(1),ve(2)),'positive');

                mod_pos=AA[currentclass][ocl]
                mod=model[int(mod_pos)]
                filtered_data=createlinearSR(Xp,mod,parameters) #Xp=data from currentclass
                current_data.append(filtered_data)
                #filtdata{cl}=unique(v,'rows');




                pdb.set_trace()
                print("aqui")

    pdb.set_trace()


def update_model(parameters,data_x,data_y,batch_size,model):
    #Implement to use the appox kernel function

    #The Batch size is a int that represent the number of instances

    data_size=len(data_y)
    integertest=batch_size==math.floor(batch_size)

    if integertest==True:
        bats=batch_size
    else:
        bats=batch_size*data_size
        bats=math.floor(bats)

    p=0

    for i in range(data_size):
        traindata1=data_x[p:p+bats,:]
        trainlabel1=data_y[p:p+bats]
        #pdb.set_trace()
        #TODO WORK ON THE INC_MODEL FUNCTION
        fbtsvm_struct,datau,labelu=inc_model(parameters,traindata1,trainlabel1,model)
        print("update model")


    classes=np.unique(data_y)
    #get the number of classes
    num_classes=len(classes)
    fbtsvm_struct=[]

    #Select a class

    for currentclass in classes:
        #print("loop over two classes only for the DAG algorithm")
        #Xp - all training data from one class
        Xpi=np.where(data_y==currentclass) # indexes from data which are equal to the respective class
        Xp=data_x[Xpi] # The data from the indexes above
        lp=len(Xpi[0]) # the number of instances
        Lp=np.ones(lp) #array of ones
        #pdb.set_trace()
        otherclasses=np.delete(classes,np.where(classes==currentclass))

        #pdb.set_trace()

        if len(fbtsvm_struct)!=0:
            for mod in fbtsvm_struct:
                cl_pos=mod.currentclass
                #pdb.set_trace()
                if mod.ocl==currentclass & currentclass>mod.currentclass:
                    otherclasses=np.delete(otherclasses,np.where(otherclasses==mod.currentclass))
                    #pdb.set_trace()


        #check if the structure already exists
        #print(currentclass)
        #pdb.set_trace()
        if len(otherclasses)>0:

            for ocl in otherclasses:
                Lpi=np.where(data_y==ocl) # indexes from data which are equal another class (negative from currentclass)
                Xn=data_x[Lpi] # The data from the indexes above
                ln=len(Lpi[0]) # the number of instances
                Ln=-1*np.ones(ln) #array of ones
                #pdb.set_trace()
                X=np.concatenate((Xp,Xn))
                L=np.concatenate((Lp,Ln))
                #The fuzzy function can be improved
                sp,sn,NXpv,NXnv=fuzzy_membership(Xp,Xn,parameters);


                #XP_one and XN_one variables
                #pdb.set_trace()
                XP_one=np.append(Xp, np.ones((len(Xp), 1)), axis=1)
                XN_one=np.append(Xn, np.ones((len(Xn),1)),axis=1)
                CCp=parameters.iloc[0].loc['CC']*sn
                CCn=parameters.iloc[0].loc['CC2']*sp
                #pdb.set_trace()

                #Implement here the main function (calc)
                CR2=parameters.iloc[0].loc['CR2'] #for the positives
                CR=parameters.iloc[0].loc['CR'] #for the negatives
                alpha,vp,iter,pgp=calc_train(XN_one,XP_one,CR2,CCn,parameters)
                beta,vn,iter2,pgn=calc_train(XP_one,XN_one,CR,CCp,parameters)
                vn=-vn
                new_structure=data_structure(sp,sn,alpha,beta,vp,vn,NXpv,NXnv,pgp,pgn,currentclass,ocl,Xpi,Lpi)
                fbtsvm_struct.append(new_structure)
            #print("Calculated")

            #pandas dataframe in currentclass and ocl
            #todo save the structure (ftsvm_struct)




    return fbtsvm_struct
