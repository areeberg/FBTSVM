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
from itertools import chain
from collections import OrderedDict

#model 0 -> 0 - 1
#model 1 -> 0 - 2
#model 2 -> 1 - 2

def get_intersect(data,label,model,parameters):
    phi=parameters.iloc[0].loc['phi']
    rep=parameters.iloc[0].loc['repetitions']
    classes=np.unique(label)
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
    BB=AA
    Cinter= np.array([])
    Cmatrix={}
    aux_var=0
    Fgt_mtx_index=np.empty((len(model),len(model)))
    Fgt_mtx_index[:]=np.nan


    for currentclass in classes:
        print("loop1")
        #print("loop over two classes only for the DAG algorithm")
            #pdb.set_trace()
        otherclasses=np.delete(classes,np.where(classes==currentclass))
        current_data=[]
        #unique_rows=[]

        if len(otherclasses)>0:

            for ocl in otherclasses:
                #send the data Xp and the respective models (currentclass and ocl), and the positive or negatives
                #S{ve(1),ve(2)}=createlinearSR(trainp,ftsvm_struct(ve(1),ve(2)),'positive');
                mod_pos=AA[currentclass][ocl]
                if math.isnan(mod_pos)==True:
                    #print("NaN")
                    continue
                else:
                    mod=model[int(mod_pos)]
                    #pdb.set_trace()
                    Cpos=mod.Xpi[0][np.nonzero(mod.alpha <=phi )]
                    Cneg=mod.Lpi[0][np.nonzero(mod.beta <=phi )]
                    #pdb.set_trace()

                    Cmatrix[aux_var,currentclass]=Cpos
                    Fgt_mtx_index[aux_var][currentclass]=1
                    Cmatrix[aux_var,ocl]=Cneg
                    Fgt_mtx_index[aux_var][ocl]=1
                    #pdb.set_trace()
                    aux_var=aux_var+1

    #pdb.set_trace()
    Inters=[]
    Inters_struct=[]

    #O erro deve estar na Cmatrix
    for inde in range(int(num_classes)):

        values=np.argwhere(~np.isnan(Fgt_mtx_index[inde]))
        #IT MAY EXISTS ONLY 1 CLASS!!! check condition and continue
        X=Cmatrix[values[0][0],inde]
        #A must loop over the number of classes except for the first class
        for val in values[1:]:
            A=Cmatrix[val[0],inde]
            #pdb.set_trace()
            X=np.intersect1d(A, X)
            Inters_struct.append(X)

            Inters.extend(X)
            #pdb.set_trace()
        #pdb.set_trace()
    #pdb.set_trace()
    return Inters, Inters_struct

def forgetn(parameters,data,label,model,score):
    #Implement to use the appox kernel function

    #The Batch size is a int that represent the number of instances




    # phi=parameters.iloc[0].loc['phi']
    rep=parameters.iloc[0].loc['repetitions']
    #
    # classes=np.unique(label)
    # num_classes=len(classes)
    # num_models=len(model)
    #
    #
    # pdb.set_trace()
    #
    #
    #
    # AA=np.empty((len(model),len(model)))
    # AA[:]=np.nan
    # i=0
    # for mod in model:
    #     currentclass=mod.currentclass
    #     ocl=mod.ocl
    #     #pdb.set_trace()
    #     AA[currentclass][ocl]=i
    #     i=i+1
    # Cinter= np.array([])
    # Cmatrix={}
    # aux_var=0
    # Fgt_mtx_index=np.empty((len(model),len(model)))
    # Fgt_mtx_index[:]=np.nan
    #
    # for currentclass in classes:
    #     print("loop1")
    #     #print("loop over two classes only for the DAG algorithm")
    #         #pdb.set_trace()
    #     otherclasses=np.delete(classes,np.where(classes==currentclass))
    #     current_data=[]
    #     #unique_rows=[]
    #
    #     if len(otherclasses)>0:
    #
    #         for ocl in otherclasses:
    #             #send the data Xp and the respective models (currentclass and ocl), and the positive or negatives
    #             #S{ve(1),ve(2)}=createlinearSR(trainp,ftsvm_struct(ve(1),ve(2)),'positive');
    #             mod_pos=AA[currentclass][ocl]
    #             if math.isnan(mod_pos)==True:
    #                 #print("NaN")
    #                 continue
    #             else:
    #                 mod=model[int(mod_pos)]
    #                 #pdb.set_trace()
    #                 Cpos=mod.Xpi[0][np.nonzero(mod.alpha <=phi )]
    #                 Cneg=mod.Lpi[0][np.nonzero(mod.beta <=phi )]
    #                 #pdb.set_trace()
    #
    #                 Cmatrix[aux_var,currentclass]=Cpos
    #                 Fgt_mtx_index[aux_var][currentclass]=1
    #                 Cmatrix[aux_var,ocl]=Cneg
    #                 Fgt_mtx_index[aux_var][ocl]=1
    #                 #pdb.set_trace()
    #                 aux_var=aux_var+1
    #
    # #pdb.set_trace()
    # Inters=[]
    #
    # #O erro deve estar na Cmatrix
    # for inde in range(int(num_classes)):
    #
    #     values=np.argwhere(~np.isnan(Fgt_mtx_index[inde]))
    #     X=Cmatrix[values[0][0],inde]
    #     #A must loop over the number of classes except for the first class
    #     for val in values[1:]:
    #         A=Cmatrix[val[0],inde]
    #         #pdb.set_trace()
    #         X=np.intersect1d(A, X)
    #         #pdb.set_trace()
    #     Inters.extend(X)
    #
    #     #pdb.set_trace()
    #     #A=Cmatrix[values[1][0],inde]
    #     #Inters.extend(np.intersect1d(A, X))
    # #Inters = list(OrderedDict.fromkeys(Inters))
    # #pdb.set_trace()

    Inters,Inters_struct=get_intersect(data,label,model,parameters)
    #pdb.set_trace()

    if len(score)==0:
        #pdb.set_trace()
        score=np.asarray(Inters)
        sc=np.ones(len(score))
        score=np.array((score,sc))

        #pdb.set_trace()
        score=score.astype(int)


        print("score empty")
    else:
        #pdb.set_trace()
        Inters=np.asarray(Inters)
        res,com1,com2=np.intersect1d(score[0],Inters,return_indices=True)
        score[1][com1]=score[1][com1]+1
        diff=np.setdiff1d(score[0],Inters)
        #pdb.set_trace()
        if len(diff)!=0:
            pdb.set_trace()
            new_ones=np.ones(len(diff))
            new_score=np.array((diff,new_ones))
            np.insert(score[0],new_score)
            np.insert(score[1],new_ones)

    if len(score)!=0:
        res = [idx for idx, val in enumerate(score[1]) if val >= rep]
        AA=np.empty((len(model),len(model)))
        AA[:]=np.nan
        i=0
        for mod in model:
            currentclass=mod.currentclass
            ocl=mod.ocl
            #pdb.set_trace()
            AA[currentclass][ocl]=i
            i=i+1

        #Remove unwanted data from: data, label and models
        if len(res)==0:
            return model,score,data,label
        else:
            Inters, Inters_struct=get_intersect(data,label,model,parameters)
            label= np.delete(label,Inters)
            data= np.delete(data,Inters,axis=0)
            #REMOVE THE ELEMENTS FROM THE SCORE
            resi=np.where(score[1]<=rep)

            npInters=np.asarray(Inters)
            sco0=np.delete(score[0],resi)

            sco1=np.delete(score[1],resi)

            if len(sco0)==0:
                score=[]
            else:
                sc1=sco0.tolist()
                sc2=sco1.tolist()
                score=[[sc1],[sc2]]




            #pdb.set_trace()

            classes=np.unique(label)
            num_classes=len(classes)
            num_models=len(model)
            for currentclass in classes:
                print("loop1")
                #print("loop over two classes only for the DAG algorithm")
                    #pdb.set_trace()
                otherclasses=np.delete(classes,np.where(classes==currentclass))
                current_data=[]
                #unique_rows=[]

                if len(otherclasses)>0:

                    for ocl in otherclasses:
                        #send the data Xp and the respective models (currentclass and ocl), and the positive or negatives
                        #S{ve(1),ve(2)}=createlinearSR(trainp,ftsvm_struct(ve(1),ve(2)),'positive');
                        mod_pos=AA[currentclass][ocl]
                        if math.isnan(mod_pos)==True:
                            #print("NaN")
                            continue
                        else:
                            mod=model[int(mod_pos)]
                            bb=Inters_struct[currentclass]
                            cc=Inters_struct[ocl]
                            res1=np.intersect1d(bb,mod.Xpi[0])
                            res2=np.intersect1d(cc,mod.Lpi[0])
                            Xpiu= np.delete(mod.Xpi[0],res1)
                            Lpiu= np.delete(mod.Lpi[0],res2)


                            alpu= np.delete(mod.alpha,bb)
                            betu=np.delete(mod.beta,cc)

                            spu=np.delete(mod.sp,bb)
                            snu=np.delete(mod.sn,cc)

                            new_structure=data_structure(spu,snu,alpu,betu,mod.vp,mod.vn,mod.NXpv,mod.NXnv,mod.pgp,mod.pgn,currentclass,ocl,Xpiu,Lpiu)
                            model[int(mod_pos)]=new_structure
                            #pdb.set_trace()
                            #print("Fdas")
                #pdb.set_trace()
                return model,score,data,label


            #Y = Inters
            #Update the models
            #need to return both Interns e Y for each class
        #pdb.set_trace()
        print("remove")





        #pdb.set_trace()
        #print("score not empty")
    #Inters=[]
    #return model,score




#    if Cinter.size==0:
#        return model score
#    else:
#        pdb.set_trace()
#        print("loop ja era")
