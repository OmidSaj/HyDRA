# Libraries
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import time
from sklearn.metrics import confusion_matrix

from OsUtils import *

# lstm autoencoder to recreate a timeseries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Flatten, Reshape

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from tensorflow.keras import backend as K

import numpy as np

import tensorflow as tf
import time

from Hyper_bin import *

# Delta function obtained from https://github.com/jameslyons/python_speech_features
def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat
    
def get_delta(submask,N=2):    
    n_batch=submask.shape[0]
    nFeat=submask.shape[2]
    comb_mask=np.zeros((submask.shape[0],submask.shape[1],int(nFeat*2)))
    comb_mask[:,:,0:nFeat]=submask   # A tensor that includes the original and their corresponding delta features.  
    for i in range(n_batch):
        comb_mask[i,:,nFeat:]=delta(submask[i,:,:], N)
    return comb_mask

# Assemble the modules
def enc_input(X_array,i_f_first,nfilt_keep,
              feat_type,delta_on, n_pad=400,subtraction=False, normalize=False):
    
    # Figure out feature indexing
    if feat_type=='MSFB':
        feat_ch=0
    if feat_type=='MFCC':
        feat_ch=1 
    i_f_last=i_f_first+nfilt_keep
    # print(X_array.shape)
    nB=X_array.shape[0]    # batch size
    # nType=X_array.shape[1] # MSFB or MFCC --> Updated after LifeLines
    nW=X_array.shape[1]    # window
    nN=X_array.shape[3]    # node 11 or 12
    # nFilt=X_array.shape[4] # Nfilt
    nCh=X_array.shape[5]   # channels X or Y
    # MSFB or MFCC: Extract the Main features w.r.t to index and feat_type
    sub_mask_X_11=X_array[:,:,feat_ch,0,i_f_first:i_f_last,0]
    sub_mask_Y_11=X_array[:,:,feat_ch,0,i_f_first:i_f_last,1]
    sub_mask_X_12=X_array[:,:,feat_ch,1,i_f_first:i_f_last,0]
    sub_mask_Y_12=X_array[:,:,feat_ch,1,i_f_first:i_f_last,1]
    
    # print(sub_mask_X_11.shape)
    # If the delta feature is active, the prevoius masks are doubled in size to include their delta features as well. 
    if delta_on==1: 
        sub_mask_X_11=get_delta(sub_mask_X_11)
        sub_mask_Y_11=get_delta(sub_mask_Y_11)
        sub_mask_X_12=get_delta(sub_mask_X_12)
        sub_mask_Y_12=get_delta(sub_mask_Y_12)
        
    nfilt_keep_new=sub_mask_X_11.shape[-1]
    
    Mask_4L=np.zeros((nB,nW,int(nfilt_keep_new*4)))
    
    Mask_4L[:,:,0:nfilt_keep_new]=sub_mask_X_11
    Mask_4L[:,:,nfilt_keep_new:nfilt_keep_new*2]=sub_mask_Y_11
    Mask_4L[:,:,nfilt_keep_new*2:nfilt_keep_new*3]=sub_mask_X_12
    Mask_4L[:,:,nfilt_keep_new*3:nfilt_keep_new*4]=sub_mask_Y_12    
    
    if subtraction:   
        input_mask=Mask_4L[:,:,2*nfilt_keep_new:4*nfilt_keep_new]-Mask_4L[:,:,0:2*nfilt_keep_new]
    else:
        input_mask=Mask_4L
        
    if normalize:
        input_mask-=np.mean(input_mask,axis=1,keepdims=True) # w.r.t time dim
        
    Nfrm=input_mask.shape[1]
    input_mask_padded=np.zeros((nB,n_pad,input_mask.shape[-1]))
    input_mask_padded[:,0:Nfrm,:]=input_mask
#     print(input_mask_padded.shape)
    return input_mask_padded
    
def proc_W(W_array,n_pad=400, subtraction=False, normalize=False):
    
    nB,nN,nW,nF,nCh=W_array.shape
    sub_mask_X_11=W_array[:,0,:,:,0]
    sub_mask_Y_11=W_array[:,0,:,:,1]
    sub_mask_X_12=W_array[:,1,:,:,0]
    sub_mask_Y_12=W_array[:,1,:,:,1]
    
    Mask_4L=np.zeros((nB,nW,int(nF*4)))
    
    Mask_4L[:,:,0:nF]=sub_mask_X_11
    Mask_4L[:,:,nF:nF*2]=sub_mask_Y_11
    Mask_4L[:,:,nF*2:nF*3]=sub_mask_X_12
    Mask_4L[:,:,nF*3:nF*4]=sub_mask_Y_12    
    
    if subtraction:   
        input_mask=Mask_4L[:,:,2*nF:4*nF]-Mask_4L[:,:,0:2*nF]
    else:
        input_mask=Mask_4L
        
    if normalize:
        input_mask-=np.mean(input_mask,axis=1,keepdims=True) # w.r.t time dim
        
    Nfrm=input_mask.shape[1]
    input_mask_padded=np.zeros((nB,n_pad,input_mask.shape[-1]))
    input_mask_padded[:,0:Nfrm,:]=input_mask
#     print(input_mask_padded.shape)
    return input_mask_padded
    

def MinMaxNorm(arr, min_val, max_val, augFact):
    Norm_arr=(arr-min_val)/(max_val*augFact-min_val)
    return Norm_arr
    

def proc_Z(obs_info_set,Z_set_bin,Eta_sub,augFact,obs_info_train_bench,Z_train_bin_bench):
    tg=obs_info_set[:,4]           # event duration
    SR=obs_info_set[:,5]           # Signal SR
    PGA_set=obs_info_set[:,6:10]   # 11X 11Y 12X 12Y
    CIM_set=Z_set_bin[:,:,:,Eta_sub]  # nLevel, nChannel, nEta_list
    
    obs_info_train_bench=np.array(obs_info_train_bench)
    Z_train_bin_bench=np.array(Z_train_bin_bench)
    
    # tg
    tg_min=np.amin(obs_info_train_bench[:,4]) # based on tg_train
    tg_max=np.amax(obs_info_train_bench[:,4])
    tg_norm=MinMaxNorm(tg,tg_min,tg_max,augFact)
    
    # SR
    SR_min=np.amin(obs_info_train_bench[:,5]) # based on SR_train
    SR_max=np.amax(obs_info_train_bench[:,5])
    SR_norm=MinMaxNorm(SR,SR_min,SR_max,augFact)   
    
    # PGA
    PGA_min=np.amin(obs_info_train_bench[:,6:10]) # based on PGA_train
    PGA_max=np.amax(obs_info_train_bench[:,6:10])
    PGA_norm=MinMaxNorm(PGA_set,PGA_min,PGA_max,augFact)   
    
    # CIM features
    R_eta=CIM_set[:,1,:,:]/CIM_set[:,0,:,:]
    
    I_eta_train=Z_train_bin_bench[:,:,:,Eta_sub]
    R_eta_train=I_eta_train[:,1,:,:]/I_eta_train[:,0,:,:]
    R_eta_train_vec=R_eta_train.reshape(-1,len(Eta_sub))
    eta_min_vec=np.amin(R_eta_train_vec,axis=0)
    eta_max_vec=np.amax(R_eta_train_vec,axis=0)
    R_eta_norm=MinMaxNorm(R_eta,eta_min_vec,eta_max_vec,augFact)   
    R_eta_norm_vec=R_eta_norm.reshape(R_eta_norm.shape[0],R_eta_norm.shape[1]*len(Eta_sub))
    
    Z_big=np.zeros((obs_info_set.shape[0],1+1+4+R_eta_norm.shape[1]*len(Eta_sub)))
    Z_big[:,0]=tg_norm
    Z_big[:,1]=SR_norm
    Z_big[:,2:6]=PGA_norm
    Z_big[:,6:]=R_eta_norm_vec
    # print('Z_big.shape: '+str(Z_big.shape))
    return Z_big

from skimage.transform import resize

def optimize_batch_Hydra_W(X_set_bin,Y_set_bin,Z_set_bin,W_set_bin,info_set_bin,
              obs_info_train_bench,Z_train_bin_bench,
              i_f_first,nfilt_keep,Eta_sub,nfilt_WLT,
              feat_type,delta_on,normFact,augFact,shuffle_bin=True):
    
    Shape_bin_set=[]
    for iPoint in X_set_bin:
        Shape_bin_set.append(iPoint.shape[0])
    time_unique=np.unique(Shape_bin_set)

    n_time_unique=len(time_unique)
    n_obs=len(X_set_bin)
    
    X_set_opt=[]
    Y_set_opt=[]
    Z_set_opt=[]
    W_set_opt=[] # Wavelet mask data
    F_set_opt=[] # Frequency range [min,max] from WLT
    info_set_opt=[]
    
    for i_time in range(n_time_unique):
        X_Batch_i=[]
        Y_Batch_i=[]
        Z_Batch_i=[]
        W_Batch_i=[]
        F_Batch_i=[]
        info_batch_i=[]
        for i_obs in range(n_obs):
            if X_set_bin[i_obs].shape[0]==time_unique[i_time]:
                X_Batch_i.append(X_set_bin[i_obs])
                Y_Batch_i.append(Y_set_bin[i_obs])
                # WLT date load, resize and append
                nNode_W,NTSP_W,nFilt_W,nCh_W=W_set_bin[i_obs][0].shape # get the WLT shapes for each observation
                W_i_res=resize(W_set_bin[i_obs][0], (nNode_W,NTSP_W,nfilt_WLT,nCh_W),anti_aliasing=True) # resize to fixed size filters
                W_Batch_i.append(W_i_res) #WLT MASKS
                min_F_WLT=W_set_bin[i_obs][1][0]
                max_F_WLT=W_set_bin[i_obs][1][-1]
                F_Batch_i.append([min_F_WLT,max_F_WLT])
                
                info_batch_i.append(info_set_bin[i_obs])
                Z_Batch_i.append(Z_set_bin[i_obs])

        
        X_Batch_i=np.array(X_Batch_i)
        Y_Batch_i=np.array(Y_Batch_i)/normFact 
        # print(i_time)
        # print(W_Batch_i[0].shape)
        # print(len(W_Batch_i))
        # for iW in range(len(W_Batch_i)):
            # print(info_batch_i[iW][0:5])
            # print(W_Batch_i[iW].shape)
            # print(X_Batch_i[iW].shape)
            
        W_Batch_i=np.array(W_Batch_i)
        F_Batch_i=np.array(F_Batch_i)
        Z_Batch_i=np.array(Z_Batch_i)
        info_batch_i=np.array(info_batch_i)
        
        if len(X_Batch_i.shape)==2:
            X_Batch_i=np.expand_dims(X_Batch_i,0)
            W_Batch_i=np.expand_dims(W_Batch_i,0)
            Y_Batch_i=np.expand_dims(Y_Batch_i,0)
            info_batch_i=np.expand_dims(info_batch_i,0)
            F_Batch_i=np.expand_dims(F_Batch_i,0)
            Z_Batch_i=np.expand_dims(Z_Batch_i,0)
            
        # Process mask data    
        X_Batch_i=enc_input(X_Batch_i,i_f_first,nfilt_keep,
                            feat_type,delta_on,subtraction=True, normalize=False)
        W_Batch_i=proc_W(W_Batch_i, subtraction=True, normalize=False)
        
        Y_Batch_i=np.expand_dims(Y_Batch_i,1)
        # New feature
        Z_Batch_i=proc_Z(info_batch_i,Z_Batch_i,Eta_sub,augFact,obs_info_train_bench,Z_train_bin_bench)
        # Nobs_i=X_Batch_i.shape[0]
        # Break the batch
        if i_time==0:
            X_set_opt=X_Batch_i
            Y_set_opt=Y_Batch_i
            W_set_opt=W_Batch_i
            Z_set_opt=Z_Batch_i
            F_set_opt=F_Batch_i
            info_set_opt=info_batch_i
        else:
            X_set_opt=np.vstack((X_set_opt,X_Batch_i))
            Y_set_opt=np.vstack((Y_set_opt,Y_Batch_i)) 
            W_set_opt=np.vstack((W_set_opt,W_Batch_i)) 
            Z_set_opt=np.vstack((Z_set_opt,Z_Batch_i)) 
            F_set_opt=np.vstack((F_set_opt,F_Batch_i)) 
            info_set_opt=np.vstack((info_set_opt,info_batch_i))
    ### Attention! the output has shffled output 
    if shuffle_bin:
        np.random.seed(0)
        indx_bin=np.arange(0,X_set_opt.shape[0],1)
        np.random.shuffle(indx_bin)
        
        X_set_opt=X_set_opt[indx_bin,:,:]
        Y_set_opt=Y_set_opt[indx_bin,:]
        Z_set_opt=Z_set_opt[indx_bin,:]
        W_set_opt=W_set_opt[indx_bin,:]
        F_set_opt=F_set_opt[indx_bin,:]
        info_set_opt=info_set_opt[indx_bin,:]
    
    # print("X_set_opt.shape: "+str(X_set_opt.shape))
    # print("Y_set_opt.shape: "+str(Y_set_opt.shape))
    # print("Z_set_opt.shape: "+str(Z_set_opt.shape))
    # print("info_set_opt.shape: "+str(info_set_opt.shape))
        
    return X_set_opt,Y_set_opt,Z_set_opt,W_set_opt,F_set_opt,info_set_opt
    
    