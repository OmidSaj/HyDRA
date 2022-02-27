# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:26:47 2022

@author: Seyed Omid Sajedi """

from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, GRU, Dropout
from tensorflow.keras import backend as K
import numpy as np

def build_hydra(hydra_id:str='MWZ',
                nFeat_M:int=48,
                nFeat_W:int=48,
                nFeat_Z:int=14,
                drop_rate_GRU:float=0.08,
                drop_rate_dense:float=0.2,
                Bayesian_BN:bool=False,
                show_model_summary:bool=False):
    """

    Parameters
    ----------
    hydra_id : str, optional
        String containing the name of Hydra model branches (without spaces).
        Acceptable choices are any combinations of M,W and Z. The order in the
        input string is not important. However, the model's input list order 
        is M, W, Z. For example, if the hydra_id='ZWM' or 'MZW', the model input 
        will always be [M, W, Z] (or any subset). The default is 'MWZ'.
    nFeat_M : int, optional
        M branch input tensor channels feature dimension. The default is 48.
    nFeat_W : int, optional
        W branch input tensor channels feature dimension. The default is 48.
    nFeat_Z : int, optional
        Z branch input tensor(vector) channels feature . The default is 14.
    drop_rate_GRU : float, optional
        GRU layer dropout ratio. The default is 0.08.
    drop_rate_dense : float, optional
        Dense layer dropout ratio. The default is 0.2.
    Bayesian_BN : bool, optional
        If true, will add a dropout layer after each dense layer in both training
        and inference. Make sure the expected values are extracted with Monte Carlo 
        Dropout Sampling if this option is set to True. The default is False.
    show_model_summary : bool, optional
        Print a summary of model architecture and the number of paramters. The
        default is False.

    Returns
    -------
    The Hydra model. Input to the model is a list of input tensor in the order
    [M, W, Z] (or any subset) and the output is a single positive float. 

    """
    hydra_branches=['M','Z','W']
    
    for branch_i in hydra_id:
        if branch_i not in hydra_branches:
            raise ValueError("Invalid branch %s, should be one of  'M','Z','W'"%(branch_i))

    reg_kernel= L1L2(l1=1e-6, l2=1e-6)
    reg_bias=L1L2(l1=1e-6, l2=1e-6)
    
    reg_kernel_D= L1L2(l1=1e-5, l2=1e-5)
    reg_bias_D=L1L2(l1=1e-5, l2=1e-5)
        
    model_branches=[m_i for m_i in hydra_id]
    net_branch_list=[]
    input_layer_list=[]
    # =============================================================================
    #  M branch   
    # =============================================================================
    if 'M' in model_branches:
        M = Input(shape=(None, nFeat_M))
        M_RNN=Masking(mask_value=0.0, input_shape=(None, nFeat_M))(M)
        M_RNN=GRU(300, activation='tanh',dropout=drop_rate_GRU,return_sequences=True,
                                bias_regularizer=reg_bias,kernel_regularizer=reg_kernel)(M_RNN)
    
        M_RNN=GRU(50, activation='tanh',dropout=drop_rate_GRU, return_sequences=False,
                                bias_regularizer=reg_bias,kernel_regularizer=reg_kernel)(M_RNN)
        input_layer_list.append(M)
        net_branch_list.append(M_RNN)

    # =============================================================================
    # W branch   
    # =============================================================================
    if 'W' in model_branches:
        W = Input(shape=(None, nFeat_W))    
        W_RNN=Masking(mask_value=0.0, input_shape=(None, nFeat_W))(W)
        W_RNN=GRU(300, activation='tanh',dropout=drop_rate_GRU,return_sequences=True,
                                bias_regularizer=reg_bias,kernel_regularizer=reg_kernel)(W_RNN)
        W_RNN=GRU(50, activation='tanh',dropout=drop_rate_GRU, return_sequences=False,
                                bias_regularizer=reg_bias,kernel_regularizer=reg_kernel)(W_RNN)
        input_layer_list.append(W)
        net_branch_list.append(W_RNN)                       
    # =============================================================================
    # Z branch   
    # =============================================================================
    if 'Z' in model_branches:
        Z = Input(shape=nFeat_Z)
        input_layer_list.append(Z)
        net_branch_list.append(Z) 
    # =============================================================================
    #  Bottleneck   
    # =============================================================================
    if len(net_branch_list)>1: 
        BNeck=concat(net_branch_list)
    else: # single input branch like: M, W, Z
        BNeck=net_branch_list[0]
    BNeck=Dense(500, activation='relu', bias_regularizer=reg_bias_D,kernel_regularizer=reg_kernel_D)(BNeck)
    if Bayesian_BN:
        BNeck=Dropout(drop_rate_dense)(BNeck,training=True)
    BNeck=Dense(500, activation='tanh', bias_regularizer=reg_bias_D,kernel_regularizer=reg_kernel_D)(BNeck)
    if Bayesian_BN:
        BNeck=Dropout(drop_rate_dense)(BNeck,training=True)
    Y=Dense(1, activation='relu', bias_regularizer=reg_bias_D,kernel_regularizer=reg_kernel_D)(BNeck)        
    
    if len(input_layer_list)==1:
        input_layer_list=input_layer_list[0]
    model_hydra =  Model(input_layer_list, Y)   
    if show_model_summary:
        print(model_hydra.summary())
    return model_hydra

def mae_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def eval_r(Y_true,Y_pred):
    diff=np.abs(Y_true-Y_pred)
    diff2=np.power(diff,2)
    MAE=np.mean(diff)
    MSE=np.mean(diff2)
    return MAE,MSE

def agg_Y(set_generator):
    for i in range(set_generator.n_batch):
        data_tuple_i=set_generator.__getitem__(i)
        if i==0:
            Y_gt=data_tuple_i[1]
        else:
            Y_gt=np.vstack((Y_gt,data_tuple_i[1]))
    return Y_gt