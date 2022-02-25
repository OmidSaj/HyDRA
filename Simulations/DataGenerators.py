# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:25:23 2022

@author: Seyed Omid Sajedi """

import numpy as np
import tensorflow as tf
import glob

class DataGenerator_MZW(tf.keras.utils.Sequence):

    def __init__(self, 
                 set_name,
                 fold_id,
                 features='MWZ',
                 shuffle=False):   
        
        'Initialization'
        self.shuffle = shuffle
        self.set_name=set_name
        self.fold_id=fold_id
        self.n_batch = len(glob.glob('Data/S_'+str(self.fold_id)+'/'+set_name+'_*.npz'))
        self.feat_list=[f for f in features]
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        data_tuple = self.__data_generation(self.indexes[index])

        return data_tuple

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0,self.n_batch)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Generate data
        fname_i='Data/S_'+str(self.fold_id)+'/'+self.set_name+'_B_'+str(int(index+1))+'.npz'
        data_i=np.load(fname_i)
        input_list=[]

        # X=data_i['X_i']
        Y=data_i['Y_i']
        # Z=data_i['Z_i']
        # W=data_i['W_i']
        # F=data_i['F_i']     
        
        for feat in self.feat_list:
            if feat=='M':
                input_list.append(data_i['X_i'])
            if feat=='W':
                input_list.append(data_i['W_i'])
            if feat=='Z':
                input_list.append(data_i['Z_i'])
            
            
        del data_i.f
        data_i.close()

        return (input_list, Y)