# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:14:14 2022

@author: Seyed Omid Sajedi """

from DataGenerators import DataGenerator_MZW
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import time
import gc 
from OsUtils import save_pickle

def mae_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

model_name=['01_X','01_W','01_Z','01_XZ','01_XW','01_ZW','01_XZW']  # '02_XZW_B','02_XZW_B50'
features_list=['M','W','Z','MZ','MW','WZ','MZW']
nFold=6
n_repeat=10

inf_time_dict={}

for i, iModel in enumerate(model_name):
    data_dir='S_OUT/'+iModel
    for sh_i in range(nFold):
        inf_time_dict[(iModel,sh_i)]=[]
        test_generator = DataGenerator_MZW('test',sh_i+1,shuffle=False,features= features_list[i])
        model_filename = data_dir+'/S_'+str(sh_i+1).zfill(2)+'/Model.h5'
        model_i = load_model(model_filename,custom_objects={'mae_loss': mae_loss})
        for i_rep in range(n_repeat+1):
            t_start=time.time()
            model_i.predict(test_generator)
            t_end=time.time()
            t_inf=t_end-t_start   
            if i_rep>0: # removing the first time prediction as the outlier (longer due to graph creation)
                inf_time_dict[(iModel,sh_i)].append(t_inf)
                print('%s, %d, t= %1.2f'%(iModel,sh_i+1,t_inf))
        
        del model_i    
        K.clear_session()
        gc.collect()
        
print(inf_time_dict)
save_pickle(inf_time_dict,'inf_time_dict')