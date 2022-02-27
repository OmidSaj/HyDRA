from UtilLibs_W import *
import gc
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Input, Dense, Lambda, Activation
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.models import Model
import glob

# tf.device("cpu:0")

selection_criteria='GM'

normFact=Drift_limit

# =============================================================================
# Learning Hyperparamtersets
# =============================================================================
n_epoch=200
patience=10
drop_rate_GRU=0.08
verbose=1
batch_size=100

Eta_list=[1,3,5,7]
augFact=1.2
# Wavelet hyperparameters 
t_win=1 # one of 0.5,1,2 sec
nfilt_WLT=24

n_sim=4

from keras.regularizers import L1L2
def mae_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def eval_r(Y_true,Y_pred):
    diff=np.abs(Y_true-Y_pred)
    # print('diff.shape:')
    # print(diff.shape)
    diff2=np.power(diff,2)
    MAE=np.mean(diff)
    MSE=np.mean(diff2)
    return MAE,MSE
    
# Pick HP set:
iFr=1          # 2 seconds
iNf=1          # 24 filters
iAlpha=2       # 700
iBeta=0        # 0.0
iShape=1       # Rec
i_f_first=0    # 
nfilt_keep=24 
delta_on=0

# batch_size_train=100
# batch_size_eval=2000

exp_id='01_MZ'   # the experiment ID

import tensorflow as tf
import numpy as np

sample_data=np.load('Data/S_1/train_B_1.npz')
X_0=sample_data['X_i'][0]
Y_0=sample_data['Y_i'][0]
Z_0=sample_data['Z_i'][0]
W_0=sample_data['W_i'][0]
F_0=sample_data['F_i'][0]
info_0=sample_data['info_i'][0]

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, 
                 set_name,
                 fold_id,
                 shuffle=False):   
        
        'Initialization'
        self.shuffle = shuffle
        self.set_name=set_name
        self.fold_id=fold_id
        self.n_batch = len(glob.glob('Data/S_'+str(self.fold_id)+'/'+set_name+'_*.npz'))

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
        X=data_i['X_i']
        Y=data_i['Y_i']
        Z=data_i['Z_i']
        # W=data_i['W_i']
        # F=data_i['F_i']     
        
        del data_i.f
        data_i.close()
        
        # input_mask-=np.mean(input_mask,axis=1,keepdims=True) # w.r.t time dim

        return ([X,Z], Y)
 
def get_Y(set_generator):
    for i in range(set_generator.n_batch):
        data_tuple_i=set_generator.__getitem__(i)
        if i==0:
            Y_gt=data_tuple_i[1]
        else:
            Y_gt=np.vstack((Y_gt,data_tuple_i[1]))
    return Y_gt

fontsize=10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = fontsize
    
def build_train_hydra(i_sim,Save_Dir,train_generator_i,val_generator_i):
    # DL module
    model_filename_save=Save_Dir+'/Model_'+str(i_sim)+'.h5'
    nFilt=X_0.shape[-1]

    reg_kernel= L1L2(l1=1e-6, l2=1e-6)
    reg_bias=L1L2(l1=1e-6, l2=1e-6)
    
    reg_kernel_D= L1L2(l1=1e-5, l2=1e-5)
    reg_bias_D=L1L2(l1=1e-5, l2=1e-5)
    
    # Hydra
    X = Input(shape=(None, nFilt))
    # W = Input(shape=(None, W_0.shape[-1]))    # Wavelet input
    
    Z = Input(shape=Z_0.shape)
    
    # Recurrent Branch
    X_RNN=Masking(mask_value=0.0, input_shape=(None, nFilt))(X)
    X_RNN=GRU(300, activation='tanh',dropout=drop_rate_GRU,return_sequences=True,
                            bias_regularizer=reg_bias,kernel_regularizer=reg_kernel)(X_RNN)

    X_RNN=GRU(50, activation='tanh',dropout=drop_rate_GRU, return_sequences=False,
                            bias_regularizer=reg_bias,kernel_regularizer=reg_kernel)(X_RNN)

    # W_RNN=Masking(mask_value=0.0, input_shape=(None, W_0.shape[-1]))(W)
    # W_RNN=GRU(300, activation='tanh',dropout=drop_rate_GRU,return_sequences=True,
    #                         bias_regularizer=reg_bias,kernel_regularizer=reg_kernel)(W_RNN)
    # W_RNN=GRU(50, activation='tanh',dropout=drop_rate_GRU, return_sequences=False,
    #                         bias_regularizer=reg_bias,kernel_regularizer=reg_kernel)(W_RNN)
                            
    BNeck=concat([X_RNN, Z])
    BNeck=Dense(500, activation='relu', bias_regularizer=reg_bias_D,kernel_regularizer=reg_kernel_D)(BNeck)
    BNeck=Dense(500, activation='tanh', bias_regularizer=reg_bias_D,kernel_regularizer=reg_kernel_D)(BNeck)
    Y=Dense(1, activation='relu', bias_regularizer=reg_bias_D,kernel_regularizer=reg_kernel_D)(BNeck)
    
    model_Hydra= Model([X, Z], Y)
    # print(model_Hydra.summary())

    # if shuffle_id==0:
    #     model_Hydra.summary()
    
    model_Hydra.compile(optimizer='Nadam', loss='mae',metrics=[mae_loss])
    
    model_callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
             ModelCheckpoint(filepath=model_filename_save,
                             monitor='val_mae_loss', save_best_only=True,mode='min')]  #plot_losses
    
    start_time = time.time()
    
    model_hist = model_Hydra.fit(train_generator_i,epochs=n_epoch,verbose=verbose,
                               callbacks=model_callbacks,
                               validation_data=val_generator_i,
                               workers=24,
                               max_queue_size=5)
    end_time = time.time()  
    
    train_dur=end_time-start_time
    
    print('training took %1.1f seconds or %1.2f minutes'%(train_dur, train_dur/60))
    
    # Loss log plot      
    train_loss=model_hist.history['loss']
    val_loss=model_hist.history['val_loss']
    train_loss_pure=model_hist.history['mae_loss']
    val_loss_pure=model_hist.history['val_mae_loss']
    
    del model_hist
    del model_Hydra

    fig,ax=plt.subplots(2,1,figsize=(12,5))
    
    plt.subplots_adjust(wspace=0.25, hspace=0.1)
    
    init_epoch=1
    
    ax[0].plot(train_loss[init_epoch:],'-b')
    ax[0].plot(val_loss[init_epoch:],'-r')
    ax[1].plot(train_loss_pure[init_epoch:],'--b')
    ax[1].plot(val_loss_pure[init_epoch:],'--r')
    ax[0].legend(['training','validation'])
    ax[1].legend(['p_training','p_validation'])
    
    fig.savefig(Save_Dir+'/Loss_'+str(i_sim)+'.png',dpi=300,bbox_inches='tight')
    plt.close()
    K.clear_session()
    gc.collect()     

# A function for training and evaluation

def DL_worker(exp_id,shuffle_id):
    # Load data
    panasas_dir_load='F:/Globus'
    HP_set_name=selection_criteria\
                +'_S_'+str(shuffle_id)\
                +'_M_'+str(model_id)\
                +'_fr_'+str(iFr)\
                +'_nf_'+str(iNf)\
                +'_al_'+str(iAlpha)\
                +'_bt_'+str(iBeta)\
                +'_sh_'+str(iShape)+'/'
    DataDir=panasas_dir_load+'/'+HP_set_name
    
    # Load data                            
    
    obs_info_train=load_pickle(DataDir+'info_train')
    obs_info_val=load_pickle(DataDir+'info_val')
    obs_info_test=load_pickle(DataDir+'info_test')
    
    train_generator=DataGenerator('train',shuffle_id,shuffle=True)
    val_generator=DataGenerator('val',shuffle_id,shuffle=False)
    test_generator=DataGenerator('test',shuffle_id,shuffle=False)
    

    make_di_path('S_OUT')
    Sub_Dir_1='S_OUT/'+exp_id
    make_di_path(Sub_Dir_1)
    Save_Dir=Sub_Dir_1+'/S_'+str(shuffle_id).zfill(2)
    delete_folder(Save_Dir) # clean up from previous runs
    make_di_path(Save_Dir)
    # Allocate storage space for this HP set
    MAE_val_bin=[]
    for i_sim in range(n_sim):
        model_filename_save_i=Save_Dir+'/Model_'+str(i_sim)+'.h5'
        build_train_hydra(i_sim,Save_Dir,train_generator,val_generator)
        # Load and eval
        model_load=load_model(model_filename_save_i,custom_objects={'mae_loss': mae_loss})
        # Y_pred_train=model_load.predict(X_train_opt)*normFact
        Y_pred_val=model_load.predict(val_generator)*normFact
        Y_pred_test=model_load.predict(test_generator)*normFact
        # Y_train_opt=Y_train_opt*normFact.
        Y_val_opt=get_Y(val_generator)
        Y_test_opt=get_Y(test_generator)
        
        Y_val_opt=Y_val_opt*normFact
        Y_test_opt=Y_test_opt*normFact
        
        MAE_val,MSE_val=eval_r(Y_val_opt,Y_pred_val)
        MAE_test,MSE_test=eval_r(Y_test_opt,Y_pred_test)
        
        MAE_val_bin.append(MAE_val)
        # Regression plots
        fig,ax=plt.subplots(1,2,figsize=(10,5))
        
        plt.subplots_adjust(wspace=0.25, hspace=0)
        
        ax[0].plot(Y_val_opt,Y_pred_val,'bx',markersize=1)
        ax[0].plot([0,1],[0,1],'-r')
        ax[0].set_xlim(0,Drift_limit)
        ax[0].set_ylim(0,Drift_limit)
        ax[0].set_title('Validation-MAE='+str(np.round(MAE_val,6)))
        
        ax[1].plot(Y_test_opt,Y_pred_test,'bx',markersize=1)
        ax[1].plot([0,1],[0,1],'-r')
        ax[1].set_xlim(0,Drift_limit)
        ax[1].set_ylim(0,Drift_limit)
        ax[1].set_title('Testing-MAE='+str(np.round(MAE_test,6)))
        
        for i in range(2):
            ax[i].set_xlabel('Ground truth')
            ax[i].set_ylabel('Prediction')
        
        fig.savefig(Save_Dir+'/Regres_plots_'+str(i_sim)+'.png',dpi=300,bbox_inches='tight')
        plt.close()
    
        
        print('Validation:  MAE= %1.4f,  MSE= %1.6f'%(MAE_val,MSE_val))
        print('Testing:     MAE= %1.4f,  MSE= %1.6f'%(MAE_test,MSE_test))
        
        if MAE_val==np.amin(MAE_val_bin):
            model_load.save(Save_Dir+'/Model.h5')
            MAE_val_best=MAE_val
            MAE_test_best=MAE_test
            
            np.savez_compressed(Save_Dir+'/EvalData_M_'+str(model_id).zfill(2)+'_'+selection_criteria+'.npz',
                               Y_true_val=Y_val_opt,Y_pred_val=Y_pred_val,
                               Y_true_test=Y_test_opt,Y_pred_test=Y_pred_test,
                               MAE_test=MAE_test,MSE_test=MSE_test,
                               MAE_val=MAE_val, MSE_val=MSE_val,
                               obs_info_val=obs_info_val,
                               obs_info_test=obs_info_test)

        np.savez_compressed(Save_Dir+'/EvalData_M_'+str(model_id).zfill(2)+'_'+selection_criteria+'_'+str(i_sim)+'.npz',
                           Y_true_val=Y_val_opt,Y_pred_val=Y_pred_val,
                           Y_true_test=Y_test_opt,Y_pred_test=Y_pred_test,
                           MAE_test=MAE_test,MSE_test=MSE_test,
                           MAE_val=MAE_val, MSE_val=MSE_val,
                           obs_info_val=obs_info_val,
                           obs_info_test=obs_info_test)
        del model_load     
        K.clear_session()
        gc.collect()
    
    del obs_info_train
    del obs_info_val
    del obs_info_test
    del Y_val_opt
    del Y_pred_val
    del Y_test_opt
    del Y_pred_test
    del train_generator
    del val_generator
    del test_generator
    gc.collect()
    return MAE_val_best,MAE_test_best
    
# Model_evaluator

def DL_evaluator(exp_id,model_S_bin):
    
    n_model=len(model_S_bin)  ## Number of models to load and evalaute
    Y_pred_test_bin=[]
    
    for i_model in range(n_model):
        h5_model_id=model_S_bin[i_model]
        
        # Load data
        panasas_dir_load='F:/Globus'
        HP_set_name=selection_criteria\
                    +'_S_'+str(h5_model_id)\
                    +'_M_'+str(model_id)\
                    +'_fr_'+str(iFr)\
                    +'_nf_'+str(iNf)\
                    +'_al_'+str(iAlpha)\
                    +'_bt_'+str(iBeta)\
                    +'_sh_'+str(iShape)+'/'
        DataDir=panasas_dir_load+'/'+HP_set_name
        
        # Load data                            
        # obs_info_train=load_pickle(DataDir+'info_train')
        # obs_info_val=load_pickle(DataDir+'info_val')
        obs_info_test=load_pickle(DataDir+'info_test')
        
        # train_generator=DataGenerator(obs_info_train,batch_size_train,shuffle=True)
        # val_generator=DataGenerator(obs_info_val,batch_size_eval,shuffle=False)
        test_generator=DataGenerator('test',h5_model_id,shuffle=False)
        

        Y_test_opt=get_Y(test_generator)
        
        # Evaluate
        Y_test_opt_scaled=Y_test_opt*normFact
        
        # DL module
        Sub_Dir_1='S_OUT/'+exp_id
        Load_Dir=Sub_Dir_1+'/S_'+str(h5_model_id).zfill(2)
        model_filename_save=Load_Dir+'/Model.h5'
    
        # Load and eval
        model_load=load_model(model_filename_save,custom_objects={'mae_loss': mae_loss})
        Y_pred_test_bin.append(model_load.predict(test_generator)*normFact)
        # Y_train_opt=Y_train_opt*normFact

    Y_pred_test_bin=np.array(Y_pred_test_bin)
    
    Y_pred_test_std=np.std(Y_pred_test_bin,axis=0)
    Y_pred_test_avg=np.mean(Y_pred_test_bin,axis=0)

        
    MAE_test,MSE_test=eval_r(Y_test_opt_scaled,Y_pred_test_avg)
    
    # Regression plots
    fig,ax=plt.subplots(1,2,figsize=(5,10))
    
    plt.subplots_adjust(wspace=0.25, hspace=0)
    
    
    ax[0].plot(Y_test_opt_scaled,Y_pred_test_avg,'bx',markersize=1)
    ax[0].plot([0,1],[0,1],'-r')
    ax[0].set_xlim(0,Drift_limit)
    ax[0].set_ylim(0,Drift_limit)
    ax[0].set_title('Testing-AVG MAE='+str(np.round(MAE_test,6)))
    
    ax[0].set_xlabel('Ground truth')
    ax[0].set_ylabel('Prediction')

    ax[1].plot(Y_pred_test_avg,Y_pred_test_std,'bx',markersize=1)
    # ax[1].plot([0,1],[0,1],'-r')
    ax[1].set_xlim(0,Drift_limit)
    # ax[1].set_ylim(0,Drift_limit)
    # ax[1].set_title('Testing-AVG MAE='+str(np.round(MAE_test,6)))
    
    ax[1].set_xlabel('Prediction')
    ax[1].set_ylabel('Uncertainty (std)')    
    fig.savefig(Sub_Dir_1+'/Avg_test_regres_plot.png',dpi=300,bbox_inches='tight')
    plt.close()
    np.savez_compressed(Sub_Dir_1+'/EvalData_test_avg.npz',
                       Y_true_test=Y_test_opt_scaled,Y_pred_test=Y_pred_test_avg,
                       MAE_test=MAE_test,MSE_test=MSE_test,
                       obs_info_test=obs_info_test)

    K.clear_session()
    gc.collect()
    
    return MAE_test
    
model_S_bin=[1,2,3,4,5,6]

MAE_val_bin=[]
MAE_test_bin=[]

for sh_i in range(len(model_S_bin)):  
    sh_id=model_S_bin[sh_i]
    MAE_val_i,MAE_test_i=DL_worker(exp_id,sh_id)
    MAE_val_bin.append(MAE_val_i)
    MAE_test_bin.append(MAE_test_i)
    gc.collect()
print(' ')
print(' ')
print('================================================')
print('Fold          MAE_val    MAE_test')

for sh_i in range(len(model_S_bin)):      
    print('%s            %1.6f   %1.6f'%(str(model_S_bin[sh_i]),MAE_val_bin[sh_i],MAE_test_bin[sh_i]))

print('---------------------------')    
print('mean         %1.6f   %1.6f'%(np.mean(MAE_val_bin),np.mean(MAE_test_bin))) 
print('std          %1.6f   %1.6f'%(np.std(MAE_val_bin),np.std(MAE_test_bin)))   

print('--------Ensemble-----------') 
MAE_test_avg=DL_evaluator(exp_id,model_S_bin)
print('Test avg MAE:           %1.6f'%(MAE_test_avg)) 

from n_ZW import *