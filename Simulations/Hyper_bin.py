# This py file is a global reference for all workers to get the main hyperparameter 
# settings for dataset generation. 
# This file can be shared with other modules in the model. 

import numpy as np

Noise_level=1
model_id=2 # second model AAA2
Noise_NSR_var=np.array([0.0,0.2])
Noise_NSR_std=Noise_NSR_var**0.5
    
Fr_size_bin=np.array([1,2])      # frame size, sec
Fr_strd_bin=np.array([0.5,1])    # frame stride, sec, consistent with frame size
NFFT_bin=np.array([512,1024])    # Number of FFT points, consistent with frame size
Nfilt_bin=np.array([12,24,36])   # Number of filters 
N_alpha=3                        # Number of different alpha values
                                 # Minimum alpha is automatically calculated based on the Nfilt
alpha_max=700                    # Maximum alpha is set to 700 as in deafualt MFCC definition
beta_bin=np.array([0,-0.2,-0.5]) # Beta value as the rate of increase in filter height w.r.t frequency 
Filt_shape_bin=['Tri','Rec']         # triangular or rectangular

pre_emphasis=0        # No pre-emphasis
hamming_filt=False
MeanNormMFCC=False
MeanNormMSFB=False

# Bin of eta values for CIM
ETA=[0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]            # hyperparameter for feature generation
Ang_list=[0,30,60,90,120,150]                        # [0,30,60,90,120,150]
Drift_limit=0.075                                    # I will not save NRHAs with higher drifts (if any). 


# The following is used to generate the alpha bin

dt_min=0.0024 # this information is adopted from the dataset
SR_max=1/dt_min
alpha_range_bin=np.arange(1,alpha_max+1,1)
alpha_bin=np.zeros((NFFT_bin.shape[0],Nfilt_bin.shape[0],N_alpha))
# Some utiliy functions first

def F2M(f,alpha):
    return np.log10(1+f/alpha)
    
def M2F(m,alpha):
    return alpha*(10**(m)-1) 
    
def df(SR,Nfilt,NFFT,alpha):
    dm=F2M(SR/2,alpha)/(Nfilt+1)
    df=M2F(dm,alpha)
    return df
    
def N_FFT_min(SR,alpha,Nfilt):
    NFFT_min=SR/(alpha*((1+SR/(2*alpha))**(1/(Nfilt+1))-1))
    return NFFT_min
    
def get_alpha(dm00,SR,Nfilt):
    return SR/(2*(10**(dm00*(Nfilt+1))-1))
def dm0(alpha,SR,Nfilt):
    return np.log10(1+SR/(2*alpha))/(Nfilt+1)
def df0(dm00,alpha):
    return alpha*(10**dm00-1)
    
# now readyy to generate alphas
for j in range(NFFT_bin.shape[0]):
    NFFT_j=NFFT_bin[j]
    for i in range(Nfilt_bin.shape[0]):
        Nf_i=Nfilt_bin[i]
        NFF_min_bin_i= N_FFT_min(SR_max,alpha_range_bin,Nf_i)
        indx_min=np.amin(np.where(NFFT_j-NFF_min_bin_i>0))
        alpha_min_i=alpha_range_bin[indx_min]

        dm_min=dm0(alpha_min_i,SR_max,Nf_i)
        dm_max=dm0(alpha_max,SR_max,Nf_i)
        dm_bin=np.linspace(dm_max,dm_min,N_alpha)

        alpha_bin_i=get_alpha(dm_bin,SR_max,Nf_i)
        alpha_bin[j,i,:]=alpha_bin_i
# will be used later
alpha_bin=np.round(alpha_bin).astype(int)

