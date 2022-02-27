# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:08:31 2020

@author: Omid
"""

import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import glob
import numpy as np
import re
import pandas as pd
# sample_rate, signal = scipy.io.wavfile.read('OSR_us_000_0010_8k.wav')  # File assumed to be in the same directory
# signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds

# =============================================================================
# Noise module from P2
# =============================================================================

def Add_noise(clean_time_history,Noise_SNR_std):
    n_noisy_set=Noise_SNR_std.shape[0]
    NTSP=clean_time_history.shape[0]
    nchannel=clean_time_history.shape[1]
    
    Signal_std_nodes=np.std(clean_time_history,axis=0)
    tmp=np.repeat(Signal_std_nodes[:, np.newaxis], n_noisy_set, axis=1)
    Noise_std_nodes=tmp*Noise_SNR_std
    
    white_noise_mat = np.random.normal(0, Noise_std_nodes, size=(NTSP,nchannel,n_noisy_set))
    clean_bin_rep=np.repeat(clean_time_history[:,:, np.newaxis], n_noisy_set, axis=2)  
    noisy_time_history_mat=np.add(clean_bin_rep,white_noise_mat)  
    return noisy_time_history_mat #,white_noise_mat,Signal_std_nodes

def get_MFCC(signal,sample_rate,frame_size,frame_stride,nfilt):
    
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    
    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
   
    NFFT=512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    h700=1
    h2595=2595
    
    low_freq_mel = 0
    high_freq_mel = (h2595 * np.log10(1 + (sample_rate / 2) / h700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (h700 * (10**(mel_points / h2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
        
    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
    
    return mfcc
    

def muli_dim_MFCC(Accl_aa,sample_rate,frame_size,frame_stride,nfilt):
    
    nChannel=Accl_aa.shape[1]
    nNoise=Accl_aa.shape[2]
    
    MFCC_bin=[]
    for i in range(nChannel):
        for j in range(nNoise):
            signal=Accl_aa[:,i,j]
            MFCC_bin.append(get_MFCC(signal,sample_rate,frame_size,frame_stride,nfilt))
            
    MFCC_bin=np.array(MFCC_bin)
    
    return MFCC_bin.reshape(nChannel,nNoise,MFCC_bin.shape[1],MFCC_bin.shape[2])