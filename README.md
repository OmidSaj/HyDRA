# HyDRA

## Introduction
This repository is the official implementation of "Filter Banks and Hybrid Deep Learning Architectures for Performance-based Seismic Assessments of Bridges" by [Seyed Omid Sajedi](https://github.com/OmidSaj) and [Xiao Liang](https://github.com/benliangxiao). Hybrid Deep learning models for Rapid Assessments (HyDRA) is introduced as a multi-branch neural network architecture that enable end-to-end training for different types of processed vibration data structures for structural damage diagnosis.

## Mel Filter Banks 
![Custom filter banks](https://github.com/OmidSaj/HyDRA/blob/main/Assets/MFB_demo.gif)

## Architecture
![HyDRA models](https://github.com/OmidSaj/HyDRA/blob/main/Assets/HyDRA.jpg)

## Getting started
* The code is tested on a Python environment and CUDA installation compatible with tensorflow 2.6 (See [here](https://www.tensorflow.org/install/source)). After cloning this repository, make sure to download the Dataset from the latest [Release](https://github.com/OmidSaj/HyDRA/releases/tag/Dataset). To optimize training and infererence, the dataset contains processed input features and labels from the seismic simulations. After downloading the dataset, extract all the files together and make sure to updates the paths in the desired scripts. A subset bin of raw signals can also be downloaded from [here](https://github.com/OmidSaj/HyDRA/releases/tag/GM_signal_sample) to provide insight on feature extraction. 

* Two example notebooks are provided for the [signal preprocessing](https://github.com/OmidSaj/HyDRA/blob/main/seismic_MFB_insights.ipynb) and [deep learning](https://github.com/OmidSaj/HyDRA/blob/main/HyDRA_example.ipynb) stages of this implementation. 

* The code used to generate the benchmark results in the paper can be found inside the [utils](https://github.com/OmidSaj/HyDRA/tree/main/utils) directory. 

## Citing this work
Please cite the following paper if you have used this repositpry. 

Sajedi, S, Liang, X. "Filter Banks and Hybrid Deep Learning Architectures for Performance-based Seismic Assessments of Bridges", Paper under review in the journal of Structural Engineering (citation record will be updated soon)

## Acknowledgment
* Parts of the code regarding the extraction of filter banks from the raw signals is based on Haytham Fayek's excellet [tutorial on Speech Processing for Machine Learning](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html). 
* [Mel Frequency Cepstral Coefficient (MFCC) tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) by James Lyons is a great resource to obtain deep insight on the theoretical background of filter banks for speech processing. 
