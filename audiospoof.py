# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 23:23:53 2017

@author: Nigel Wee

This whole thing takes about 10mins to compile 
"""
import os
import numpy as np
from sklearn import mixture
import librosa
import matplotlib.pyplot as plt
import functions as fn

# The paths where the full training files are stored
path1 = r"C:\Users\Nigel Wee\Desktop\train_dev_test\training\genuine"
path2 = r"C:\Users\Nigel Wee\Desktop\train_dev_test\training\spoofed"

#Partial training files
#path1 = r"C:\Users\Nigel Wee\Desktop\train_subdivided_samples\genuine"
#path2 = r"C:\Users\Nigel Wee\Desktop\train_subdivided_samples\spoofed"

# A list of all audiofiles 
audiofile1 = os.listdir(path1)  # genuine
audiofile2 = os.listdir(path2)  # spoofed


# For the first file only   
# Load the audiofile
y, sr = librosa.load(os.path.join(path1, audiofile1[0])) # y = audio time series, sr = sampling rate
   
# Extract MFCC features and append
mfcc = librosa.feature.mfcc(y=y, sr=sr)
#g_mfccObj = fn.DataHolder(mfcc)
   
# Extract MFCC Delta and append
mfcc_delta = librosa.feature.delta(mfcc)
#g_deltaObj = fn.DataHolder(mfcc_delta)
   
# Extract MFCC Delta-Delta and append
mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
#g_delDelObj = fn.DataHolder(mfcc_delta_delta)  

g_feature_obj = fn.DataHolder(mfcc, mfcc_delta, mfcc_delta_delta)

# Extract Mfcc features from all audio files
for filename in audiofile1[1:]:
    
   # Concatenate path with filename 
   filename = os.path.join(path1, filename)
   
   # Load the audiofile
   y, sr = librosa.load(filename) # y = audio time series, sr = sampling rate
   
   # Extract MFCC features and append
   mfcc = librosa.feature.mfcc(y=y, sr=sr)
   #g_mfccObj = fn.DataHolder(mfcc, g_mfccObj)
   
   # Extract MFCC Delta and append
   mfcc_delta = librosa.feature.delta(mfcc)
   #g_deltaObj = fn.DataHolder(mfcc_delta, g_deltaObj)
   
   # Extract MFCC Delta-Delta and append
   mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
   #g_delDelObj = fn.DataHolder(mfcc_delta_delta, g_delDelObj)   
   
   g_feature_obj = fn.DataHolder(mfcc, mfcc_delta, mfcc_delta_delta, g_feature_obj)   

# For the first file only   
# Load the audiofile
y, sr = librosa.load(os.path.join(path2, audiofile2[0])) # y = audio time series, sr = sampling rate
   
# Extract MFCC features and append
mfcc = librosa.feature.mfcc(y=y, sr=sr)
#s_mfccObj = fn.DataHolder(mfcc)
   
# Extract MFCC Delta and append
mfcc_delta = librosa.feature.delta(mfcc)
#s_deltaObj = fn.DataHolder(mfcc_delta)
   
# Extract MFCC Delta-Delta and append
mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
#s_delDelObj = fn.DataHolder(mfcc_delta_delta)  
   
s_feature_obj = fn.DataHolder(mfcc, mfcc_delta, mfcc_delta_delta)

for filename in audiofile2[1:]:
   
   # Concatenate path with filename 
   filename = os.path.join(path2, filename)
   
   # Load the audio file
   y, sr = librosa.load(filename) # y = audio time series, sr = sampling rate
   
   # Extract MFCC features
   mfcc = librosa.feature.mfcc(y=y, sr=sr)
   #s_mfccObj = fn.DataHolder(mfcc, s_mfccObj)
   
   # Extract MFCC Delta
   mfcc_delta = librosa.feature.delta(mfcc)
   #s_deltaObj = fn.DataHolder(mfcc_delta, s_deltaObj)
   
   # Extract MFCC Delta-Delta
   mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
   #s_delDelObj = fn.DataHolder(mfcc_delta_delta, s_delDelObj)  
   s_feature_obj = fn.DataHolder(mfcc, mfcc_delta, mfcc_delta_delta, s_feature_obj)   
   
mean_g = fn.calMeanOfFrames(g_feature_obj) 
mean_s = fn.calMeanOfFrames(s_feature_obj)  

genuine = np.array(fn.vStackFeatures(mean_g, g_feature_obj))
spoofed= np.array(fn.vStackFeatures(mean_s, s_feature_obj))

gmm1 = mixture.GaussianMixture(n_components = 20)
gmm1.fit(genuine)

gmm2 = mixture.GaussianMixture(n_components = 20)
gmm2.fit(spoofed)  

 