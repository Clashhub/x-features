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
#import matplotlib.pyplot as plt
#import functions as fn

#import joblib to save model
from sklearn.externals import joblib

# The paths where the full training files are stored
path1 = r"C:\train_dev_test\training\genuine"
path2 = r"C:\train_dev_test\training\spoofed"

#Partial training files
#path1 = r"C:\train_subdivided_samples\genuine"
#path2 = r"C:\train_subdivided_samples\spoofed"

# A list of all audiofiles 
audiofile1 = os.listdir(path1)  # genuine
audiofile2 = os.listdir(path2)  # spoofed

sum = 0
gen_list = []
spo_list = []
frames = 100

# Extract Mfcc features from all audio files
for filename in audiofile1:
    
   # Concatenate path with filename 
   filename = os.path.join(path1, filename)
   
   # Load the audiofile
   y, sr = librosa.load(filename) # y = audio time series, sr = sampling rate
   
   # Extract MFCC features and append
   mfcc = librosa.feature.mfcc(y=y, sr=sr)
   sum += len(mfcc[0])
   
   # Extract MFCC Delta and append
   mfcc_delta = librosa.feature.delta(mfcc)
  
   # Extract MFCC Delta-Delta and append
   mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
     
   a = np.vstack([mfcc, mfcc_delta, mfcc_delta_delta])
   
   if len(a[0]) < frames:
       a = np.pad(a, [(0,0),(0, frames-len(a[0]))], 'constant')
   elif len(a[0]) > frames:
       a = a[0:60, :frames]
       
   gen_list.append(np.transpose(a))
        
genuine = np.vstack(gen_list)      

sum = 0

for filename in audiofile2:
   
   # Concatenate path with filename 
   filename = os.path.join(path2, filename)
   
   # Load the audio file
   y, sr = librosa.load(filename) # y = audio time series, sr = sampling rate
   
   # Extract MFCC features
   mfcc = librosa.feature.mfcc(y=y, sr=sr)
   sum += len(mfcc[0])
   
   # Extract MFCC Delta
   mfcc_delta = librosa.feature.delta(mfcc)

   # Extract MFCC Delta-Delta
   mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
   
   b = np.vstack([mfcc, mfcc_delta, mfcc_delta_delta])
   
   if len(b[0]) < frames:
       b = np.pad(b, [(0,0),(0, frames-len(b[0]))], 'constant')
   elif len(b[0]) > frames:
       b = b[0:60, :frames]
       
   spo_list.append(np.transpose(b))
   
spoofed = np.vstack(spo_list) 

gmm1 = mixture.GaussianMixture(n_components = 20)
gmm1.fit(genuine)
#
gmm2 = mixture.GaussianMixture(n_components = 20)
gmm2.fit(spoofed)  

#persist model for future use
joblib.dump(gmm1, 'gmm1_g')
joblib.dump(gmm2, 'gmm2_s')

 