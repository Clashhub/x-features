# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:44:19 2017

@author: Nigel Wee

"""
import numpy as np
import librosa

class DataHolder(object):
    def __init__(self, mfcc, delta, deldel, previous = None):        
        self.data = np.vstack([mfcc, delta, deldel])
        self.previous = previous     
        
def calMeanOfFrames(object):
   sum = 0
   x = 0
   feature_object = object
   while feature_object.previous != None:
       x += 1
       sum += len(feature_object.data[0])
       feature_object = feature_object.previous
    
   #For the first object 
   sum += len(feature_object.data[0])   
   x += 1
   
   mean = sum // x 
   return mean

def vStackFeatures(mean, object):
    feature_object = object
    
    if len(feature_object.data[0]) < mean:
      feature_object.data = np.pad(feature_object.data, [(0,0),(0, mean-len(feature_object.data[0]))], 'constant')
    
    a = np.array(feature_object.data[0:20, :mean])
    feature_object = feature_object.previous
    
    while feature_object.previous != None:
        if len(feature_object.data[0]) < mean:
            feature_object.data = np.pad(feature_object.data, [(0,0),(0, mean-len(feature_object.data[0]))], 'constant')
            
        a = np.vstack((feature_object.data[0:20, :mean],a))
        feature_object = feature_object.previous
     
    if len(feature_object.data[0]) < mean:
      feature_object.data = np.pad(feature_object.data, [(0,0),(0, mean-len(feature_object.data[0]))], 'constant')  
      
    a = np.vstack((feature_object.data[0:20, :mean],a)) 
    
    return a
    
def getFeatures(mean_g, mean_s):
  # For the first file only   
  # Load the audiofile
  y, sr = librosa.load(r"C:\train_dev_test\devolpment\genuine\D_1000001.wav") # y = audio time series, sr = sampling rate
    
  # Extract MFCC features and append
  mfcc = librosa.feature.mfcc(y=y, sr=sr)
   
  # Extract MFCC Delta and append
  mfcc_delta = librosa.feature.delta(mfcc)
    
  # Extract MFCC Delta-Delta and append
  mfcc_delta_delta = librosa.feature.delta(mfcc_delta)

  a = np.vstack([mfcc, mfcc_delta, mfcc_delta_delta])
  b = a
  if len(a[0]) < mean_g:
     a = np.pad(a, [(0,0),(0, mean_g-len(a[0]))], 'constant')
     
  if len(b[0]) < mean_s:
     b = np.pad(b, [(0,0),(0, mean_s-len(b[0]))], 'constant')
        
  return a[0:20, :mean_g], b[0:20, :mean_s]   