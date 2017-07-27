# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:06:19 2017

@author: Nigel Wee
"""
from sklearn import mixture
from sklearn import preprocessing
import librosa
import numpy as np
#import joblib to save model
from sklearn.externals import joblib

Genuine = np.load(r'C:\Users\Nigel Wee\Desktop\PythonProjects\Genuine.npy')
Spoofed = np.load(r'C:\Users\Nigel Wee\Desktop\PythonProjects\Spoofed.npy')

Genuine = preprocessing.scale(Genuine)

Spoofed = preprocessing.scale(Spoofed)

gmm1 = mixture.GaussianMixture(n_components = 20)
gmm1.fit(Genuine)
#
gmm2 = mixture.GaussianMixture(n_components = 20)
gmm2.fit(Spoofed)  

#persist model for future use
joblib.dump(gmm1, 'gmm1_g')
joblib.dump(gmm2, 'gmm2_s')