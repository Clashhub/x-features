# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:49:10 2017

@author: Nigel Wee

The name of the gmm models from joblib are
gmm1_g for genuine and gmm2_s for spoofed
"""
from sklearn.externals import joblib
from functions import getFeatures
from functions import getDir
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import numpy as np


#load the models first
gmm1 = joblib.load('gmm1_g')
gmm2 = joblib.load('gmm2_s')

def predictFile():
    a = gmm1.score(feature_list)
    b = gmm2.score(feature_list)
    c = a - b
    
    if c < 0:
        result = "Spoofed"
    elif c > 0:
        result = "Genuine"
          
    
#Instantiate main top level window
root = Tk()
root.title("Anti-spoofing") 

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
       
result = StringVar() 
dirname = ""
feature_list = 0


ttk.Button(mainframe, text="Select directory", command =getDir(dirname).grid(column=3, row=1, sticky=W)
#ttk.Button(mainframe, text="Extract features", command=getFeatures(feature_list,path)).grid(column=3, row=2, sticky=(W,N))
ttk.Button(mainframe, text="Predict", command=predictFile).grid(column=3, row=3, sticky=W)


ttk.Label(mainframe, text="result:").grid(column=1,row=3, sticky=W)
ttk.Label(mainframe, variabletext = result ).grid(colum=3,row=3, sticky=(N,W,E,S))
#Add a list box and scrollbar
#populate list box