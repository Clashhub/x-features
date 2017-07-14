# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:49:10 2017

@author: Nigel Wee
"""
from functions import getFeatures
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import numpy as np

#Instantiate main top level window
root = Tk()

root.title("Anti-spoofing") 

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

ttk.Button(mainframe, text="Extract features", command=getFeatures).grid(column=3, row=3, sticky=W)