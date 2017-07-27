# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:44:19 2017

@author: Nigel Wee

"""

class DataHolder(object):
    def __init__(self, mfcc, mfcc_d, mfcc_d2):
        self.data = mfcc, mfcc_d, mfcc_d2
    
          