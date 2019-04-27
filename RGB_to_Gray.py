#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:50:05 2019

@author: sohel
"""

class rgb2gray(object):
    
    def _init_(self):
        pass
    
    def convert_img(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    
    def _del_(self):
        pass
    