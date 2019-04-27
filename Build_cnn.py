#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:32:53 2019

@author: sohel
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

class build_cnn(object):
    
    def _init_(self):
        pass
    
    def create_model(self):
        classifier=Sequential()
        
        classifier.add(Convolution2D(32,5,5, input_shape=(28,28,1),activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2,2)))
        
        classifier.add(Convolution2D(32,3,3, activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2,2)))
        
        classifier.add(Dropout(0.2))
        
        classifier.add(Flatten())
        
        classifier.add(Dense(output_dim=128,activation='relu'))
        classifier.add(Dense(output_dim=64,activation='relu'))
        
        classifier.add(Dense(output_dim=10,activation='softmax'))
        
        return classifier
    
    def compile_model(self,classifier):
        classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

        return classifier
    
    def _del_(self):
        pass