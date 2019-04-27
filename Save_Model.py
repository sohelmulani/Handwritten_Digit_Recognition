#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:28:37 2019

@author: sohel
"""
#importing library required to save model
from keras.models import model_from_json

class save_model(object):
    
    def _init_(self):
        pass
    
    def save(self,classifier):
        model_json = classifier.to_json()
        
        #writing a json file to write a model in it
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
         
        #saving weights of model    
        classifier.save_weights("model.h5")
        print("Saved model to disk")
    
    def use_model(self):
        
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights("model.h5")
        
        return loaded_model
    
    def _del_(self):
        pass
    
        