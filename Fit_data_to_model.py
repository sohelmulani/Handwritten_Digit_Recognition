#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:41:40 2019

@author: sohel
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import image
from keras.datasets import mnist
from keras.utils import np_utils

from Build_cnn import build_cnn
from RGB_to_Gray import rgb2gray
from Save_Model import save_model
model=build_cnn()
gray=rgb2gray()
obj=save_model()

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_train=x_train.astype('float32')
x_train/=255

x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_test=x_test.astype('float32')
x_test/=255

print(x_train.shape)
print(x_test.shape)

from  keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test= np_utils.to_categorical(y_test)
print(y_train[0])

classifier=model.create_model()    

classifier=model.compile_model(classifier)

classifier.summary()

classifier.fit(x_train,y_train,
               validation_data=(x_test,y_test),
               epochs=5,
               batch_size=200,
               verbose=2)

score=classifier.evaluate(x_test,y_test)
print("accuracy={} %".format(score[1]*100))

obj.save(classifier)

loaded_classifier=obj.use_model()
