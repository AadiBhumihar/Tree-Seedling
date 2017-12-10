#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:31:34 2017

@author: bhumihar
"""
#==============================================================================
    #### importing Library ####

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import np_utils

#==============================================================================

        # dimensions of our images.
img_width, img_height = 350, 350

datagen = ImageDataGenerator(rescale=1. / 255)

top_model_weights_path = 'cnn_bnf_model1.h5'

train_data_dir = './train'
nb_train_samples = 4750
epochs = 50
batch_size = 19

#==============================================================================

    #### build the VGG16 network ######
    
def bottleneck_feature() :
    model = applications.VGG16(include_top=False, weights='imagenet')

               ##### Generating Image Data to Extract Feature and to tune it with VGC16 Model
    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False) 
    
    #### Extracting Features   #####
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size,verbose=1)   
    
    file = open('bottleneck_features_train.npy', 'wb')      #### Saving Features To file
    np.savez(file, bottleneck_features_train)

#==============================================================================
    
def train_top_model() :
    
    #Loading Feature from File
    npyfile = np.load("bottleneck_features_train.npy")
    train_data = npyfile['arr_0'] 
    
    
    ## Size of each Dataset
    bg_size ,ch_size,cl_size,cc_size,cw_size,fh_size ,ls_size,m_size,sm_size,sp_size,sf_size,sb_size = 263,390,287,611,221,475,654,221,516,231,
    496,365
    
    ### Creating Label for each class according to their size and label them ##### 
    train_labels =np.array(([0] * int(bg_size)) + ([1] * int(ch_size)) + ([2] * int(cl_size))
                          + ([3] * int(cc_size))+ ([4] * int(cw_size))+ ([5] * int(fh_size))
                          +([6] * int(ls_size)) + ([7] * int(m_size)) + ([8] * int(sm_size))
                          +([9] * int(sp_size)) + ([10] * int(sf_size)) + ([11] * int(sb_size)))
      
    ### One-Hot Encoding ##### 
    train_labels = np_utils.to_categorical(train_labels, 12)
    
    
       #####  Sequential Model of Neural Network #####
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
     #### Loss Function and Optimizer ####
    model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])
          #### Fitting Our Model ####
    model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size)
    
    #### Saving our model weight into file for future use #####
    model.save_weights(top_model_weights_path)

#==============================================================================
    
    #### Calling bottle_neck() method #####
bottleneck_feature() 
    #### Calling train_neck() method #####
train_top_model() 