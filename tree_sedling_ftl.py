#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:53:40 2017

@author: bhumihar
"""
#==============================================================================
    #### importing Library ####

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K

#==============================================================================

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = './train'  


    ### Rescaled dimensions of images ####.
img_width, img_height = 350, 350

nb_train_samples = 4750
nb_epoch = 15
batch_size = 19

#==============================================================================

          #### Setting Input Shape depending on Image channel type
if K.image_data_format() == 'channels_first':
    inp_shape = (3, img_width, img_height)
else:
    inp_shape = (img_width, img_height, 3)

     #### Input Shape ####
inp_tensor = Input(shape=inp_shape)

#==============================================================================

    #### build the VGG16 network  #####
model = Sequential()
model.add(applications.VGG16(weights='imagenet', include_top=False,input_tensor=inp_tensor))
print('Model loaded.')

    #Reshaping and Connecting to Fully Connected Neural Network ####
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
    
    #### 1st Hidden Layer  #### 

top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))

        #### OutPut Layer  ####
top_model.add(Dense(12, activation='softmax'))

    #### Loading Weight of BottleNeckModel Parameter ####
top_model.load_weights(top_model_weights_path)

    #### Adding our neural network model to our VGC16 Model  #####
model.add(top_model)

    #### Setting first 25 layer of VGC16 Model not ot train #####
for layer in model.layers[:25]:
    layer.trainable = False

    #### Loss Function and Optimizer ####
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

#==============================================================================

     ##### Image Augementation ####
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

#==============================================================================

      #### Fitting Our Model ####
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=nb_epoch)

#==============================================================================

   #### Saving our model weight into file for future use #####
model.save('cnn_finetn_model1.h5')

#==============================================================================
