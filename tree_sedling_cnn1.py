#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:59:12 2017

@author: bhumihar
"""

#==============================================================================
tree_seedling = ['Black-grass',
'Charlock',
'Cleavers',
'Common Chickweed',
'Common wheat',
'Fat Hen',
'Loose Silky-bent',
'Maize',
'Scentless Mayweed',
'Shepherds Purse',
'Small-flowered Cranesbill',
'Sugar beet']

#==============================================================================
    #### importing Library ####

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.image as mpimg
import os
import numpy as np
from skimage .transform import resize

#==============================================================================
   
train_data_dir = './train'
nb_train_samples = 4750

epochs = 10
batch_size = 95

    ### Rescaled dimensions of images ####.
img_width, img_height = 350, 350

#==============================================================================

          #### Setting Input Shape depending on Image channel type
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
      
#==============================================================================


    ### Sequential Model 
model = Sequential()

     ##### 1st convolutional Layer ######
model.add(Conv2D(32, (3, 3), input_shape=input_shape))    ## Filter  
model.add(Activation('relu'))                             ## Activation Function  
model.add(MaxPooling2D(pool_size=(2, 2)))                 ## MaxPooling  


     ##### 1st convolutional Layer ######
model.add(Conv2D(32, (3, 3)))                             ## Filter 
model.add(Activation('relu'))                             ## Activation Function  
model.add(MaxPooling2D(pool_size=(2, 2)))                 ## MaxPooling 

     ##### 1st convolutional Layer ######
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

    #Reshaping and Connecting to Fully Connected Neural Network ####
    
    #### 1st Hidden Layer  #### 
    
model.add(Flatten())                                ### Reshaping Our Parameter
model.add(Dense(64))                               ## Neuron
model.add(Activation('relu'))

    #### OutPut Layer  ####
model.add(Dropout(0.5))
model.add(Dense(12))                                ## Neuron
model.add(Activation('softmax'))

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


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

#==============================================================================
         
      #### Fitting Our Model ####
for i in range(3) :
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)

   #### Saving our model weight into file for future use #####
model.save('cnn_model1.h5')
#==============================================================================

      #### Function for Predicting For Test Data ######
      
def predict_file(path):
    data = []
    classes = []
    root = path    
    i =0
    for path, subdirs, files in os.walk(root):
        for name in files:
            i +=1
            img = os.path.join(path, name);
            im1 = mpimg.imread(img)
            im1 = resize(im1,(img_width,img_height))
            im1 = np.reshape(im1,[1,img_width,img_height,3])
            classes = model.predict_classes(im1)
            tree_inx = classes[0]%12 ;
            t = [name,tree_seedling[tree_inx]]
            data.append(t)
            
        with open('pedict.csv','w') as out:
            out.write('file,species'+'\n')
            for row in data:
                out.write(row[0]+','+row[1]) + '\n' ;
                
                
#==============================================================================
         #### CAlling Predict Function ###### 
predict_file("./test")
