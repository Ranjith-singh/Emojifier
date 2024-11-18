# Import the required libraries

import numpy as np
import cv2
# from keras import Sequential
# from keras.layers import Dense
from keras import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.layers import Flatten
from keras._tf_keras.keras.layers import Conv2D
from keras._tf_keras.keras.layers import MaxPooling2D
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.optimizers import adam_v2
# from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.all_utils import plot_model

# Initialize the training and validation generators:

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
#training generator for CNN
train_generator = train_datagen.flow_from_directory(
       train_dir,
       target_size=(48,48),
       batch_size=64,
       color_mode="grayscale",
       class_mode='categorical')
#validation generator for CNN
validation_generator = val_datagen.flow_from_directory(
       val_dir,
       target_size=(48,48),
       batch_size=64,
       color_mode="grayscale",
       class_mode='categorical')


# # To display the train data

# for i in os.listdir("train/"):
#     print(str(len(os.listdir("train/"+i))) +" "+ i +" images")

# Build the convolution network architecture:

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))#output=(48-3+0)/1+1=46
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))#output=(46-3+0)/1+1=44
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#output=devided input by 2 it means 22,22,64
emotion_model.add(Dropout(0.25))#reduce 25% module at a time of output
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',input_shape=(48,48,1)))#(22-3+0)/1+1=20
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#10
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))#(10-3+0)/1+1=8
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))#output=4
emotion_model.add(Dropout(0.25))#nothing change
emotion_model.add(Flatten())#here we get multidimension output and pass as linear to the dense so that 4*4*128=2048
emotion_model.add(Dense(1024, activation='relu'))#hddien of 1024 neurons of input 
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))#hddien of 7 neurons of input
# plot_model(emotion_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)#save model leyer as model_plot.png
emotion_model.summary()

# Compile and train the model

emotion_model.compile(loss='categorical_crossentropy',optimizer=adam_v2.Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
# emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam)
emotion_model_info = emotion_model.fit( #to fetch the model info from validation generator
       train_generator,
       steps_per_epoch=28709 // 64,
       epochs=5,
       validation_data=validation_generator,
       validation_steps=7178 // 64)

# Save the model weights:

emotion_model.save_weights('model.h5')#to save the model