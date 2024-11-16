import streamlit as st
import cv2
from PIL import Image
import os
import numpy as np
import cv2
from keras import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.layers import Flatten
from keras._tf_keras.keras.layers import Conv2D
from keras._tf_keras.keras.layers import MaxPooling2D
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.optimizers import adam_v2 as Adam
import threading

# from keras._tf_keras.keras.layers import MaxPooling2D
# from keras.preprocessing.image import ImageDataGenerator

start = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
text_placeholder = st.empty()
while start:
    flag, frame = camera.read()
    if not flag:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    global last_frame1    #emoji dictionary is created with images for every emotion present ion dataset                               
    last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    global cap1
    global frame_number
    show_text=[0]

    em_dict = {
        0: "   Angry   ",
        1: "Disgusted",
        2: "  Fearful  ",
        3: "   Happy   ",
        4: "  Neutral  ",
        5: "    Sad    ",
        6: "Surprised"
        }

    cur_path = os.path.dirname(os.path.abspath(__file__))
    # emoji_dist={
    #     0:"emojis/angry.png",
    #     1:"emojis/disgusted.png",
    #     2:"emojis/fearful.png",
    #     3:"emojis/happy.png",
    #     4:"emojis/neutral.png",
    #     5:"emojis/sad.png",
    #     6:"emojis/surprised.png"
    #     }
    emoji_dist={
        0:"😡",
        1:"🤢",
        2:"😨",
        3:"😄",
        4:"😃",
        5:"😔",
        6:"😯"
        }

    emotion_model = Sequential()#to extract the features in model
    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))
    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(7, activation='softmax'))
    emotion_model.load_weights('model.h5')

    frame = cv2.resize(frame,(600,500))#to resize the image frame
    bound_box = cv2.CascadeClassifier('haar_cascade_frontal_face.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#to color the frame
    num_faces = bound_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces: #for n different faces of a video
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_frame = gray_frame[y:y + h, x:x + w]
        crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)#crop the image and save only emotion contating face
        prediction = emotion_model.predict(crop_img)#predict the emotion from the cropped image
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, em_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex #store the emotion found in image from emotion dictionary
        # print(show_text[0])
        # print(emoji_dist[show_text[0]])
        
    # frame2=cv2.imread(emoji_dist[show_text[0]])#to store the emoji with respect to the emotion
    # print(frame2)
    # pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    # img2=Image.fromarray(frame2)
    # st_img = st.image(img2)
    
    font_size = "<h1 style='font-size: 148px;'>" + emoji_dist[show_text[0]] + "</h1>"
    text_placeholder.write(font_size, unsafe_allow_html=True)

    