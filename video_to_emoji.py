# Import the Libraries

import tkinter as tk
from tkinter import *
import cv2
from PIL import Image
from PIL import ImageTk
import os
import numpy as np
import cv2
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
# from keras.optimizers import adam

from tensorflow.python.keras.optimizers import adam_v2 as Adam
import threading

# Model Creation

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
cv2.ocl.setUseOpenCL(False)

# Mapping of facial emotion with Avtar

#emotion dictionary contains the emotions present in the dataset
em_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}

cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dist={0:cur_path+"/emojis/angry.png", 1:cur_path+"/emojis/disgusted.png", 2:cur_path+"/emojis/fearful.png",3:cur_path+"/emojis/happy.png",4:cur_path+"/emojis/neutral.png",5:cur_path+"/emojis/sad.png",6:cur_path+"/emojis/surprised.png"}

# emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgust.png",2:"./emojis/fear.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surprise.png"}


global last_frame1    #emoji dictionary is created with images for every emotion present ion dataset                               
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
global frame_number
show_text=[0]

def show_subject():    #to open the camera and to record video
   # cap1 = cv2.VideoCapture(r'C:\Users\Aryaman\Pictures\Camera Roll\surprised.mp4')      #it starts capturing         
   cap1 = cv2.VideoCapture(0)      #it starts capturing                      
   if not cap1.isOpened():  #if camera is not open                          
      print("Can't open the camera")
   global frame_number
   length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
   frame_number +=1
   # if frame_number >= length:
   #    exit()
   # cap1.set(1, frame_number)
   flag1, frame1 = cap1.read()
   frame1 = cv2.resize(frame1,(600,500))#to resize the image frame
   bound_box = cv2.CascadeClassifier('haar_cascade_frontal_face.xml')
   gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)#to color the frame
   num_faces = bound_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
   for (x, y, w, h) in num_faces: #for n different faces of a video
         cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
         roi_frame = gray_frame[y:y + h, x:x + w]
         crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)#crop the image and save only emotion contating face
         prediction = emotion_model.predict(crop_img)#predict the emotion from the cropped image
         maxindex = int(np.argmax(prediction))
         cv2.putText(frame1, em_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
         show_text[0]=maxindex #store the emotion found in image from emotion dictionary
   if flag1 is None:
      print ("Major error!")
   elif flag1:
      global last_frame1
      last_frame1 = frame1.copy()
      pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB) #to store the image   
      img = Image.fromarray(pic)
      imgtk = ImageTk.PhotoImage(image=img)
      lmain.imgtk = imgtk
      lmain.configure(image=imgtk)
      root.update()
      lmain.after(10, show_subject)
   if cv2.waitKey(1) & 0xFF == ord('q'):
         exit()

def show_avatar():
   frame2=cv2.imread(emoji_dist[show_text[0]])#to store the emoji with respect to the emotion
   pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
   img2=Image.fromarray(frame2)
   imgtk2=ImageTk.PhotoImage(image=img2)
   lmain2.imgtk2=imgtk2
   lmain3.configure(text=em_dict[show_text[0]],font=('arial',45,'bold'))#to configure image and text
   lmain2.configure(image=imgtk2)
   root.update()
   lmain2.after(10, show_avatar)


if __name__ == '__main__':
   frame_number = 0
   root=tk.Tk()
   lmain = tk.Label(master=root,padx=50,bd=10)
   lmain2 = tk.Label(master=root,bd=10)
   lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
   lmain.pack(side=LEFT)
   lmain.place(x=50,y=250)
   lmain3.pack()
   lmain3.place(x=960,y=250)
   lmain2.pack(side=RIGHT)
   lmain2.place(x=900,y=350)

   root.title("Photo To Emoji")           
   root.geometry("1400x900+100+10")
   root['bg']='black'
   exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
   threading.Thread(target=show_subject).start()
   threading.Thread(target=show_avatar).start()
   root.mainloop()