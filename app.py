from flask import Flask, request, render_template
import cv2

import os
from datetime import date
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)

nimages = 10

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
   os.makedirs('Attendance')
if not os.path.isdir('static'):
   os.makedirs('static')
if not os.path.isdir('static/faces'):
   os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
   with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
      f.write('Name,Roll,Time')


def totalreg():
   return len(os.listdir('static/faces'))


def extract_faces(img):
   try:
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      face_points = face_detector.detectMultiScale(
          gray, 1.2, 5, minSize=(20, 20))
      return face_points
   except:
      return []


def identify_face(facearray):
   model = joblib.load('static/face_recognition_model.pkl')
   return model.predict(facearray)

def train_model():
   faces = []
   labels = []
   userlist = os.listdir('static/faces')
   for user in userlist:
      for imgname in os.listdir(f'static/faces/{user}'):
         img = cv2.imread(f'static/faces/{user}/{imgname}')
         resized_face = cv2.resize(img, (50, 50))
         faces.append(resized_face.ravel())
         labels.append(user)
   faces = np.array(faces)
   knn = KNeighborsClassifier(n_neighbors=5)
   knn.fit(faces, labels)
   joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
   df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
   names = df['Name']
   rolls = df['Roll']
   times = df['Time']
   l = len(df)
   return names, rolls, times, l


def add_attendance(name):
   username = name.split('_')[0]
   userid = name.split('_')[1]
   current_time = datetime.now().strftime("%H:%M:%S")

   df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
   if int(userid) not in list(df['Roll']):
      with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
         f.write(f'\n{username},{userid},{current_time}')


def getallusers():
   userlist = os.listdir('static/faces')
   names = []
   rolls = []
   l = len(userlist)

   for i in userlist:
      name, roll = i.split('_')
      names.append(name)
      rolls.append(roll)

   return userlist, names, rolls, l


def deletefolder(duser):
   pics = os.listdir(duser)
   for i in pics:
      os.remove(duser+'/'+i)
   os.rmdir(duser)
