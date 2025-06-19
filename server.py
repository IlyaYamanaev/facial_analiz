from app import *


@app.route('/')
def home():
   names, rolls, times, l = extract_attendance()
   return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/listusers')
def listusers():
   userlist, names, rolls, l = getallusers()
   return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/deleteuser', methods=['GET'])
def deleteuser():
   duser = request.args.get('user')
   deletefolder('static/faces/'+duser)
   if os.listdir('static/faces/') == []:
      os.remove('static/face_recognition_model.pkl')
   try:
      train_model()
   except:
      pass
   userlist, names, rolls, l = getallusers()
   return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
   names, rolls, times, l = extract_attendance()
   if 'face_recognition_model.pkl' not in os.listdir('static'):
      return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

   ret = True
   cap = cv2.VideoCapture(0)
   while ret:
      ret, frame = cap.read()
      if len(extract_faces(frame)) > 0:
         (x, y, w, h) = extract_faces(frame)[0]
         cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
         cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
         face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
         identified_person = identify_face(face.reshape(1, -1))[0]
         add_attendance(identified_person)
         cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
      cv2.imshow('Attendance', frame)
      if cv2.waitKey(1) == 27:
         break

   cap.release()
   cv2.destroyAllWindows()
   names, rolls, times, l = extract_attendance()
   return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['GET', 'POST'])
def add():
   newusername = request.form['newusername']
   newuserid = request.form['newuserid']
   userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)

   if not os.path.isdir(userimagefolder):
      os.makedirs(userimagefolder)

   i, j = 0, 0
   cap = cv2.VideoCapture(0)
   while 1:
      _, frame = cap.read()
      faces = extract_faces(frame)
      for (x, y, w, h) in faces:
         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
         cv2.putText(frame, f'Images Captured: {i}/{nimages}', (30, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
         if j % 5 == 0:
            name = newusername+'_'+str(i)+'.jpg'
            cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
            i += 1
         j += 1
      if j == nimages*5:
         break
      cv2.imshow('Adding new User', frame)
      if cv2.waitKey(1) == 27:
         break

   cap.release()
   cv2.destroyAllWindows()
   print('Training Model')
   train_model()
   names, rolls, times, l = extract_attendance()
   return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


if __name__ == '__main__':
   app.run(debug=True)
