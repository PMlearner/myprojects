import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
##load json file
json_file = open('model1/emotion_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
##load weights of the model
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model1/emotion_model.h5")
frame=cv2.imread("../demos/anger.jpg")
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on image
num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the image and Preprocess it
for (x, y, w, h) in num_faces:
    cv2.rectangle(frame, (x, y-10), (x+w, y+h+5), (0, 255, 0), 4)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imshow('Emotion Detection', frame)
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
