from __future__ import division
from django.shortcuts import render
from .forms import UploadFile
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from keras.models import model_from_json
import os
import numpy as np
import cv2

# Create your views here.


def handle_upload_file(f):
    json_file = open('Emt/fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('Emt/fer.h5')
    # print("Loaded model from disk")

    # setting image resizing parameters
    WIDTH = 48
    HEIGHT = 48
    x = None
    y = None
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # loading image
    full_size_image = cv2.imread("media/"+f.name)
    # print("Image Loaded")
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('Emt/haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)

    # detecting faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # predicting the emotion
        yhat = loaded_model.predict(cropped_img)
        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                    1, cv2.LINE_AA)
        # print("Emotion: " + labels[int(np.argmax(yhat))])

        return labels[int(np.argmax(yhat))]


def detect_image(request):
    if request.method == 'POST':
        upload = request.FILES['img']
        fs = FileSystemStorage()
        fs.save(upload.name, upload)
        e = handle_upload_file(upload)
        return render(request, 'predict.html', {'pred':e})
    else:
        return render(request, 'home.html')
