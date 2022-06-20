from tensorflow import keras
import tensorflow as tf
import cv2
import time
import os
import numpy as np
from PIL import Image

FRAME_SIZE = (256, 256)

classes = ["me", "nothing", "victory", "yes"]

cap = cv2.VideoCapture(1) 
if cap is None or not cap.isOpened():
    cap = cv2.VideoCapture(0)

model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models\\"))


v = 0

while v < 5:
        ret, frame = cap.read()
        frame = cv2.resize(frame, FRAME_SIZE)
        cv2.imshow("Acquisition", frame)
        cv2.waitKey(1)
        frame = Image.fromarray(frame)
        frame = np.asarray(frame)
        frame = tf.expand_dims(frame, 0)
        prediction = model.predict(frame)
        pred_class = classes[np.argmax(prediction[0])]
        pred_prec = 100 * np.max(prediction[0])
        if pred_class == "victory":
            v += 1
        else:
            v = 0

        print(pred_class, pred_prec, v)

os.startfile("C:/Users/aless/AppData/Local/Discord/app-1.0.9005/Discord.exe")