from tensorflow import keras
import tensorflow as tf
import cv2
import time
import os
import numpy as np
from PIL import Image
import glob

FRAME_SIZE = (256, 256)

list_of_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfportrait\\" + "*.jpg"))
if list_of_files:
    latest_file = max(list_of_files, key=os.path.getctime)
    last = int(latest_file.split("\\selfportrait\\")[1].split('.')[0])
else:
    last = -1

last += 1

classes = ["me", "nothing", "victory", "yes"]

cap = cv2.VideoCapture(1) 
if cap is None or not cap.isOpened():
    cap = cv2.VideoCapture(0)

model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models\\"))


v = 0

while 1:
        ret, frame = cap.read()
        img = cv2.resize(frame, FRAME_SIZE)
        cv2.imshow("Acquisition", img)
        cv2.waitKey(1)
        img = Image.fromarray(img)
        img = np.asarray(img)
        img = tf.expand_dims(img, 0)
        prediction = model.predict(img)
        pred_class = classes[np.argmax(prediction[0])]
        pred_prec = 100 * np.max(prediction[0])
        if pred_class == "victory":
            v += 1
        else:
            v = 0

        print(pred_class, pred_prec, v)

        if v >= 5:
            time.sleep(3)
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfportrait\\" + str(last) + ".jpg")
            cv2.imwrite(filename=path, img=frame)
            last += 1


