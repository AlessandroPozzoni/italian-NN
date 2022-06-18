import string
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os

FRAMES = 10
INTERVAL = 250

SUBJECT = "me"

FRAME_SIZE = (256, 256)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset\\" + SUBJECT)

print(path)

cap = cv2.VideoCapture(1) 
if cap is None or not cap.isOpened():
    cap = cv2.VideoCapture(0)

for i in range(FRAMES):
    ret, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, FRAME_SIZE)
    imgplot = plt.imshow(frame)
    framePath = os.path.join(path, str(i) + ".jpg")
    cv2.imwrite(filename=framePath, img=frame)
    cv2.imshow("culo", frame)
    cv2.waitKey(INTERVAL)
    


