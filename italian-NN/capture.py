import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

FRAMES = 10
INTERVAL = 0.5

cap = cv2.VideoCapture(1)

for _ in range(FRAMES):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(frame)
    plt.show(block=False)
    plt.pause(INTERVAL)
    plt.close()
    


