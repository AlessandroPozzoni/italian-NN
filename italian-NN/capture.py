import string
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import glob

INTERVAL = 0.250

FRAME_SIZE = (256, 256)

subjects = ["me", "nothing", "victory", "yes"]

reply = "N"

while reply == "N":
    FRAMES = 0
    SUBJECT = " "
    while not SUBJECT in subjects:
        SUBJECT = input("Data collection: which subject are you collecting data for?\n")

    list_of_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset\\" + SUBJECT + "\\*.jpg"))
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        last = int(latest_file.split('\\' + SUBJECT + "\\")[1].split('-')[0])
    else:
        last = -1

    last += 1


    while FRAMES == 0:
        FRAMES = int(input("How many frames would you like to collect?\n"))

    print("Starting acquisition...")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset\\" + SUBJECT)

    cap = cv2.VideoCapture(1) 
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    frames = {}


    for i in range(FRAMES):
        ret, frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, FRAME_SIZE)
        frames[i] = frame;
        print("Capturing...")
        time.sleep(INTERVAL)
        #cv2.imwrite(filename=framePath, img=frame)
        #cv2.imshow("culo", frame)
        #cv2.waitKey(INTERVAL)

    print("These are the frames that have been captured!\n")
    time.sleep(3)

    for key in frames.keys():
        cv2.imshow("culo", frames[key])
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    reply = str.capitalize(input("Do you want to keep these frames? (Y/N)"))

for i in range(FRAMES):
    framePath = os.path.join(path, str(last) + '-' + str(i) + ".jpg")
    cv2.imwrite(filename=framePath, img=frames[i])
    


print("Completed!")
    


