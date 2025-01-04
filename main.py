import cv2
import mediapipe as mp
import time
import os
import numpy as np
from HandDetector import HandDetector


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)

def video_time(start_time, image, text_color):
    elapsed_time = int(time.time() - start_time)
    seconds = elapsed_time % 60
    minutes = (elapsed_time // 60) % 60
    hours = elapsed_time // 3600
    cv2.putText(image,
        f"{hours:02}:{minutes:02}:{seconds:02}", (10,700),
        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, text_color, 3, cv2.LINE_AA)
    
def folder_path():
    folderPath = ("Header")
    myList = os.listdir(folderPath)
    overlayList = []

    for imPath in myList:
        image = cv2.imread(f"{folderPath}/{imPath}")
        overlayList.append(image)
    header = overlayList[0]
    return header



def main():
    drawcolor = hex_to_rgb("#EAA23F")
    header = folder_path()
    capture = cv2.VideoCapture(0)
    capture.set(3,1280)
    capture.set(4, 720)
    if not capture.isOpened():
        print("Error: Unable to access the camera.")
        return
    detector = HandDetector(detectionCon=0.85)
    start_time = time.time()
    xp , yp = 0, 0
    brushThickness = 15
    eraserThickness = 100
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    while True:
        ret, image = capture.read()
        if not ret:
            print("Failed to grab frame.")
            break
        image = detector.find_hands(image)
        LmList = detector.find_pos_landmarks(image, draw=False)
        video_time(start_time, image, detector.text_color)
        image[0:125, 0:1280] = header
        if len(LmList) != 0:
            x1, y1 = LmList[8][1:]
            x2, y2 = LmList[12][1:]
            fingers = detector.fingers_up()
            if fingers[1] and fingers[2]:
                xp , yp = 0, 0
                if y1 < 125:
                    if 299 < x1 < 518:
                        drawcolor = hex_to_rgb("#00C0FF")
                    elif 553 < x1 < 772:
                        drawcolor = hex_to_rgb("#08A045")
                    elif 807 < x1 < 1026:
                        drawcolor = hex_to_rgb("#FE3939")
                    elif 1061 < x1 < 1280:
                        drawcolor = (0, 0, 0)

                cv2.rectangle(image, (x1, y1 - 25), (x2, y2 + 25), drawcolor, cv2.FILLED)
                print("Selection Mode")

            if fingers[1] and fingers[2] ==  False:
                cv2.circle(image, (x1, y1), 15, drawcolor, cv2.FILLED)
                print("Drawing mode")
                if xp == 0 and yp == 0:
                    xp , yp = x1, y1
                elif drawcolor == (0, 0, 0) :
                    cv2.line(image, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                else:
                    cv2.line(image, (xp, yp), (x1, y1), drawcolor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushThickness)

                xp , yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, imgInv)
        image =cv2.bitwise_or(image, imgCanvas)
        cv2.imshow('Lady Killer', image)
        if cv2.waitKey(20) & 0xFF == ord('x'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


