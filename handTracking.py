import cv2
import mediapipe as mp
import time
from HandDetector import HandDetector

def video_time(start_time, image, text_color):
    elapsed_time = int(time.time() - start_time)
    seconds = elapsed_time % 60
    minutes = (elapsed_time // 60) % 60
    hours = elapsed_time // 3600
    cv2.putText(image,
        f"{hours:02}:{minutes:02}:{seconds:02}", (10,35),
        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, text_color, 3, cv2.LINE_AA)
    
def resize(image, scale = 1.75):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimentions = (width,height)
    return cv2.resize(image, dimentions, interpolation = cv2.INTER_AREA)

def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Unable to access the camera.")
        return
    detector = HandDetector()
    start_time = time.time()
    while True:
        ret, image = capture.read()
        if not ret:
            print("Failed to grab frame.")
            break
        image = resize(image)
        image = detector.find_hands(image)
        video_time(start_time, image, detector.text_color)
        cv2.imshow('Lady Killer', image)
        if cv2.waitKey(20) & 0xFF == ord('x'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


