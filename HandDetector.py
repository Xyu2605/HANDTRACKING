import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.text_color = self.hex_to_rgb("#00C0FF")
        self.results = None
        self.tipIDs = [4, 8, 12, 16, 20]
        self.Lmlist = None
        
    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)
    
    def find_hands(self, image, draw=True):
        image = cv2.cvtColor(cv2.flip(image, 1),cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = self.mp_drawing.DrawingSpec(color = self.text_color , thickness = 3)
                )
        return image
    
    def find_pos_landmarks(self, image, handNo=0, draw=True):
        self.Lmlist = []
        if self.results and self.results.multi_hand_landmarks:
            try:
                my_hand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(my_hand.landmark):
                    height, width, chanels = image.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    self.Lmlist.append([id, cx, cy])
            except IndexError:
                print(f"Hand {handNo} not found.")
        return self.Lmlist
    
    def fingers_up(self):
        fingers = []
        
        if self.Lmlist[self.tipIDs[0]][1] < self.Lmlist[self.tipIDs[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.Lmlist[self.tipIDs[id]][2] < self.Lmlist[self.tipIDs[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
        
    



