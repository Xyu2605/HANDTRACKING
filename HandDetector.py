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

        self.landmark_color = self.hex_to_rgb("#40E0D0")
        self.text_color = self.hex_to_rgb("#00C0FF")
        
    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)
    
    def find_hands(self, image):
        image = cv2.cvtColor(cv2.flip(image, 1),cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id , landmark_pos in enumerate(hand_landmarks.landmark):
                    height, width, chanels = image.shape
                    cx, cy = int(landmark_pos.x * width), int(landmark_pos.y * height)
                    if id == 8 :
                        cv2.circle(image, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = self.mp_drawing.DrawingSpec(color = self.landmark_color, thickness = 3)
                )
        return image