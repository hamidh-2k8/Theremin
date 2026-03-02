import cv2
import numpy as np

class Landmarks():
    
    WRIST = 0
    
    THUMB_BASE = 1
    THUMB_KNUCKLE_1 = 2
    THUMB_KNUCKLE_2 = 3
    THUMB_TIP = 4
    
    INDEX_BASE = 5
    INDEX_KNUCKLE_1 = 6
    INDEX_KNUCKLE_2 = 7
    INDEX_TIP = 8
    
    MIDDLE_BASE = 9
    MIDDLE_KNUCKLE_1 = 10
    MIDDLE_KNUCKLE_2 = 11
    MIDDLE_TIP = 12
    
    RING_BASE = 13
    RING_KNUCKLE_1 = 14
    RING_KNUCKLE_2 = 15
    RING_TIP = 16
    
    PINKY_BASE = 17
    PINKY_KNUCKLE_1 = 18
    PINKY_KNUCKLE_2 = 19
    PINKY_TIP = 20
    
class Result():
    
    def __init__(self):
        self.processed = {
            "lift": 0, # Moving closer/further from camera
            "pitch": 0, # Tilting hand up/down
            "yaw": 0, # Twisting hand along palm normal
            "roll": 0, # Rolling hand along wrist axis
            "indexCurl": 0, # Curling individual fingers, whole hand
            "middleCurl": 0,
            "ringCurl": 0,
            "pinkyCurl": 0,
            "fullCurl": 0,
            "spread": 0, # Spreading fingers apart
            "pinch": 0 # Bringing index to thumb
        }
        
        self.raw = None
        self.image = None #TODO: Replace with placeholder
        
    def update(self, landmarks, image):

        self.raw = landmarks
        self.image = image

        palmAspectX = distance(landmarks[Landmarks.INDEX_BASE], landmarks[Landmarks.PINKY_BASE])
        palmAspectY = distance(
            landmarks[Landmarks.WRIST],
            np.mean([landmarks[Landmarks.INDEX_BASE].x, landmarks[Landmarks.INDEX_BASE].y], [landmarks[Landmarks.PINKY_BASE].x, landmarks[Landmarks.PINKY_BASE].y])
        )

        self.processed["lift"] = np.max([palmAspectX, palmAspectY])
        
        self.processed["pitch"] = np.clip((4 / np.pi) * np.arctan(palmAspectX / palmAspectY) + 1, 0, 1)
        
        self.processed["yaw"] = np.clip((4 / np.pi) * np.arctan2(palmAspectY, palmAspectX) + 1, 0, 1)
        
        self.processed["roll"] = (1 / np.pi) * np.arctan2(
            landmarks[Landmarks.MIDDLE_TIP].y - landmarks[Landmarks.WRIST].y,
            landmarks[Landmarks.MIDDLE_TIP].x - landmarks[Landmarks.WRIST].x)

        self.processed["indexCurl"] = distance(landmarks[Landmarks.INDEX_TIP], landmarks[Landmarks.INDEX_BASE])

        self.processed["middleCurl"] = distance(landmarks[Landmarks.MIDDLE_TIP], landmarks[Landmarks.MIDDLE_BASE])
        self.processed["ringCurl"] = distance(landmarks[Landmarks.RING_TIP], landmarks[Landmarks.RING_BASE])
        self.processed["pinkyCurl"] = distance(landmarks[Landmarks.PINKY_TIP], landmarks[Landmarks.PINKY_BASE])
        self.processed["fullCurl"] = np.mean([
            self.processed["indexCurl"],
            self.processed["middleCurl"],
            self.processed["ringCurl"],
            self.processed["pinkyCurl"]
        ])
        
        self.processed["spread"] = np.mean([
            distance(landmarks[Landmarks.INDEX_BASE], landmarks[Landmarks.INDEX_KNUCKLE_1]),
            distance(landmarks[Landmarks.MIDDLE_BASE], landmarks[Landmarks.MIDDLE_KNUCKLE_1]),
            distance(landmarks[Landmarks.RING_BASE], landmarks[Landmarks.RING_KNUCKLE_1]),
            distance(landmarks[Landmarks.PINKY_BASE], landmarks[Landmarks.PINKY_KNUCKLE_1])
        ])

        self.processed["pinch"] = 1 - (distance(landmarks[Landmarks.THUMB_TIP], landmarks[Landmarks.INDEX_TIP]) / self.processed["lift"])

    def getAnnotatedImage(image, sidelen, landmarks):
        for i in range(len(landmarks)):
            
            color = (255, 128, 0) # Orange

            if i in [Landmarks.THUMB_BASE, Landmarks.INDEX_BASE, Landmarks.MIDDLE_BASE, Landmarks.RING_BASE, Landmarks.PINKY_BASE]:
                color = (255, 0, 0) # Red
            if i in [Landmarks.THUMB_TIP, Landmarks.INDEX_TIP, Landmarks.MIDDLE_TIP, Landmarks.RING_TIP, Landmarks.PINKY_TIP]:
                color = (0, 255, 0) # Green
            if i == Landmarks.WRIST:
                color = (0, 255, 255) # Cyan

            cv2.circle(image, (int(landmarks[i].x * sidelen), int(landmarks[i].y * sidelen)), 5, color, -1)
        return image

def distance(p1, p2):
    return np.linalg.norm(np.array((p1.x, p1.y)) - np.array((p2.x, p2.y)))