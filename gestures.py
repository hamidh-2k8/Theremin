import threading
import time
import cv2
import numpy as np

placeholder = cv2.cvtColor(cv2.imread("placeholder.jpg"), cv2.COLOR_BGR2RGB)

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
            "lift": 0.0, # Moving closer/further from camera
            "pitch": 0.0, # Tilting hand up/down
            "yaw": 0.0, # Twisting hand along palm normal
            "roll": 0.0, # Rolling hand along wrist axis
            "indexCurl": 0.0, # Curling individual fingers, whole hand
            "middleCurl": 0.0,
            "ringCurl": 0.0,
            "pinkyCurl": 0.0,
            "fullCurl": 0.0,
            "spread": 0.0, # Spreading fingers apart
            "pinch": 0.0 # Bringing index to thumb
        }
        
        self.raw = None
        self.image = placeholder

    def update(self, landmarks):

        self.raw = landmarks

        palmAspectX = distance(landmarks[Landmarks.INDEX_BASE], landmarks[Landmarks.PINKY_BASE])
        palmAspectY = distance(
            landmarks[Landmarks.WRIST],
            type('',(object,),{
                "x": np.mean([landmarks[Landmarks.INDEX_BASE].x, landmarks[Landmarks.PINKY_BASE].x]),
                "y": np.mean([landmarks[Landmarks.INDEX_BASE].y, landmarks[Landmarks.PINKY_BASE].y])
            })()
        )

        self.processed["lift"] = np.max([palmAspectX, palmAspectY])
        
        self.processed["pitch"] = (4 / np.pi) * np.arctan(palmAspectX / palmAspectY) + 1
        
        self.processed["yaw"] = (4 / np.pi) * np.arctan2(palmAspectY, palmAspectX) + 1
        
        self.processed["roll"] = (1 / np.pi) * np.arctan2(
            landmarks[Landmarks.MIDDLE_TIP].y - landmarks[Landmarks.WRIST].y,
            landmarks[Landmarks.MIDDLE_TIP].x - landmarks[Landmarks.WRIST].x)

        self.processed["indexCurl"] = distance(landmarks[Landmarks.INDEX_TIP], landmarks[Landmarks.INDEX_BASE]) / self.processed["lift"]
        self.processed["middleCurl"] = distance(landmarks[Landmarks.MIDDLE_TIP], landmarks[Landmarks.MIDDLE_BASE]) / self.processed["lift"]
        self.processed["ringCurl"] = distance(landmarks[Landmarks.RING_TIP], landmarks[Landmarks.RING_BASE]) / self.processed["lift"]
        self.processed["pinkyCurl"] = distance(landmarks[Landmarks.PINKY_TIP], landmarks[Landmarks.PINKY_BASE]) / self.processed["lift"]
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

        for key in self.processed.keys():
            self.processed[key] = np.clip(self.processed[key], 0, 1)

    def getAnnotatedImage(self):

        image = self.image.copy()
        if image is None:
            return cv2.putText(placeholder.copy(), time.strftime("%H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        sidelen = image.shape[0]
        landmarks = self.raw
        
        if landmarks is None:
            return cv2.putText(image, time.strftime("%H:%M:%S"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        posList = np.ndarray((21, 2), dtype=np.int32)
        for i in range(len(landmarks)):
            color = (255, 128, 0) # Orange

            if i in [Landmarks.THUMB_BASE, Landmarks.INDEX_BASE, Landmarks.MIDDLE_BASE, Landmarks.RING_BASE, Landmarks.PINKY_BASE]:
                color = (255, 0, 0) # Red
            if i in [Landmarks.THUMB_TIP, Landmarks.INDEX_TIP, Landmarks.MIDDLE_TIP, Landmarks.RING_TIP, Landmarks.PINKY_TIP]:
                color = (0, 255, 0) # Green
            if i == Landmarks.WRIST:
                color = (0, 255, 255) # Cyan

            cv2.circle(image, (int(landmarks[i].x * sidelen), int(landmarks[i].y * sidelen)), 5, color, -1)
            posList[i] = [int(landmarks[i].x * sidelen), int(landmarks[i].y * sidelen)]

        cv2.polylines(image, [np.array([
            np.array(posList[Landmarks.INDEX_BASE]),
            np.array(posList[Landmarks.PINKY_BASE]),
            np.add(posList[Landmarks.PINKY_BASE], np.subtract(posList[Landmarks.WRIST], posList[Landmarks.RING_BASE])),
            np.add(posList[Landmarks.INDEX_BASE], np.subtract(posList[Landmarks.WRIST], posList[Landmarks.RING_BASE])),
        ]).reshape(-1, 1, 2)], isClosed=True, color=(255, 255, 255), thickness=2)

        for i in [Landmarks.INDEX_BASE, Landmarks.MIDDLE_BASE, Landmarks.RING_BASE, Landmarks.PINKY_BASE, Landmarks.THUMB_BASE]:
            cv2.polylines(image, [np.array([
                np.array(posList[i]),       # BASE
                np.array(posList[i + 1]),   # KNUCKLE 1
                np.array(posList[i + 2]),   # KNUCKLE 2
                np.array(posList[i + 3]),   # TIP
            ]).reshape(-1, 1, 2)], isClosed=False, color=(255, 255, 255), thickness=2)

        return image

def distance(p1, p2):
    return np.linalg.norm(np.array((p1.x, p1.y)) - np.array((p2.x, p2.y)))