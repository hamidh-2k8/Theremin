import os
import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from rtmidi.midiutil import open_midioutput
from rtmidi.midiconstants import CONTROL_CHANGE

# File Directories (Change project_dir on demo computer)
project_dir = "/home/hamid/Documents/Robots/Theremin"
model_path = "hand_landmarker.task"
cam_id = 0
midi_port = 0

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

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
    
class CC_IDs():
    
    LEFT_LIFT  =  0x01
    RIGHT_LIFT =  0x02
    LEFT_CURL  =  0x03
    RIGHT_CURL =  0x04
    
    
class Gestures():
    
    def __init__(self, lift, curl):
        
        self.lift = lift
        self.curl = curl
        
    def update(self, landmarks):
        
        updated = False
    
        # Liftage Gesture: Regular theremin operation, based on the estimated distance of the hand from the camera.
        # Calculated using the distance between WRIST and MIDDLE_BASE, which is also affected by hand pitch.
        
        try:
            prev_lift = self.lift
            self.lift = np.clip(abs(
                    distance(landmarks[Landmarks.WRIST], landmarks[Landmarks.MIDDLE_BASE])
                    ), 0, 1)
            if self.lift != prev_lift:
                updated = True
        except:
            pass
        
        # Curl Gesture: The amount that the fingers are curled, bringing the tips of the fingers to their bases.
        # Calculated using the distance between MIDDLE_TIP and MIDDLE_BASE. Not a full squeeze/fist, the palm is still flat.
        
        try:
            prev_curl = self.curl
            self.curl = 1 - np.clip(abs(
                    distance(landmarks[Landmarks.MIDDLE_TIP], landmarks[Landmarks.MIDDLE_BASE]) / self.lift
                    ), 0, 1)
            if self.curl != prev_curl:
                updated = True
        except: 
            pass
        
        return updated
            
def distance(pos1, pos2):
    return np.sqrt(np.square(pos1.x - pos2.x) + np.square(pos1.y - pos2.y))

def landmarker_callback(result, output_image: mp.Image, timestamp_ms: int):    
    left_landmarks = None
    left_world_landmarks = None
    right_landmarks = None
    right_world_landmarks = None
    
    for i in range(len(result.handedness)):
        if result.handedness[i][0].category_name == "Left":
            left_landmarks = result.hand_landmarks[i]
            left_world_landmarks = result.hand_world_landmarks[i]
        if result.handedness[i][0].category_name == "Right":
            right_landmarks = result.hand_landmarks[i]
            right_world_landmarks = result.hand_world_landmarks[i]
            
    # Recalculate Gesture data based on new landmarks
    l_update = l_gestures.update(left_landmarks)
    r_update = r_gestures.update(right_landmarks)
    
    #Gesture -> MIDI Assignments
    if l_update:   
        midiout.send_message([CONTROL_CHANGE, CC_IDs.LEFT_LIFT, int(l_gestures.lift * 127)])
        midiout.send_message([CONTROL_CHANGE, CC_IDs.LEFT_CURL, int(l_gestures.curl * 127)])
        
    if r_update:
        midiout.send_message([CONTROL_CHANGE, CC_IDs.RIGHT_LIFT, int(r_gestures.lift * 127)])
        midiout.send_message([CONTROL_CHANGE, CC_IDs.RIGHT_CURL, int(r_gestures.curl * 127)])
    
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(project_dir, model_path)),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=landmarker_callback,
    num_hands=2)

cam = cv2.VideoCapture(cam_id)

midiout, port_name = open_midioutput(midi_port)

landmarker = HandLandmarker.create_from_options(options)

start_time = time.perf_counter()

global l_gestures, r_gestures

l_gestures = Gestures(0.5, 0.5)
r_gestures = Gestures(0.5, 0.5)

while cam.isOpened():
    success, image = cam.read()
    
    if not success:
        print(f"[ERROR] Reading from camera {cam_id} failed.")
        break
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    landmarker.detect_async(mp_image, int((time.perf_counter() - start_time) * 1000) )
    
midiout.close_port()