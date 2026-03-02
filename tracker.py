import os
import cv2
import time
import logging
import threading
import mediapipe as mp
from gestures import Result
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

class CameraType():
    TOWER = 0
    BASE = 1

LOGGER = logging.getLogger("Tracker")

modelPath = "./hand_landmarker.task"


towerResult, baseResult = Result(), Result()
        
def towerCallback(result, output_image: mp.Image, timestamp_ms: int):
    towerResult.update(result.hand_landmarks[0], output_image)
    
def baseCallback(result, output_image: mp.Image, timestamp_ms: int):
    baseResult.update(result.hand_landmarks[0], output_image)

# Initialize HandLandmarker
towerLandmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=modelPath),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=towerCallback,
        num_hands=1)
)

baseLandmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=modelPath),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=baseCallback,
        num_hands=1)
)

# To be changed/saved from UI
config: dict = {
    "tower_id": 0,
    "base_id": 1,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30
}

def updateConfig(newConfig: dict):
    global config
    config = newConfig

def startTracking():
    global towerLandmarker, baseLandmarker
    global towerCam, baseCam

    towerCam = cv2.VideoCapture(config["tower_id"])
    baseCam = cv2.VideoCapture(config["base_id"])

    towerCam.set(cv2.CAP_PROP_FRAME_WIDTH, config["frame_width"])
    towerCam.set(cv2.CAP_PROP_FRAME_HEIGHT, config["frame_height"])
    towerCam.set(cv2.CAP_PROP_FPS, config["fps"])

    baseCam.set(cv2.CAP_PROP_FRAME_WIDTH, config["frame_width"])
    baseCam.set(cv2.CAP_PROP_FRAME_HEIGHT, config["frame_height"])
    baseCam.set(cv2.CAP_PROP_FPS, config["fps"])

    if not towerCam.open():
        raise Exception("Could not open tower camera")

    if not baseCam.open():
        raise Exception("Could not open base camera")
    
    global trackStartTime
    trackStartTime = time.perf_counter()

    global towerThread, baseThread

    towerThread = threading.Thread(target=trackerThreadLoop, args=(CameraType.TOWER,))
    towerThread.start()

    baseThread = threading.Thread(target=trackerThreadLoop, args=(CameraType.BASE,))
    baseThread.start()

def stopTracking():
    global towerCam, baseCam, towerThread, baseThread

    if towerCam.isOpened():
        towerCam.release()

    if baseCam.isOpened():
        baseCam.release()

    towerThread.join()
    baseThread.join()

def trackerThreadLoop(type: CameraType):

    cam = towerCam if type == CameraType.TOWER else baseCam
    landmarker = towerLandmarker if type == CameraType.TOWER else baseLandmarker

    while True:
        if not cam.isOpened():
            LOGGER.warning("Could not open stream for camera.")
            continue

        success, frame = cam.read()
        
        if not success:
            LOGGER.warning("Could not read frame from camera.")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Crop to centered square
        h, w, _ = frame.shape
        min_dim = min(h, w)
        x = (w - min_dim) // 2
        y = (h - min_dim) // 2
        frame = frame[y:y + min_dim, x:x + min_dim]

        landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        )