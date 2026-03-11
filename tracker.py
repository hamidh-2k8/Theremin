import os
import cv2
import time
import logging
import threading
import numpy as np
import mediapipe as mp
import cv2_enumerate_cameras
from gestures import Result
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

class CameraType():
    TOWER = 0
    BASE = 1

LOGGER = logging.getLogger("Tracker")

towerResult, baseResult = Result(), Result()
towerCam, baseCam = None, None
        
def towerCallback(result, output_image: mp.Image, timestamp_ms: int):
    if len(result.hand_landmarks) > 0:
        towerResult.update(result.hand_landmarks[0])

def baseCallback(result, output_image: mp.Image, timestamp_ms: int):
    if len(result.hand_landmarks) > 0:
        baseResult.update(result.hand_landmarks[0])

# To be changed/saved from UI
config: dict = {
    "tower_id": "/dev/video0",
    "base_id": "/dev/video2",
    "frame_width": 800,
    "frame_height": 600,
    "fps": 30
}

def updateConfig(newConfig: dict):
    global config
    config = newConfig
    LOGGER.info(f"Updated tracking configuration: {str(config)}")

def getCaptureDevices():
    return cv2_enumerate_cameras.enumerate_cameras()

def startTracking():
    global towerCam, baseCam

    towerCam = cv2.VideoCapture()
    baseCam = cv2.VideoCapture()

    towerCam.set(cv2.CAP_PROP_FRAME_WIDTH, config["frame_width"])
    towerCam.set(cv2.CAP_PROP_FRAME_HEIGHT, config["frame_height"])
    towerCam.set(cv2.CAP_PROP_FPS, config["fps"])

    baseCam.set(cv2.CAP_PROP_FRAME_WIDTH, config["frame_width"])
    baseCam.set(cv2.CAP_PROP_FRAME_HEIGHT, config["frame_height"])
    baseCam.set(cv2.CAP_PROP_FPS, config["fps"])

    if not towerCam.open(config["tower_id"]):
        raise Exception("Could not open tower camera")

    if not baseCam.open(config["base_id"]):
        raise Exception("Could not open base camera")
    
    global trackStartTime
    trackStartTime = time.perf_counter()

    global towerThread, baseThread

    towerThread = threading.Thread(target=trackerThreadLoop, args=(CameraType.TOWER,), daemon=True)
    towerThread.start()

    baseThread = threading.Thread(target=trackerThreadLoop, args=(CameraType.BASE,), daemon=True)
    baseThread.start()

def stopTracking():
    global towerCam, baseCam, towerThread, baseThread

    if towerCam.isOpened():
        towerCam.release()

    if baseCam.isOpened():
        baseCam.release()

    towerThread.join()
    baseThread.join()
    
    LOGGER.info("Tracking threads stopped.")

def trackerThreadLoop(camType: CameraType):

    LOGGER.info(f"Starting tracker thread for {'tower' if camType == CameraType.TOWER else 'base'} camera.")

    cam = towerCam if camType == CameraType.TOWER else baseCam
    result = towerResult if camType == CameraType.TOWER else baseResult
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="/home/hamid/Documents/Robots/Theremin/hand_landmarker.task"),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=(towerCallback if camType == CameraType.TOWER else baseCallback),
        num_hands=1
    )
    
    landmarker = HandLandmarker.create_from_options(options)

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
        frame = np.array(frame[y:y + min_dim, x:x + min_dim], dtype=np.uint8)

        result.image = frame
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        landmarker.detect_async(mp_image, int((time.perf_counter() - trackStartTime) * 1000))