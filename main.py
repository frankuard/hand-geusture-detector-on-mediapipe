import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VisionRunningMode

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")


latest_result= [None]

def on_result(result):
    latest_result[0] = result
    
options = HandLandmarkerOptions(base_options=base_options,running_mode=VisionRunningMode.LIVE_STREAM,num_hands=2,min_hand_detection_confidence = 0.5,result_callback= on_result)


landmarker = HandLandmarker.create_from_options(options)
print("HandLandMarker created successfully")

