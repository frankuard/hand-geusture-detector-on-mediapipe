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



cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame= cv2.flip(frame,1)
    
    cv2.imshow("Hand Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
