import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VisionRunningMode

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")


latest_result= [None]

def on_result(result,output_image,timestamps_ms):
    
    latest_result[0] = result
    
options = HandLandmarkerOptions(base_options=base_options,running_mode=VisionRunningMode.LIVE_STREAM,num_hands=2,min_hand_detection_confidence = 0.5,result_callback= on_result)


landmarker = HandLandmarker.create_from_options(options)
print("HandLandMarker created successfully")



cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame= cv2.flip(frame,1)
    
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format= mp.ImageFormat.SRGB,data=frame_rgb)
    
    timestamp_ms = int(time.time()* 1000)
    
    landmarker.detect_async(mp_image,timestamp_ms)
    
    result = latest_result[0]
    
    if result and result.hand_landmarks:
        
        for hand_lms in result.hand_landmarks:
    
            h,w = frame.shape[:2]
            
            Connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
            
            for a,b in Connections:
                x1 = int(hand_lms[a].x*w)
                y1 = int(hand_lms[a].y*h)
        
                x2 = int(hand_lms[b].x*w)
                y2 = int(hand_lms[b].y*h)

                cv2.line(frame,(x1,y1),(x2,y2),(255,255,255),2)
            
            landmarkers = list(range(21))
            
            for i in landmarkers:
                lm = hand_lms[i]
                x = int(lm.x*w)
                y = int(lm.y*h)    
                
                cv2.circle(frame,(x,y),3,(0,200,0),-1)
                
                cv2.putText(frame,str(i),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1)
             
            tips = [4,8,12,16,20]
            pips = [3,6,10,14,18]
            
            fingers = []
            
            text = ''
            
        
            if hand_lms[4].x < hand_lms[3].x:
                fingers.append(1)
            
            else:
                fingers.append(0)
                 
            
            for tip,pip in zip(tips[1:],pips[1:]):
                
                if hand_lms[tip].y < hand_lms[pip].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
                    
            if fingers == [1,1,1,1,1]:
                text = "HELLO 👋"
                
            elif fingers == [0,1,1,0,0]:
                text = "PEACE ✌️"
                
            elif fingers == [0,0,0,0,0]:
                text = "BRO FIST 👊"
              
            cv2.putText(frame,text, (50,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)               
    cv2.imshow("Hand Tracking", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
