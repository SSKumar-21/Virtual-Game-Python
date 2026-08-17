import cv2
import mediapipe as mp
import time
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options=python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options=vision.PoseLandmarkerOptions(base_options=base_options,running_mode=vision.RunningMode.VIDEO,num_poses=1)
landmarker=vision.PoseLandmarker.create_from_options(options)

connections=[
(0,11),(0,12),(11,12),
(11,13),(13,15),
(12,14),(14,16),
(15,17),(15,19),(15,21),
(16,18),(16,20),(16,22),
(11,23),(12,24),(23,24),
(23,25),(25,27),(27,29),(27,31),
(24,26),(26,28),(28,30),(28,32)
]

def press_key(key):
    print(f"KEY PRESS → {key}")
    pyautogui.press(key)

cam=cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Could not open camera")
    exit()

frame_timestamp=0
previous_horizontal="CENTER"
last_special_action=None
special_action_until=0

while True:
    success,frame=cam.read()
    if not success:
        print("❌ Could not read camera")
        break

    frame=cv2.flip(frame,1)
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
    frame_timestamp+=33
    result=landmarker.detect_for_video(mp_image,frame_timestamp)

    if result.pose_landmarks:
        landmarks=result.pose_landmarks[0]
        points=[]

        for landmark in landmarks:
            x=int(landmark.x*frame.shape[1])
            y=int(landmark.y*frame.shape[0])
            points.append((x,y))

        for start,end in connections:
            x1,y1=points[start]
            x2,y2=points[end]
            cv2.line(frame,(x1,y1),(x2,y2),(123,0,231),3)

        for i in range(11,33):
            x,y=points[i]
            cv2.circle(frame,(x,y),5,(123,0,231),-1)

        nose_x,nose_y=points[0]
        cv2.circle(frame,(nose_x,nose_y),7,(123,0,231),-1)

        left_shoulder=landmarks[11]
        right_shoulder=landmarks[12]
        left_hip=landmarks[23]
        right_hip=landmarks[24]

        base_x=(left_shoulder.x+right_shoulder.x+left_hip.x+right_hip.x)/4
        shoulder_line=(left_shoulder.y+right_shoulder.y)/2

        if base_x<0.40:
            horizontal_action="LEFT"
        elif base_x>0.60:
            horizontal_action="RIGHT"
        else:
            horizontal_action="CENTER"

        current_time=time.time()
        special_detected=None

        if shoulder_line<0.30:
            special_detected="JUMP"
        elif shoulder_line>0.60:
            special_detected="SIT"

        if special_detected is not None:
            if special_detected!=last_special_action or current_time>=special_action_until:
                if special_detected=="JUMP":
                    press_key("up")
                elif special_detected=="SIT":
                    press_key("down")
                last_special_action=special_detected
            special_action_until=current_time+1.0

        if current_time<special_action_until:
            action=special_detected if special_detected is not None else last_special_action
        else:
            action=horizontal_action
            last_special_action=None

            if action!=previous_horizontal:
                if previous_horizontal=="LEFT" and action=="RIGHT":
                    press_key("right")
                    press_key("right")
                elif previous_horizontal=="RIGHT" and action=="LEFT":
                    press_key("left")
                    press_key("left")
                elif previous_horizontal=="LEFT" and action=="CENTER":
                    press_key("right")
                elif previous_horizontal=="RIGHT" and action=="CENTER":
                    press_key("left")
                elif previous_horizontal=="CENTER" and action=="LEFT":
                    press_key("left")
                elif previous_horizontal=="CENTER" and action=="RIGHT":
                    press_key("right")

                previous_horizontal=action

        cv2.putText(frame,f"Action: {action}",(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.2,(123,0,231),3)
        cv2.putText(frame,f"Base X: {base_x:.2f}",(30,105),cv2.FONT_HERSHEY_SIMPLEX,0.8,(123,0,231),2)
        cv2.putText(frame,f"Shoulder Y: {shoulder_line:.2f}",(30,140),cv2.FONT_HERSHEY_SIMPLEX,0.8,(123,0,231),2)
        cv2.putText(frame,f"Previous: {previous_horizontal}",(30,175),cv2.FONT_HERSHEY_SIMPLEX,0.8,(123,0,231),2)

    cv2.imshow("Subway Surfers Controller",frame)

    if cv2.waitKey(1)&0xFF==ord("x"):
        break

cam.release()
cv2.destroyAllWindows()