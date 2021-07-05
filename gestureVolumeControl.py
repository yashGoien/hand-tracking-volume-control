import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


##################################
wCam, hCam = 640, 480
##################################

cTime , pTime = 0 , 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))       
volRange = volume.GetVolumeRange()
vol = 0
volBar = 400
volPer = 0


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


detector = htm.HandDetector(detectionConf=0.6)

while(True) :
    _ , img = cap.read()

    img = detector.find_hands(img)
    lmList = detector.find_position(img,draw = False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx , cy = int((x1+x2)/2), int((y1+y2)/2)
        length = math.hypot(x2-x1,y2-y1)

        cv2.circle(img, (x1, y1), 10,(240,240,16), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10,(240,240,16), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15,(240,240,16), cv2.FILLED)

        cv2.line(img, (x1,y1), (x2,y2),(240,240,16), 2)

        # VOLUME CHANGE
        vol = np.interp(length, [20,300], [volRange[0], volRange[1]])
        volBar = np.interp(length, [20,300], [400,140])
        volPer = np.interp(length, [20,300], [0,100])
        volume.SetMasterVolumeLevel(vol, None)

        # print(length)
        

    cv2.rectangle(img, (50, 130), (85, 400), (240,240,16), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (240,240,16), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40,450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (240,240,16), 2)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (40,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (220,30,0), 2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()