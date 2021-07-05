import cv2
import mediapipe as mp
import time

class HandDetector():

    def __init__(self, static = False, maxHands = 2, detectionConf = 0.5, trackConf = 0.5):
        self.static = static
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static, self.maxHands, self.detectionConf, self.trackConf)

        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.res = self.hands.process(imgRGB)

        if self.res.multi_hand_landmarks:
            for handLms in self.res.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) 
                   
        return img


    def find_position(self, img, handNo = 0, draw = True):

        lmList = []

        if self.res.multi_hand_landmarks:
            myHand = self.res.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
        
        return lmList


def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0
    detector = HandDetector()
    while(True):
        _, img = cap.read()

        img = detector.find_hands(img)
        lmList = detector.find_position(img)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,255), 3)


        cv2.imshow('img',img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
