"""
Hand Module
"""

import cv2
import mediapipe as mp
from constrants import Screen_Width, GRAY, BLUR_GREEN, BLUR_RED

leftDot = Screen_Width // 2


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # convert image to RGB color
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # process find hand

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True) -> list:
        leftList = []
        rightList = []
        isLeft = True
        if self.results.multi_hand_landmarks:
            for myHand in self.results.multi_hand_landmarks:
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if int(id) == 0 and cx < leftDot:
                        isLeft = True
                    elif int(id) == 0:
                        isLeft = False

                    if isLeft:
                        leftList.append([id, cx, cy])
                    else:
                        rightList.append([id, cx, cy])
                    if draw and isLeft:
                        cv2.circle(img, (cx, cy), 10,
                                   BLUR_GREEN, cv2.FILLED)
                        cv2.putText(img, str(id), (cx, cy),
                                    cv2.FONT_HERSHEY_PLAIN, 2, GRAY, 3)
                    else:
                        cv2.circle(img, (cx, cy), 10,
                                   BLUR_RED, cv2.FILLED)
                        cv2.putText(img, str(id), (cx, cy),
                                    cv2.FONT_HERSHEY_PLAIN, 2, GRAY, 3)
        return leftList, rightList
