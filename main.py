# main file
# terminal run project: py main.py

from hand import handDetector
import cv2
from constrants import *
import cvzone
import random

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, Screen_Width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Screen_Height)

# load image
imgBackGround = cv2.imread(urlBackGround)
imgGameOver = cv2.imread(urlGameOver)
imgBall = cv2.imread(urlBall, cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread(urlBat1, cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread(urlBat2, cv2.IMREAD_UNCHANGED)
imgControl = cv2.imread(urlControl, cv2.IMREAD_UNCHANGED)

detector = handDetector()  # hand detector from hand module

h1, _, _ = imgBat1.shape
h2, _, _ = imgBat1.shape
y_bat1 = Screen_Height // 2 - h1
y_bat2 = Screen_Height // 2 - h2


def _getPos(posY: int, h1: int) -> int:
    tmp = posY - 200 - h1//2
    if tmp < 0:
        return 5
    elif tmp > Screen_Height - 300:
        return Screen_Height - 300
    return tmp


def _start_Rand() -> None:
    global ballPos, player, speedX
    ballPos = [random.randint(100, 1180), random.randint(
        30, Screen_Height - 350)]
    if ballPos[0] < Screen_Width // 2:
        player = 2
        speedX = 15
    else:
        player = 1
        speedX = -15


_start_Rand()
while True:
    success, img = cap.read()  # screen shot
    if not success:  # check read image done or not
        print("Camera error!!!")
        break

    img = cv2.flip(img, 1)  # flip image
    imgRaw = img.copy()

    # detector hand
    img = detector.findHands(img)  # find hands
    # left, right hand position
    leftList, rightList = detector.findPosition(img)  # find landmark

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, imgBackGround, 0.8, 0)

    # get position base hand landmarks
    if leftList:
        y_bat1 = _getPos(leftList[0][2], imgBat1.shape[0])
    if rightList:
        y_bat2 = _getPos(rightList[0][2], imgBat2.shape[0])

    # draw image bat1 and bat2
    if not(y_bat1 is None):
        img = cvzone.overlayPNG(img, imgBat1, (59, y_bat1))
    if not(y_bat2 is None):
        img = cvzone.overlayPNG(img, imgBat2, (1195, y_bat2))

    # check change x
    h1, w1, _ = imgBat1.shape
    if 59 < ballPos[0] < 59 + w1 and y_bat1 < ballPos[1] < y_bat1 + h1:
        speedX = abs(speedX) + 1 if speedX < 0 else -(speedX + 1)
        ballPos[0] += 30
        if player == 1:
            score[0] += 1
            player = 2
    h1, w1, _ = imgBat2.shape
    if 1195 - 55 < ballPos[0] < 1195 + w1 and y_bat2 < ballPos[1] < y_bat2 + h1:
        speedX = abs(speedX) + 1 if speedX < 0 else - \
            (speedX + 1)  # Change OX direction
        ballPos[0] -= 30
        if player == 2:
            score[1] += 1
            player = 1
    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True
    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[0] + score[1]).zfill(2), (585, 360),
                    cv2.FONT_HERSHEY_COMPLEX, 2.5, RED, 5)
    else:
        if ballPos[1] >= 500 or ballPos[1] <= 20:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        # Display score on the image
        # Left player (hand)
        cv2.putText(img, str(score[0]), (300, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, GREEN, 5)
        # Rigth player (hand)
        cv2.putText(img, str(score[1]), (900, 650),
                    cv2.FONT_HERSHEY_COMPLEX, 3, RED, 5)
        # speed
        cv2.putText(img, "SPEED:", (540, 600),
                    cv2.FONT_HERSHEY_COMPLEX, 2, BLUE, 2)
        cv2.putText(img, str(abs(speedX)), (580, 690),
                    cv2.FONT_HERSHEY_COMPLEX, 3, BLUE, 2)

    # Cam show
    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    # show screen
    cv2.imshow("Pong Game", img)
    # wait key press
    key = cv2.waitKey(5)
    # Reload the game by pressing "r"
    if gameOver and key == ord("r"):
        speedX = speedY = 15
        score = [0, 0]
        player = 2
        _start_Rand()
        gameOver = False
        imgGameOver = cv2.imread(urlGameOver)
    # Exit loop if 'q' key is pressed
    if key and key == ord('q'):
        break
    if not gameOver and key and key == ord('s'):
        while True:  # wait press s to continue
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
# Release resources
cap.release()
cv2.destroyAllWindows()
