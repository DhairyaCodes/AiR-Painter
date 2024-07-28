import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Mediapipe uses RGB Images
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks)
    h, w, c = img.shape
    if result.multi_hand_landmarks:
        for handsLms in result.multi_hand_landmarks:
            for id, lm in enumerate(handsLms.landmark):
                # print(id, lm)
                cx, cy = int(w * lm.x), int(h * lm.y)
                # print(id, cx, cy)
                # if(id == 20):
                #     cv2.circle(img, (cx, cy), 8, (255, 0, 249), cv2.FILLED)

            mpDraw.draw_landmarks(img, handsLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), thickness=2)


    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
