import streamlit as st
import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

# Initialize variables
brushThickness = 32
eraserThickness = 64

folderPath = 'Header'
myFiles = os.listdir(folderPath)
overlayList = []

for imPath in myFiles:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (0, 0, 255)

detector = htm.HandDetector(min_detection_confidence=0.85, min_tracking_confidence=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Streamlit app layout
st.title('Air Canvas')
st.subheader('Draw with your hands in the air!')

# Button to clear canvas
if st.button('Clear Canvas'):
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Streamlit webcam input
stframe = st.empty()
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.write("Ignoring empty camera frame.")
        continue

    img = cv2.flip(img, 1)

    # 2. Find the Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        # Tip of Index Finger
        x1, y1 = lmList[8][1:]
        # Tip of Middle Finger
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If Selection Mode (Two Fingers up) - SELECT MODE
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 32), (x2, y2 + 32), drawColor, cv2.FILLED)
            if y1 < 128:
                if 250 <= x1 <= 350:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 480 <= x1 <= 580:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 720 <= x1 <= 820:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                elif 950 <= x1 <= 1050:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

        # 5. Index Finger up - DRAW MODE
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 16, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness=eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness=eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness=brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness=brushThickness)

            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the Header Image
    img[0:128, 0:1280] = header

    # Display the image in Streamlit
    stframe.image(img, channels="BGR")

cap.release()
