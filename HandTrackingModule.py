import cv2
import mediapipe as mp
import time


class HandDetector:

    def __init__(self, static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Mediapipe uses RGB Images
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        h, w, c = img.shape
        if self.result.multi_hand_landmarks:
            for handsLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handsLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=False):
        self.lmList = []
        # print(result.multi_hand_landmarks)
        h, w, c = img.shape
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                cx, cy = int(w * lm.x), int(h * lm.y)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 249), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []

        # Thumb -> Open Close by left or Right offset
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers -> Finger tip above or below the index 2 below it shows open or close
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        # if(len(lmList) > 0):
        #     print(lmList[20])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), thickness=2)

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == '__main__':
    main()