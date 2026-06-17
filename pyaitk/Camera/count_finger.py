import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
cv2.setLogLevel(0)

import cv2
import mediapipe as mp

def countF():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    count = 0

    for _ in range(10):  # Check for ~0.5 seconds
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                fingers = []

                fingers.append(1 if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x else 0)
                for tip in finger_tips:
                    fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)

                count = sum(fingers)
                break
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    return count
