## Motion_Capture_Test
# 20260615 ASH
# 손가락을 인식하는지만을 확인하는코드.
# mediapipe가 잘 작동하는지, 웹캠이 잘 연결되는지, 랜드마크가 잘 그려지는지만을 확인하기 위한 테스트 코드.

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)  # 0번 = 맥북 내장 웹캠

with mp_hands.Hands(
    max_num_hands=2,              # 최대 추적할 손 개수
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,          # 0=빠름, 1=정확
) as hands:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)              # 거울처럼 좌우 반전
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)            # 손 추정

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:          # ESC 키로 종료
            break

cap.release()
cv2.destroyAllWindows()
