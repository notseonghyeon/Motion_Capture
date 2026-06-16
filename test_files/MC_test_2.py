## Hand_Quad_Filter
# 20260615 ASH
# 양손의 엄지끝과 검지끝을 꼭짓점으로 하는 사각형 영역에 필터를 씌우는 코드.
# 키보드로 필터를 변경할 수 있음.


import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# 필터 모드: 'glass' = 불투명 유리(블러), 'noise' = 노이즈
mode = "glass"


def order_points(pts):
    """네 점을 무게중심 기준 각도순으로 정렬해 다각형이 꼬이지 않게 함"""
    pts = np.array(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    return pts[order].astype(np.int32)


def get_fingertip(hand_landmarks, idx, w, h):
    """랜드마크 인덱스의 (x, y)를 픽셀 좌표로 변환"""
    lm = hand_landmarks.landmark[idx]
    return [int(lm.x * w), int(lm.y * h)]


with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
) as hands:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # 왼손/오른손의 엄지끝·검지끝 좌표를 담을 딕셔너리
        tips = {"Left": None, "Right": None}

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # flip 했으므로 라벨도 좌우가 화면과 일치함
                label = handedness.classification[0].label  # 'Left' or 'Right'
                thumb = get_fingertip(hand_landmarks, 4, w, h)   # 엄지끝
                index = get_fingertip(hand_landmarks, 8, w, h)   # 검지끝
                tips[label] = (thumb, index)

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # 양손이 모두 잡혔을 때만 사각형 영역 생성
        if tips["Left"] is not None and tips["Right"] is not None:
            quad = [
                tips["Left"][0],   # 왼손 엄지끝
                tips["Left"][1],   # 왼손 검지끝
                tips["Right"][1],  # 오른손 검지끝
                tips["Right"][0],  # 오른손 엄지끝
            ]
            quad = order_points(quad)

            # 다각형 마스크 생성
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [quad], 255)

            # 필터 적용된 전체 프레임을 미리 만들고, 마스크 영역만 합성
            if mode == "glass":
                filtered = cv2.GaussianBlur(frame, (0, 0), sigmaX=25)

            elif mode == "noise":
                noise = np.random.randint(0, 80, (h, w, 3), dtype=np.uint8)
                filtered = cv2.add(frame, noise)

            elif mode == "pixel":  # 모자이크(픽셀화)
                small = cv2.resize(frame, (w // 20, h // 20),
                                   interpolation=cv2.INTER_LINEAR)
                filtered = cv2.resize(small, (w, h),
                                      interpolation=cv2.INTER_NEAREST)

            elif mode == "invert":  # 색반전
                filtered = cv2.bitwise_not(frame)

            elif mode == "thermal":  # 열화상 느낌
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                filtered = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
            frame = np.where(mask3, filtered, frame)

            # 영역 테두리 표시 (원하지 않으면 주석 처리)
            cv2.polylines(frame, [quad], True, (0, 255, 0), 2)

        cv2.imshow("Hand Quad Filter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:            # ESC 종료
            break
        elif key == ord("g"):    # 유리
            mode = "glass"
        elif key == ord("n"):    # 노이즈
            mode = "noise"
        elif key == ord("p"):    # 모자이크(픽셀화)
            mode = "pixel"
        elif key == ord("i"):    # 색반전
            mode = "invert"
        elif key == ord("t"):    # 열화상
            mode = "thermal"

cap.release()
cv2.destroyAllWindows()