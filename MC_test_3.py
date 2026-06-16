## Hand_Quad_Filter
# 20260615 ASH

import cv2
import numpy as np
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# 필터 목록과 현재 인덱스
FILTERS = ["glass", "noise", "pixel", "invert", "thermal"]
filter_idx = 0
mode = FILTERS[filter_idx]

# 제스처 상태 추적용
prev_fist = False         # 직전 프레임에서 주먹이었는지
last_switch_time = 0.0    # 마지막으로 필터 바꾼 시각
COOLDOWN = 0.5            # 초 단위 쿨다운


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


def is_fist(hand_landmarks):
    """검지·중지·약지·새끼가 모두 접혔으면 주먹으로 판단"""
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    folded = 0
    for tip, pip in finger_pairs:
        # y는 아래로 갈수록 커짐. 끝이 관절보다 아래(=값이 큼)면 접힌 것
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded += 1
    return folded == 4


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

                # 오른손 기준으로 주먹/폄 감지해 필터 전환
                if label == "Right":
                    fist_now = is_fist(hand_landmarks)
                    now = time.time()
                    # 직전이 주먹이고 지금 폈으면 = 펴는 순간
                    if prev_fist and not fist_now and (now - last_switch_time) > COOLDOWN:
                        filter_idx = (filter_idx + 1) % len(FILTERS)
                        mode = FILTERS[filter_idx]
                        last_switch_time = now
                    prev_fist = fist_now

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

        # 현재 필터 이름 화면에 표시
        cv2.putText(
            frame,
            f"mode: {mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Hand Quad Filter", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC 종료
            break

cap.release()
cv2.destroyAllWindows()