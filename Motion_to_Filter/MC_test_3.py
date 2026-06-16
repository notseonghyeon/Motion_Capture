## Hand_Quad_Filter
# 20260616 ASH

import cv2
import numpy as np
import mediapipe as mp
import time
import random
from filters import FILTER_REGISTRY

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# 필터 목록과 현재 인덱스
FILTERS = list(FILTER_REGISTRY.keys())
filter_idx = 0
mode = FILTERS[filter_idx]

# 제스처 상태 추적용 (오른손: 필터 전환)
prev_fist = False         # 직전 프레임에서 주먹이었는지
last_switch_time = 0.0    # 마지막으로 필터 바꾼 시각
COOLDOWN = 0.5            # 초 단위 쿨다운

# 왼손 제스처 상태 추적용 (고정 영역)
prev_left_fist = False    # 직전 프레임에서 왼손이 주먹이었는지
left_open_time = 0.0      # 왼손을 편 시각 (0이면 대기 중 아님)
FREEZE_DELAY = 0.5        # 손을 편 뒤 고정까지 딜레이(초)
frozen_quads = []         # 고정된 (quad, mode) 쌍 목록 — 중첩 누적
current_quad = None       # 이번 프레임의 라이브 사각형

def order_points(pts):
    """네 점을 무게중심 기준 각도순으로 정렬해 사각형이 꼬이지 않게 함"""
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
    max_num_hands=2,                           # 최대 추적할 손 개수
    min_detection_confidence=0.5,              # 손이 인식되었다고 판단하려면 최소한 이 정도 확신이 필요
    min_tracking_confidence=0.5,               # 손 추적이 얼마나 안정적이어야 하는지
    model_complexity=1,                        # 손 인식에 쓸 모델의 정밀도
) as hands:
    while cap.isOpened():
        ok, frame = cap.read()                        # 웹캠에서 프레임 읽기
        if not ok:
            break

        frame = cv2.flip(frame, 1)                    # 거울처럼 좌우 반전
        h, w = frame.shape[:2]                        # 프레임 높이와 너비
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe는 RGB 입력을 기대함
        results = hands.process(rgb)                  # 손 추정 수행

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
                    if prev_fist and not fist_now and (now - last_switch_time) > COOLDOWN:
                        filter_idx = (filter_idx + 1) % len(FILTERS)
                        mode = FILTERS[filter_idx]
                        last_switch_time = now
                    prev_fist = fist_now

                # 왼손 기준으로 주먹/폄 감지해 영역 고정
                if label == "Left":
                    left_fist_now = is_fist(hand_landmarks)
                    now = time.time()
                    if prev_left_fist and not left_fist_now:
                        left_open_time = now
                    prev_left_fist = left_fist_now

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # 딜레이 경과 후 현재 라이브 quad를 고정
        now = time.time()
        if left_open_time > 0 and (now - left_open_time) >= FREEZE_DELAY:
            if current_quad is not None:
                frozen_quads.append((current_quad.copy(), mode))  # 누적
            left_open_time = 0.0

        # 고정 영역들 순서대로 합성 (각자 고정 시점 필터 유지)
        for fq, fm in frozen_quads:
            frozen_filtered = FILTER_REGISTRY[fm](frame)
            frozen_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(frozen_mask, [fq], 255)
            frozen_mask3 = cv2.cvtColor(frozen_mask, cv2.COLOR_GRAY2BGR) > 0
            frame = np.where(frozen_mask3, frozen_filtered, frame)
            cv2.polylines(frame, [fq], True, (0, 0, 255), 2)

        # 양손이 모두 잡혔을 때 라이브 사각형 영역 생성
        current_quad = None
        if tips["Left"] is not None and tips["Right"] is not None:
            quad = [
                tips["Left"][0],   # 왼손 엄지끝
                tips["Left"][1],   # 왼손 검지끝
                tips["Right"][1],  # 오른손 검지끝
                tips["Right"][0],  # 오른손 엄지끝
            ]
            quad = order_points(quad)
            current_quad = quad

            # 라이브 영역에 현재 필터 합성 (고정 영역 위에 덮어씌움)
            live_filtered = FILTER_REGISTRY[mode](frame)
            live_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(live_mask, [quad], 255)
            live_mask3 = cv2.cvtColor(live_mask, cv2.COLOR_GRAY2BGR) > 0
            frame = np.where(live_mask3, live_filtered, frame)

            # 초록 테두리
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

        display = cv2.resize(frame, (640, 360)) # 화면 크기 조정
        cv2.imshow("Hand Quad Filter", display)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC 종료
            break

cap.release()
cv2.destroyAllWindows()