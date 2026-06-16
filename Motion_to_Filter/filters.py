## filters.py — 각 필터 함수 정의

import cv2
import numpy as np
import random

def filter_glass(frame): # 유리창
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=25)


def filter_noise(frame): # 노이즈
    h, w = frame.shape[:2]
    noise = np.random.randint(0, 200, (h, w, 3), dtype=np.uint8)
    return cv2.add(frame, noise)


def filter_pixel(frame): # 픽셀
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 20, h // 20), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def filter_invert(frame): # 색 반전
    return cv2.bitwise_not(frame)


def filter_thermal(frame): # 열화상
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def filter_pencil(frame): # 펜슬
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    _, sk_color = cv2.pencilSketch(small, sigma_s=10, sigma_r=0.07, shade_factor=0.1)
    return cv2.resize(sk_color, (w, h))

def filter_glitch(frame, intensity=1.0):
    glitch_frame = frame.copy()
    h, w = frame.shape[:2]

    # 1) 가로 밀기: 개수와 거리를 intensity로 키움
    num_slices = int(10 * intensity)        # 3 -> 10+
    max_shift = int(60 * intensity)         # 20 -> 60+
    band = max(5, int(20 / intensity))      # 밴드 높이도 다양하게

    for _ in range(num_slices):
        bh = random.randint(3, band)        # 밴드 높이 무작위
        y = random.randint(0, h - bh)
        shift = random.randint(-max_shift, max_shift)

        if shift > 0:
            glitch_frame[y:y+bh, shift:w] = frame[y:y+bh, 0:w-shift]
        elif shift < 0:
            glitch_frame[y:y+bh, 0:w+shift] = frame[y:y+bh, -shift:w]

    # 2) RGB 채널 분리 (색 번짐) — BGR 기준
    cshift = int(15 * intensity)
    if cshift > 0:
        b, g, r = cv2.split(glitch_frame)
        r = np.roll(r, cshift, axis=1)      # 빨강 오른쪽으로
        b = np.roll(b, -cshift, axis=1)     # 파랑 왼쪽으로
        glitch_frame = cv2.merge([b, g, r])

    # 3) 블록 깨짐: 무작위 직사각형을 다른 위치에서 복사
    num_blocks = int(5 * intensity)
    for _ in range(num_blocks):
        bw = random.randint(20, 80)
        bh = random.randint(10, 40)
        x1 = random.randint(0, w - bw)
        y1 = random.randint(0, h - bh)
        x2 = random.randint(0, w - bw)
        y2 = random.randint(0, h - bh)
        glitch_frame[y1:y1+bh, x1:x1+bw] = frame[y2:y2+bh, x2:x2+bw]

    return glitch_frame


# 이름 → 함수 매핑. 새 필터를 추가할 때 여기에만 등록하면 됨.
FILTER_REGISTRY = {
    # "glass":   filter_glass,
    "noise":   filter_noise,
    # "pixel":   filter_pixel,
    "invert":  filter_invert,
    "thermal": filter_thermal,
    "pencil":  filter_pencil,
    "glitch": filter_glitch,

}
