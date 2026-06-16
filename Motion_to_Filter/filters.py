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
    h, w = frame.shape[:2]

    # 절반 해상도에서 처리 → 픽셀 수 1/4, 블록감은 오히려 강해짐
    sw, sh = w // 2, h // 2
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    g = small.copy()

    # 1) 가로 밀기
    num_slices = int(10 * intensity)
    max_shift = int(30 * intensity)   # 해상도 절반이므로 거리도 절반
    band = max(3, int(10 / intensity))

    for _ in range(num_slices):
        bh = random.randint(2, band)
        y  = random.randint(0, sh - bh)
        shift = random.randint(-max_shift, max_shift)
        if shift > 0:
            g[y:y+bh, shift:sw] = small[y:y+bh, 0:sw-shift]
        elif shift < 0:
            g[y:y+bh, 0:sw+shift] = small[y:y+bh, -shift:sw]

    # 2) 채널 번짐 — split/merge 대신 배열 슬라이싱으로 복사 최소화
    cshift = int(8 * intensity)
    if cshift > 0:
        out = g.copy()
        out[:, cshift:,  2] = g[:, :sw-cshift, 2]   # R 오른쪽
        out[:, :sw-cshift, 0] = g[:, cshift:,  0]   # B 왼쪽
        g = out

    # 3) 블록 깨짐
    num_blocks = int(5 * intensity)
    for _ in range(num_blocks):
        bw = random.randint(10, 40)
        bh = random.randint(5, 20)
        x1 = random.randint(0, sw - bw)
        y1 = random.randint(0, sh - bh)
        x2 = random.randint(0, sw - bw)
        y2 = random.randint(0, sh - bh)
        g[y1:y1+bh, x1:x1+bw] = small[y2:y2+bh, x2:x2+bw]

    # 원래 해상도로 복원 (NEAREST → 블록 픽셀 느낌 유지)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)


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
