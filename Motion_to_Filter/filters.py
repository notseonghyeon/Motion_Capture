## filters.py — 각 필터 함수 정의

import cv2
import numpy as np


def filter_glass(frame):
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=25)


def filter_noise(frame):
    h, w = frame.shape[:2]
    noise = np.random.randint(0, 80, (h, w, 3), dtype=np.uint8)
    return cv2.add(frame, noise)


def filter_pixel(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 20, h // 20), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def filter_invert(frame):
    return cv2.bitwise_not(frame)


def filter_thermal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


# 이름 → 함수 매핑. 새 필터를 추가할 때 여기에만 등록하면 됨.
FILTER_REGISTRY = {
    "glass":   filter_glass,
    "noise":   filter_noise,
    "pixel":   filter_pixel,
    "invert":  filter_invert,
    "thermal": filter_thermal,
}
