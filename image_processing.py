import numpy as np
import cv2

minValue = 70


def remove_background(frame):
    # Convert to YCrCb (works better than HSV for skin detection)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Define skin color range
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    # Create mask for skin
    mask = cv2.inRange(ycrcb, lower, upper)

    # Apply mask
    skin = cv2.bitwise_and(frame, frame, mask=mask)

    return skin


def func(path):
    frame = cv2.imread(path)

    # --- Step 1: Remove background, keep only skin regions ---
    skin_only = remove_background(frame)

    # --- Step 2: Convert to grayscale ---
    gray = cv2.cvtColor(skin_only, cv2.COLOR_BGR2GRAY)

    # --- Step 3: Gaussian Blur ---
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    # --- Step 4: Adaptive thresholding ---
    th3 = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # --- Step 5: Otsu’s thresholding ---
    _, res = cv2.threshold(
        th3, minValue, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return res
