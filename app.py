import os
import time
import math
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

# ---------- CONFIG ----------
MODEL_PATH_H5 = "sign_model.h5"
CLASSES_ROOT_FOR_ORDER = "smaller-data/train"
INPUT_SIZE = (128, 128)
SMOOTH_WINDOW = 12
CONF_THRESH = 0.75
HOLD_SECONDS = 3.0
COOLDOWN_SECONDS = 0.8
MAX_FPS_AVG = 30

# NEW: GUI scaling (only for display; capture stays 1080p)
DISPLAY_SCALE = 0.60              # shown window scale (0.4–1.0)
DISPLAY_MIN, DISPLAY_MAX = 0.40, 1.00


def get_class_names_from_dir(root):
    names = [d for d in os.listdir(
        root) if os.path.isdir(os.path.join(root, d))]
    names = sorted(names)
    return names


if not os.path.isdir(CLASSES_ROOT_FOR_ORDER):
    raise SystemExit(
        f"[ERROR] Could not find class root: {CLASSES_ROOT_FOR_ORDER}")

CLASS_NAMES = get_class_names_from_dir(CLASSES_ROOT_FOR_ORDER)
NUM_CLASSES = len(CLASS_NAMES)
print("[INFO] Classes:", CLASS_NAMES)

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH_H5)
print("[INFO] Model loaded.")

# --------------- MediaPipe (optional) ---------------
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    USE_MP = True
except Exception:
    print("[WARN] mediapipe not available; running WITHOUT hand mask. pip install mediapipe")
    USE_MP = False

hands = mp_hands.Hands(
    model_complexity=0, max_num_hands=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) if USE_MP else None


def preprocess_frame(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    g = g.astype("float32") / 255.0
    g = np.expand_dims(g, axis=(0, -1))
    return g


def hand_mask_from_landmarks(frame_bgr, results):
    h, w, _ = frame_bgr.shape
    if not results or not results.multi_hand_landmarks:
        return frame_bgr, (0, 0, w, h)

    lm = results.multi_hand_landmarks[0]
    pts = []
    for p in lm.landmark:
        x, y = int(p.x * w), int(p.y * h)
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

    x, y, bw, bh = cv2.boundingRect(hull)
    margin = int(0.15 * max(bw, bh))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)
    return masked, (x1, y1, x2, y2)


probs_queue = deque(maxlen=SMOOTH_WINDOW)


def smooth_probs(current_probs):
    probs_queue.append(current_probs)
    avg = np.mean(probs_queue, axis=0) if len(
        probs_queue) > 0 else current_probs
    return avg


last_top = None
top_since = None
last_commit_time = 0.0
typed_text = ""

cap = cv2.VideoCapture(0)
# Keep full resolution capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Make the window resizable; we’ll control size via scaling
cv2.namedWindow("ISL Real-time Translator", cv2.WINDOW_NORMAL)

fps_hist = deque(maxlen=MAX_FPS_AVG)
last_time = time.time()

print("[INFO] Press:")
print("  [Space] to insert space")
print("  [Backspace/B] to delete last char")
print("  [C] to clear text")
print("  [+/-] to zoom GUI (display only)")
print("  [Esc] to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if USE_MP:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = hands.process(rgb)
        masked, bbox = hand_mask_from_landmarks(frame, mp_res)
    else:
        mp_res = None
        masked, bbox = frame, (0, 0, w, h)

    x1, y1, x2, y2 = bbox
    roi = masked[y1:y2, x1:x2].copy()

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    top_class = None
    top_conf = 0.0
    top3 = []

    if roi.size > 0 and (y2 - y1) > 20 and (x2 - x1) > 20:
        inp = preprocess_frame(roi)
        probs = model.predict(inp, verbose=0)[0]
        probs = smooth_probs(probs)
        order = np.argsort(-probs)[:3]
        top_class = int(order[0])
        top_conf = float(probs[top_class])
        top3 = [(CLASS_NAMES[i], float(probs[i])) for i in order]
    else:
        probs_queue.clear()

    panel = frame.copy()
    base_x, base_y = 25, 50
    cv2.putText(panel, "ISL -> Text (hold 3s to commit)", (base_x, base_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if top3:
        for k, (lbl, c) in enumerate(top3):
            y = base_y + 28 * k
            bar_w = int(300 * c)
            cv2.rectangle(panel, (base_x, y),
                          (base_x + 300, y + 18), (40, 40, 40), 2)
            cv2.rectangle(panel, (base_x, y), (base_x +
                          bar_w, y + 18), (0, 180, 0), -1)
            cv2.putText(panel, f"{lbl}: {c*100:.1f}%", (base_x + 310, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "No hand detected", (base_x, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)

    now = time.time()
    committed = False
    if top3 and top_conf >= CONF_THRESH:
        current_label = CLASS_NAMES[top_class]
        if last_top == current_label:
            if top_since is None:
                top_since = now
            elif (now - top_since) >= HOLD_SECONDS and (now - last_commit_time) >= COOLDOWN_SECONDS:
                typed_text += current_label
                last_commit_time = now
                top_since = None
                probs_queue.clear()
                committed = True
        else:
            last_top = current_label
            top_since = now
    else:
        top_since = None
        last_top = None

    if top3:
        label = CLASS_NAMES[top_class]
        cv2.putText(panel, f"Candidate: {label}  ({top_conf*100:.1f}%)",
                    (base_x, base_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cx, cy, r = base_x + 30, base_y + 160, 18
        cv2.circle(panel, (cx, cy), r, (200, 200, 200), 2)
        if top_since and not committed:
            progress = min(1.0, (now - top_since) / HOLD_SECONDS)
            end_angle = int(360 * progress)
            cv2.ellipse(panel, (cx, cy), (r, r), -90,
                        0, end_angle, (0, 255, 0), 4)
            cv2.putText(panel, f"Hold: {HOLD_SECONDS - (now - top_since):.1f}s",
                        (base_x + 60, base_y + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (180, 255, 180), 2, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Hold 3s to type", (base_x + 60, base_y + 168),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

    t = time.time()
    fps = 1.0 / max(1e-6, (t - last_time))
    last_time = t
    fps_hist.append(fps)
    fps_avg = sum(fps_hist) / len(fps_hist) if fps_hist else 0
    cv2.putText(panel, f"FPS: {fps_avg:.1f}", (w - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.rectangle(panel, (20, h - 90), (w - 20, h - 30), (30, 30, 30), -1)
    cv2.putText(panel, f"Typed: {typed_text}", (30, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    if USE_MP and mp_res and mp_res.multi_hand_landmarks:
        for hand_lms in mp_res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                panel, hand_lms, mp_hands.HAND_CONNECTIONS)

    if roi.size > 0:
        preview = cv2.resize(roi, (200, 200))
        # keep preview inside bounds (top-right)
        y0, y1 = 20, 220
        x0, x1 = w - 220, w - 20
        panel[y0:y1, x0:x1] = preview
        cv2.putText(panel, "ROI", (w - 80, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ---------- NEW: display scaling (only the shown window is resized) ----------
    scale = np.clip(DISPLAY_SCALE, DISPLAY_MIN, DISPLAY_MAX)
    if abs(scale - 1.0) > 1e-3:
        disp = cv2.resize(panel, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    else:
        disp = panel

    cv2.imshow("ISL Real-time Translator", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:      # ESC
        break
    elif key == 32:    # SPACE
        typed_text += " "
    elif key in (8, ord('b'), ord('B')):  # Backspace or 'b'
        typed_text = typed_text[:-1] if typed_text else typed_text
    elif key in (ord('c'), ord('C')):
        typed_text = ""
    elif key in (ord('+'), ord('=')):     # zoom in
        DISPLAY_SCALE = min(DISPLAY_MAX, DISPLAY_SCALE + 0.05)
    elif key in (ord('-'), ord('_')):     # zoom out
        DISPLAY_SCALE = max(DISPLAY_MIN, DISPLAY_SCALE - 0.05)

cap.release()
cv2.destroyAllWindows()
if hands:
    hands.close()
