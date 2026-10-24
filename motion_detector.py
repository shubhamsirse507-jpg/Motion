"""
Advanced Motion Detector, Hand Posture Recognition, and Gesture Math Engine.

Features:
- MOG2 background modeling and motion tracking
- YCrCb skin segmentation and hand contour analysis
- Robust palm center calculation via distance transform
- Real-time fingertip detection and posture classification (Fist, 1, 2/Peace, 3, 4, 5/Palm)
- Dual-hand mathematical calculation (Left Hand op Right Hand = Sum, Difference, Product, Quotient)
- Interactive quiz & game telemetry for the Code Eye visualizer
- Non-blocking threaded camera capture
- Rich cybernetic HUD overlay on preview stream
"""

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionState:
    motion: bool = False
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    blink_id: int = 0
    fps: float = 0.0

    # Hand Posture & Math Fields
    hand_detected: bool = False
    left_hand_fingers: int = 0
    right_hand_fingers: int = 0
    active_fingers: int = 0
    math_add: int = 0
    math_sub: int = 0
    math_mul: int = 0
    math_div: float = 0.0
    gesture_left: str = "NONE"
    gesture_right: str = "NONE"


class ThreadedCamera:
    """Non-blocking threaded webcam reader that eliminates OpenCV buffer lag."""

    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.running = self.cap.isOpened()
        self.lock = threading.Lock()
        if self.running:
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()

    def _reader_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.grabbed = True
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def is_opened(self) -> bool:
        return self.running and self.cap.isOpened()

    def release(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


GESTURE_NAMES = {
    0: "FIST (0)",
    1: "POINT (1)",
    2: "PEACE (2)",
    3: "THREE (3)",
    4: "FOUR (4)",
    5: "PALM (5)",
}


class AdvancedMotionDetector:
    """Advanced motion detection, hand posture recognition, and math calculator."""

    def __init__(
        self,
        min_area: int = 80,
        history: int = 150,
        var_threshold: int = 25,
        blink_cooldown: float = 0.7,
        preview_width: int = 320,
        preview_height: int = 240,
    ):
        self.min_area = min_area
        self.blink_cooldown = blink_cooldown
        self.preview_w = preview_width
        self.preview_h = preview_height

        # MOG2 Background Subtractor
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )

        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_hand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Motion tracking state
        self.sx = 0.0
        self.sy = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.was_moving = False
        self.last_blink_time = 0.0
        self.blink_id = 0

        # Performance metrics
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.fps_filter = 30.0

        # Temporal smoothing buffers for fingers
        self.left_history = deque(maxlen=5)
        self.right_history = deque(maxlen=5)

    def process(self, frame: np.ndarray) -> Tuple[MotionState, bytes]:
        now = time.time()
        dt = max(0.001, now - self.last_frame_time)
        self.last_frame_time = now

        # Instant FPS
        instant_fps = 1.0 / dt
        self.fps_filter += (instant_fps - self.fps_filter) * 0.1
        self.fps = round(self.fps_filter, 1)

        # 1. Motion Tracking
        proc_w, proc_h = 160, 120
        small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small, (7, 7), 0)

        fg_mask = self.subtractor.apply(blurred)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.dilate(fg_mask, self.kernel_dilate, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = 0
        sum_cx, sum_cy = 0.0, 0.0
        motion_detected = False

        sig_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= 30:
                sig_contours.append((c, area))
                total_area += area

        if total_area >= self.min_area and sig_contours:
            motion_detected = True
            for c, area in sig_contours:
                m = cv2.moments(c)
                if m["m00"] > 0:
                    sum_cx += (m["m10"] / m["m00"]) * area
                    sum_cy += (m["m01"] / m["m00"]) * area

            raw_cx = sum_cx / max(1.0, total_area)
            raw_cy = sum_cy / max(1.0, total_area)

            target_nx = -((raw_cx / proc_w) * 2.0 - 1.0)
            target_ny = (raw_cy / proc_h) * 2.0 - 1.0
            target_nx = max(-1.0, min(1.0, target_nx))
            target_ny = max(-1.0, min(1.0, target_ny))

            self.vx = round((target_nx - self.prev_x) / dt, 3)
            self.vy = round((target_ny - self.prev_y) / dt, 3)
            self.prev_x = target_nx
            self.prev_y = target_ny
            speed = math.hypot(self.vx, self.vy)

            alpha = min(0.88, max(0.38, 0.38 + speed * 0.18))
            self.sx += (target_nx - self.sx) * alpha
            self.sy += (target_ny - self.sy) * alpha

            if not self.was_moving and (now - self.last_blink_time > self.blink_cooldown):
                self.blink_id += 1
                self.last_blink_time = now
            self.was_moving = True
        else:
            motion_detected = False
            self.was_moving = False
            self.vx *= 0.6
            self.vy *= 0.6
            speed = 0.0

        # 2. Hand Posture Recognition & Math Calculation
        # Flip frame horizontally for mirror preview (320x240)
        disp = cv2.flip(frame, 1)
        disp = cv2.resize(disp, (self.preview_w, self.preview_h), interpolation=cv2.INTER_LINEAR)

        hands_data = self._detect_hands(disp)
        left_val, right_val, left_g, right_g = self._analyze_hand_math(hands_data)

        # Mathematical calculations
        m_add = left_val + right_val
        m_sub = left_val - right_val
        m_mul = left_val * right_val
        m_div = round(left_val / max(1, right_val), 2)
        has_hands = bool(hands_data)
        active_f = left_val if left_val > 0 else right_val

        # Construct MotionState
        motion_state = MotionState(
            motion=motion_detected,
            x=round(float(self.sx), 3),
            y=round(float(self.sy * 0.85), 3),
            vx=round(float(self.vx), 2),
            vy=round(float(self.vy), 2),
            speed=round(float(speed), 2),
            blink_id=self.blink_id,
            fps=self.fps,
            hand_detected=has_hands,
            left_hand_fingers=left_val,
            right_hand_fingers=right_val,
            active_fingers=active_f,
            math_add=m_add,
            math_sub=m_sub,
            math_mul=m_mul,
            math_div=m_div,
            gesture_left=left_g,
            gesture_right=right_g,
        )

        # 3. Render Hand Math Telemetry on Preview Image
        self._render_hud(disp, motion_state, hands_data)

        # Encode JPEG preview
        ok_enc, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = buf.tobytes() if ok_enc else b""

        return motion_state, jpeg_bytes

    def _detect_hands(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Detects hand contours, palm centers, and fingertips using YCrCb skin segmentation."""
        h, w = frame_bgr.shape[:2]
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

        # Skin range in YCrCb
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 177, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower, upper)

        # Morphological clean
        skin_mask = cv2.erode(skin_mask, self.kernel_hand, iterations=1)
        skin_mask = cv2.dilate(skin_mask, self.kernel_hand, iterations=2)
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hands = []

        # Find significant hand contours (up to 2 hands)
        valid_cnts = []
        for c in contours:
            area = cv2.contourArea(c)
            # Area between 1400 and 40000 pixels on 320x240 image
            if 1400 <= area <= 42000:
                valid_cnts.append((c, area))

        valid_cnts.sort(key=lambda x: x[1], reverse=True)
        top_hands = valid_cnts[:2]

        for c, area in top_hands:
            # Palm center via distance transform
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            dist_map = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
            cx, cy = max_loc
            radius = max_val

            if radius < 9:
                continue

            hull = cv2.convexHull(c)
            fingertips = []
            min_dist = radius * 1.32

            # Find points extending outward from palm
            for pt in hull[:, 0]:
                px, py = int(pt[0]), int(pt[1])
                d = np.hypot(px - cx, py - cy)
                if d > min_dist and py < cy + radius * 0.45:
                    # Deduplicate nearby cluster points
                    if not any(np.hypot(px - fx, py - fy) < radius * 0.42 for fx, fy in fingertips):
                        fingertips.append((px, py))

            f_count = min(5, len(fingertips))
            # If area is substantial and fingers > 0
            is_left = cx < (w / 2)

            hands.append({
                "count": f_count,
                "center": (cx, cy),
                "radius": int(radius),
                "tips": fingertips,
                "is_left": is_left,
                "area": area,
            })

        return hands

    def _analyze_hand_math(self, hands: List[Dict]) -> Tuple[int, int, str, str]:
        """Calculates stabilized left and right hand finger counts and gesture names."""
        left_c = None
        right_c = None

        for h in hands:
            if h["is_left"]:
                if left_c is None:
                    left_c = h["count"]
            else:
                if right_c is None:
                    right_c = h["count"]

        # Temporal smoothing
        if left_c is not None:
            self.left_history.append(left_c)
        else:
            if self.left_history:
                self.left_history.append(0)

        if right_c is not None:
            self.right_history.append(right_c)
        else:
            if self.right_history:
                self.right_history.append(0)

        left_final = round(np.median(self.left_history)) if self.left_history else 0
        right_final = round(np.median(self.right_history)) if self.right_history else 0

        left_final = int(max(0, min(5, left_final)))
        right_final = int(max(0, min(5, right_final)))

        g_left = GESTURE_NAMES.get(left_final, "NONE") if (hands and left_c is not None) else "NONE"
        g_right = GESTURE_NAMES.get(right_final, "NONE") if (hands and right_c is not None) else "NONE"

        return left_final, right_final, g_left, g_right

    def _render_hud(self, disp: np.ndarray, state: MotionState, hands: List[Dict]):
        w, h = self.preview_w, self.preview_h

        # Top status bar
        cv2.rectangle(disp, (0, 0), (w, 24), (10, 14, 32), -1)
        cv2.line(disp, (0, 24), (w, 24), (70, 140, 240), 1)

        # Status text & Math header
        if state.hand_detected:
            # Active Math HUD banner
            math_txt = f"MATH: {state.left_hand_fingers} + {state.right_hand_fingers} = {state.math_add}  |  {state.left_hand_fingers} x {state.right_hand_fingers} = {state.math_mul}"
            cv2.circle(disp, (12, 12), 4, (0, 255, 120), -1)
            cv2.putText(disp, math_txt, (24, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 245, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(disp, (12, 12), 4, (80, 160, 255), -1)
            cv2.putText(disp, "SHOW HANDS TO CALCULATE", (24, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, f"FPS:{state.fps:.0f}", (265, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 180, 220), 1, cv2.LINE_AA)

        # Render Hand skeletons, palm circles and fingertips
        for hand in hands:
            cx, cy = hand["center"]
            r = hand["radius"]
            side_txt = "L" if hand["is_left"] else "R"

            # Inscribed palm circle
            cv2.circle(disp, (cx, cy), r, (60, 120, 255), 1, cv2.LINE_AA)
            cv2.circle(disp, (cx, cy), 3, (255, 255, 255), -1)

            # Draw fingertips and ray lines from palm
            for tx, ty in hand["tips"]:
                cv2.line(disp, (cx, cy), (tx, ty), (0, 220, 255), 1, cv2.LINE_AA)
                cv2.circle(disp, (tx, ty), 5, (0, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(disp, (tx, ty), 2, (255, 255, 255), -1)

            # Label above hand
            f_num = hand["count"]
            badge = f"{side_txt}:{f_num}"
            cv2.rectangle(disp, (cx - 24, cy - r - 20), (cx + 24, cy - r - 4), (12, 18, 40), -1)
            cv2.rectangle(disp, (cx - 24, cy - r - 20), (cx + 24, cy - r - 4), (0, 220, 255), 1)
            cv2.putText(disp, badge, (cx - 18, cy - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)
