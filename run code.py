import sys
import warnings
import logging
import numpy as np
import torch
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import leap
from kan import KAN
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import deque, Counter
import threading
import time
import os
import queue
import serial
import math

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*std().*")
logging.getLogger("kan").setLevel(logging.ERROR)


def create_fake_numpy_core():
    if hasattr(np, '_core'):
        return
    import types
    fake_core = types.ModuleType('numpy._core')
    fake_core.__path__ = []
    fake_core.__file__ = '<fake>'
    import numpy.core as real_core
    import numpy.core.multiarray as real_multiarray
    import numpy.core.umath as real_umath
    fake_multiarray = types.ModuleType('numpy._core.multiarray')
    for attr in dir(real_multiarray):
        if not attr.startswith('__'):
            try:
                setattr(fake_multiarray, attr, getattr(real_multiarray, attr))
            except Exception:
                pass
    if hasattr(real_multiarray, '_reconstruct'):
        fake_multiarray._reconstruct = real_multiarray._reconstruct
    else:
        fake_multiarray._reconstruct = lambda self, *args, **kwargs: self
    fake_multiarray.ndarray = np.ndarray
    fake_multiarray.dtype = np.dtype
    fake_multiarray.generic = np.generic
    fake_multiarray.ufunc = type(np.add)
    fake_multiarray.scalar = np.generic
    fake_multiarray.getbuffer = memoryview
    fake_umath = types.ModuleType('numpy._core.umath')
    for attr in dir(real_umath):
        if not attr.startswith('__'):
            try:
                setattr(fake_umath, attr, getattr(real_umath, attr))
            except Exception:
                pass
    if not hasattr(fake_umath, '_ufunc_reconstruct'):
        fake_umath._ufunc_reconstruct = lambda *args, **kwargs: None
    if not hasattr(fake_umath, 'absolute'):
        fake_umath.absolute = np.abs
    fake_core.multiarray = fake_multiarray
    fake_core.umath = fake_umath
    if not hasattr(fake_core, '_ufunc_reconstruct'):
        fake_core._ufunc_reconstruct = fake_umath._ufunc_reconstruct
    sys.modules['numpy._core'] = fake_core
    sys.modules['numpy._core.multiarray'] = fake_multiarray
    sys.modules['numpy._core.umath'] = fake_umath
    np._core = fake_core

create_fake_numpy_core()


class Config:
    # ---- Runtime settings (من الكود الأول) ----
    USE_FRAME_SAMPLING = True
    SAMPLE_RATE = 1
    WINDOW_SIZE = 40
    SMOOTHING_WINDOW = 4
    CONFIRMATION_THRESHOLD = 4
    CONFIDENCE_THRESHOLD = 0.75
    COOLDOWN_SECONDS = 0.6

    MIN_VELOCITY = 0.03
    STILL_FRAMES_TO_CLEAR = 25
    MIN_MOTION_RANGE = 0.005

    SERVO_MAX_SPEED_DEG_PER_SEC = 50.0
    SERVO_MAX_ACCEL_DEG_PER_SEC2 = 120.0
    SERVO_EMA_ALPHA = 0.45
    SERVO_DEADZONE_DEG = 0.3

    DEBUG_PERIOD = 0.5

    # 🛡️ عتبات Semantic Guard
    CLOSE_GRAB_MAX = 0.35
    CLOSE_PINCH_MAX = 0.35
    OPEN_GRAB_MIN = 0.75
    OPEN_PINCH_MIN = 0.75
    FINGER_STREAK_REQUIRED = 2

    REQUIRED_FEATURES = [
        'palm_x', 'palm_y', 'palm_z',
        'palm_velocity_x', 'palm_velocity_y', 'palm_velocity_z',
        'grab_strength', 'pinch_strength', 'pinch_distance',
        'thumb_extended', 'index_extended', 'middle_extended',
        'ring_extended', 'pinky_extended'
    ]
    FEATURES = len(REQUIRED_FEATURES)

    GRU_HIDDEN_SIZE = 38
    GRU_NUM_LAYERS = 1
    ATTENTION_HEADS = 2
    KAN_WIDTH = [38, 19, 9]
    KAN_GRID_SIZE = 3
    KAN_SPLINE_ORDER = 3
    DROPOUT_RATE = 0.4

    MODEL_SAVE_PATH = "hand_gesture_fast_smart_9classes.pth"

L1 = 0.15
L2 = 0.15
L3 = 0.16
L2_EFF = L2 + L3

HOME_X = 0.0
HOME_Y = 0.15
HOME_Z = 0.20

X_MIN, X_MAX = -0.40,  0.40
Y_MIN, Y_MAX =  0.05,  0.45
Z_MIN, Z_MAX =  0.10,  0.45

SCALE = 0.8

_NEUTRAL = {"x": None, "y": None, "z": None}
_NEUTRAL_LOCK = threading.Lock()

# 🆕 آخر delta محفوظة لكل محور (raw قبل SCALE) — الفلترة بين الإيماءات
_LAST_DELTA = {"x": 0.0, "y": 0.0, "z": 0.0}
_LAST_DELTA_LOCK = threading.Lock()

DEBUG_VERBOSE = False

REALTIME_HZ = 30
SMOOTHING_ALPHA = 0.4

_ACTIVE_GESTURE = {"name": None, "time": 0.0, "conf": 0.0}
ACTIVE_GESTURE_TIMEOUT = 1.5

HOME_J1 = 90
HOME_J2 = 150
HOME_J3 = 0
HOME_J4 = 110
HOME_J5 = 120
HOME_J6 = 90

DIR_J1   = -1
DIR_J2   = -1
DIR_J5_Y = -1
DIR_J5_Z = +1

J4_MODE = "fixed"

def compute_j4(j2):
    if J4_MODE == "coupled":
        const = HOME_J2 + HOME_J4
        return int(np.clip(const - j2, 0, 180))
    elif J4_MODE == "follow_j2":
        delta = j2 - HOME_J2
        return int(np.clip(HOME_J4 + delta, 0, 180))
    elif J4_MODE == "mirror_j2":
        return int(np.clip(180 - j2, 0, 180))
    else:
        return HOME_J4

_LATEST_HAND = {"features": None, "time": 0.0}
_LATEST_HAND_LOCK = threading.Lock()

_FINGER_STATE = {"j3": HOME_J3, "j6": HOME_J6}

_current_arm_angles = [float(HOME_J1), float(HOME_J2), float(HOME_J5)]
_current_arm_vels   = [0.0, 0.0, 0.0]
_ema_targets        = [float(HOME_J1), float(HOME_J2), float(HOME_J5)]

# ============================================================
# CYLINDRICAL INVERSE KINEMATICS
# ============================================================
_IK_NEUTRAL = None

def _raw_ik(x, y, z):
    j1_geo = math.degrees(math.atan2(x, z))

    r_planar = abs(z) if abs(z) > 1e-6 else 1e-6

    d = math.sqrt(r_planar**2 + y**2)
    max_reach = L1 + L2_EFF
    min_reach = abs(L1 - L2_EFF)

    if d > max_reach and d > 1e-6:
        s = max_reach / d
        r_planar *= s; y *= s; d = max_reach
    if d < min_reach and d > 1e-6:
        s = min_reach / d
        r_planar *= s; y *= s; d = min_reach
    if d < 1e-6:
        return (0.0, 90.0, 90.0)

    cos_beta = (L1**2 + L2_EFF**2 - d**2) / (2 * L1 * L2_EFF)
    beta = math.degrees(math.acos(np.clip(cos_beta, -1.0, 1.0)))

    phi   = math.degrees(math.atan2(y, r_planar))
    alpha = math.degrees(math.acos(
        np.clip((L1**2 + d**2 - L2_EFF**2) / (2 * L1 * d), -1.0, 1.0)
    ))
    shoulder_geo = phi + alpha

    return (j1_geo, beta, shoulder_geo)


def compute_ik_xyz(x, y, z, push_pull=False):
    global _IK_NEUTRAL

    j1_geo, j2_geo, j5_geo = _raw_ik(x, y, z)

    if _IK_NEUTRAL is None:
        _IK_NEUTRAL = _raw_ik(HOME_X, HOME_Y, HOME_Z)
        print("=" * 60)
        print(f"📍 IK NEUTRAL CACHED")
        print("=" * 60)

    j1_n, j2_n, j5_n = _IK_NEUTRAL

    d1 = j1_geo - j1_n
    d2 = j2_geo - j2_n
    d5 = j5_geo - j5_n

    j1 = HOME_J1 + DIR_J1 * d1
    j2 = HOME_J2 + DIR_J2 * d2

    dir_j5 = DIR_J5_Z if push_pull else DIR_J5_Y
    j5 = HOME_J5 + dir_j5 * d5

    return [float(np.clip(j1, 0, 180)),
            float(np.clip(j2, 0, 160)),
            float(np.clip(j5, 0, 160))]


def hand_to_target(features, gesture=None, invert_z=False):
    hx = float(features[0]) / 1000.0
    hy = float(features[1]) / 1000.0
    hz = float(features[2]) / 1000.0

    if abs(hx) > 1.0 or abs(hy) > 1.0 or abs(hz) > 1.0:
        return [HOME_X, HOME_Y, HOME_Z]

    with _NEUTRAL_LOCK:
        nx, ny, nz = _NEUTRAL["x"], _NEUTRAL["y"], _NEUTRAL["z"]

    if nx is None:
        return [HOME_X, HOME_Y, HOME_Z]

    raw_dx = hx - nx
    raw_dy = hy - ny
    raw_dz = hz - nz

    with _LAST_DELTA_LOCK:
        if gesture in ["left", "right"]:
            _LAST_DELTA["x"] = raw_dx
            dx = _LAST_DELTA["x"]     # ← يُحدَّث
            dy = _LAST_DELTA["y"]     # ← آخر قيمة محفوظة
            dz = _LAST_DELTA["z"]     # ← آخر قيمة محفوظة
        elif gesture in ["raise", "lower"]:
            _LAST_DELTA["y"] = raw_dy
            dx = _LAST_DELTA["x"]
            dy = _LAST_DELTA["y"]     # ← يُحدَّث
            dz = _LAST_DELTA["z"]
        elif gesture in ["push", "pull"]:
            _LAST_DELTA["z"] = raw_dz
            dx = _LAST_DELTA["x"]
            dy = _LAST_DELTA["y"]
            dz = _LAST_DELTA["z"]     # ← يُحدَّث
        else:
            # بدون إيماءة ذراع → تحديث الكل (سلوك افتراضي)
            dx = raw_dx
            dy = raw_dy
            dz = raw_dz

    dx *= SCALE
    dy *= SCALE
    dz *= SCALE

    if invert_z:
        dz = -dz

    tx = float(np.clip(HOME_X + dx, X_MIN, X_MAX))
    ty = float(np.clip(HOME_Y + dy, Y_MIN, Y_MAX))
    tz = float(np.clip(HOME_Z + dz, Z_MIN, Z_MAX))

    return [tx, ty, tz]


SERIAL_PORT = 'COM4'
BAUDRATE = 9600
ser = None
_SERIAL_LOCK = threading.Lock()

_last_sent_cmd = None

def _build_current_cmd():
    j1 = int(round(np.clip(_current_arm_angles[0], 0, 180)))
    j2 = int(round(np.clip(_current_arm_angles[1], 0, 160)))
    j5 = int(round(np.clip(_current_arm_angles[2], 0, 180)))
    j3 = int(_FINGER_STATE["j3"])
    j6 = int(_FINGER_STATE["j6"])
    j4 = compute_j4(j2)
    return f"{j1},{j2},{j3},{j4},{j5},{j6}\n"

def _send_current_cmd(force=False, label=""):
    global _last_sent_cmd
    cmd = _build_current_cmd()
    if not force and cmd == _last_sent_cmd:
        return False
    with _SERIAL_LOCK:
        if ser and ser.is_open:
            try:
                ser.write(cmd.encode())
                _last_sent_cmd = cmd
                if label:
                    print(f"📤 {label}: {cmd.strip()}")
                return True
            except Exception as e:
                print(f"❌ Serial error: {e}")
        else:
            print("❌ Serial not available")
    return False

# ============================================================
# REAL-TIME WORKER — Cylindrical IK
# ============================================================
_realtime_running = False
_realtime_thread = None
_smoothed_target = [HOME_X, HOME_Y, HOME_Z]
_first_cycle = True

def _realtime_worker():
    global _realtime_running, _smoothed_target, _first_cycle
    global _current_arm_angles, _current_arm_vels, _ema_targets

    print(f"⚡ Real-time worker بدأ ({REALTIME_HZ} Hz)")
    print(f"   🎯 Cylindrical IK: J1 من x، J2/J5 من y و z")
    print(f"   🔥 Filtering: كل إيماءة تستخدم محورها فقط")
    print(f"   🧊 المحاور الأخرى تحتفظ بآخر قيمة (لا ترجع لصفر)")
    print(f"   Servo speed = {Config.SERVO_MAX_SPEED_DEG_PER_SEC}°/s")
    print("=" * 90)

    interval = 1.0 / REALTIME_HZ
    last_time = time.perf_counter()
    prev_t = time.perf_counter()

    arm_gestures = ["push", "pull", "left", "right", "raise", "lower"]
    _ema_targets = list(_current_arm_angles)
    _last_debug = 0.0

    while _realtime_running:
        now = time.perf_counter()
        dt = now - prev_t
        prev_t = now

        if now - last_time < interval:
            time.sleep(max(0, interval - (now - last_time) - 0.001))
            continue

        dt_real = max(0.005, min(dt, 0.2))

        with _LATEST_HAND_LOCK:
            feat = _LATEST_HAND["features"]
            hand_time = _LATEST_HAND["time"]

        hand_ok = feat is not None and (time.time() - hand_time) <= 0.5

        gesture = _ACTIVE_GESTURE["name"]
        gesture_time = _ACTIVE_GESTURE["time"]

        if gesture is not None and (time.time() - gesture_time) > ACTIVE_GESTURE_TIMEOUT:
            _ACTIVE_GESTURE["name"] = None
            gesture = None

        hand_raw = None
        hand_delta = None
        robot_target = None
        ik_raw = None

        if hand_ok and (gesture in arm_gestures):
            hx = float(feat[0])
            hy = float(feat[1])
            hz = float(feat[2])
            hand_raw = (hx, hy, hz)

            invert_z = gesture in ["push", "pull"]
            push_pull = gesture in ["push", "pull"]

            target = hand_to_target(feat, gesture=gesture, invert_z=invert_z)
            robot_target = target

            with _NEUTRAL_LOCK:
                nx = _NEUTRAL["x"]
                ny = _NEUTRAL["y"]
                nz = _NEUTRAL["z"]

            if nx is not None:
                dx = (hx / 1000.0 - nx) * SCALE
                dy = (hy / 1000.0 - ny) * SCALE
                dz = (hz / 1000.0 - nz) * SCALE
                hand_delta = (dx, dy, dz)

            if _first_cycle:
                _smoothed_target = list(target)
                _first_cycle = False
            else:
                for i in range(3):
                    _smoothed_target[i] = (SMOOTHING_ALPHA * target[i] +
                                           (1 - SMOOTHING_ALPHA) * _smoothed_target[i])

            try:
                j1_ik, j2_ik, j5_ik = compute_ik_xyz(*_smoothed_target, push_pull=push_pull)
                ik_raw = (j1_ik, j2_ik, j5_ik)
            except Exception as e:
                print(f"IK error: {e}")
                j1_ik, j2_ik, j5_ik = _current_arm_angles
        else:
            j1_ik, j2_ik, j5_ik = _current_arm_angles

        raw_target = [j1_ik, j2_ik, j5_ik]

        a = Config.SERVO_EMA_ALPHA
        for i in range(3):
            _ema_targets[i] = a * raw_target[i] + (1 - a) * _ema_targets[i]

        max_v = Config.SERVO_MAX_SPEED_DEG_PER_SEC
        max_a = Config.SERVO_MAX_ACCEL_DEG_PER_SEC2
        max_dv = max_a * dt_real

        for i in range(3):
            err = _ema_targets[i] - _current_arm_angles[i]
            desired_v = err / max(dt_real, 0.02)
            desired_v = max(-max_v, min(max_v, desired_v))
            dv = desired_v - _current_arm_vels[i]
            dv = max(-max_dv, min(max_dv, dv))
            _current_arm_vels[i] += dv
            _current_arm_angles[i] += _current_arm_vels[i] * dt_real
            if abs(_ema_targets[i] - _current_arm_angles[i]) < Config.SERVO_DEADZONE_DEG:
                _current_arm_angles[i] = _ema_targets[i]
                _current_arm_vels[i] = 0.0

        if time.time() - _last_debug > Config.DEBUG_PERIOD:
            _last_debug = time.time()
            print("─" * 90)
            print(f"🎯 gesture={gesture}   hand_ok={hand_ok}")
            if hand_raw is not None:
                print(f"📷 Hand RAW (mm):     x={hand_raw[0]:+8.1f}  y={hand_raw[1]:+8.1f}  z={hand_raw[2]:+8.1f}")
            if hand_delta is not None:
                print(f"📐 Hand Delta (m):    dx={hand_delta[0]:+7.3f}  dy={hand_delta[1]:+7.3f}  dz={hand_delta[2]:+7.3f}")
            with _LAST_DELTA_LOCK:
                print(f"💾 Last Delta (raw):  x={_LAST_DELTA['x']:+7.3f}  y={_LAST_DELTA['y']:+7.3f}  z={_LAST_DELTA['z']:+7.3f}")
            with _NEUTRAL_LOCK:
                if _NEUTRAL["x"] is not None:
                    print(f"🏠 Neutral (m):       x={_NEUTRAL['x']:+7.3f}  y={_NEUTRAL['y']:+7.3f}  z={_NEUTRAL['z']:+7.3f}")
                else:
                    print(f"🏠 Neutral:           (NOT SET — اضغط زر الكالبريشن)")
            if robot_target is not None:
                print(f"🤖 Robot Target (m):  x={robot_target[0]:+7.3f}  y={robot_target[1]:+7.3f}  z={robot_target[2]:+7.3f}")
            if ik_raw is not None:
                print(f"🧮 IK Raw (deg):      J1={ik_raw[0]:6.1f}  J2={ik_raw[1]:6.1f}  J5={ik_raw[2]:6.1f}")
            print(f"⚙️  EMA Target (deg):  J1={_ema_targets[0]:6.1f}  J2={_ema_targets[1]:6.1f}  J5={_ema_targets[2]:6.1f}")
            print(f"✅ Servo Actual (deg): J1={_current_arm_angles[0]:6.1f}  J2={_current_arm_angles[1]:6.1f}  J5={_current_arm_angles[2]:6.1f}")

        _send_current_cmd()
        last_time = now

    print("⚡ Real-time worker توقف")


def start_realtime():
    global _realtime_running, _realtime_thread
    if _realtime_running:
        return
    _realtime_running = True
    _realtime_thread = threading.Thread(target=_realtime_worker, daemon=True)
    _realtime_thread.start()


def stop_realtime():
    global _realtime_running
    _realtime_running = False
    if _realtime_thread:
        _realtime_thread.join(timeout=1.0)


def set_active_gesture(name, conf=1.0):
    _ACTIVE_GESTURE["name"] = name
    _ACTIVE_GESTURE["time"] = time.time()
    _ACTIVE_GESTURE["conf"] = conf


def capture_neutral_from_current():
    global _IK_NEUTRAL, _first_cycle, _smoothed_target

    with _LATEST_HAND_LOCK:
        feat = _LATEST_HAND["features"]

    if feat is None:
        print("❌ لا توجد بيانات من اليد")
        return False

    hx = float(feat[0]) / 1000.0
    hy = float(feat[1]) / 1000.0
    hz = float(feat[2]) / 1000.0

    with _NEUTRAL_LOCK:
        _NEUTRAL["x"] = hx
        _NEUTRAL["y"] = hy
        _NEUTRAL["z"] = hz

    # 🆕 صفّر آخر delta عند كالبريشن جديد
    with _LAST_DELTA_LOCK:
        _LAST_DELTA["x"] = 0.0
        _LAST_DELTA["y"] = 0.0
        _LAST_DELTA["z"] = 0.0

    _IK_NEUTRAL = None
    _first_cycle = True
    _smoothed_target = [HOME_X, HOME_Y, HOME_Z]

    print("=" * 60)
    print(f"📍 NEUTRAL CAPTURED")
    print(f"   Hand raw: x={hx:+.3f}  y={hy:+.3f}  z={hz:+.3f} m")
    print("=" * 60)
    return True

# ============================================================
# FINGER GESTURE
# ============================================================
def apply_finger_gesture(gesture_name, live_features):
    if gesture_name == "open":
        _FINGER_STATE["j3"] = HOME_J3
        _FINGER_STATE["j6"] = 100
    elif gesture_name == "close":
        _FINGER_STATE["j3"] = 140
        _FINGER_STATE["j6"] = 0
    elif gesture_name == "grab":
        grab  = float(np.clip(live_features[6], 0.0, 1.0))
        pinch = float(np.clip(live_features[7], 0.0, 1.0))
        _FINGER_STATE["j3"] = int(np.clip(round(grab * 140), 0, 140))
        _FINGER_STATE["j6"] = int(np.clip(round((1.0 - pinch) * 100), 0, 100))
    else:
        return False

    print(f"\n🎯 FINGER: {gesture_name.upper()} → J3={_FINGER_STATE['j3']}°, J6={_FINGER_STATE['j6']}°\n")
    return True

class HandGestureModel(nn.Module):
    def __init__(self, input_features=14, num_classes=9,
                 gru_hidden_size=38, gru_num_layers=1,
                 attention_heads=2,
                 kan_width=None,
                 kan_grid=3, kan_k=3,
                 dropout=0.4):
        super().__init__()

        if kan_width is None:
            kan_width = [gru_hidden_size, gru_hidden_size // 2, num_classes]

        # Normalize kan_width to flat list of ints
        kan_width = list(kan_width)
        if len(kan_width) > 0 and isinstance(kan_width[0], (list, tuple, np.ndarray)):
            kan_width = [int(w[0]) for w in kan_width]
        else:
            kan_width = [int(w) for w in kan_width]

        assert kan_width[0] == gru_hidden_size, \
            f"kan_width[0]={kan_width[0]} must equal gru_hidden_size={gru_hidden_size}"

        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(input_features)

        self.input_dropout = nn.Dropout(0.15)
        self.gru_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Attention Pooling
        self.pool_attention = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(gru_hidden_size // 2, 1)
        )

        # Residual Projection
        self.residual_proj = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size),
            nn.LayerNorm(gru_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.kan = KAN(
            width=kan_width,
            grid=kan_grid,
            k=kan_k,
            seed=42
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.input_dropout(x)

        gru_out, _ = self.gru(x)
        gru_out = self.gru_dropout(gru_out)

        attn_out, _ = self.attention(gru_out, gru_out, gru_out)

        weights = self.pool_attention(attn_out)
        weights = torch.softmax(weights, dim=1)
        context = (attn_out * weights).sum(dim=1)

        context = self.dropout(context)
        context = context + self.residual_proj(context)

        return self.kan(context)

# ============================================================
# 3D CAMERA
# ============================================================
class HighRes3DCamera:
    def __init__(self, width=800, height=600):
        self.w = width
        self.h = height
        self.fc = 0

    def generate_3d_frame(self, hands, fc, gesture=""):
        frame = np.zeros((self.h, self.w, 3), np.uint8)
        for y in range(self.h):
            c = 20 + int(10 * np.sin(y / 50 + fc / 20))
            frame[y, :] = [c, c, c + 10]
        # 🔥 تم إزالة الشبكة (المربعات) من الخلفية
        self._draw_axes(frame)
        if hands:
            for h in hands:
                self._draw_hand(frame, h)
        if gesture:
            self._draw_gesture(frame, gesture)
        self._draw_info(frame, fc, len(hands) if hands else 0)
        return frame

    def _draw_grid(self, f):
        # ⚠️ غير مُستخدَم حالياً
        cx, cy = self.w // 2, self.h // 2
        gs, st = 200, 20
        for i in range(-gs, gs + 1, st):
            alpha = 1 - abs(i) / gs
            col = int(80 * alpha)
            cv2.line(f, (cx - gs, cy + i), (cx + gs, cy + i), (col, col, col), 1)
            cv2.line(f, (cx + i, cy - gs), (cx + i, cy + gs), (col, col, col), 1)

    def _draw_axes(self, f):
        cx, cy = self.w // 2, self.h // 2
        L = 80
        cv2.arrowedLine(f, (cx, cy), (cx + L, cy), (0, 0, 255), 2, tipLength=0.1)
        cv2.putText(f, "X", (cx + L + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.arrowedLine(f, (cx, cy), (cx, cy - L), (0, 255, 0), 2, tipLength=0.1)
        cv2.putText(f, "Y", (cx, cy - L - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.arrowedLine(f, (cx, cy), (cx - L, cy + L), (255, 0, 0), 2, tipLength=0.1)
        cv2.putText(f, "Z", (cx - L - 5, cy + L + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def _draw_hand(self, f, hand):
        cx, cy = self.w // 2, self.h // 2
        scale = 2.0
        def proj(x, y, z):
            ry, rz = z, -y
            xr, yr, zr = x, ry, rz
            pf = 1 + (zr * 0.002)
            ds = scale * pf
            sx = int(cx + xr * ds + zr * 0.5)
            sy = int(cy + yr * ds + zr * 0.1)
            if -100 <= sx < self.w + 100 and -100 <= sy < self.h + 100:
                return (sx, sy)
            return None
        palm = hand.palm.position
        pp = proj(palm.x, palm.y, palm.z)
        if pp:
            cv2.circle(f, pp, 12, (0, 255, 255), -1)
            cv2.circle(f, pp, 12, (255, 255, 255), 2)
        colors = [(0, 165, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 255, 0)]
        for fi, dig in enumerate(hand.digits):
            col = colors[fi % len(colors)]
            for bi in range(4):
                b = dig.bones[bi]
                sp = proj(b.prev_joint.x, b.prev_joint.y, b.prev_joint.z)
                ep = proj(b.next_joint.x, b.next_joint.y, b.next_joint.z)
                if sp and ep:
                    cv2.line(f, sp, ep, col, max(1, 4 - bi))
                    cv2.circle(f, sp, max(2, 6 - bi), col, -1)
                    cv2.circle(f, ep, max(2, 6 - bi), col, -1)

    def _draw_gesture(self, f, txt):
        overlay = f.copy()
        cv2.rectangle(overlay, (self.w - 350, 10), (self.w - 10, 110), (0, 0, 0), -1)
        f = cv2.addWeighted(overlay, 0.6, f, 0.4, 0)
        cv2.putText(f, "GESTURE:", (self.w - 340, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(f, txt, (self.w - 340, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        active = _ACTIVE_GESTURE["name"]
        if active:
            cv2.putText(f, f"ACTIVE: {active.upper()}", (self.w - 340, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return f

    def _draw_info(self, f, fc, hc):
        cv2.putText(f, "ULTRALEAP + GRU+ATT+KAN", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(f, f"Hands: {hc}  Frame: {fc}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

# ============================================================
# TRACKING LISTENER
# ============================================================
class TrackingListener(leap.Listener):
    def __init__(self, app):
        self.app = app
        self.fps_counter = 0
        self.last_time = time.time()

    def on_tracking_event(self, event):
        self.fps_counter += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.app.update_status(f"Tracking FPS: {self.fps_counter}")
            self.fps_counter = 0
            self.last_time = now

        hands = event.hands if event.hands else []
        self.app.update_hand_count(len(hands))

        if hands:
            best = max(hands, key=lambda h: h.confidence)
            feat = self.app.extract_features(best)
            if feat is not None:
                with _LATEST_HAND_LOCK:
                    _LATEST_HAND["features"] = feat
                    _LATEST_HAND["time"] = time.time()
            self.app.queue_hand_for_inference(best)

        self.app.last_hands = hands

# ============================================================
# MAIN APP
# ============================================================
class UltraleapRealTimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultraleap Real‑Time Gesture Recognition + Robot Control")
        self.root.geometry("1300x900")

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Inference device: {self.device}")

        self.lock = threading.Lock()
        self.frame_counter = 0
        self.feature_buffer = deque(maxlen=Config.WINDOW_SIZE)
        self.prediction_history = deque(maxlen=Config.SMOOTHING_WINDOW)
        self.current_gesture = "Initialising..."
        self.current_confidence = 0.0
        self.window_ready = False

        self.cooldown_until = 0.0
        self.still_counter = 0
        self.last_motion_time = 0.0

        # 🔒 Hysteresis لإيماءات الأصابع
        self._last_finger_gesture = None
        self._finger_gesture_streak = 0

        self.inference_lock = threading.Lock()
        self.pending_hand = None
        self.inference_event = threading.Event()
        self.inference_thread = None
        self.running = True

        self.connection = None
        self.listener = None
        self.leap_thread = None
        self.last_hands = []

        self.camera_running = False
        self.camera_thread = None

        self.camera_3d = HighRes3DCamera(width=1200, height=800)

        self._last_ui_update = 0.0
        self._last_buffer_ui_update = 0.0
        self._last_conf_print = 0.0

        global ser
        ser = None
        try:
            ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
            time.sleep(2)
            print(f"✅ Serial connected to {SERIAL_PORT}")
        except Exception as e:
            print(f"❌ Failed to open serial: {e}")
            print(f"   → تأكد من توصيل الأردوينو وضبط SERIAL_PORT")

        self.setup_ui()
        self.load_model()
        self.start_inference_worker()
        self.init_leap_motion()
        start_realtime()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        ttk.Label(left_panel, text="Device Information", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        ttk.Label(left_panel, text="Device: Leap Motion Controller").pack(anchor=tk.W)
        ttk.Label(left_panel, text="Model: GRU + Attention + KAN").pack(anchor=tk.W)
        ttk.Label(left_panel, text=f"Numpy: {np.__version__}").pack(anchor=tk.W)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=5)

        ttk.Label(left_panel, text="Camera Control", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        ttk.Button(left_panel, text="Start Camera", command=self.start_camera).pack(fill=tk.X, pady=2)
        ttk.Button(left_panel, text="Stop Camera", command=self.stop_camera).pack(fill=tk.X, pady=2)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=5)

        ttk.Label(left_panel, text="Calibration", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        ttk.Button(left_panel, text="🎯 التقاط Neutral (يد وسط)",
                   command=self.capture_neutral).pack(fill=tk.X, pady=2)
        ttk.Button(left_panel, text="🔁 إعادة كالبريشن",
                   command=self.recalibrate).pack(fill=tk.X, pady=2)
        ttk.Button(left_panel, text="🏠 العودة للـ HOME",
                   command=self.go_home).pack(fill=tk.X, pady=2)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=5)

        ttk.Label(left_panel, text="Status", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        self.status_label = ttk.Label(left_panel, text="Initialising...")
        self.status_label.pack(anchor=tk.W)
        self.hands_label = ttk.Label(left_panel, text="Hands: 0")
        self.hands_label.pack(anchor=tk.W)
        self.active_label = ttk.Label(left_panel, text="Active: ---",
                                       foreground="blue", font=('Arial', 10, 'bold'))
        self.active_label.pack(anchor=tk.W)
        self.buffer_label = ttk.Label(left_panel, text=f"Buffer: 0/{Config.WINDOW_SIZE}")
        self.buffer_label.pack(anchor=tk.W)
        self.motion_label = ttk.Label(left_panel, text="Motion: ---", foreground="gray")
        self.motion_label.pack(anchor=tk.W)
        self.finger_label = ttk.Label(left_panel, text="Fingers: J3=0  J6=90",
                                       foreground="purple", font=('Arial', 9, 'bold'))
        self.finger_label.pack(anchor=tk.W)
        self.neutral_label = ttk.Label(left_panel, text="Neutral: NOT SET",
                                        foreground="red", font=('Arial', 9, 'bold'))
        self.neutral_label.pack(anchor=tk.W)

        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=5)

        ttk.Label(left_panel, text="Real‑time Gesture", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        self.gesture_label = ttk.Label(left_panel, text="---", font=('Arial', 14, 'bold'), foreground="green")
        self.gesture_label.pack(anchor=tk.W)
        self.confidence_label = ttk.Label(left_panel, text="Confidence: 0.00")

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        camera_frame = ttk.LabelFrame(right_panel, text="3D Hand Tracking", padding="5")
        camera_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(camera_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_text(400, 300, text="Camera not started", fill="white", font=('Arial', 14), tag="info_text")

    def _update_finger_label(self):
        try:
            self.finger_label.config(text=f"Fingers: J3={_FINGER_STATE['j3']}  J6={_FINGER_STATE['j6']}")
        except Exception:
            pass

    def _update_neutral_label(self):
        try:
            with _NEUTRAL_LOCK:
                nx = _NEUTRAL["x"]
                if nx is not None:
                    self.neutral_label.config(
                        text=f"Neutral: x={_NEUTRAL['x']:+.3f}  y={_NEUTRAL['y']:+.3f}",
                        foreground="green")
                else:
                    self.neutral_label.config(text="Neutral: NOT SET", foreground="red")
        except Exception:
            pass

    def capture_neutral(self):
        if capture_neutral_from_current():
            self._update_neutral_label()
            messagebox.showinfo("Calibration", "✅ Neutral تم التقاطه!\nالآن حرّك يدك.")
        else:
            messagebox.showwarning("Calibration", "❌ لا توجد بيانات من اليد.\nمد يدك فوق Leap ثم اضغط الزر.")

    def recalibrate(self):
        global _IK_NEUTRAL, _first_cycle, _last_sent_cmd, _smoothed_target
        global _current_arm_angles, _current_arm_vels, _ema_targets

        with _NEUTRAL_LOCK:
            _NEUTRAL["x"] = None
            _NEUTRAL["y"] = None
            _NEUTRAL["z"] = None

        # 🆕 صفّر آخر delta
        with _LAST_DELTA_LOCK:
            _LAST_DELTA["x"] = 0.0
            _LAST_DELTA["y"] = 0.0
            _LAST_DELTA["z"] = 0.0

        _IK_NEUTRAL = None
        _first_cycle = True
        _smoothed_target = [HOME_X, HOME_Y, HOME_Z]
        _last_sent_cmd = None
        _ACTIVE_GESTURE["name"] = None

        _current_arm_angles = [float(HOME_J1), float(HOME_J2), float(HOME_J5)]
        _current_arm_vels = [0.0, 0.0, 0.0]
        _ema_targets = list(_current_arm_angles)

        self.still_counter = 0
        self.last_motion_time = 0.0

        self._last_finger_gesture = None
        self._finger_gesture_streak = 0

        with self.lock:
            self.prediction_history.clear()
            self.feature_buffer.clear()

        self.current_gesture = "idle"
        self.current_confidence = 0.0
        self.root.after(0, self.update_gesture_display)
        self.root.after(0, self.update_active_label)
        self.root.after(0, lambda: self.motion_label.config(text="Motion: ---", foreground="gray"))
        self.root.after(0, self._update_finger_label)
        self.root.after(0, self._update_neutral_label)
        print("\n🔁 Calibration RESET\n")

    def go_home(self):
        global _IK_NEUTRAL, _first_cycle, _current_arm_angles, _current_arm_vels, _ema_targets, _last_sent_cmd, _smoothed_target

        with _NEUTRAL_LOCK:
            _NEUTRAL["x"] = None
            _NEUTRAL["y"] = None
            _NEUTRAL["z"] = None

        # 🆕 صفّر آخر delta
        with _LAST_DELTA_LOCK:
            _LAST_DELTA["x"] = 0.0
            _LAST_DELTA["y"] = 0.0
            _LAST_DELTA["z"] = 0.0

        _IK_NEUTRAL = None
        _FINGER_STATE["j3"] = HOME_J3
        _FINGER_STATE["j6"] = HOME_J6
        _ACTIVE_GESTURE["name"] = None
        _first_cycle = True
        _smoothed_target = [HOME_X, HOME_Y, HOME_Z]

        _current_arm_angles = [float(HOME_J1), float(HOME_J2), float(HOME_J5)]
        _current_arm_vels = [0.0, 0.0, 0.0]
        _ema_targets = list(_current_arm_angles)

        self.still_counter = 0

        cmd = f"{HOME_J1},{HOME_J2},{HOME_J3},{HOME_J4},{HOME_J5},{HOME_J6}\n"
        with _SERIAL_LOCK:
            if ser and ser.is_open:
                try:
                    ser.write(cmd.encode())
                    _last_sent_cmd = cmd
                    print(f"🏠 HOME → {cmd.strip()}")
                except Exception as e:
                    print(f"❌ Serial write error: {e}")
            else:
                print("❌ Serial not available")
        self.root.after(0, self._update_finger_label)
        self.root.after(0, self._update_neutral_label)

    def start_inference_worker(self):
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()

    def queue_hand_for_inference(self, hand):
        with self.inference_lock:
            self.pending_hand = hand
        self.inference_event.set()

    def _inference_worker(self):
        while self.running:
            if not self.inference_event.wait(timeout=0.1):
                continue
            self.inference_event.clear()
            with self.inference_lock:
                hand = self.pending_hand
                self.pending_hand = None
            if hand is None:
                continue
            try:
                self._do_inference(hand)
            except Exception as e:
                print(f"Inference worker error: {e}")

    def _do_inference(self, hand):
        if self.model is None:
            return

        feat = self.extract_features(hand)
        if feat is None:
            return

        vx, vy, vz = feat[3], feat[4], feat[5]
        vel_mag = math.sqrt(vx*vx + vy*vy + vz*vz)
        is_moving = vel_mag >= Config.MIN_VELOCITY

        if not is_moving:
            self.still_counter += 1
            now_ui = time.time()
            if now_ui - self._last_ui_update > 0.15:
                self._last_ui_update = now_ui
                sc = self.still_counter
                self.root.after(0, lambda: self.motion_label.config(
                    text=f"Motion: STILL ({sc})", foreground="gray"))

            if self.still_counter >= Config.STILL_FRAMES_TO_CLEAR:
                _ACTIVE_GESTURE["name"] = None
                if self.current_gesture not in ("idle", "Initialising...",
                                                "open", "close", "grab"):
                    self.current_gesture = "idle"
                    self.current_confidence = 0.0
                    self.root.after(0, self.update_gesture_display)
                    self.root.after(0, self.update_active_label)
        else:
            self.still_counter = 0
            self.last_motion_time = time.time()
            now_ui = time.time()
            if now_ui - self._last_ui_update > 0.15:
                self._last_ui_update = now_ui
                vm = vel_mag
                self.root.after(0, lambda: self.motion_label.config(
                    text=f"Motion: {vm:.2f} m/s", foreground="green"))

        with self.lock:
            self.frame_counter += 1
            if Config.SAMPLE_RATE > 1 and self.frame_counter % Config.SAMPLE_RATE != 0:
                return

            self.feature_buffer.append(feat)

            now_ui = time.time()
            if now_ui - self._last_buffer_ui_update > 0.2:
                self._last_buffer_ui_update = now_ui
                blen = len(self.feature_buffer)
                self.root.after(0, lambda: self.buffer_label.config(
                    text=f"Buffer: {blen}/{Config.WINDOW_SIZE}"))

            if len(self.feature_buffer) < Config.WINDOW_SIZE:
                return

            window = np.array(self.feature_buffer, dtype=np.float32)
            window_positions = window[:, :3]
            pos_range = np.ptp(window_positions, axis=0)
            motion_range = float(np.linalg.norm(pos_range))

            if motion_range < Config.MIN_MOTION_RANGE:
                finger_range = np.ptp(window[:, 6:9], axis=0)
                finger_motion = float(np.linalg.norm(finger_range))
                if finger_motion < 0.005:
                    return

            try:
                norm_window = self.scaler.transform(window)
            except Exception as e:
                print(f"Scaling error: {e}")
                return

            input_tensor = torch.FloatTensor(norm_window).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.inference_mode():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(logits, dim=1).item()
                confidence = probs[0, pred_idx].item()

            gesture = self.label_encoder.inverse_transform([pred_idx])[0]

            if time.time() - self._last_conf_print > 0.5:
                self._last_conf_print = time.time()
                print(f"🔍 conf={confidence:.3f}  gesture={gesture}")

            if confidence < Config.CONFIDENCE_THRESHOLD:
                return

            self.prediction_history.append(gesture)

            if len(self.prediction_history) >= Config.SMOOTHING_WINDOW:
                counter = Counter(self.prediction_history)
                most_common_gesture, count = counter.most_common(1)[0]

                if count >= Config.CONFIRMATION_THRESHOLD:
                    now = time.time()
                    if now < self.cooldown_until:
                        return

                    arm_gestures = ["push", "pull", "left", "right", "raise", "lower"]
                    if most_common_gesture in arm_gestures and not is_moving:
                        return

                    # ═══════════════════════════════════════════════════════
                    # 🛡️ Semantic Guard — امنع close/open الكاذبة
                    # ═══════════════════════════════════════════════════════
                    if most_common_gesture == "close":
                        grab  = float(feat[6])
                        pinch = float(feat[7])
                        if grab < Config.CLOSE_GRAB_MAX and pinch < Config.CLOSE_PINCH_MAX:
                            print(f"🛡️  رفض close كاذبة (grab={grab:.2f}, pinch={pinch:.2f})")
                            return
                    elif most_common_gesture == "open":
                        grab  = float(feat[6])
                        pinch = float(feat[7])
                        if grab > Config.OPEN_GRAB_MIN or pinch > Config.OPEN_PINCH_MIN:
                            print(f"🛡️  رفض open كاذبة (grab={grab:.2f}, pinch={pinch:.2f})")
                            return

                    finger_gestures = ["close", "grab", "open"]
                    if most_common_gesture in finger_gestures:
                        if most_common_gesture == self._last_finger_gesture:
                            self._finger_gesture_streak += 1
                        else:
                            self._last_finger_gesture = most_common_gesture
                            self._finger_gesture_streak = 1

                        if self._finger_gesture_streak < Config.FINGER_STREAK_REQUIRED:
                            print(f"🔒 Hysteresis: {most_common_gesture} ({self._finger_gesture_streak}/{Config.FINGER_STREAK_REQUIRED})")
                            return
                    else:
                        self._last_finger_gesture = None
                        self._finger_gesture_streak = 0
                    # ═══════════════════════════════════════════════════════

                    self.current_gesture = most_common_gesture
                    self.current_confidence = confidence
                    self.window_ready = True
                    self.root.after(0, self.update_gesture_display)

                    set_active_gesture(most_common_gesture, confidence)
                    self.root.after(0, self.update_active_label)

                    if most_common_gesture in finger_gestures:
                        apply_finger_gesture(most_common_gesture, feat)
                        _send_current_cmd(force=True, label="FINGER")
                        self.root.after(0, self._update_finger_label)
                        print(f"✅ Activated FINGER: {most_common_gesture} (conf={confidence:.2f})")
                    else:
                        print(f"✅ Activated ARM: {most_common_gesture} (conf={confidence:.2f})")

                    # 🔥 الاحتفاظ بآخر 15 فريم بدل تفريغ البَفر كاملاً
                    self.prediction_history.clear()

                    KEEP_FRAMES = 15
                    if len(self.feature_buffer) > KEEP_FRAMES:
                        recent = list(self.feature_buffer)[-KEEP_FRAMES:]
                        self.feature_buffer.clear()
                        self.feature_buffer.extend(recent)

                    self.cooldown_until = now + Config.COOLDOWN_SECONDS

                    blen = len(self.feature_buffer)
                    self.root.after(0, lambda: self.buffer_label.config(
                        text=f"Buffer: {blen}/{Config.WINDOW_SIZE}"))

    def update_active_label(self):
        name = _ACTIVE_GESTURE["name"]
        if name:
            self.active_label.config(text=f"Active: {name.upper()}")
        else:
            self.active_label.config(text="Active: ---")

    def extract_features(self, hand):
        try:
            px = float(hand.palm.position.x)
            py = float(hand.palm.position.y)
            pz = float(hand.palm.position.z)

            vx = float(hand.palm.velocity.x)
            vy = float(hand.palm.velocity.y)
            vz = float(hand.palm.velocity.z)

            grab  = float(hand.grab_strength)
            pinch = float(hand.pinch_strength)
            pdist = float(hand.pinch_distance)

            thumb  = float(hand.thumb.is_extended)
            index  = float(hand.index.is_extended)
            middle = float(hand.middle.is_extended)
            ring   = float(hand.ring.is_extended)
            pinky  = float(hand.pinky.is_extended)
        except Exception:
            return None

        if not (0.0 <= grab <= 1.5) or not (0.0 <= pinch <= 1.5):
            return None

        grab  = float(np.clip(grab,  0.0, 1.0))
        pinch = float(np.clip(pinch, 0.0, 1.0))

        if abs(px) > 1000.0 or abs(py) > 1000.0 or abs(pz) > 1000.0:
            return None

        return np.array([
            px, py, pz,
            vx, vy, vz,
            grab, pinch, pdist,
            thumb, index, middle, ring, pinky
        ], dtype=np.float32)

    def update_gesture_display(self):
        self.gesture_label.config(text=self.current_gesture.upper())
        self.confidence_label.config(text=f"Confidence: {self.current_confidence:.2f}")


    def load_model(self):
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            messagebox.showerror("Model not found", f"File not found: {Config.MODEL_SAVE_PATH}")
            self.root.destroy()
            return
        try:
            checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device, weights_only=False)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model.\n{str(e)}")
            self.root.destroy()
            return

        num_classes = len(checkpoint['label_encoder_classes'])
        saved_cfg = checkpoint.get('config', {})

        # قراءة المعمارية من الـ checkpoint (تدعم مفاتيح النسختين)
        gru_hidden   = saved_cfg.get('gru_hidden_size', Config.GRU_HIDDEN_SIZE)
        gru_layers   = saved_cfg.get('gru_num_layers', Config.GRU_NUM_LAYERS)
        attn_heads   = saved_cfg.get('attention_heads', Config.ATTENTION_HEADS)
        kan_grid     = saved_cfg.get('kan_grid', saved_cfg.get('kan_grid_size', Config.KAN_GRID_SIZE))
        kan_k        = saved_cfg.get('kan_spline_order', Config.KAN_SPLINE_ORDER)
        dropout_rate = saved_cfg.get('dropout_rate', Config.DROPOUT_RATE)

        # ⭐⭐⭐ معالجة kan_width بكلا الشكلين: [38,19,9] أو [[38,0],[19,0],[9,0]]
        kan_width = saved_cfg.get('kan_width', None)

        if kan_width is None:
            kan_width = [gru_hidden, gru_hidden // 2, num_classes]
        else:
            kan_width = list(kan_width)
            if len(kan_width) > 0 and isinstance(kan_width[0], (list, tuple, np.ndarray)):
                kan_width = [int(w[0]) for w in kan_width]
            else:
                kan_width = [int(w) for w in kan_width]

        # Safety check
        if kan_width[0] != gru_hidden or kan_width[-1] != num_classes:
            print(f"⚠️  Rebuilding kan_width: gru_hidden={gru_hidden}, classes={num_classes}")
            kan_width = [gru_hidden, gru_hidden // 2, num_classes]

        print(f"\n📋 Architecture from checkpoint:")
        print(f"   GRU hidden   = {gru_hidden}")
        print(f"   GRU layers   = {gru_layers}")
        print(f"   Attn heads   = {attn_heads}")
        print(f"   KAN width    = {kan_width}")
        print(f"   KAN grid     = {kan_grid}  |  KAN k = {kan_k}")
        print(f"   Dropout      = {dropout_rate}")

        self.model = HandGestureModel(
            input_features=Config.FEATURES,
            num_classes=num_classes,
            gru_hidden_size=gru_hidden,
            gru_num_layers=gru_layers,
            attention_heads=attn_heads,
            kan_width=kan_width,
            kan_grid=kan_grid,
            kan_k=kan_k,
            dropout=dropout_rate
        ).to(self.device)

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            messagebox.showerror("Weights Load Error", f"Failed to load weights.\n{str(e)}")
            self.root.destroy()
            return

        self.model.eval()

        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
                print("⚡ Model converted to FP16")
            except Exception:
                pass

        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = checkpoint['label_encoder_classes']
        self.scaler = StandardScaler()
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']

        if saved_cfg:
            ws = saved_cfg.get('window_size', Config.WINDOW_SIZE)
            if ws != Config.WINDOW_SIZE:
                Config.WINDOW_SIZE = ws
                self.feature_buffer = deque(maxlen=Config.WINDOW_SIZE)

        print(f"\n✅ Model loaded:")
        print(f"   Classes ({num_classes}): {list(self.label_encoder.classes_)}")
        print(f"   WINDOW_SIZE = {Config.WINDOW_SIZE}")
        print(f"   🎯 Cylindrical IK + Filtering")
        print(f"   🧊 المحاور غير المفلترة تحتفظ بآخر قيمة")
        print(f"   🛡️  Semantic Guard: ACTIVE")
        print(f"   🔒 Hysteresis: {Config.FINGER_STREAK_REQUIRED} تأكيدات متتالية\n")

        self.update_status(f"Model loaded: {num_classes} gestures")

    def init_leap_motion(self):
        try:
            self.connection = leap.Connection()
            self.listener = TrackingListener(self)
            self.connection.add_listener(self.listener)
            self.leap_thread = threading.Thread(target=self._leap_loop, daemon=True)
            self.leap_thread.start()
            self.update_status("Leap Motion connected")
        except Exception as e:
            self.update_status(f"Leap error: {e}")

    def _leap_loop(self):
        with self.connection.open():
            self.connection.set_tracking_mode(leap.TrackingMode.Desktop)
            while self.running:
                time.sleep(0.01)

    def start_camera(self):
        if not self.camera_running:
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            self.update_status("3D Camera started")
            self.canvas.delete("info_text")

    def stop_camera(self):
        self.camera_running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        self.update_status("3D Camera stopped")
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="Camera stopped", fill="white", font=('Arial', 14), tag="info_text")

    def _camera_loop(self):
        frame_counter = 0
        interval = 1.0 / 30.0
        last_time = time.perf_counter()
        while self.camera_running:
            now = time.perf_counter()
            if now - last_time < interval:
                time.sleep(max(0, interval - (now - last_time) - 0.001))
                continue
            with self.lock:
                if self.window_ready:
                    gesture_text = f"{self.current_gesture.upper()}  ({self.current_confidence:.2f})"
                else:
                    gesture_text = "Initialising..."
            frame = self.camera_3d.generate_3d_frame(self.last_hands, frame_counter, gesture_text)
            frame_counter += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            try:
                self.root.after(0, self._update_canvas, imgtk)
            except Exception:
                break
            last_time = now

    def _update_canvas(self, imgtk):
        if not self.camera_running:
            return
        try:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk
        except Exception:
            pass

    def update_status(self, msg):
        try:
            self.status_label.config(text=msg)
        except Exception:
            pass

    def update_hand_count(self, count):
        try:
            self.hands_label.config(text=f"Hands: {count}")
        except Exception:
            pass

    def on_closing(self):
        self.running = False
        stop_realtime()
        self.stop_camera()
        if self.connection:
            try:
                self.connection.remove_listener(self.listener)
                self.connection.close()
            except Exception:
                pass
        global ser
        if ser and ser.is_open:
            ser.close()
        try:
            self.root.destroy()
        except Exception:
            pass

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  ULTRALEAP + GRU + ATTENTION + KAN + ROBOT ARM")
    print("  🎯 CYLINDRICAL IK + FILTERING")
    print("  🧊 المحاور غير المفلترة تحتفظ بآخر قيمة")
    print("  🛡️  SEMANTIC GUARD + 🔒 HYSTERESIS")
    print("  🧠 Attention Pooling + Residual (Architecture v2)")
    print("=" * 60)
    print(f"  left/right  → x فقط (y, z = آخر قيمة محفوظة)")
    print(f"  raise/lower → y فقط (x, z = آخر قيمة محفوظة)")
    print(f"  push/pull   → z فقط (x, y = آخر قيمة محفوظة)")
    print(f"  🧹 Keep last 15 frames بعد كل إيماءة")
    print("=" * 60)
    print(f"  SERVO speed = {Config.SERVO_MAX_SPEED_DEG_PER_SEC}°/s")
    print(f"  EMA alpha = {Config.SERVO_EMA_ALPHA}")
    print(f"  Conf threshold = {Config.CONFIDENCE_THRESHOLD}")
    print(f"  Confirmations required = {Config.CONFIRMATION_THRESHOLD}")
    print(f"  Finger streak required = {Config.FINGER_STREAK_REQUIRED}")
    print("=" * 60 + "\n")

    root = tk.Tk()
    app = UltraleapRealTimeGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
        app.on_closing()
    sys.exit(0)

if __name__ == "__main__":
    main()
