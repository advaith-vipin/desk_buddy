"""Clean pixelated robot face: centered, polished, expressive.

A minimal robot face with glowing eyes and mouth on transparent background.
Eyes change per mood, mouth driven by viseme sequence for lip-sync.
Natural blinking, saccades, micro-expressions, emotional mouth shapes.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

import pytoon

_VISEME_DIR = os.path.join(os.path.dirname(pytoon.__file__), "assets", "visemes", "positive")

# Colors - clean robot palette
EYE_COLOR = (80, 220, 255)       # Bright cyan
EYE_CORE = (200, 245, 255)       # White-hot center
EYE_DIM = (40, 140, 180)         # Dim glow
BROW_COLOR = (100, 200, 255)     # Brow accent
MOUTH_COLOR = (80, 220, 255)     # Matching cyan
MOUTH_CORE = (220, 250, 255)     # Bright core
MOUTH_DIM = (30, 120, 160)       # Dim glow
PINK = (255, 110, 180)           # Hearts
TEAR = (80, 190, 255)            # Tears

_MOUTH_OPEN: dict[str, float] = {}
_MOUTH_MINMAX: tuple[float, float] | None = None


def _mouth_minmax() -> tuple[float, float]:
    global _MOUTH_MINMAX
    if _MOUTH_MINMAX is None:
        raws = [_viseme_raw(f"{i}.png") for i in range(1, 12)]
        _MOUTH_MINMAX = (min(raws), max(raws))
    return _MOUTH_MINMAX


def _viseme_raw(key: str) -> float:
    try:
        img = Image.open(os.path.join(_VISEME_DIR, key)).convert("RGBA")
        g = np.asarray(img)
        alpha = g[:, :, 3] if g.shape[2] == 4 else np.full(g.shape[:2], 255)
        dark = (g[:, :, :3].mean(axis=2) < 200) & (alpha > 128)
        rows = np.where(dark.any(axis=1))[0]
        if len(rows):
            return float((rows[-1] - rows[0]) / img.height)
    except Exception:
        pass
    return 0.3


def _viseme_openness(name: str) -> float:
    key = os.path.basename(name or "")
    if key in _MOUTH_OPEN:
        return _MOUTH_OPEN[key]
    lo, hi = _mouth_minmax()
    span = max(hi - lo, 1e-6)
    openness = float(np.clip((_viseme_raw(key) - lo) / span, 0.0, 1.0))
    _MOUTH_OPEN[key] = openness
    return openness


# Viseme to mouth shape mapping for more realistic lip sync
_VISEME_SHAPES = {
    # viseme_name: (width_mult, height_mult, corner_pull, upper_lip_raise, lower_lip_drop, teeth_show)
    "1.png":  (0.8, 0.15, 0.0, 0.0, 0.0, False),   # sil - closed
    "2.png":  (0.9, 0.15, 0.0, 0.0, 0.0, False),   # PP - closed
    "3.png":  (0.9, 0.15, 0.0, 0.0, 0.0, False),   # FF - closed
    "4.png":  (1.0, 0.35, 0.1, 0.1, 0.2, True),    # TH - teeth
    "5.png":  (1.1, 0.55, 0.0, 0.2, 0.5, False),   # DD - open
    "6.png":  (1.2, 0.75, 0.0, 0.3, 0.7, False),   # kk - wide open
    "7.png":  (0.9, 0.40, 0.2, 0.0, 0.3, False),   # CH - rounded
    "8.png":  (0.9, 0.45, 0.1, 0.1, 0.4, False),   # SS - narrow
    "9.png":  (1.1, 0.65, 0.0, 0.2, 0.6, False),   # nn - open
    "10.png": (1.0, 0.50, 0.1, 0.1, 0.4, False),   # RR - mid
    "11.png": (1.0, 0.45, 0.0, 0.1, 0.4, False),   # aa - open
}


# Natural blinking state machine
_BLINK_STATE = {
    "phase": "open",        # "open", "closing", "closed", "opening"
    "next_blink": 0.0,      # time of next blink
    "blink_progress": 0.0,  # 0.0 to 1.0
    "blink_speed": 0.0,     # frames per phase
}

# Saccade (quick eye movement) state machine
_SACCADE_STATE = {
    "active": False,
    "progress": 0.0,
    "duration": 0,
    "start_glance": 0.0,
    "target_glance": 0.0,
    "next_saccade": 0.0,
}

# Micro-expression state
_MICRO_STATE = {
    "brow_offset": 0.0,     # subtle brow movement
    "eye_squint": 0.0,      # subtle squint
    "mouth_tension": 0.0,   # subtle mouth tension
    "timer": 0.0,
}


def _pupil_glance(frame_index: float, fps: int = 48) -> float:
    """Combined idle glance + saccades for natural eye movement."""
    global _SACCADE_STATE
    t = frame_index / fps

    # Handle saccade (quick gaze shift)
    if _SACCADE_STATE["active"]:
        _SACCADE_STATE["progress"] += 1.0 / _SACCADE_STATE["duration"]
        if _SACCADE_STATE["progress"] >= 1.0:
            _SACCADE_STATE["active"] = False
            _SACCADE_STATE["progress"] = 0.0
            _SACCADE_STATE["next_saccade"] = t + np.random.uniform(2.0, 5.0)
            return _SACCADE_STATE["target_glance"]
        # Saccade uses fast ease-out
        eased = 1.0 - (1.0 - _SACCADE_STATE["progress"]) ** 3
        return _SACCADE_STATE["start_glance"] + (_SACCADE_STATE["target_glance"] - _SACCADE_STATE["start_glance"]) * eased

    # Check for new saccade
    if t >= _SACCADE_STATE["next_saccade"]:
        _SACCADE_STATE["active"] = True
        _SACCADE_STATE["start_glance"] = _SACCADE_STATE.get("current_glance", 0.0)
        _SACCADE_STATE["target_glance"] = np.random.uniform(-0.35, 0.35)
        _SACCADE_STATE["duration"] = int(np.random.uniform(3, 8))  # very fast
        _SACCADE_STATE["progress"] = 0.0
        _SACCADE_STATE["current_glance"] = _SACCADE_STATE["target_glance"]
        return _pupil_glance(frame_index, fps)  # recurse to start saccade

    # Slow idle drift
    if not hasattr(_pupil_glance, "_idle_state"):
        _pupil_glance._idle_state = {
            "target": 0.0, "current": 0.0, "next_change": 0.0
        }
    idle = _pupil_glance._idle_state
    if t >= idle["next_change"]:
        idle["target"] = np.random.uniform(-0.15, 0.15)
        idle["next_change"] = t + np.random.uniform(4.0, 10.0)
    # Smooth drift toward target
    idle["current"] += (idle["target"] - idle["current"]) * 0.02
    _SACCADE_STATE["current_glance"] = idle["current"]
    return idle["current"]


def _blink_factor(frame_index: float, fps: int = 48) -> float:
    """Returns eyelid closure factor (0=open, 1=closed) with natural timing."""
    global _BLINK_STATE
    t = frame_index / fps

    if _BLINK_STATE["phase"] == "open":
        if t >= _BLINK_STATE["next_blink"]:
            _BLINK_STATE["phase"] = "closing"
            _BLINK_STATE["blink_progress"] = 0.0
            _BLINK_STATE["blink_speed"] = max(2, int(np.random.uniform(2, 4)))  # closing speed
        return 0.0

    elif _BLINK_STATE["phase"] == "closing":
        _BLINK_STATE["blink_progress"] += 1.0 / _BLINK_STATE["blink_speed"]
        if _BLINK_STATE["blink_progress"] >= 1.0:
            _BLINK_STATE["phase"] = "closed"
            _BLINK_STATE["blink_progress"] = 0.0
            _BLINK_STATE["blink_speed"] = max(1, int(np.random.uniform(1, 2)))  # hold closed
        return min(_BLINK_STATE["blink_progress"], 1.0)

    elif _BLINK_STATE["phase"] == "closed":
        _BLINK_STATE["blink_progress"] += 1.0 / _BLINK_STATE["blink_speed"]
        if _BLINK_STATE["blink_progress"] >= 1.0:
            _BLINK_STATE["phase"] = "opening"
            _BLINK_STATE["blink_progress"] = 0.0
            _BLINK_STATE["blink_speed"] = max(2, int(np.random.uniform(2, 5)))  # opening speed
        return 1.0

    else:  # opening
        _BLINK_STATE["blink_progress"] += 1.0 / _BLINK_STATE["blink_speed"]
        if _BLINK_STATE["blink_progress"] >= 1.0:
            _BLINK_STATE["phase"] = "open"
            # Next blink in 2-6 seconds (natural rate ~15-20/min)
            _BLINK_STATE["next_blink"] = t + np.random.uniform(2.0, 6.0)
        return max(1.0 - _BLINK_STATE["blink_progress"], 0.0)


def _micro_expressions(frame_index: float, fps: int = 48, mood: str = "neutral") -> dict:
    """Subtle micro-expressions that add life."""
    global _MICRO_STATE
    t = frame_index / fps

    # Update timer
    _MICRO_STATE["timer"] += 1.0 / fps

    # Periodic subtle changes
    if _MICRO_STATE["timer"] > np.random.uniform(0.5, 2.0):
        _MICRO_STATE["timer"] = 0.0
        # Mood-appropriate micro-expressions
        if mood == "sad" or mood == "lonely":
            _MICRO_STATE["brow_offset"] = np.random.uniform(-0.02, 0.05)  # inner brows up slightly
            _MICRO_STATE["eye_squint"] = np.random.uniform(0.0, 0.05)
            _MICRO_STATE["mouth_tension"] = np.random.uniform(-0.03, 0.0)
        elif mood == "angry":
            _MICRO_STATE["brow_offset"] = np.random.uniform(-0.05, 0.02)  # brows down
            _MICRO_STATE["eye_squint"] = np.random.uniform(0.02, 0.1)
            _MICRO_STATE["mouth_tension"] = np.random.uniform(0.0, 0.05)
        elif mood == "happy":
            _MICRO_STATE["brow_offset"] = np.random.uniform(-0.02, 0.02)
            _MICRO_STATE["eye_squint"] = np.random.uniform(0.0, 0.08)  # happy squint
            _MICRO_STATE["mouth_tension"] = np.random.uniform(-0.02, 0.03)
        elif mood == "excited":
            _MICRO_STATE["brow_offset"] = np.random.uniform(0.0, 0.05)
            _MICRO_STATE["eye_squint"] = np.random.uniform(0.0, 0.03)
            _MICRO_STATE["mouth_tension"] = np.random.uniform(0.0, 0.03)
        else:  # neutral, calm, etc.
            _MICRO_STATE["brow_offset"] = np.random.uniform(-0.015, 0.015)
            _MICRO_STATE["eye_squint"] = np.random.uniform(0.0, 0.02)
            _MICRO_STATE["mouth_tension"] = np.random.uniform(-0.01, 0.01)

    # Smooth interpolation toward targets
    return {
        "brow_offset": _MICRO_STATE["brow_offset"],
        "eye_squint": _MICRO_STATE["eye_squint"],
        "mouth_tension": _MICRO_STATE["mouth_tension"],
    }


_PIXEL_SCALE = 8


def _rect(d: ImageDraw.ImageDraw, cx: int, cy: int, w: int, h: int, color: tuple) -> None:
    x0, y0 = cx - w // 2, cy - h // 2
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=color)


def _hline(d: ImageDraw.ImageDraw, y: int, x0: int, x1: int, color: tuple) -> None:
    if x0 > x1: x0, x1 = x1, x0
    d.rectangle([x0, y, x1, y], fill=color)


def _rounded_rect(d: ImageDraw.ImageDraw, cx: int, cy: int, w: int, h: int, r: int, color: tuple) -> None:
    r = min(r, w // 2, h // 2)
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    d.rectangle([x0 + r, y0, x1 - r, y1], fill=color)
    d.rectangle([x0, y0 + r, x1, y1 - r], fill=color)
    for dx, dy in [(0, 0), (w - r, 0), (0, h - r), (w - r, h - r)]:
        d.rectangle([x0 + dx, y0 + dy, x0 + dx + r, y0 + dy + r], fill=color)


def _glow(d: ImageDraw.ImageDraw, cx: int, cy: int, w: int, h: int, base: tuple, layers: int = 2) -> None:
    for i in range(layers, 0, -1):
        scale = 1.0 + i * 0.12
        gw, gh = int(w * scale), int(h * scale)
        alpha = 35 // i
        _rect(d, cx, cy, gw, gh, base[:3] + (alpha,))


def _ellipse_approx(d: ImageDraw.ImageDraw, cx: int, cy: int, w: int, h: int, color: tuple) -> None:
    """Approximate ellipse using rounded rect with large radius."""
    _rounded_rect(d, cx, cy, w, h, min(w, h) // 2, color)


def render_cube_frame(viseme_name: str, frame_index: int, fps: int = 48,
                      size: tuple = (640, 640), mood: str = "neutral") -> np.ndarray:
    W, H = int(size[0]), int(size[1])
    S = min(W, H)
    mood = mood if mood in ("neutral", "happy", "excited", "playful", "silly",
                            "sad", "lonely", "angry", "calm", "loving") else "neutral"

    base_W, base_H = max(1, W // _PIXEL_SCALE), max(1, H // _PIXEL_SCALE)
    base_S = min(base_W, base_H)

    canvas = Image.new("RGBA", (base_W, base_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)

    cx, cy = base_W // 2, base_H // 2

    # Eye layout
    eye_sep = max(3, int(0.19 * base_S))
    ew = max(3, int(0.095 * base_S))
    eh = max(4, int(0.15 * base_S))
    lx, rx = cx - eye_sep, cx + eye_sep
    ey = cy - int(0.10 * base_S)

    # Dynamic factors
    glance = _pupil_glance(frame_index, fps)
    blink = _blink_factor(frame_index, fps)
    micro = _micro_expressions(frame_index, fps, mood)
    glance_px = int(glance * ew * 0.22)

    # Breathing micro-movement (subtle vertical oscillation)
    breath = np.sin(frame_index / fps * 0.8) * 0.3
    breath_px = int(breath)

    def eye(cx, cy, w, h, color=EYE_COLOR, glow=True, highlight=True, round_r=None, lid_factor=0.0):
        """Draw eye with optional lid closure."""
        if h <= 0: return
        # Apply blink/squint to height
        effective_h = max(1, int(h * (1.0 - lid_factor)))
        effective_cy = cy + int((h - effective_h) / 2)

        if glow: _glow(d, cx, effective_cy, w, effective_h, color)
        if round_r: _rounded_rect(d, cx, effective_cy, w, effective_h, round_r, color + (255,))
        else: _rect(d, cx, effective_cy, w, effective_h, color + (255,))
        if highlight and effective_h > 2:
            hx, hy = cx + int(w * 0.2), effective_cy - int(effective_h * 0.2)
            r = max(1, min(w, effective_h) // 6)
            _rect(d, hx + glance_px, hy, r, r, (255, 255, 255, 255))
            _rect(d, hx + glance_px - r, hy - r, 1, 1, (255, 255, 255, 160))

    def brow(x0, y0, x1, y1, w=1):
        dx, dy = x1 - x0, y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            _hline(d, y, x - w // 2, x + w // 2, BROW_COLOR + (255,))

    def mouth_shape(cx, cy, base_w, base_h, openness, shape_params, mood):
        """Draw mouth with viseme-specific shape and emotional modulation."""
        w_mult, h_mult, corner_pull, upper_raise, lower_drop, teeth = shape_params

        # Emotional modulation
        if mood == "happy":
            corner_pull += 0.25
            upper_raise += 0.1
        elif mood == "sad" or mood == "lonely":
            corner_pull -= 0.2
            upper_raise -= 0.05
            lower_drop += 0.1
        elif mood == "angry":
            corner_pull -= 0.1
            upper_raise -= 0.05
        elif mood == "loving":
            corner_pull += 0.15
            upper_raise += 0.05

        # Micro-expression modulation
        micro = _micro_expressions(frame_index, fps, mood)
        corner_pull += micro["mouth_tension"] * 0.5

        w = max(2, int(base_w * w_mult))
        h = max(1, int(base_h * h_mult * max(0.1, openness)))

        # Corner pull affects width and horizontal position
        corner_offset = int(corner_pull * w * 0.3)
        upper_offset = int(upper_raise * h * 0.5)
        lower_offset = int(lower_drop * h * 0.5)

        # Glow
        _glow(d, cx, cy, w + 8, h + 8, MOUTH_COLOR, layers=2)
        _rect(d, cx, cy, w + 4, h + 4, MOUTH_DIM + (70,))

        # Mouth shape - rounded rectangle with asymmetric modification
        r = max(1, h // 2)
        x0, y0 = cx - w // 2, cy - h // 2
        x1, y1 = x0 + w, y0 + h

        # Apply corner pull
        x0 += corner_offset
        x1 -= corner_offset

        # Apply upper/lower lip offsets
        y0 += upper_offset
        y1 -= lower_offset

        # Ensure valid rect
        if x1 > x0 and y1 > y0:
            _rounded_rect(d, (x0 + x1) // 2, (y0 + y1) // 2, x1 - x0, y1 - y0, r, MOUTH_CORE + (255,))

        # Teeth
        if teeth and openness > 0.5:
            tw = max(1, int(w * 0.12))
            th = max(1, int(h * 0.5))
            _rect(d, cx - w // 3, cy, tw, th, (255, 255, 255, 230))
            _rect(d, cx + w // 3, cy, tw, th, (255, 255, 255, 230))

    k = max(1, base_S / 640.0)
    bw = max(1, int(8 * k))

    # Get viseme shape
    viseme_key = os.path.basename(viseme_name or "1.png")
    shape_params = _VISEME_SHAPES.get(viseme_key, (1.0, 0.5, 0.0, 0.0, 0.0, False))
    openness = _viseme_openness(viseme_name)

    # --- EYES PER MOOD ---
    # Combined blink + squint factor
    total_lid = min(blink + micro["eye_squint"], 1.0)

    if mood == "happy":
        # Happy: slight squint, cheeks up
        for ex in (lx, rx):
            eye(ex + glance_px, ey, int(ew * 1.3), int(eh * 1.1), round_r=max(1, int(eh*0.35)), lid_factor=total_lid + 0.15)
            # Cheek blush
            _rect(d, ex + int(ew * 1.0), ey + int(eh * 0.5), max(1, int(2*k)), max(1, int(2*k)), PINK + (120,))

    elif mood == "excited":
        # Wide eyes, raised brows, sparkles
        for ex in (lx, rx):
            eye(ex + glance_px, ey, int(ew * 1.4), int(eh * 1.4), round_r=max(1, int(eh*0.3)), lid_factor=total_lid)
        # Sparkles
        for sx in (lx - ew, rx + ew):
            r = max(1, int(4 * k))
            _rect(d, sx, ey - int(eh * 0.6), r, r, EYE_CORE + (255,))

    elif mood in ("playful", "silly"):
        # Asymmetric: one eye wider, one wink
        eye(lx + glance_px, ey, int(ew * 1.2), int(eh * 1.2), round_r=max(1, int(eh*0.3)), lid_factor=total_lid)
        wx, wy = rx, ey + int(eh * 0.15)
        wink_lid = 0.7 + total_lid * 0.3
        eye(wx, wy, int(ew * 1.1), int(eh * 0.8), round_r=max(1, int(eh*0.3)), lid_factor=wink_lid, highlight=False)
        if mood == "silly":
            _rect(d, cx, cy + int(0.28 * base_S), max(1, int(0.07*base_S)), max(1, int(0.05*base_S)), PINK + (255,))

    elif mood in ("sad", "lonely"):
        scale = 0.75 if mood == "sad" else 0.6
        # Inner brows up (sad brow)
        brow_y0 = cy - int(0.16 * base_S) + int(micro["brow_offset"] * base_S * 0.5)
        brow(cx - int(0.22*base_S), brow_y0, cx - int(0.05*base_S), cy - int(0.26*base_S), bw)
        brow(cx + int(0.22*base_S), brow_y0, cx + int(0.05*base_S), cy - int(0.26*base_S), bw)
        for ex in (lx, rx):
            eye(ex + glance_px, ey + int(eh * 0.1) + breath_px, int(ew * scale), max(1, int(eh * scale)),
                color=EYE_DIM, round_r=max(1, int(eh*0.25)), lid_factor=total_lid + 0.2)
            # Droopy upper lid line
            _hline(d, ey - int(eh * 0.05) + breath_px, ex - int(ew * 0.5), ex + int(ew * 0.5), (0,0,0,160))
        if mood == "sad":
            tx, ty = lx - int(ew * 0.9), ey + int(eh * 0.9) + breath_px
            tw, th = max(1, int(4*k)), max(1, int(7*k))
            _rect(d, tx, ty, tw*2, th*2, TEAR + (220,))
            _rect(d, tx - tw//2, ty - th//2, max(1, tw//2), max(1, th//2), (255,255,255,150))

    elif mood == "angry":
        # Steep angled brows down
        brow_y0 = cy - int(0.18 * base_S) + int(micro["brow_offset"] * base_S * 0.5)
        brow(cx - int(0.20*base_S), brow_y0, cx - int(0.05*base_S), cy - int(0.28*base_S), bw)
        brow(cx + int(0.20*base_S), brow_y0, cx + int(0.05*base_S), cy - int(0.28*base_S), bw)
        for ex in (lx, rx):
            eye(ex + glance_px, ey + int(eh * 0.05) + breath_px, int(ew * 1.0), max(1, int(eh * 0.35)),
                color=EYE_COLOR, round_r=max(1, int(eh*0.15)), lid_factor=total_lid + micro["eye_squint"])
            # Glare line
            _hline(d, ey + int(eh * 0.02) + breath_px, ex - int(ew * 0.55), ex + int(ew * 0.55), EYE_DIM + (200,))

    elif mood == "calm":
        # Half-lidded, relaxed
        for ex in (lx, rx):
            eye(ex, ey + int(eh * 0.12) + breath_px, int(ew * 1.15), max(1, int(3 * k)),
                color=EYE_DIM, highlight=False, lid_factor=total_lid + 0.35)
            hx, hy = ex + int(ew * 0.15), ey - int(eh * 0.05) + breath_px
            r = max(1, min(ew, eh) // 7)
            _rect(d, hx + glance_px, hy, r, r, (255,255,255,120))

    elif mood == "loving":
        # Heart eyes with subtle pulse
        pulse = np.sin(frame_index / fps * 3) * 0.05 + 1.0
        for ex in (lx, rx):
            hs = int(ew * 0.75 * pulse)
            _glow(d, ex, ey - int(hs * 0.25), hs + 2, hs + 2, PINK, layers=2)
            _rect(d, ex - hs//2, ey - int(hs*0.65), hs, hs, PINK + (255,))
            _rect(d, ex + hs//2, ey - int(hs*0.65), hs, hs, PINK + (255,))
            for i in range(hs):
                w = hs - i
                _rect(d, ex, ey - int(hs*0.65) + hs + i, w, 1, PINK + (255,))
            _rect(d, ex - hs//3, ey - int(hs*0.85), max(1, int(2*k)), max(1, int(2*k)), (255,255,255,255,))

    else:  # neutral
        for ex in (lx, rx):
            eye(ex + glance_px, ey + breath_px, ew, eh, round_r=max(1, int(eh*0.28)), lid_factor=total_lid)

    # --- MOUTH ---
    mx, my = cx, cy + int(0.23 * base_S) + breath_px
    mouth_shape(mx, my, max(2, int(0.13 * base_S)), max(4, int(0.15 * base_S)), openness, shape_params, mood)

    canvas = canvas.resize((W, H), Image.NEAREST)
    return np.asarray(canvas)