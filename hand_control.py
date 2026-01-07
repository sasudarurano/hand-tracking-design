import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc import udp_client
import math
import time
import numpy as np

# OSC config
IP = "127.0.0.1"
PORT = 5005
client = udp_client.SimpleUDPClient(IP, PORT)

# MediaPipe Hands
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Person segmentation (for background blur)
seg_base = python.BaseOptions(model_asset_path='selfie_segmenter.tflite')
seg_options = vision.ImageSegmenterOptions(base_options=seg_base, output_category_mask=True)
segmenter = vision.ImageSegmenter.create_from_options(seg_options)

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

# Set higher resolution for webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get video dimensions
ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape
else:
    h, w = 720, 1280

print(f"Video resolution: {w}x{h}")

# Trail effect buffer
trail_buffer = np.zeros((h, w, 3), dtype=np.uint8)

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

class SimplePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def get_palm_center(hand_landmarks):
    """Approximate palm center using wrist + MCP joints (0,5,9,13,17). Returns SimplePoint in normalized coords."""
    idxs = [0, 5, 9, 13, 17]
    xs = [hand_landmarks[i].x for i in idxs]
    ys = [hand_landmarks[i].y for i in idxs]
    return SimplePoint(sum(xs) / len(xs), sum(ys) / len(ys))

def get_person_mask(mp_image, shape):
    """Run selfie segmentation and return uint8 mask (255=person, 0=background)."""
    h, w, _ = shape
    seg_result = segmenter.segment(mp_image)
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        if hasattr(seg_result, 'category_mask') and seg_result.category_mask is not None:
            cm = seg_result.category_mask.numpy_view()
            # Resize to frame size if needed
            if cm.shape[0] != h or cm.shape[1] != w:
                cm = cv2.resize(cm, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (cm > 0).astype(np.uint8) * 255
        elif hasattr(seg_result, 'confidence_masks') and seg_result.confidence_masks:
            conf = seg_result.confidence_masks[0].numpy_view()
            if conf.shape[0] != h or conf.shape[1] != w:
                conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (conf > 0.5).astype(np.uint8) * 255
    except Exception:
        pass
    return mask

def get_hand_rotation(hand_landmarks):
    """Calculate hand rotation angle from wrist to middle finger"""
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]  # Middle finger base
    
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    angle = math.atan2(dy, dx)
    return angle

def get_hand_openness(hand_landmarks):
    """Calculate how open the hand is (0 = closed fist, 1 = open hand)"""
    # Measure distances from palm center to each fingertip
    palm_center = hand_landmarks[0]  # Wrist as reference
    
    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    finger_bases = [2, 5, 9, 13, 17]  # Finger bases
    
    total_extension = 0
    for tip_idx, base_idx in zip(fingertips, finger_bases):
        tip = hand_landmarks[tip_idx]
        base = hand_landmarks[base_idx]
        extension = get_distance(base, tip)
        total_extension += extension
    
    # Normalize (typical range is 0.3 to 0.8)
    normalized = (total_extension - 0.3) / 0.5
    return max(0, min(1, normalized))

def hsv_to_bgr(h, s, v):
    """Convert HSV to BGR color"""
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(map(int, bgr[0][0]))

def rw_mix(alpha):
    """Blend between white (alpha=0) and red (alpha=1), BGR tuple."""
    alpha = max(0.0, min(1.0, float(alpha)))
    r = int(255 * alpha + 255 * (1 - alpha))  # R from white->red stays 255
    g = int(255 * (1 - alpha))
    b = int(255 * (1 - alpha))
    return (b, g, 255)  # ensure strong red channel

def draw_futuristic_hud(canvas, center_x, center_y, rotation_angle, scale, t, motion_level=0.0, pos_norm=0.5):
    """Futuristic, multicolor HUD with segmented arcs, number rings, sweep, and particles.
    Returns a secondary glow layer for bloom effect.
    """
    h, w, _ = canvas.shape
    base_radius = int(150 * scale)

    # Red/White palette: vary intensity by rotation/position
    rot_norm = (rotation_angle % (2 * math.pi)) / (2 * math.pi)
    pos_norm = min(1.0, max(0.0, pos_norm))
    base_alpha = 0.35 + 0.4 * rot_norm + 0.25 * pos_norm  # 0..1 for red weight

    glow = np.zeros_like(canvas)

    # Outer guide
    col_outer = rw_mix(base_alpha * 0.9)
    cv2.circle(canvas, (center_x, center_y), base_radius, col_outer, 2)

    # Sunburst core
    num_rays = 54
    for i in range(num_rays):
        ang = (i * 2 * math.pi / num_rays) + rotation_angle
        inner_r = int(18 * scale)
        outer_r = int((38 + 10 * math.sin(i * 0.4 + t * 2)) * scale)
        x1 = int(center_x + inner_r * math.cos(ang))
        y1 = int(center_y + inner_r * math.sin(ang))
        x2 = int(center_x + outer_r * math.cos(ang))
        y2 = int(center_y + outer_r * math.sin(ang))
        col = rw_mix((base_alpha + 0.5 * (i % 2)) % 1.0)
        cv2.line(canvas, (x1, y1), (x2, y2), col, 2)
        cv2.line(glow, (x1, y1), (x2, y2), col, 2)

    # Segmented arcs (dash-like)
    # Vary segments/gaps/thickness with motion_level
    motion_level = max(0.0, min(1.0, float(motion_level)))
    seg_boost = int(10 * motion_level)
    gap_reduce = int(4 * motion_level)
    th_boost = int(4 * motion_level)
    ring_specs = [
        (int(80 * scale), 14 + seg_boost, 10 - gap_reduce, 7 + th_boost),
        (int(115 * scale), 22 + seg_boost, 7 - gap_reduce, 10 + th_boost),
        (int(155 * scale), 32 + seg_boost, 5 - gap_reduce, 12 + th_boost)
    ]
    for r_idx, (radius, segs, gap_deg, thick) in enumerate(ring_specs):
        seg_span = (360 - segs * gap_deg) / segs
        for s in range(segs):
            start = s * (seg_span + gap_deg) + math.degrees(rotation_angle) * (0.7 + 0.2 * r_idx)
            end = start + seg_span
            # alternate white/red pattern drifting with rotation
            alt = (s + r_idx + int(rot_norm * 10)) % 2
            col = rw_mix(0.85 if alt else 0.2)
            cv2.ellipse(canvas, (center_x, center_y), (radius, radius), 0, start, end, col, thick, lineType=cv2.LINE_AA)
            cv2.ellipse(glow, (center_x, center_y), (radius, radius), 0, start, end, col, 2, lineType=cv2.LINE_AA)

    # Number ring (0-9 repeated)
    num_count = 48
    r_num = int(135 * scale)
    for i in range(num_count):
        ang = (i * 2 * math.pi / num_count) + rotation_angle
        tx = int(center_x + r_num * math.cos(ang))
        ty = int(center_y + r_num * math.sin(ang))
        col = rw_mix(0.9 if (i % 3) else 0.1)  # mostly red, occasional white
        cv2.putText(canvas, str(i % 10), (tx - 7, ty + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, col, 2, lineType=cv2.LINE_AA)

    # Sweep beam
    sweep_ang = rotation_angle + t * 0.6
    sweep_len = int(base_radius * 1.15)
    tri = np.array([
        [center_x, center_y],
        [center_x + int(sweep_len * math.cos(sweep_ang - 0.05)), center_y + int(sweep_len * math.sin(sweep_ang - 0.05))],
        [center_x + int(sweep_len * math.cos(sweep_ang + 0.05)), center_y + int(sweep_len * math.sin(sweep_ang + 0.05))]
    ])
    col_sweep = rw_mix(0.6)
    cv2.fillConvexPoly(glow, tri, col_sweep)

    # Orbiting particles
    sats = 10
    orbit_r = int(200 * scale)
    for i in range(sats):
        ang = (i * 2 * math.pi / sats) - rotation_angle * (0.9 + 0.1 * i) + t * (0.3 + 0.05 * i)
        x = int(center_x + orbit_r * math.cos(ang))
        y = int(center_y + orbit_r * math.sin(ang))
        col = rw_mix(0.75 if i % 2 else 0.15)
        cv2.circle(canvas, (x, y), int(12 * scale), col, 2)
        cv2.circle(canvas, (x, y), int(3 * scale), col, -1)
        cv2.circle(glow, (x, y), int(10 * scale), col, 1)

    return glow


try:
    print("Hand tracking with visualization started!")
    print("- Rotate your hand left/right to spin the circle")
    print("- Close your fist to shrink the circle")
    print("- Open your hand to enlarge the circle")
    print("Press Q in the video window to quit.\n")
    
    rotation_angle = 0
    target_scale = 1.0
    current_scale = 1.0
    
    # Create window with specific size
    window_name = "Hand Control Visualization (press Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    prev_center_px = None
    prev_rot = None
    motion_level = 0.0
    invert_blur = False  # False = blur background (keep person sharp)
    blur_levels = [21, 31, 45, 61]
    blur_idx = 2
    
    # Optimization: cache masks and skip heavy ops
    cached_person_mask = None
    cached_keep_mask = None
    seg_frame_skip = 2  # Run segmentation every N frames
    seg_counter = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Transparent-style overlay: draw on a blank layer then blend with camera
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert to MediaPipe Image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Rotation from wrist -> middle MCP
                hand_angle = get_hand_rotation(hand_landmarks)
                rotation_angle = hand_angle

                # Palm-centered position
                palm_center = get_palm_center(hand_landmarks)

                # Use fingertip spread from palm center to determine scale
                tip_indices = [4, 8, 12, 16, 20]
                dists_norm = [get_distance(palm_center, hand_landmarks[i]) for i in tip_indices]
                mean_tip_dist_norm = sum(dists_norm) / len(dists_norm)

                # Map normalized distance to pixels then to scale factor (base unit 150px)
                target_radius_px = mean_tip_dist_norm * min(w, h) * 0.95
                target_scale = max(0.35, min(1.6, target_radius_px / 150.0))

                # Smooth scale
                current_scale += (target_scale - current_scale) * 0.25

                # Pixel center
                center_x = int(palm_center.x * w)
                center_y = int(palm_center.y * h)
                
                # Motion metric from center/rotation deltas
                if prev_center_px is not None and prev_rot is not None:
                    dx = (center_x - prev_center_px[0]) / max(1, w)
                    dy = (center_y - prev_center_px[1]) / max(1, h)
                    drot = (rotation_angle - prev_rot)
                    speed = math.sqrt(dx*dx + dy*dy) + abs(drot) * 0.2
                    motion_level = motion_level * 0.80 + min(1.0, speed * 6.0) * 0.20
                prev_center_px = (center_x, center_y)
                prev_rot = rotation_angle

                # Draw the futuristic HUD visualization (red/white, motion-reactive)
                t = time.time()
                pos_norm = center_x / max(1, w)
                glow_layer = draw_futuristic_hud(canvas, center_x, center_y, rotation_angle, current_scale, t, motion_level, pos_norm)
                
                # Draw hand landmarks on top
                for idx, landmark in enumerate(hand_landmarks):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    if idx in [0, 4, 8, 12, 16, 20]:  # Key points
                        cv2.circle(canvas, (cx, cy), 6, (0, 255, 0), -1)
                    else:
                        cv2.circle(canvas, (cx, cy), 3, (0, 200, 0), -1)
                
                # Draw hand skeleton (light red lines)
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm
                ]
                for connection in connections:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(canvas, start_point, end_point, (60, 60, 200), 2)
                
                # Send OSC messages
                index_tip = hand_landmarks[8]
                thumb_tip = hand_landmarks[4]
                pinch_dist = get_distance(index_tip, thumb_tip)
                
                # Send palm-centered coordinates
                client.send_message("/center/x", palm_center.x)
                client.send_message("/center/y", palm_center.y)
                client.send_message("/control/pinch", pinch_dist)
                client.send_message("/index/x", index_tip.x)
                client.send_message("/index/y", index_tip.y)
                client.send_message("/rotation", rotation_angle)
                client.send_message("/scale", current_scale)
                
                # Display info with better visibility
                # Show scale and center info for debugging alignment
                cv2.putText(canvas, f"Scale: {current_scale:.2f}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(canvas, f"Rotation: {math.degrees(rotation_angle):.1f}°", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Apply trail effect with more persistence (only on overlay)
        trail_buffer = cv2.addWeighted(trail_buffer, 0.90, canvas, 0.10, 0)
        display = cv2.addWeighted(trail_buffer, 0.7, canvas, 0.3, 0)

        # Optimized background blur: skip segmentation every other frame
        seg_counter += 1
        if seg_counter % seg_frame_skip == 0:
            # Run segmentation only every N frames
            person_mask = 255 - get_person_mask(mp_image, frame.shape)
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    pts = np.array([[int(l.x * w), int(l.y * h)] for l in hand_landmarks], dtype=np.int32)
                    if len(pts) >= 3:
                        hull = cv2.convexHull(pts)
                        cv2.fillConvexPoly(hand_mask, hull, 255)
            keep_mask = cv2.bitwise_or(person_mask, hand_mask)
            # Simplified feathering: dilate only
            keep_mask = cv2.dilate(keep_mask, np.ones((7, 7), np.uint8), iterations=1)
            keep_mask = cv2.GaussianBlur(keep_mask, (15, 15), 0)
            cached_keep_mask = keep_mask
        
        # Use cached mask if available
        if cached_keep_mask is not None:
            keep_mask = cached_keep_mask
        else:
            keep_mask = np.ones((h, w), dtype=np.uint8) * 255  # Fallback: no blur
        
        k = blur_levels[blur_idx]
        # Faster blur with smaller kernel on alternate frames, ensure odd
        blur_strength = (k + max(21, k - 10)) // 2 if seg_counter % 2 == 0 else k
        blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1  # Ensure odd
        blurred_bg = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        
        keep_mask_3 = cv2.merge([keep_mask, keep_mask, keep_mask])
        if invert_blur:
            # Blur person, keep background sharp (toggle mode)
            base_frame = np.where(keep_mask_3 > 128, blurred_bg, frame)
        else:
            # Default: keep person sharp, blur background
            base_frame = np.where(keep_mask_3 > 128, frame, blurred_bg)

        # Bloom/glow: reduced kernel for speed
        try:
            blur = cv2.GaussianBlur(glow_layer, (15, 15), 0)
            display = cv2.addWeighted(display, 1.0, blur, 0.6, 0)
        except Exception:
            pass
        except Exception:
            pass

        # Blend overlay with the live camera fully (transparent background)
        final = cv2.addWeighted(base_frame, 1.0, display, 1.0, 0)

        # Show small HUD with blur settings
        cv2.putText(final, f"Blur:{blur_levels[blur_idx]}  [1-4], Toggle[B]  Mode:{'Invert' if invert_blur else 'BG'}",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(window_name, final)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            blur_idx = int(chr(key)) - 1
            blur_idx = max(0, min(len(blur_levels)-1, blur_idx))
        elif key in [ord('b'), ord('B')]:
            invert_blur = not invert_blur
finally:
    cap.release()
    cv2.destroyAllWindows()
