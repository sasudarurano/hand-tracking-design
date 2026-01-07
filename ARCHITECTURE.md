# Hand Tracking OSC + HUD Architecture

## Overview
- **Goal:** Capture webcam frames, detect hand landmarks, send OSC to TouchDesigner, and render a holographic HUD overlay with optional background blur.
- **Main script:** `hand_control.py`
- **Key deps:** OpenCV (video + drawing), MediaPipe Hands (landmarks), MediaPipe Image Segmenter (person mask), python-osc (OSC), NumPy (math/buffers).

## Data Flow (per frame)
1) **Capture & prep**
   - Grab frame via OpenCV; flip horizontally for mirror UX.
   - Convert BGR→RGB to feed MediaPipe.

2) **Hand tracking**
   - MediaPipe Hands returns up to one hand: 21 landmarks.
   - Compute palm center (wrist + MCP average), fingertip spread, hand rotation (wrist→middle MCP).
   - Smooth scale based on fingertip spread; update motion metric from position/rotation deltas.

3) **HUD rendering (overlay)**
   - Draw futuristic red/white HUD anchored at palm center: segmented arcs, sunburst, number ring, sweep beam, orbiting particles.
   - Colors shift with rotation and horizontal position; arc density/thickness react to motion level.
   - Trail buffer adds persistence; glow layer is blurred for a soft bloom.

4) **Segmentation & blur composite**
   - MediaPipe selfie segmenter produces person mask; hand hull is added to the keep-mask.
   - Mask is feathered; background is Gaussian-blurred (odd kernel enforced). Person/hand stay sharp by default; toggle invert via key.
   - Final frame = (sharp subject) + (blurred background) + HUD overlay.

5) **OSC output**
   - Sends `/center/x`, `/center/y` (palm), `/control/pinch`, `/index/x`, `/index/y`, `/rotation`, `/scale` to `127.0.0.1:5005` for TouchDesigner.

## Controls (runtime)
- **Q**: Quit
- **1/2/3/4**: Blur strength presets
- **B**: Toggle blur mode (default: blur background; invert: blur person)

## Performance notes
- Blur kernel forced odd to avoid OpenCV assertion.
- Blur uses smaller kernel on alternate frames; segmentation mask cached until next run.
- Hand scale/rotation smoothed to reduce jitter.

## Customization hints
- Change OSC target: edit IP/PORT near top of `hand_control.py`.
- Adjust HUD size: tweak base radius/scales inside `draw_futuristic_hud` and fingertip→scale mapping.
- Disable blur entirely: skip segmentation block and set `base_frame = frame`.
- Theme: colors are red/white via `rw_mix`; you can swap to other palettes easily.
