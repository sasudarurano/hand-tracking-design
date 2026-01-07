# Hand Tracking OSC → TouchDesigner

This folder contains `hand_control.py` to send wrist and index finger data via OSC to TouchDesigner, plus setup notes to build the rotating numbers and trails.

## Setup (Windows)

### 1) Create venv and install deps
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Run the sender
```powershell
python hand_control.py
```
Press `Q` to quit. OSC is sent to `127.0.0.1:5005` with channels `/center/x`, `/center/y`, `/control/pinch`, `/index/x`, `/index/y`.

## TouchDesigner Wiring (Summary)

- OSC In CHOP: Port `5005`.
- Center position: Select `/center/x` `/center/y` → Math (0→1 to -0.5→0.5) → Lag (≈0.1) → use to drive parent `Geometry COMP` translate X/Y.
- Control value: Select `/control/pinch` → Math (map to 0→5) → Speed CHOP → parent `Geometry COMP` Rotate Z.
- Instancing for numbers:
  - `Text SOP` (e.g., "0") then instance via Pattern CHOP (0→360) → Math (deg→rad) to make `sin`/`cos` offsets (`tx_offset`, `ty_offset`) and `rz`.
  - Merge offsets + `tz=0` via Constant CHOP and map TX/TY/TZ/RZ in the `Geo COMP` Instancing page.
- Trails:
  - Render TOP → Feedback TOP → Transform TOP (Scale ~1.01, Opacity ~0.95) → back into Feedback.
  - Composite TOP with original render (Add/Screen).
- Final composite:
  - Over TOP with `Video Device In TOP` as background.

Adjust ranges/radii to taste; add center gear via `Circle SOP` with instancing for teeth.
