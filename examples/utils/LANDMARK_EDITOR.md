# Landmark Editor

An interactive OpenCV editor for correcting pose landmarks produced by the
`body_trajectory` pipeline. Edits are saved back to the same `landmarks.json`
file and picked up automatically on the next render via `--use_cached_landmarks`.

---

## Workflow

### 1. Generate landmarks

Run the body trajectory pipeline with landmark export enabled:

```bash
python examples/scripts/body_trajectory_demo.py \
  --pose_backend vitpose \
  --video_path path/to/climb.mov \
  --export_landmarks
```

This produces a `*_landmarks.json` file next to the video (or in `output/`
depending on your settings).

### 2. Open the editor

```bash
python examples/utils/landmark_editor.py \
  --video_path path/to/climb.mov \
  --landmarks path/to/climb_landmarks.json
```

### 3. Correct landmarks frame by frame

Use the controls below to navigate and edit. Press **S** to save when done.

### 4. Re-render with corrections

```bash
python examples/scripts/body_trajectory_demo.py \
  --pose_backend vitpose \
  --use_cached_landmarks \
  --video_path path/to/climb.mov
```

The renderer will load your corrected landmarks instead of re-running pose
estimation.

---

## Controls

| Key / Action | Description |
|---|---|
| `[` | Previous frame |
| `]` | Next frame |
| `,` | Jump back 10 frames |
| `.` | Jump forward 10 frames |
| **Click** a dot | Select the joint (highlights it and its connected bones in cyan) |
| **Drag** selected dot | Move the joint to a new position |
| **Shift + click** empty space | Place a missing/invisible landmark at the cursor — you will be prompted in the terminal to enter the joint index or name |
| `R` | Reset the current frame to the originally loaded landmarks |
| `Z` | Undo the last edit on the current frame (up to 50 steps) |
| `S` | Save all changes and quit |
| `Q` / `Esc` | Quit without saving |

---

## Visual Guide

| Colour | Meaning |
|---|---|
| White dot | Normal visible landmark (COCO/ViTPose keypoint) |
| Cyan dot + bones | Currently selected joint and its connections |
| Green dot | Landmark placed manually by the user |
| Dark grey dot | Landmark with very low confidence (below threshold) |
| Dim dot / bone | MediaPipe-only slot not populated by ViTPose |

---

## Tips

- The HUD in the top-left shows the current frame number and how many frames
  have been modified in this session.
- When a joint is selected, its name and confidence score appear next to it.
- **Shift + click** is the primary way to recover a completely missing landmark
  (e.g. a wrist that ViTPose failed to detect). After placing, you can drag it
  to fine-tune.
- Use `R` freely — it resets only the current frame and is itself undoable via `Z`.
- The JSON is only written when you press **S**. Closing the window with `Q` or
  `Esc` discards all changes.
- For ViTPose, only the 17 COCO keypoints are meaningful to edit:
  nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles (indices 0, 2, 5, 7, 8, 11–16, 23–28).
