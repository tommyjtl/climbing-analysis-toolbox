"""
Landmark Editor
===============
A standalone OpenCV-based editor for correcting pose landmarks produced by the
body_trajectory pipeline.

Workflow:
  1. Run body_trajectory with --export_landmarks to produce a landmarks JSON.
  2. Run this editor: python landmark_editor.py --video path/to/video.mp4 --landmarks path/to/landmarks.json
  3. Correct the landmarks frame by frame, then press S to save.
  4. Re-run body_trajectory with --use_cached_landmarks to render with corrections.

Controls
--------
  [ / ]           : previous / next frame
  , / .               : jump back / forward 10 frames
  Click landmark dot  : select it (highlights the joint and its bones)
  Drag selected dot   : reposition the joint
  Shift + click empty : place a missing/invisible landmark at the cursor
                        (you will be prompted in the terminal for which index)
  R                   : reset current frame's landmarks to the original loaded data
  Z                   : undo last edit on the current frame
  S                   : save the (potentially modified) landmarks JSON and quit
  Q / Esc             : quit without saving
"""

import argparse
import copy
import json
import os
import sys

import cv2
import numpy as np

# ── MediaPipe landmark index names (for display) ────────────────────────────
LANDMARK_NAMES = [
    "nose",  # 0
    "left_eye_inner",  # 1
    "left_eye",  # 2
    "left_eye_outer",  # 3
    "right_eye_inner",  # 4
    "right_eye",  # 5
    "right_eye_outer",  # 6
    "left_ear",  # 7
    "right_ear",  # 8
    "mouth_left",  # 9
    "mouth_right",  # 10
    "left_shoulder",  # 11
    "right_shoulder",  # 12
    "left_elbow",  # 13
    "right_elbow",  # 14
    "left_wrist",  # 15
    "right_wrist",  # 16
    "left_pinky",  # 17
    "right_pinky",  # 18
    "left_index",  # 19
    "right_index",  # 20
    "left_thumb",  # 21
    "right_thumb",  # 22
    "left_hip",  # 23
    "right_hip",  # 24
    "left_knee",  # 25
    "right_knee",  # 26
    "left_ankle",  # 27
    "right_ankle",  # 28
    "left_heel",  # 29
    "right_heel",  # 30
    "left_foot_index",  # 31
    "right_foot_index",  # 32
]

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]

# Indices that ViTPose actually populates (COCO-17 mapped to MediaPipe slots).
# All others will always be invisible; we still allow editing them but show
# them in a dimmer colour.
COCO_MP_INDICES = {0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

# Visual constants
DOT_RADIUS = 7
DOT_RADIUS_SELECTED = 10
HIT_RADIUS = 12  # pixels — click detection radius
COLOR_BONE = (180, 180, 180)
COLOR_BONE_SELECTED = (0, 220, 255)
COLOR_BONE_DIM = (60, 60, 60)
COLOR_DOT = (255, 255, 255)
COLOR_DOT_SELECTED = (0, 220, 255)
COLOR_DOT_LOW_VIS = (80, 80, 80)
COLOR_DOT_DIM = (40, 40, 40)
COLOR_PLACED = (0, 255, 120)  # freshly user-placed landmark
VISIBILITY_THRESHOLD = 0.15
DEFAULT_PLACE_VISIBILITY = 0.75


# ── Helpers ──────────────────────────────────────────────────────────────────


def _lm_px(lm, w, h):
    """Convert normalised landmark to pixel coords."""
    return (int(round(lm["x"] * w)), int(round(lm["y"] * h)))


def _connected_indices(idx):
    """All landmark indices directly connected to idx via POSE_CONNECTIONS."""
    result = set()
    for a, b in POSE_CONNECTIONS:
        if a == idx:
            result.add(b)
        elif b == idx:
            result.add(a)
    return result


def _draw_frame(canvas, landmarks, selected_idx, w, h):
    """Draw skeleton and dots onto canvas in-place."""
    if landmarks is None:
        cv2.putText(
            canvas,
            "No landmarks for this frame",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (80, 80, 200),
            2,
        )
        return

    connected = _connected_indices(selected_idx) if selected_idx is not None else set()

    # Build pixel coords map for visible landmarks
    coords = {}
    for i, lm in enumerate(landmarks):
        vis = lm.get("visibility") or 0.0
        if vis >= VISIBILITY_THRESHOLD or lm.get("_placed", False):
            coords[i] = _lm_px(lm, w, h)

    # Draw bones
    for a, b in POSE_CONNECTIONS:
        if a not in coords or b not in coords:
            continue
        is_selected_bone = (selected_idx in (a, b)) and selected_idx is not None
        if is_selected_bone:
            color = COLOR_BONE_SELECTED
            thickness = 3
        elif a in COCO_MP_INDICES and b in COCO_MP_INDICES:
            color = COLOR_BONE
            thickness = 2
        else:
            color = COLOR_BONE_DIM
            thickness = 1
        cv2.line(canvas, coords[a], coords[b], color, thickness, cv2.LINE_AA)

    # Draw dots
    for i, px in coords.items():
        lm = landmarks[i]
        vis = lm.get("visibility") or 0.0
        placed = lm.get("_placed", False)

        if i == selected_idx:
            color = COLOR_DOT_SELECTED
            r = DOT_RADIUS_SELECTED
        elif placed:
            color = COLOR_PLACED
            r = DOT_RADIUS
        elif vis < VISIBILITY_THRESHOLD:
            color = COLOR_DOT_LOW_VIS
            r = DOT_RADIUS - 2
        elif i not in COCO_MP_INDICES:
            color = COLOR_DOT_DIM
            r = DOT_RADIUS - 3
        else:
            color = COLOR_DOT
            r = DOT_RADIUS

        cv2.circle(canvas, px, r, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, px, r, (0, 0, 0), 1, cv2.LINE_AA)

    # Label selected landmark
    if selected_idx is not None and selected_idx in coords:
        name = (
            LANDMARK_NAMES[selected_idx]
            if selected_idx < len(LANDMARK_NAMES)
            else str(selected_idx)
        )
        lm = landmarks[selected_idx]
        vis = lm.get("visibility") or 0.0
        label = f"[{selected_idx}] {name}  vis={vis:.2f}"
        px = coords[selected_idx]
        cv2.putText(
            canvas,
            label,
            (px[0] + 12, px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_DOT_SELECTED,
            2,
            cv2.LINE_AA,
        )


def _draw_hud(canvas, frame_idx, total_frames, modified_frames, selected_idx):
    """Overlay HUD info in the top-left corner."""
    h, w = canvas.shape[:2]
    modified = frame_idx in modified_frames
    mod_str = " [MODIFIED]" if modified else ""
    lines = [
        f"Frame {frame_idx + 1}/{total_frames}{mod_str}",
        f"Modified frames: {len(modified_frames)}",
        "Drag: move  |  Shift+click: place  |  R: reset  |  Z: undo",
        "[ ]: prev/next  |  , .: -10/+10  |  S: save & quit  |  Q: quit",
    ]
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def _hit_test(landmarks, mx, my, w, h):
    """Return index of the landmark closest to (mx, my) within HIT_RADIUS, or None."""
    best_idx = None
    best_dist = HIT_RADIUS**2
    if landmarks is None:
        return None
    for i, lm in enumerate(landmarks):
        vis = lm.get("visibility") or 0.0
        if vis < VISIBILITY_THRESHOLD and not lm.get("_placed", False):
            continue
        px, py = _lm_px(lm, w, h)
        d = (px - mx) ** 2 + (py - my) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


# ── Editor state ─────────────────────────────────────────────────────────────


class LandmarkEditor:
    def __init__(self, video_path, landmarks_path):
        self.video_path = video_path
        self.landmarks_path = landmarks_path

        # Load landmarks JSON
        with open(landmarks_path, "r", encoding="utf-8") as f:
            self.payload = json.load(f)
        self.frames_data = self.payload["frames"]  # list[None | list[dict]]
        # Deep-copy originals for per-frame reset
        self.original_frames = copy.deepcopy(self.frames_data)
        self.total_frames = len(self.frames_data)

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        self.vid_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # State
        self.frame_idx = 0
        self._cached_frame_idx = -1
        self._cached_bgr = None

        self.selected_idx = None
        self.dragging = False
        self.modified_frames = set()
        # Per-frame undo stacks: frame_idx -> list of snapshots
        self.undo_stacks = {}

    def _get_video_frame(self, idx):
        if self._cached_frame_idx == idx and self._cached_bgr is not None:
            return self._cached_bgr.copy()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((self.vid_h, self.vid_w, 3), dtype=np.uint8)
        self._cached_frame_idx = idx
        self._cached_bgr = frame.copy()
        return frame.copy()

    def _render(self):
        canvas = self._get_video_frame(self.frame_idx)
        lms = self.frames_data[self.frame_idx]
        _draw_frame(canvas, lms, self.selected_idx, self.vid_w, self.vid_h)
        _draw_hud(
            canvas,
            self.frame_idx,
            self.total_frames,
            self.modified_frames,
            self.selected_idx,
        )
        return canvas

    def _push_undo(self):
        fi = self.frame_idx
        if fi not in self.undo_stacks:
            self.undo_stacks[fi] = []
        self.undo_stacks[fi].append(copy.deepcopy(self.frames_data[fi]))
        # Cap stack depth
        if len(self.undo_stacks[fi]) > 50:
            self.undo_stacks[fi].pop(0)

    def _pop_undo(self):
        fi = self.frame_idx
        stack = self.undo_stacks.get(fi, [])
        if stack:
            self.frames_data[fi] = stack.pop()
            self.modified_frames.add(fi)
            print(f"Undo on frame {fi}. Undo stack depth: {len(stack)}")
        else:
            print("Nothing to undo on this frame.")

    def _reset_frame(self):
        fi = self.frame_idx
        self._push_undo()
        self.frames_data[fi] = copy.deepcopy(self.original_frames[fi])
        self.modified_frames.discard(fi)
        self.selected_idx = None
        print(f"Frame {fi} reset to original.")

    def _move_landmark(self, lm_idx, mx, my):
        fi = self.frame_idx
        if self.frames_data[fi] is None:
            return
        self._push_undo()
        lm = self.frames_data[fi][lm_idx]
        lm["x"] = max(0.0, min(1.0, mx / self.vid_w))
        lm["y"] = max(0.0, min(1.0, my / self.vid_h))
        # Boost visibility if it was very low so it stays visible
        if (lm.get("visibility") or 0.0) < VISIBILITY_THRESHOLD:
            lm["visibility"] = DEFAULT_PLACE_VISIBILITY
            lm["presence"] = DEFAULT_PLACE_VISIBILITY
        self.modified_frames.add(fi)

    def _place_landmark(self, mx, my):
        """Interactively prompt which landmark index to place at (mx, my)."""
        fi = self.frame_idx
        print(
            "\nPlace landmark — enter index (0-32) or name (partial ok), blank to cancel:"
        )
        for i, name in enumerate(LANDMARK_NAMES):
            print(f"  {i:2d}  {name}")
        raw = input("Index or name > ").strip()
        if not raw:
            return

        lm_idx = None
        if raw.isdigit():
            lm_idx = int(raw)
        else:
            # partial name match
            matches = [i for i, n in enumerate(LANDMARK_NAMES) if raw.lower() in n]
            if len(matches) == 1:
                lm_idx = matches[0]
            elif len(matches) > 1:
                print(
                    f"Ambiguous: {[LANDMARK_NAMES[i] for i in matches]}. Be more specific."
                )
                return
            else:
                print("No matching landmark name found.")
                return

        if lm_idx is None or not (0 <= lm_idx < 33):
            print("Invalid index.")
            return

        if self.frames_data[fi] is None:
            # Create a blank 33-slot landmark list
            self.frames_data[fi] = [
                {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0, "presence": 0.0}
                for _ in range(33)
            ]
        self._push_undo()
        self.frames_data[fi][lm_idx] = {
            "x": max(0.0, min(1.0, mx / self.vid_w)),
            "y": max(0.0, min(1.0, my / self.vid_h)),
            "z": 0.0,
            "visibility": DEFAULT_PLACE_VISIBILITY,
            "presence": DEFAULT_PLACE_VISIBILITY,
            "_placed": True,
        }
        self.modified_frames.add(fi)
        self.selected_idx = lm_idx
        print(f"Placed [{lm_idx}] {LANDMARK_NAMES[lm_idx]} at ({mx}, {my})")

    def save(self):
        # Strip internal _placed flags before saving
        clean_frames = []
        for frame_lms in self.frames_data:
            if frame_lms is None:
                clean_frames.append(None)
            else:
                clean_frames.append(
                    [
                        {k: v for k, v in lm.items() if k != "_placed"}
                        for lm in frame_lms
                    ]
                )
        self.payload["frames"] = clean_frames
        with open(self.landmarks_path, "w", encoding="utf-8") as f:
            json.dump(self.payload, f)
        print(
            f"Saved {len(self.modified_frames)} modified frames to {self.landmarks_path}"
        )

    def run(self):
        win = "Landmark Editor"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.vid_w, self.vid_h)

        mouse_state = {"down": False, "shift": False, "mx": 0, "my": 0}

        def on_mouse(event, mx, my, flags, _):
            shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
            mouse_state["mx"] = mx
            mouse_state["my"] = my

            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_state["down"] = True
                mouse_state["shift"] = shift
                if shift:
                    self._place_landmark(mx, my)
                else:
                    hit = _hit_test(
                        self.frames_data[self.frame_idx], mx, my, self.vid_w, self.vid_h
                    )
                    self.selected_idx = hit
                    self.dragging = hit is not None

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging and self.selected_idx is not None:
                    self._move_landmark(self.selected_idx, mx, my)

            elif event == cv2.EVENT_LBUTTONUP:
                mouse_state["down"] = False
                self.dragging = False

        cv2.setMouseCallback(win, on_mouse)

        while True:
            canvas = self._render()
            cv2.imshow(win, canvas)
            key = cv2.waitKey(16) & 0xFF

            if key in (ord("q"), 27):  # Q or Esc
                print("Quit without saving.")
                break
            elif key == ord("s"):
                self.save()
                break
            elif key == ord("z"):
                self._pop_undo()
            elif key == ord("r"):
                self._reset_frame()
            elif key in (ord("["), ord(",")):
                self.frame_idx = max(0, self.frame_idx - (10 if key == ord(",") else 1))
                self.selected_idx = None
                self.dragging = False
            elif key in (ord("]"), ord(".")):
                self.frame_idx = min(
                    self.total_frames - 1,
                    self.frame_idx + (10 if key == ord(".") else 1),
                )
                self.selected_idx = None
                self.dragging = False

        self.cap.release()
        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Interactive landmark editor for body_trajectory landmarks JSON."
    )
    parser.add_argument(
        "--video_path",
        "-v",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--landmarks",
        "-l",
        type=str,
        required=True,
        help="Path to the landmarks JSON file (will be overwritten on save).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: video not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.landmarks):
        print(f"Error: landmarks file not found: {args.landmarks}", file=sys.stderr)
        sys.exit(1)

    editor = LandmarkEditor(args.video_path, args.landmarks)
    print(f"Loaded {editor.total_frames} frames from landmarks JSON.")
    print(f"Video: {editor.vid_w}x{editor.vid_h}")
    editor.run()


if __name__ == "__main__":
    main()
