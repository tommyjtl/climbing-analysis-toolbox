"""
Debug: compare raw vs SmoothNet-refined landmark positions.

Usage
-----
# Run SmoothNet and plot inline:
python examples/utils/debug_smoothing.py \
    --landmarks /path/to/2_landmarks.json

# Compare two already-saved JSON files:
python examples/utils/debug_smoothing.py \
    --landmarks /path/to/2_landmarks.json \
    --smoothed  /path/to/2_landmarks_smoothed.json

# Also render an overlay video (raw=red, smoothed=green) for N frames:
python examples/utils/debug_smoothing.py \
    --landmarks /path/to/2_landmarks.json \
    --video     /path/to/2.mov \
    --overlay_frames 200
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cruxes.utils.smoothnet import (
    SmoothNet,
    auto_detect_device,
    train_smoothnet,
    infer_smoothnet,
    _fill_invalid_frames,
)


def run_gaussian(coords, valid, sigma):
    """Apply per-joint Gaussian filter over valid-frame subsequences."""
    from scipy.ndimage import gaussian_filter1d
    T, n_joints, _ = coords.shape
    sm = coords.copy()
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return sm
    # Build compact sequence of valid frames, smooth, scatter back
    for j in range(n_joints):
        for ax in range(2):
            seq = coords[valid_idx, j, ax]
            sm[valid_idx, j, ax] = gaussian_filter1d(seq, sigma=sigma)
    return sm


# MediaPipe joint names for the subset we care about
_JOINT_NAMES = {
    0:  "nose",
    11: "l_shoulder", 12: "r_shoulder",
    13: "l_elbow",    14: "r_elbow",
    15: "l_wrist",    16: "r_wrist",
    23: "l_hip",      24: "r_hip",
}

# Skeleton connections (MediaPipe-33 subset used by ViTPose)
_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
]


def load_landmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload["frames"]
    T = len(frames)
    coords = np.full((T, 33, 2), np.nan, dtype=np.float32)
    confs  = np.zeros((T, 33), dtype=np.float32)
    valid  = np.zeros(T, dtype=bool)
    for t, frame in enumerate(frames):
        if frame is None:
            continue
        valid[t] = True
        for i, lm in enumerate(frame):
            coords[t, i, 0] = float(lm.get("x") or 0.0)
            coords[t, i, 1] = float(lm.get("y") or 0.0)
            vis = lm.get("visibility") or 0.0
            confs[t, i] = float(np.clip(vis, 0.0, 1.0))
    return coords, confs, valid, payload


def run_smoothnet(coords, confs, valid, window_size, epochs, lambda_accel, device):
    T = len(valid)
    n_joints = 33
    coords_flat = coords.reshape(T, n_joints * 2).copy()
    confs_flat  = np.repeat(confs, 2, axis=1)
    # fill NaN slots before interpolation
    coords_flat[np.isnan(coords_flat)] = 0.0
    coords_flat = _fill_invalid_frames(coords_flat, valid)
    confs_flat[~valid] = 0.0

    model = SmoothNet(
        window_size=window_size, n_channels=66,
        hidden_size=512, res_hidden_size=128, num_blocks=5, dropout=0.25,
    ).to(device)

    train_smoothnet(model, coords_flat, confs_flat,
                    window_size=window_size, epochs=epochs,
                    lambda_accel=lambda_accel, device=device)

    refined_flat = infer_smoothnet(model, coords_flat, window_size, device)

    # Affine correction: match mean AND std of each channel over valid frames.
    raw_mean = coords_flat[valid].mean(axis=0)
    sm_mean  = refined_flat[valid].mean(axis=0)
    raw_std  = coords_flat[valid].std(axis=0) + 1e-8
    sm_std   = refined_flat[valid].std(axis=0) + 1e-8
    scale    = raw_std / sm_std
    refined_flat = (refined_flat - sm_mean) * scale + raw_mean
    bias = sm_mean - raw_mean
    if np.abs(bias).max() > 1e-4 or np.abs(scale - 1.0).max() > 1e-3:
        print(
            f"  Affine correction: max|bias|={np.abs(bias).max():.4f}  "
            f"scale=[{scale.min():.3f}, {scale.max():.3f}]"
        )

    return np.clip(refined_flat.reshape(T, n_joints, 2), 0.0, 1.0)


def plot_comparison(raw_coords, sm_coords, valid, joints, out_path):
    """Plot raw (blue) vs smoothed (orange) x and y for each joint over time."""
    n = len(joints)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    frames = np.arange(raw_coords.shape[0])

    for row, j in enumerate(joints):
        name = _JOINT_NAMES.get(j, f"joint_{j}")
        raw_x = np.where(valid, raw_coords[:, j, 0], np.nan)
        raw_y = np.where(valid, raw_coords[:, j, 1], np.nan)
        sm_x  = np.where(valid, sm_coords[:, j, 0], np.nan)
        sm_y  = np.where(valid, sm_coords[:, j, 1], np.nan)

        ax_x, ax_y = axes[row]
        ax_x.plot(frames, raw_x, lw=0.8, color="steelblue",  label="raw",      alpha=0.8)
        ax_x.plot(frames, sm_x,  lw=0.8, color="darkorange", label="smoothed", alpha=0.8)
        ax_x.set_ylabel(f"{name}\nx (normalised)")
        ax_x.legend(fontsize=7)

        ax_y.plot(frames, raw_y, lw=0.8, color="steelblue",  alpha=0.8)
        ax_y.plot(frames, sm_y,  lw=0.8, color="darkorange", alpha=0.8)
        ax_y.set_ylabel("y (normalised)")

    axes[-1][0].set_xlabel("frame")
    axes[-1][1].set_xlabel("frame")
    plt.suptitle("Raw (blue) vs Smoothed (orange)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved coordinate plot → {out_path}")


def print_stats(raw_coords, sm_coords, valid, joints):
    print("\n── Per-joint displacement stats (raw → smoothed, valid frames only) ──")
    print(f"  {'joint':<14}  {'mean |Δ|':>9}  {'max |Δ|':>9}  {'mean Δx':>9}  {'mean Δy':>9}")
    for j in joints:
        name = _JOINT_NAMES.get(j, f"joint_{j}")
        raw = raw_coords[valid, j, :]   # (V, 2)
        sm  = sm_coords[valid, j, :]
        diff = sm - raw
        dist = np.linalg.norm(diff, axis=1)
        print(
            f"  {name:<14}  {dist.mean():>9.5f}  {dist.max():>9.5f}"
            f"  {diff[:,0].mean():>+9.5f}  {diff[:,1].mean():>+9.5f}"
        )
    print()


def render_overlay_video(video_path, raw_coords, sm_coords, valid, out_path, n_frames):
    """Render a video with raw skeleton (red) and smoothed skeleton (green) overlaid."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    while cap.isOpened() and frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if valid[frame_idx]:
            for jA, jB in _CONNECTIONS:
                # raw — blue
                pA = (int(raw_coords[frame_idx, jA, 0] * w),
                      int(raw_coords[frame_idx, jA, 1] * h))
                pB = (int(raw_coords[frame_idx, jB, 0] * w),
                      int(raw_coords[frame_idx, jB, 1] * h))
                cv2.line(frame, pA, pB, (200, 80, 80), 2)

                # smoothed — green
                sA = (int(sm_coords[frame_idx, jA, 0] * w),
                      int(sm_coords[frame_idx, jA, 1] * h))
                sB = (int(sm_coords[frame_idx, jB, 0] * w),
                      int(sm_coords[frame_idx, jB, 1] * h))
                cv2.line(frame, sA, sB, (80, 200, 80), 2)

            for j in _JOINT_NAMES:
                rx, ry = (int(raw_coords[frame_idx, j, 0] * w),
                          int(raw_coords[frame_idx, j, 1] * h))
                sx, sy = (int(sm_coords[frame_idx, j, 0] * w),
                          int(sm_coords[frame_idx, j, 1] * h))
                cv2.circle(frame, (rx, ry), 4, (200, 80, 80), -1)   # raw   blue dot
                cv2.circle(frame, (sx, sy), 4, (80, 200, 80), -1)   # smooth green dot

        # legend
        cv2.putText(frame, "raw",      (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 80, 80),  2)
        cv2.putText(frame, "smoothed", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 80), 2)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved overlay video ({frame_idx} frames) → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--landmarks", "-l", required=True,
                        help="Raw landmarks JSON.")
    parser.add_argument("--smoothed", "-s", default=None,
                        help="Pre-computed smoothed landmarks JSON. "
                             "If omitted, SmoothNet is run here.")
    parser.add_argument("--video", "-v", default=None,
                        help="Source video path for overlay rendering.")
    parser.add_argument("--overlay_frames", type=int, default=0,
                        help="Number of frames to render in overlay video (0 = skip).")
    parser.add_argument("--method", type=str, default="smoothnet",
                        choices=["smoothnet", "gaussian"],
                        help="Smoothing method to run (default: smoothnet).")
    parser.add_argument("--gaussian_sigma", type=float, default=3.0,
                        help="Sigma (frames) for Gaussian smoothing (default: 3.0).")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--lambda_accel",type=float, default=0.1)
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--joints", type=int, nargs="+",
                        default=[0, 15, 16, 11, 12],
                        help="Joint indices to plot.")
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device else auto_detect_device())
    # lazy torch import only needed for actual training
    import torch

    stem   = Path(args.landmarks).stem
    parent = Path(args.landmarks).parent

    # ── Load raw ──
    print(f"Loading raw: {args.landmarks}")
    raw_coords, confs, valid, payload = load_landmarks(args.landmarks)
    raw_for_plot = raw_coords.copy()
    raw_for_plot[~valid] = np.nan

    # ── Load or compute smoothed ──
    if args.smoothed:
        print(f"Loading smoothed: {args.smoothed}")
        sm_coords, _, _, _ = load_landmarks(args.smoothed)
        sm_coords[~valid] = np.nan
    elif args.method == "gaussian":
        print(f"Running Gaussian filter (sigma={args.gaussian_sigma})...")
        sm_coords = run_gaussian(raw_coords.copy(), valid, args.gaussian_sigma)
        sm_path = str(parent / f"{stem}_smoothed_gaussian.json")
        out_frames = copy.deepcopy(payload["frames"])
        for t, frame in enumerate(out_frames):
            if frame is None or not valid[t]:
                continue
            for i, lm in enumerate(frame):
                lm["x"] = float(np.clip(sm_coords[t, i, 0], 0.0, 1.0))
                lm["y"] = float(np.clip(sm_coords[t, i, 1], 0.0, 1.0))
        out_payload = copy.deepcopy(payload)
        out_payload["frames"] = out_frames
        with open(sm_path, "w") as f:
            json.dump(out_payload, f)
        print(f"Saved smoothed JSON → {sm_path}")
    else:
        device = auto_detect_device() if args.device is None else torch.device(args.device)
        print("Running SmoothNet...")
        sm_coords = run_smoothnet(
            raw_coords.copy(), confs, valid,
            args.window_size, args.epochs, args.lambda_accel, device,
        )
        # Save smoothed JSON
        sm_path = str(parent / f"{stem}_smoothed.json")
        out_frames = copy.deepcopy(payload["frames"])
        for t, frame in enumerate(out_frames):
            if frame is None:
                continue
            for i, lm in enumerate(frame):
                lm["x"] = float(np.clip(sm_coords[t, i, 0], 0.0, 1.0))
                lm["y"] = float(np.clip(sm_coords[t, i, 1], 0.0, 1.0))
        out_payload = copy.deepcopy(payload)
        out_payload["frames"] = out_frames
        with open(sm_path, "w") as f:
            json.dump(out_payload, f)
        print(f"Saved smoothed JSON → {sm_path}")

    # ── Stats ──
    print_stats(raw_coords, sm_coords, valid, args.joints)

    # ── Plot ──
    plot_path = str(parent / f"{stem}_smoothing_debug.png")
    plot_comparison(raw_for_plot, sm_coords, valid, args.joints, plot_path)

    # ── Overlay video ──
    if args.overlay_frames > 0 and args.video:
        overlay_path = str(parent / f"{stem}_overlay.mp4")
        render_overlay_video(
            args.video, raw_coords, sm_coords, valid,
            overlay_path, args.overlay_frames,
        )


if __name__ == "__main__":
    main()
