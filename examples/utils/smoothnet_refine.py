"""
SmoothNet Temporal Refinement for Pose Landmarks
=================================================
Self-supervised SmoothNet (ECCV 2022) applied to pose landmark JSON files.

SmoothNet architecture: Zeng et al., "SmoothNet: A Plug-and-Play Network for
Refining Human Poses in Videos", ECCV 2022. Apache-2.0 license.
https://github.com/cure-lab/SmoothNet

The model is trained on the video's own landmarks using:
  - Visibility-weighted reconstruction loss  (keeps confident detections anchored)
  - Acceleration penalty                     (smoothness prior — penalises jerk)

High-visibility/confidence frames act as anchors.
Low-visibility/occluded frames are pulled toward what the temporal context
from the surrounding window predicts.

Workflow:
  1. Run body_trajectory with --export_landmarks to produce a landmarks JSON.
  2. Run this script to refine the landmarks.
  3. Re-run body_trajectory with --use_cached_landmarks pointing at the refined file.

Usage:
    python examples/utils/smoothnet_refine.py \\
        --landmarks path/to/2_landmarks.json

    # custom output path and tuning knobs
    python examples/utils/smoothnet_refine.py \\
        --landmarks path/to/2_landmarks.json \\
        --output    path/to/2_landmarks_refined.json \\
        --window_size 32 \\
        --epochs 120 \\
        --lambda_accel 1.0

Arguments
---------
--landmarks       Input landmarks JSON (required).
--output          Output path. Defaults to <stem>_smoothed.json next to input.
--window_size     Temporal window fed to the network (default: 32 frames).
                  Reduce for very short clips; increase for long, smooth moves.
--epochs          Training epochs (default: 100). ~30 s on MPS / CPU.
--lr              Learning rate (default: 1e-3).
--lambda_accel    Weight for the acceleration smoothness loss (default: 1.0).
                  Higher = smoother output, but may blur fast dynamic moves.
--device          'cpu', 'cuda', or 'mps'. Auto-detected if omitted.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cruxes.utils.smoothnet import (  # noqa: E402
    SmoothNet,
    auto_detect_device,
    train_smoothnet,
    infer_smoothnet,
    _fill_invalid_frames,
)

# ── Data loading / saving ──────────────────────────────────────────────────────


def load_landmarks(path: str):
    """Load a landmarks JSON into numpy arrays.

    Returns:
        coords      (T, 33, 2)  normalised x,y for each frame and joint
        confs       (T, 33)     visibility score per joint per frame
        payload     the raw JSON dict (needed for saving back)
        valid_mask  (T,) bool   True if the frame has landmarks
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload["frames"]
    T = len(frames)
    coords = np.zeros((T, 33, 2), dtype=np.float32)
    confs = np.zeros((T, 33), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)

    for t, frame in enumerate(frames):
        if frame is None:
            continue
        valid_mask[t] = True
        for i, lm in enumerate(frame):
            coords[t, i, 0] = float(lm.get("x") or 0.0)
            coords[t, i, 1] = float(lm.get("y") or 0.0)
            vis = lm.get("visibility") or 0.0
            confs[t, i] = float(np.clip(vis, 0.0, 1.0))

    return coords, confs, payload, valid_mask


def save_landmarks(payload: dict, refined_coords: np.ndarray, path: str):
    """Write refined x/y back into a copy of payload, preserving all other fields."""
    out_frames = copy.deepcopy(payload["frames"])
    for t, frame in enumerate(out_frames):
        if frame is None:
            continue
        for i, lm in enumerate(frame):
            lm["x"] = float(np.clip(refined_coords[t, i, 0], 0.0, 1.0))
            lm["y"] = float(np.clip(refined_coords[t, i, 1], 0.0, 1.0))
    out_payload = copy.deepcopy(payload)
    out_payload["frames"] = out_frames
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f)
    print(f"Saved refined landmarks → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Self-supervised SmoothNet temporal refinement for pose landmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--landmarks", "-l", required=True, help="Input landmarks JSON path."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path. Defaults to <stem>_smoothed.json.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=32,
        help="Temporal window size. Reduce for short clips (<150 frames).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--lambda_accel",
        type=float,
        default=0.1,
        help="Acceleration smoothness loss weight. "
        "Higher = smoother, lower = stays closer to raw detections.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="SmoothNet encoder/decoder hidden size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cpu', 'cuda', 'mps'.",
    )
    args = parser.parse_args()

    # ── Device ──
    device = torch.device(args.device) if args.device else auto_detect_device()
    print(f"Device: {device}")

    # ── Output path ──
    if args.output is None:
        stem = Path(args.landmarks).stem
        parent = Path(args.landmarks).parent
        args.output = str(parent / f"{stem}_smoothed.json")

    # ── Load ──
    print(f"Loading: {args.landmarks}")
    coords, confs, payload, valid_mask = load_landmarks(args.landmarks)
    T = coords.shape[0]
    n_valid = int(valid_mask.sum())
    print(f"  {T} total frames, {n_valid} with landmarks")

    if n_valid < args.window_size:
        suggested = max(4, n_valid // 2)
        print(
            f"Error: need at least --window_size ({args.window_size}) valid frames "
            f"but only {n_valid} are available.\n"
            f"Try: --window_size {suggested}"
        )
        sys.exit(1)

    # ── Prepare flat arrays ──
    # Shape: (T, 66) — 33 joints × 2 coords (x, y interleaved as pairs)
    coords_flat = coords.reshape(T, 66)
    # Repeat confidence for both x and y of each joint
    confs_flat = np.repeat(confs, 2, axis=1)  # (T, 66)

    # Fill invalid frames with interpolated values so the acceleration penalty
    # sees a smooth trajectory rather than discontinuous zero-spikes.
    coords_flat = _fill_invalid_frames(coords_flat, valid_mask)
    confs_flat[~valid_mask] = 0.0

    # ── Build model ──
    model = SmoothNet(
        window_size=args.window_size,
        n_channels=66,
        hidden_size=args.hidden_size,
        res_hidden_size=args.hidden_size // 4,
        num_blocks=5,
        dropout=0.25,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SmoothNet: {n_params:,} parameters")

    # ── Train ──
    train_smoothnet(
        model,
        coords_flat,
        confs_flat,
        window_size=args.window_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_accel=args.lambda_accel,
        device=device,
    )

    # ── Infer ──
    print("Running sliding-window inference...")
    refined_flat = infer_smoothnet(model, coords_flat, args.window_size, device)
    refined_coords = refined_flat.reshape(T, 33, 2)

    # Only update frames that originally had landmarks; leave None frames as-is
    # (save_landmarks already handles None frames via deepcopy)

    # ── Save ──
    save_landmarks(payload, refined_coords, args.output)

    print()
    print("Re-render with refined landmarks:")
    print(f"  python examples/scripts/body_trajectory_demo.py \\")
    print(f"    --use_cached_landmarks \\")
    print(f"    --landmarks_json_path '{args.output}' \\")
    print(f"    --video_path <your_video>")


if __name__ == "__main__":
    main()
