"""
SmoothNet temporal refinement — shared model and training utilities.

SmoothNet architecture: Zeng et al., "SmoothNet: A Plug-and-Play Network for
Refining Human Poses in Videos", ECCV 2022. Apache-2.0 license.
https://github.com/cure-lab/SmoothNet

This module is imported lazily inside body_trajectory.py (only when
smoothing="smoothnet" is requested), so PyTorch is not a hard dependency
for users who use savgol or no smoothing.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# ── Device helpers ─────────────────────────────────────────────────────────────


def auto_detect_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model ──────────────────────────────────────────────────────────────────────


class SmoothNetResBlock(nn.Module):
    """Residual FC block. Input/output shape: (*, in_channels)."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.25):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        return x + identity


class SmoothNet(nn.Module):
    """SmoothNet temporal refinement network.

    Input:  (N, C, window_size)
    Output: (N, C, window_size)

    C = n_joints * 2  (x,y pair per joint, flattened).
    window_size       = temporal window width.
    """

    def __init__(
        self,
        window_size: int,
        n_channels: int,
        hidden_size: int = 512,
        res_hidden_size: int = 128,
        num_blocks: int = 5,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels

        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[
                SmoothNetResBlock(hidden_size, res_hidden_size, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.decoder = nn.Linear(hidden_size, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T) where T == window_size.  Returns (N, C, T)."""
        N, C, T = x.shape
        assert T == self.window_size, f"Expected window T={self.window_size}, got {T}"
        x = x.float()
        x = self.encoder(x)  # (N, C, hidden_size)
        x = self.res_blocks(x)  # (N, C, hidden_size)
        x = self.decoder(x)  # (N, C, window_size)
        return x


# ── Data helpers ───────────────────────────────────────────────────────────────


def _fill_invalid_frames(coords_flat: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill invalid frames with linearly interpolated values from valid neighbours.

    This prevents the acceleration penalty from seeing discontinuous zero-spikes
    at invalid frames, which would otherwise pull the surrounding valid frames'
    predictions toward the origin and compress the skeleton.

    The reconstruction loss is *not* affected — confidence stays 0 for invalid
    frames so they contribute zero gradient there regardless.
    """
    filled = coords_flat.copy()
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return filled

    for t in range(len(valid_mask)):
        if valid_mask[t]:
            continue
        before = valid_indices[valid_indices < t]
        after = valid_indices[valid_indices > t]
        if len(before) > 0 and len(after) > 0:
            t0, t1 = int(before[-1]), int(after[0])
            alpha = float(t - t0) / float(t1 - t0)
            filled[t] = (1.0 - alpha) * coords_flat[t0] + alpha * coords_flat[t1]
        elif len(before) > 0:
            filled[t] = coords_flat[int(before[-1])]
        elif len(after) > 0:
            filled[t] = coords_flat[int(after[0])]
    return filled


def extract_windows(sequence: np.ndarray, window_size: int, stride: int = 1):
    """sequence: (T, C).  Returns windows (N, C, W) and list of start indices."""
    T, C = sequence.shape
    windows, starts = [], []
    for s in range(0, T - window_size + 1, stride):
        windows.append(sequence[s : s + window_size].T)  # (C, W)
        starts.append(s)
    if not windows:
        return np.empty((0, C, window_size), dtype=np.float32), []
    return np.stack(windows).astype(np.float32), starts


# ── Training ───────────────────────────────────────────────────────────────────


def train_smoothnet(
    model: SmoothNet,
    coords_flat: np.ndarray,  # (T, C)
    confs_flat: np.ndarray,  # (T, C)  visibility weight per channel
    window_size: int,
    epochs: int = 100,
    lr: float = 1e-3,
    lambda_accel: float = 0.1,
    device: torch.device | None = None,
    verbose: bool = True,
):
    """Self-supervised training loop.

    Loss = visibility_weighted_reconstruction + lambda_accel * acceleration_penalty

    High-visibility frames anchor the network.
    Low-visibility/occluded frames are free to be pulled by surrounding context.
    """
    if device is None:
        device = auto_detect_device()

    stride = max(1, window_size // 4)
    windows, starts = extract_windows(coords_flat, window_size, stride)
    conf_windows, _ = extract_windows(confs_flat, window_size, stride)

    if len(windows) == 0:
        print(
            "  Warning: not enough frames for window size — skipping SmoothNet training."
        )
        return

    X = torch.from_numpy(windows).to(device)  # (N, C, W)
    W = torch.from_numpy(conf_windows).to(device)  # (N, C, W)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.97)

    if verbose:
        print(
            f"  SmoothNet: {len(windows)} windows × {coords_flat.shape[1]} channels"
            f" × {window_size} frames  [{device}]"
        )

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        refined = model(X)  # (N, C, W)

        # Weighted reconstruction — confident frames are hard anchors
        recon_loss = (W * (refined - X).pow(2)).sum() / (W.sum() + 1e-8)

        # Acceleration penalty: ||x[t+2] - 2*x[t+1] + x[t]||^2
        accel = refined[:, :, 2:] - 2 * refined[:, :, 1:-1] + refined[:, :, :-2]
        accel_loss = accel.pow(2).mean()

        loss = recon_loss + lambda_accel * accel_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(
                f"  [{epoch + 1:4d}/{epochs}]  loss={loss.item():.6f}"
                f"  recon={recon_loss.item():.6f}  accel={accel_loss.item():.6f}"
            )

    model.eval()


# ── Inference ──────────────────────────────────────────────────────────────────


def infer_smoothnet(
    model: SmoothNet,
    coords_flat: np.ndarray,  # (T, C)
    window_size: int,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:  # (T, C) refined
    """Inference without any padding.

    For each frame t, build a window clamped entirely within [0, T):
        window_start = clamp(t - window_size//2, 0, T - window_size)
        position_in_window = t - window_start

    This guarantees:
    - No reflected or zero-padded frames enter the network.
    - Every frame uses real data only.
    - Frames near the edges share the same window but read different positions
      within it (the window is not re-centred, just clamped).
    """
    if device is None:
        device = auto_detect_device()

    T, C = coords_flat.shape
    half = window_size // 2
    refined = np.zeros((T, C), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for b_start in range(0, T, batch_size):
            b_end = min(b_start + batch_size, T)
            batch_windows = []
            positions = []

            for t in range(b_start, b_end):
                start = int(np.clip(t - half, 0, T - window_size))
                pos = t - start  # index inside the window for frame t
                batch_windows.append(coords_flat[start : start + window_size].T)
                positions.append(pos)

            batch = torch.from_numpy(np.stack(batch_windows).astype(np.float32)).to(
                device
            )
            out = model(batch).cpu().numpy()  # (B, C, W)

            for i, pos in enumerate(positions):
                refined[b_start + i] = out[i, :, pos]

    return refined


# ── High-level pipeline integration ───────────────────────────────────────────


def apply_smoothnet_to_landmarks(
    all_pose_landmarks,
    window_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    lambda_accel: float = 0.1,
    device=None,
) -> list:
    """Apply SmoothNet to a list of per-frame pose landmark lists.

    Parameters
    ----------
    all_pose_landmarks : list[None | list[NormalizedPoseLandmark]]
        One entry per video frame; None for frames with no detection.
    window_size, epochs, lr, lambda_accel :
        SmoothNet training/inference hyperparameters.
    device : torch.device | str | None
        Compute device.  Auto-detected when None.

    Returns
    -------
    smoothed_pose_landmarks : list[dict]
        Same format as the savgol output used by _build_render_pose_landmarks:
        a list (one per frame) of dicts mapping joint_index → {x, y, z, visibility, presence}.
        Frames with no detections produce empty dicts.
    """
    if device is None:
        device = auto_detect_device()
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    T = len(all_pose_landmarks)
    n_joints = 33

    coords = np.zeros((T, n_joints, 2), dtype=np.float32)
    confs = np.zeros((T, n_joints), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)

    for t, frame_lms in enumerate(all_pose_landmarks):
        if frame_lms is None:
            continue
        valid_mask[t] = True
        for j, lm in enumerate(frame_lms):
            coords[t, j, 0] = float(lm.x)
            coords[t, j, 1] = float(lm.y)
            vis = lm.visibility
            confs[t, j] = float(np.clip(vis if vis is not None else 0.0, 0.0, 1.0))

    n_valid = int(valid_mask.sum())
    if n_valid < window_size:
        print(
            f"  Warning: SmoothNet needs >= {window_size} valid frames "
            f"but only {n_valid} available.  Falling back to raw landmarks."
        )
        return [{} for _ in all_pose_landmarks]

    coords_flat = coords.reshape(T, n_joints * 2)  # (T, 66)
    confs_flat = np.repeat(confs, 2, axis=1)  # (T, 66) — same weight for x and y

    # Fill invalid frames with interpolated values so the acceleration penalty
    # sees a smooth trajectory rather than discontinuous zero-spikes.
    # Reconstruction loss still ignores these frames (confidence stays 0).
    coords_flat = _fill_invalid_frames(coords_flat, valid_mask)
    confs_flat[~valid_mask] = 0.0

    model = SmoothNet(
        window_size=window_size,
        n_channels=n_joints * 2,
        hidden_size=512,
        res_hidden_size=128,
        num_blocks=5,
        dropout=0.25,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SmoothNet: {n_params:,} parameters")

    train_smoothnet(
        model,
        coords_flat,
        confs_flat,
        window_size=window_size,
        epochs=epochs,
        lr=lr,
        lambda_accel=lambda_accel,
        device=device,
    )

    print("  Running SmoothNet inference...")
    refined_flat = infer_smoothnet(model, coords_flat, window_size, device)

    # Per-channel affine correction: the network (trained from scratch on a short
    # clip) introduces a systematic DC shift AND compresses coordinate variance.
    # Enforce that every channel's mean AND standard deviation over valid frames
    # matches the raw input.  This preserves the temporal smoothing shape while
    # anchoring both the position AND the spread to the original distribution.
    raw_mean = coords_flat[valid_mask].mean(axis=0)  # (66,)
    sm_mean = refined_flat[valid_mask].mean(axis=0)  # (66,)
    raw_std = coords_flat[valid_mask].std(axis=0) + 1e-8  # (66,)
    sm_std = refined_flat[valid_mask].std(axis=0) + 1e-8  # (66,)
    scale = raw_std / sm_std
    refined_flat = (refined_flat - sm_mean) * scale + raw_mean
    bias = sm_mean - raw_mean
    if np.abs(bias).max() > 1e-4 or np.abs(scale - 1.0).max() > 1e-3:
        print(
            f"  Affine correction applied  "
            f"max|bias|={np.abs(bias).max():.4f}  "
            f"scale range=[{scale.min():.3f}, {scale.max():.3f}]"
        )

    refined_coords = np.clip(refined_flat.reshape(T, n_joints, 2), 0.0, 1.0)

    # Build the smoothed_pose_landmarks list of dicts (same shape as savgol output)
    smoothed_pose_landmarks: list[dict] = [{} for _ in range(T)]
    for t in range(T):
        if not valid_mask[t]:
            continue
        frame_lms = all_pose_landmarks[t]
        for j in range(n_joints):
            smoothed_pose_landmarks[t][j] = {
                "x": float(refined_coords[t, j, 0]),
                "y": float(refined_coords[t, j, 1]),
                "z": float(frame_lms[j].z),  # z unchanged (2D model)
                "visibility": frame_lms[j].visibility,
                "presence": frame_lms[j].presence,
            }

    return smoothed_pose_landmarks
