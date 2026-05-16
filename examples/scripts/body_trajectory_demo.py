import argparse
from termcolor import colored

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cruxes import Cruxes

# References for pose smoothing techniques:
# - https://stackoverflow.com/questions/52450681/how-can-i-use-smoothing-techniques-to-remove-jitter-in-pose-estimation)
# - https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
# - https://en.wikipedia.org/wiki/Kalman_filter
# - Papers
#   - https://arxiv.org/abs/2011.00250
#   - https://ailingzeng.site/smoothnet
#   - https://dellaert.github.io/files/Ranganathan07iros.pdf
#   - https://www.youtube.com/watch?v=yrQ3ZU4zB6Q
#   - https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Towards_Robust_and_Smooth_3D_Multi-Person_Pose_Estimation_from_Monocular_ICCV_2023_paper.pdf


def main():
    parser = argparse.ArgumentParser(
        description="Extract pose and draw trajectory from a video."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--json_only",
        action="store_true",
        default=False,
        help="Export JSON artifacts only, without rendering an output video. This also enables the separate pose world landmarks export.",
    )
    parser.add_argument(
        "--export_world_landmarks",
        action="store_true",
        default=False,
        help="Export MediaPipe pose world landmarks to a separate WebGPU-friendly JSON file.",
    )
    parser.add_argument(
        "--world_landmarks_json_path",
        type=str,
        default=None,
        help="Optional path to the pose world landmarks JSON output.",
    )
    parser.add_argument(
        "--pose_backend",
        type=str,
        default="mediapipe",
        choices=["mediapipe", "vitpose"],
        help="Pose estimation backend to use (default: mediapipe).",
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        default=None,
        choices=["savgol", "gaussian", "smoothnet"],
        help="Temporal smoothing method applied after pose estimation (default: none).",
    )
    parser.add_argument(
        "--savgol_window",
        type=int,
        default=11,
        help="Savgol filter window length (must be odd, default: 11).",
    )
    parser.add_argument(
        "--savgol_order",
        type=int,
        default=3,
        help="Savgol filter polynomial order (default: 3).",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=3.0,
        help="Standard deviation (in frames) for Gaussian smoothing filter (default: 3.0).",
    )
    parser.add_argument(
        "--smoothnet_window_size",
        type=int,
        default=32,
        help="SmoothNet temporal window size in frames (default: 32).",
    )
    parser.add_argument(
        "--smoothnet_epochs",
        type=int,
        default=100,
        help="SmoothNet self-supervised training epochs (default: 100).",
    )
    parser.add_argument(
        "--smoothnet_lambda_accel",
        type=float,
        default=0.1,
        help="SmoothNet acceleration loss weight — higher = smoother (default: 0.1).",
    )
    parser.add_argument(
        "--trajectory_thickness",
        type=int,
        default=None,
        help="Thickness in pixels for trajectory lines and velocity arrows (default: 5).",
    )
    parser.add_argument(
        "--velocity_arrow_length",
        type=float,
        default=None,
        help="Scale factor for velocity arrow length (default: 40).",
    )
    parser.add_argument(
        "--use_cached_landmarks",
        action="store_true",
        default=False,
        help="Load landmarks from a previously saved JSON cache instead of re-running pose detection. Falls back to re-detection if no valid cache exists.",
    )
    args = parser.parse_args()
    if not args.video_path or args.video_path == "":
        print(
            colored(
                "Warning: No video path supplied. Please provide --video_path.",
                "red",
            )
        )
        return
    target_video_path = args.video_path

    # Print colored messages for debugging
    print(colored("Target video path:", "blue"), target_video_path)
    print(
        colored("Export world landmarks:", "blue"),
        args.export_world_landmarks,
    )
    print(
        colored("World landmarks JSON path:", "blue"),
        args.world_landmarks_json_path,
    )
    print(colored("Pose backend:", "blue"), args.pose_backend)
    print(colored("Smoothing:", "blue"), args.smoothing)
    print(colored("Use cached landmarks:", "blue"), args.use_cached_landmarks)

    cruxes = Cruxes()
    cruxes.body_trajectory(
        target_video_path,
        json_only=args.json_only,
        # tracking relevant
        track_point=[
            "hip_mid",
            # "upper_body_center",
            # "head",
            "left_hand",
            "right_hand",
            "left_foot",
            "right_foot",
        ],
        draw_pose=True,
        pose_color=(0, 0, 255),
        show_trajectory=True,
        show_gauges=False,
        trajectory_history_seconds=0.2,
        use_cached_landmarks=args.use_cached_landmarks,
        # use_cached_trajectory_metadata=True,
        # export_landmarks=True,
        export_metadata=True,
        overlay_mask=True,
        hide_original_video=False,
        kalman_settings=[  # Kalman filter settings: [use_kalman : bool, kalman_gain : float]
            True,  # Set this to false if you don't want to apply Kalman filter
            0.5e0,  # >=1e0 for higher noise, <=1e-1 for lower noise
        ],
        # additional args
        smoothing=args.smoothing,
        savgol_window=args.savgol_window,
        savgol_order=args.savgol_order,
        gaussian_sigma=args.gaussian_sigma,
        smoothnet_window_size=args.smoothnet_window_size,
        smoothnet_epochs=args.smoothnet_epochs,
        smoothnet_lambda_accel=args.smoothnet_lambda_accel,
        export_world_landmarks=args.export_world_landmarks,
        world_landmarks_json_path=args.world_landmarks_json_path,
        pose_backend=args.pose_backend,
        trajectory_thickness=args.trajectory_thickness,
        velocity_arrow_length=args.velocity_arrow_length,
    )


if __name__ == "__main__":
    main()
