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
        "--track_point_visibility_threshold",
        type=float,
        default=0.6,
        help="Minimum visibility required for tracked joints and derived track points.",
    )
    parser.add_argument(
        "--pose_visibility_threshold",
        type=float,
        default=0.4,
        help="Minimum visibility required to render a pose landmark.",
    )
    parser.add_argument(
        "--pose_presence_threshold",
        type=float,
        default=0.4,
        help="Minimum presence required to render a pose landmark.",
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
        "--trajectory_thickness",
        type=int,
        default=None,
        help="Thickness in pixels for trajectory lines and velocity arrows (default: 5).",
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
        colored("Track point visibility threshold:", "blue"),
        args.track_point_visibility_threshold,
    )
    print(
        colored("Pose visibility threshold:", "blue"),
        args.pose_visibility_threshold,
    )
    print(
        colored("Pose presence threshold:", "blue"),
        args.pose_presence_threshold,
    )
    print(
        colored("Export world landmarks:", "blue"),
        args.export_world_landmarks,
    )
    print(
        colored("World landmarks JSON path:", "blue"),
        args.world_landmarks_json_path,
    )
    print(colored("Pose backend:", "blue"), args.pose_backend)

    cruxes = Cruxes()
    cruxes.body_trajectory(
        target_video_path,
        json_only=args.json_only,
        # tracking relevant
        track_point=[
            "hip_mid",
            "upper_body_center",
            # "head",
            "left_hand",
            "right_hand",
            "left_foot",
            "right_foot",
        ],
        # trajectory_only=True,
        #
        draw_pose=True,
        pose_color=(255, 255, 255),
        show_trajectory=True,
        show_gauges=False,
        trajectory_history_seconds=0.5,
        use_cached_landmarks=False,
        # use_cached_trajectory_metadata=True,
        export_landmarks=True,
        # export_metadata=True,
        overlay_mask=False,
        hide_original_video=False,
        kalman_settings=[  # Kalman filter settings: [use_kalman : bool, kalman_gain : float]
            True,  # Set this to false if you don't want to apply Kalman filter
            0.5e0,  # >=1e0 for higher noise, <=1e-1 for lower noise
        ],
        # Savitzky-Golay filter settings: [use_savgol : bool, window_length : int, polyorder : int]
        # Window length must be odd and > polyorder
        savgol_settings=[
            True,  # Set to True to smooth the pose skeleton after pose estimation
            15,  # Window length (must be odd, typical: 5-15)
            4,  # Polynomial order (typical: 2-4, must be < window_length)
        ],
        track_point_visibility_threshold=args.track_point_visibility_threshold,
        pose_visibility_threshold=args.pose_visibility_threshold,
        pose_presence_threshold=args.pose_presence_threshold,
        export_world_landmarks=args.export_world_landmarks,
        world_landmarks_json_path=args.world_landmarks_json_path,
        pose_backend=args.pose_backend,
        trajectory_thickness=args.trajectory_thickness,
    )


if __name__ == "__main__":
    main()
