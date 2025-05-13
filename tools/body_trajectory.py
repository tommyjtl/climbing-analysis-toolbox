import argparse
from termcolor import colored

from utils.body_trajectory import extract_pose_and_draw_trajectory


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

    extract_pose_and_draw_trajectory(
        target_video_path,
        track_point=[
            # "hip_mid",
            "upper_body_center",
            # "head",
            "left_hand",
            "right_hand",
            "left_foot",
            "right_foot",
        ],
        overlay_trajectory=True,
        show_gauges=False,
        draw_pose=True,
        kalman_settings=[  # Kalman filter settings: [use_kalman, kalman_gain]
            True,
            1e0,  # >=1e0 for higher noise, <=1e-1 for lower noise
        ],
    )


if __name__ == "__main__":
    main()
