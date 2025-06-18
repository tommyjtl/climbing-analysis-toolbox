import argparse
from termcolor import colored

from utils.body_trajectory import extract_pose_and_draw_trajectory
from utils.file_operations import get_output_path

from tools import Cruxes

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

    cruxes = Cruxes()
    cruxes.body_trajectory(
        target_video_path,
        track_point=[
            "hip_mid",
            # "upper_body_center",
            # "head",
            "left_hand",
            "right_hand",
            # "left_foot",
            # "right_foot",
        ],
        overlay_trajectory=False,
        draw_pose=True,
        kalman_settings=[  # Kalman filter settings: [use_kalman : bool, kalman_gain : float]
            True,  # Set this to false if you don't want to apply Kalman filter
            1e0,  # >=1e0 for higher noise, <=1e-1 for lower noise
        ],
        trajectory_png_path=None,
    )


if __name__ == "__main__":
    main()
