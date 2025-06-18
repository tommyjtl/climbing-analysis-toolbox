import argparse
from termcolor import colored
import os

from matching import get_matcher

from utils.file_operations import get_output_path
from utils.warp_video import warp_video_with_per_frame_homography
from utils.warp_video import warp_video_with_fixed_homography

from tools import Cruxes


def main():
    parser = argparse.ArgumentParser(
        description="Warp video based on a reference image."
    )
    parser.add_argument(
        "--src_video_path",
        type=str,
        default=None,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--ref_img",
        type=str,
        default=None,
        help="Path to the reference image file.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="dynamic",
        help="Type of homography use: fixed or dynamic",
    )
    parser.add_argument(
        "--overlay_text",
        type=bool,
        default=False,
        help="Whether to overlay text on the video frames",
    )
    args = parser.parse_args()

    # Check if required arguments are provided
    # If not, print a warning and return
    if not args.src_video_path or not args.ref_img:
        print(
            colored(
                "Warning: No video path or reference image supplied. Please provide --src_video_path and --ref_img",
                "red",
            )
        )
        return

    reference_image = args.ref_img
    target_video = args.src_video_path
    warp_type = args.type
    overlay_text = args.overlay_text

    # Print colored messages for debugging
    print(colored(f"Reference Image: {reference_image}", "blue"))
    print(colored(f"Target Video: {target_video}", "blue"))
    print(colored(f"Warp Type: {warp_type}", "blue"))
    print(colored(f"Overlay Text: {overlay_text}", "blue"))

    cruxes = Cruxes()
    cruxes.warp_video(reference_image, target_video, warp_type, overlay_text)


if __name__ == "__main__":
    main()
