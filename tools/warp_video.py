import argparse
from termcolor import colored
import os

from matching import get_matcher

from utils.file_operations import get_output_path
from utils.warp_video import warp_video_with_per_frame_homography
from utils.warp_video import warp_video_with_fixed_homography


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

    matcher = get_matcher(
        # Available models:
        # https://github.com/alexstoken/image-matching-models?tab=readme-ov-file#available-models
        "superpoint-lg",
        device="mps",  # default is set to `cpu`, set to `mps` is you want to utilize apple silicon
    )

    reference_image = args.ref_img
    target_video = args.src_video_path
    warp_type = args.type
    overlay_text = args.overlay_text

    # extract parent directory from reference_image
    parent_dir = os.path.dirname(reference_image)

    # Check if reference_image and target_video exist
    if not os.path.exists(reference_image):
        print(f"Warning: Reference image not found: {reference_image}")
        return
    if not os.path.exists(target_video):
        print(f"Warning: Target video not found: {target_video}")
        return

    output_prefix = "warped"
    # Derive output video path using get_output_path
    output_video_path = get_output_path(
        target_video,
        None,
        output_prefix=output_prefix,
    )

    if warp_type not in ["fixed", "dynamic"]:
        print(
            colored(
                "Warning: Invalid warp type. Please use 'fixed' or 'dynamic'.",
                "red",
            )
        )
        return

    if warp_type == "fixed":
        # Option 1: Compute homography once using first frame of video, then warp all frames
        print("Using fixed homography for warping video.")
        warp_video_with_fixed_homography(
            reference_image,
            target_video,
            matcher,
            parent_dir,
            output_video_path,
            overlay_text=overlay_text,
        )
    else:
        # Option 2: Compute homography for every frame
        warp_video_with_per_frame_homography(
            reference_image,
            target_video,
            matcher,
            parent_dir,
            output_video_path,
            overlay_text=overlay_text,
        )


if __name__ == "__main__":
    main()
