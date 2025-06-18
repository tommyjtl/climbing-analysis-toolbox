from termcolor import colored
import os

from matching import get_matcher

from .utils.file_operations import get_output_path
from .utils.warp_video import warp_video_with_per_frame_homography
from .utils.warp_video import warp_video_with_fixed_homography
from .utils.body_trajectory import extract_pose_and_draw_trajectory
from .utils.compare_trajectories import extract_pose_and_draw_trajectory_compare


class Cruxes:
    def __init__(self):
        # Nothing to initialize for now
        pass

    def warp_video(
        self, ref_img, src_video_path, warp_type="dynamic", overlay_text=False
    ):
        matcher = get_matcher(
            # Available models:
            # https://github.com/alexstoken/image-matching-models?tab=readme-ov-file#available-models
            "superpoint-lg",
            device="mps",  # default is set to `cpu`, set to `mps` is you want to utilize apple silicon
        )

        reference_image = ref_img
        target_video = src_video_path
        overlay_text = overlay_text

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

    def body_trajectory(
        self,
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
    ):
        output_prefix = "pose_trajectory"
        # Derive output video path using get_output_path
        output_video_path = get_output_path(
            target_video_path,
            None,
            output_prefix=output_prefix,
        )

        # (Optional) Derive PNG path with same prefix and .png extension
        trajectory_png_path = output_video_path.rsplit(".", 1)[0] + ".png"

        extract_pose_and_draw_trajectory(
            target_video_path,
            output_path=output_video_path,
            track_point=track_point,
            overlay_trajectory=overlay_trajectory,
            draw_pose=draw_pose,
            kalman_settings=kalman_settings,
            trajectory_png_path=trajectory_png_path,
        )

    def compare_trajectories(
        self,
        input_video_paths=[],
        track_points=[
            "hip_mid",
            # "upper_body_center",
            # "head",
            "left_hand",
            "right_hand",
            # "left_foot",
            # "right_foot",
        ],
    ):
        # Check if input_video_paths is empty
        if not input_video_paths:
            print(
                colored(
                    "Warning: No input video paths provided. Please provide at least one video path.",
                    "red",
                )
            )
            return

        output_prefix = "compare_trajectories"
        # Derive output video path using get_output_path
        output_video_path = get_output_path(
            input_video_paths[0] if input_video_paths else None,
            None,
            output_prefix=output_prefix,
        )

        extract_pose_and_draw_trajectory_compare(
            # [
            #     "/Volumes/Climbing Videos/previous/v4-6_green_tagmay12/successful2.mp4",
            #     "/Volumes/Climbing Videos/previous/v4-6_green_tagmay12/successful1.mp4",
            #     "/Volumes/Climbing Videos/previous/v4-6_green_tagmay12/failed1.mp4",
            #     "/Volumes/Climbing Videos/previous/v4-6_green_tagmay12/failed2.mp4",
            #     "/Volumes/Climbing Videos/previous/v4-6_green_tagmay12/failed3.mp4",
            #     "/Volumes/Climbing Videos/previous/v4-6_green_tagmay12/failed4.mp4",
            # ],
            video_paths=input_video_paths,
            output_path=output_video_path,
            track_points=track_points,
        )
