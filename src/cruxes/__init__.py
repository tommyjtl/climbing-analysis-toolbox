from termcolor import colored
import os

from .utils.file_operations import get_output_path
from .utils.warp_video import warp_image_to_reference
from .utils.warp_video import warp_video_with_per_frame_homography
from .utils.warp_video import warp_video_with_fixed_homography
from .utils.body_trajectory import extract_pose_and_draw_trajectory


class Cruxes:
    def __init__(
        self,
        matcher_model_name="superpoint-lightglue",
        matcher_device="auto",
    ):
        self.matcher_model_name = matcher_model_name
        self.matcher_device = matcher_device

    def set_matcher_model_name(self, matcher_model_name):
        self.matcher_model_name = matcher_model_name
        return self.matcher_model_name

    def set_matcher_device(self, matcher_device):
        self.matcher_device = matcher_device
        return self.matcher_device

    def _get_default_matcher_device(self):
        if self.matcher_device not in [None, "auto"]:
            return self.matcher_device

        import torch

        device = "cpu"
        try:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
        except Exception as e:
            print(
                colored(
                    "Warning: Falling back to CPU device selection.",
                    "red",
                )
            )
            print(f"Error: {e}")

        return device

    def _get_default_matcher(self):
        from imm import get_matcher

        device = self._get_default_matcher_device()

        return get_matcher(
            # Available models:
            # https://github.com/alexstoken/image-matching-models?tab=readme-ov-file#available-models
            self.matcher_model_name,
            device=device,
        )

    def _get_output_image_path(self, target_image_path, output_image_path=None):
        if output_image_path is None:
            input_dir = os.path.dirname(target_image_path) or "."
            file_name = os.path.basename(target_image_path)
            derived_path = os.path.join(input_dir, f"warped_{file_name}")
            print(
                colored(
                    f"Output image will be saved to {derived_path}",
                    "green",
                    attrs=["bold"],
                )
            )
            return derived_path

        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(
                f"Output path {output_image_path} does not exist. Please specify a valid path."
            )

        return output_image_path

    def warp_video(
        self,
        ref_img,
        src_video_path,
        warp_type="dynamic",
        overlay_text=False,
        use_gradient_blending=False,
        blend_mode="none",
        feather_amount=10,
    ):
        """
        Warp a video to align with a reference image.

        Args:
            ref_img: Path to reference image
            src_video_path: Path to source video
            warp_type: 'fixed' or 'dynamic' homography
            overlay_text: Whether to overlay text on frames
            use_gradient_blending: If True, use advanced blending (deprecated, use blend_mode)
            blend_mode: Blending mode:
                        - 'none': Direct masking (fastest, hard edges)
                        - 'feathered': Full Gaussian alpha blending (may cause shadows)
                        - 'edge_feather': Distance transform edge blending (recommended, no shadows)
                        - 'smart': Morphological edge-only blending (very clean)
                        - 'multiband': Laplacian pyramid blending (high quality, slower)
                        - 'poisson': Poisson/gradient blending (may make foreground transparent)
            feather_amount: Pixels to feather at boundary (default: 10, recommended: 5-15)
        """
        reference_image = ref_img
        target_video = src_video_path
        overlay_text = overlay_text

        # Check if reference_image and target_video exist
        if not os.path.exists(reference_image):
            print(f"Warning: Reference image not found: {reference_image}")
            return
        if not os.path.exists(target_video):
            print(f"Warning: Target video not found: {target_video}")
            return

        matcher = self._get_default_matcher()

        # extract parent directory from reference_image
        parent_dir = os.path.dirname(reference_image)

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
                use_gradient_blending=use_gradient_blending,
                blend_mode=blend_mode,
                feather_amount=feather_amount,
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
                use_gradient_blending=use_gradient_blending,
                blend_mode=blend_mode,
                feather_amount=feather_amount,
            )

    def warp_image(
        self,
        ref_img,
        src_img_path,
        output_image_path=None,
        overlay_text=False,
        text_to_overlay=None,
        use_gradient_blending=False,
        blend_mode="edge_feather",
        feather_amount=15,
    ):
        """
        Warp a target image to align with a reference image.

        Args:
            ref_img: Path to reference image
            src_img_path: Path to source image
            output_image_path: Optional output image path. Defaults to warped_<input_name> in the same folder as src_img_path
            overlay_text: Whether to overlay text on the warped image
            text_to_overlay: Optional custom overlay text
            use_gradient_blending: If True, use advanced blending (deprecated, use blend_mode)
            blend_mode: Blending mode:
                        - 'none': Direct masking (fastest, hard edges)
                        - 'feathered': Full Gaussian alpha blending (may cause shadows)
                        - 'edge_feather': Distance transform edge blending (recommended, no shadows)
                        - 'smart': Morphological edge-only blending (very clean)
                        - 'multiband': Laplacian pyramid blending (high quality, slower)
                        - 'poisson': Poisson/gradient blending (may make foreground transparent)
            feather_amount: Pixels to feather at boundary (default: 15)
        """
        reference_image = ref_img
        source_image = src_img_path

        if not os.path.exists(reference_image):
            print(f"Warning: Reference image not found: {reference_image}")
            return False
        if not os.path.exists(source_image):
            print(f"Warning: Source image not found: {source_image}")
            return False

        matcher = self._get_default_matcher()
        output_image_path = self._get_output_image_path(
            source_image,
            output_image_path,
        )

        return warp_image_to_reference(
            reference_image,
            source_image,
            output_image_path,
            matcher,
            overlay_text=overlay_text,
            text_to_overlay=text_to_overlay,
            use_gradient_blending=use_gradient_blending,
            blend_mode=blend_mode,
            feather_amount=feather_amount,
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
        json_only=False,
        overlay_mask=False,
        overlay_trajectory=None,  # deprecated alias
        hide_original_video=False,
        draw_pose=True,
        show_limb_reach_circles=None,  # list of region names: "left_upper", "right_upper", "left_lower", "right_lower", "all". None disables.
        pose_color=(
            255,
            255,
            255,
        ),  # Color for pose skeleton in BGR format (default: white)
        show_gauges=False,
        show_trajectory=True,
        trajectory_history_seconds=None,
        use_cached_landmarks=False,
        export_landmarks=False,
        landmarks_json_path=None,
        export_world_landmarks=False,
        world_landmarks_json_path=None,
        use_cached_trajectory_metadata=False,
        export_metadata=False,
        metadata_path=None,
        export_trajectory_metadata=None,
        trajectory_metadata_path=None,
        kalman_settings=[  # Kalman filter settings: [use_kalman : bool, kalman_gain : float]
            True,  # Set this to false if you don't want to apply Kalman filter
            1e0,  # >=1e0 for higher noise, <=1e-1 for lower noise
        ],
        trajectory_png_path=None,
        smoothing="gaussian",  # None | "savgol" | "gaussian" | "smoothnet"
        savgol_window=11,
        savgol_order=3,
        gaussian_sigma=3.0,
        smoothnet_window_size=32,
        smoothnet_epochs=100,
        smoothnet_lambda_accel=0.1,
        track_point_visibility_threshold=0.6,
        pose_visibility_threshold=0.2,
        pose_presence_threshold=0.2,
        pose_backend="mediapipe",  # "mediapipe" or "vitpose"
        vitpose_device=None,  # device for ViTPose ("mps", "cpu", or None for auto)
        vitpose_det_frequency=3,  # how often YOLOX re-runs detection (every N frames)
        trajectory_thickness=None,  # thickness for trajectory lines and velocity arrows
        velocity_arrow_length=None,  # scale for velocity arrow length
    ):
        if overlay_trajectory is not None:
            overlay_mask = overlay_trajectory

        if export_trajectory_metadata is not None:
            export_metadata = export_metadata or export_trajectory_metadata
        if metadata_path is None and trajectory_metadata_path is not None:
            metadata_path = trajectory_metadata_path

        output_video_path = None
        if not json_only:
            output_prefix = "pose_trajectory"
            output_video_path = get_output_path(
                target_video_path,
                None,
                output_prefix=output_prefix,
            )

        extract_pose_and_draw_trajectory(
            target_video_path,
            output_path=output_video_path,
            track_point=track_point,
            json_only=json_only,
            overlay_mask=overlay_mask,
            hide_original_video=hide_original_video,
            draw_pose=draw_pose,
            show_limb_reach_circles=show_limb_reach_circles,
            pose_color=pose_color,
            show_gauges=show_gauges,
            show_trajectory=show_trajectory,
            trajectory_history_seconds=trajectory_history_seconds,
            use_cached_landmarks=use_cached_landmarks,
            export_landmarks=export_landmarks,
            landmarks_json_path=landmarks_json_path,
            export_world_landmarks=export_world_landmarks,
            world_landmarks_json_path=world_landmarks_json_path,
            use_cached_trajectory_metadata=use_cached_trajectory_metadata,
            export_metadata=export_metadata,
            metadata_path=metadata_path,
            export_trajectory_metadata=export_trajectory_metadata,
            trajectory_metadata_path=trajectory_metadata_path,
            kalman_settings=kalman_settings,
            trajectory_png_path=trajectory_png_path,
            smoothing=smoothing,
            savgol_window=savgol_window,
            savgol_order=savgol_order,
            gaussian_sigma=gaussian_sigma,
            smoothnet_window_size=smoothnet_window_size,
            smoothnet_epochs=smoothnet_epochs,
            smoothnet_lambda_accel=smoothnet_lambda_accel,
            track_point_visibility_threshold=track_point_visibility_threshold,
            pose_visibility_threshold=pose_visibility_threshold,
            pose_presence_threshold=pose_presence_threshold,
            pose_backend=pose_backend,
            vitpose_device=vitpose_device,
            vitpose_det_frequency=vitpose_det_frequency,
            trajectory_thickness=trajectory_thickness,
            velocity_arrow_length=velocity_arrow_length,
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

        from .utils.compare_trajectories import (
            extract_pose_and_draw_trajectory_compare,
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
