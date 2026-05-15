import argparse
from cruxes import Cruxes
from cruxes.utils.body_trajectory import DEFAULT_TRACK_POINT_VISIBILITY_THRESHOLD
from cruxes.utils.pose_backend import PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD

BLEND_MODES = [
    "none",
    "feathered",
    "edge_feather",
    "smart",
    "multiband",
    "poisson",
]


def main():
    parser = argparse.ArgumentParser(
        description="Cruxes: Climbing Analysis Toolbox CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    """
    Warp subcommand
    """
    warp_parser = subparsers.add_parser(
        "warp", help="Warp a video to match a reference image."
    )
    warp_parser.add_argument("--ref_img", required=True, help="Reference image path.")
    warp_parser.add_argument(
        "--src_video_path", required=True, help="Source video path."
    )
    warp_parser.add_argument(
        "--type", default="dynamic", choices=["dynamic", "fixed"], help="Warp type."
    )

    """
    Warp image subcommand
    """
    warp_image_parser = subparsers.add_parser(
        "warp-image", help="Warp an image to match a reference image."
    )
    warp_image_parser.add_argument(
        "--ref_img", required=True, help="Reference image path."
    )
    warp_image_parser.add_argument(
        "--src_img_path",
        dest="src_img_path",
        required=True,
        help="Source image path.",
    )
    warp_image_parser.add_argument(
        "--target_img_path",
        dest="src_img_path",
        help=argparse.SUPPRESS,
    )
    warp_image_parser.add_argument(
        "--output_img_path",
        default=None,
        help="Optional output image path. Defaults to warped_<input_name> in the same folder as src_img_path.",
    )
    warp_image_parser.add_argument(
        "--overlay_text",
        action="store_true",
        default=False,
        help="Overlay the target image name on the warped output.",
    )
    warp_image_parser.add_argument(
        "--text_to_overlay",
        default=None,
        help="Optional custom text to overlay when --overlay_text is enabled.",
    )
    warp_image_parser.add_argument(
        "--blend_mode",
        default="edge_feather",
        choices=BLEND_MODES,
        help="Blending mode for compositing the warped image onto the reference.",
    )
    warp_image_parser.add_argument(
        "--feather_amount",
        type=int,
        default=15,
        help="Feathering width in pixels for supported blend modes.",
    )
    warp_image_parser.add_argument(
        "--use_gradient_blending",
        action="store_true",
        default=False,
        help="Use the legacy Poisson blending path. Prefer --blend_mode poisson.",
    )

    """
    Body trajectory subcommand
    """
    body_parser = subparsers.add_parser(
        "body-trajectory", help="Draw body movement trajectories on a video."
    )
    body_parser.add_argument("--video_path", required=True, help="Input video path.")
    body_parser.add_argument(
        "--track_point",
        type=str,
        default="hip_mid,left_hand,right_hand",
        help="Comma-separated points of interest to track. Available: hip_mid, upper_body_center, head, left_hand, right_hand, left_foot, right_foot",
    )
    body_parser.add_argument(
        "--trajectory_only",
        action="store_true",
        default=False,
        help="Render only the trajectory on a black background. This disables pose drawing and telemetry, forces trajectory drawing on, and prefers cached trajectory metadata if available.",
    )
    body_parser.add_argument(
        "--json_only",
        action="store_true",
        default=False,
        help="Export JSON artifacts only, without writing a rendered video. This also enables the separate pose world landmarks export.",
    )
    body_parser.add_argument(
        "--overlay_mask",
        dest="overlay_mask",
        action="store_true",
        help="Overlay a semi-transparent mask for trajectories/gauges.",
    )
    body_parser.add_argument(
        "--overlay_trajectory",
        dest="overlay_mask",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    body_parser.add_argument(
        "--hide_original_video",
        action="store_true",
        help="Use black background instead of original video.",
    )
    body_parser.add_argument(
        "--draw_pose", action="store_true", help="Draw pose skeleton."
    )
    body_parser.add_argument(
        "--show_trajectory",
        action="store_true",
        default=False,
        help="Draw trajectories.",
    )
    body_parser.add_argument(
        "--show_gauges",
        action="store_true",
        default=False,
        help="Show top-left telemetry info for each tracked joint.",
    )
    body_parser.add_argument(
        "--trajectory_history_seconds",
        type=float,
        default=None,
        help="Only show the last N seconds of each trajectory. By default the full trajectory is shown.",
    )
    body_parser.add_argument(
        "--use_cached_landmarks",
        action="store_true",
        default=False,
        help="Reuse a matching landmarks JSON cache if one exists.",
    )
    body_parser.add_argument(
        "--export_landmarks",
        action="store_true",
        default=False,
        help="Export detected landmarks to JSON for reuse in later runs.",
    )
    body_parser.add_argument(
        "--landmarks_json_path",
        default=None,
        help="Optional path to the landmarks JSON cache. Defaults to <video_stem>_landmarks.json next to the input video.",
    )
    body_parser.add_argument(
        "--export_world_landmarks",
        action="store_true",
        default=False,
        help="Export MediaPipe pose world landmarks to a separate WebGPU-friendly JSON file.",
    )
    body_parser.add_argument(
        "--world_landmarks_json_path",
        default=None,
        help="Optional path to the pose world landmarks JSON file. Defaults to <video_stem>_pose_world_landmarks.json next to the input video.",
    )
    body_parser.add_argument(
        "--use_cached_trajectory_metadata",
        action="store_true",
        default=False,
        help="Reuse a matching trajectory metadata JSON file if one exists. This selects the trajectory data source; use --show_trajectory to control drawing.",
    )
    body_parser.add_argument(
        "--export_metadata",
        action="store_true",
        default=False,
        help="Export unified frontend-facing metadata JSON for downstream rendering or analysis.",
    )
    body_parser.add_argument(
        "--metadata_path",
        default=None,
        help="Optional path to the metadata JSON file. Defaults to <video_stem>_trajectory_metadata.json next to the input video.",
    )
    body_parser.add_argument(
        "--export_trajectory_metadata",
        dest="export_trajectory_metadata",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    body_parser.add_argument(
        "--trajectory_metadata_path",
        dest="trajectory_metadata_path",
        default=None,
        help=argparse.SUPPRESS,
    )
    body_parser.add_argument(
        "--kalman_settings",
        type=float,
        default=None,
        help="Kalman filter gain (float). If not supplied, Kalman filter is disabled.",
    )
    body_parser.add_argument(
        "--trajectory_png_path", default=None, help="Output PNG path for trajectory."
    )
    body_parser.add_argument(
        "--track_point_visibility_threshold",
        type=float,
        default=DEFAULT_TRACK_POINT_VISIBILITY_THRESHOLD,
        help="Minimum visibility required for tracked joints and derived track points.",
    )
    body_parser.add_argument(
        "--pose_visibility_threshold",
        type=float,
        default=VISIBILITY_THRESHOLD,
        help="Minimum visibility required to render a pose landmark.",
    )
    body_parser.add_argument(
        "--pose_presence_threshold",
        type=float,
        default=PRESENCE_THRESHOLD,
        help="Minimum presence required to render a pose landmark.",
    )
    body_parser.add_argument(
        "--pose_backend",
        type=str,
        default="mediapipe",
        choices=["mediapipe", "vitpose"],
        help="Pose estimation backend to use (default: mediapipe).",
    )
    body_parser.add_argument(
        "--smoothing",
        type=str,
        default=None,
        choices=["gaussian", "savgol", "smoothnet"],
        help="Smoothing method to apply to the pose skeleton (default: none).",
    )

    args = parser.parse_args()
    cruxes = Cruxes()

    if args.command == "warp":
        cruxes.warp_video(
            args.ref_img,
            args.src_video_path,
            warp_type=args.type,
        )
    elif args.command == "warp-image":
        cruxes.warp_image(
            args.ref_img,
            args.src_img_path,
            output_image_path=args.output_img_path,
            overlay_text=args.overlay_text,
            text_to_overlay=args.text_to_overlay,
            use_gradient_blending=args.use_gradient_blending,
            blend_mode=args.blend_mode,
            feather_amount=args.feather_amount,
        )
    elif args.command == "body-trajectory":
        if args.kalman_settings is not None:
            kalman_settings = [True, args.kalman_settings]
        else:
            kalman_settings = [False, 1e-1]  # default gain if disabled
        track_points = [p.strip() for p in args.track_point.split(",") if p.strip()]
        export_metadata = args.export_metadata or args.export_trajectory_metadata
        metadata_path = args.metadata_path or args.trajectory_metadata_path
        cruxes.body_trajectory(
            args.video_path,
            track_point=track_points,
            json_only=args.json_only,
            trajectory_only=args.trajectory_only,
            hide_original_video=args.hide_original_video,
            draw_pose=args.draw_pose,
            overlay_mask=args.overlay_mask,
            show_gauges=args.show_gauges,
            show_trajectory=args.show_trajectory,
            trajectory_history_seconds=args.trajectory_history_seconds,
            use_cached_landmarks=args.use_cached_landmarks,
            export_landmarks=args.export_landmarks,
            landmarks_json_path=args.landmarks_json_path,
            export_world_landmarks=args.export_world_landmarks,
            world_landmarks_json_path=args.world_landmarks_json_path,
            use_cached_trajectory_metadata=args.use_cached_trajectory_metadata,
            export_metadata=export_metadata,
            metadata_path=metadata_path,
            export_trajectory_metadata=args.export_trajectory_metadata,
            trajectory_metadata_path=args.trajectory_metadata_path,
            kalman_settings=kalman_settings,
            trajectory_png_path=args.trajectory_png_path,
            track_point_visibility_threshold=args.track_point_visibility_threshold,
            pose_visibility_threshold=args.pose_visibility_threshold,
            pose_presence_threshold=args.pose_presence_threshold,
            pose_backend=args.pose_backend,
            smoothing=args.smoothing,
        )


if __name__ == "__main__":
    main()
