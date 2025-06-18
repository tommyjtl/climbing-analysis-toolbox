import argparse
from cruxes import Cruxes


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
        help="Comma-separated points of interest to track \n(currently available: hip_mid,upper_body_center,head,left_hand,right_hand,left_foot,right_foot)",
    )
    body_parser.add_argument(
        "--overlay_trajectory", action="store_true", help="Overlay trajectory mask."
    )
    body_parser.add_argument(
        "--draw_pose", action="store_true", help="Draw pose skeleton."
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

    args = parser.parse_args()
    cruxes = Cruxes()

    if args.command == "warp":
        cruxes.warp_video(
            args.ref_img,
            args.src_video_path,
            warp_type=args.type,
        )
    elif args.command == "body-trajectory":
        if args.kalman_settings is not None:
            kalman_settings = [True, args.kalman_settings]
        else:
            kalman_settings = [False, 1e-1]  # default gain if disabled
        track_points = [p.strip() for p in args.track_point.split(",") if p.strip()]
        cruxes.body_trajectory(
            args.video_path,
            track_point=track_points,
            overlay_trajectory=args.overlay_trajectory,
            draw_pose=args.draw_pose,
            kalman_settings=kalman_settings,
            trajectory_png_path=args.trajectory_png_path,
        )


if __name__ == "__main__":
    main()
