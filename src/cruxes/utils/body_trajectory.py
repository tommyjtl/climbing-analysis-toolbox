import cv2
import numpy as np

# There's a package called `mediapipe-silicon` but it doesn't work with the latest version of `mediapipe`
import mediapipe as mp

from .kamlan_filter import SimpleKalmanFilter
from .file_operations import get_output_path
from .draw_helpers import (
    draw_trajectory,
    draw_velocity_arrow,
    draw_gauge,
    draw_label,
    GaugeConfig,
    get_gauge_centers,
)

trajectory_colors = {
    "matisse": {
        # in BGR format
        "red": (59, 67, 183),
        "green": (127, 161, 85),
        "blue": (192, 112, 78),
        "orange": (63, 125, 217),
        "yellow": (70, 181, 223),
        "magenta": (131, 78, 176),
        "purple": (188, 125, 160),
        "beige": (201, 218, 217),
    }
}


colors = {
    # B, G, R
    "hip_mid": trajectory_colors["matisse"]["red"],
    "upper_body_center": trajectory_colors["matisse"]["green"],
    "head": trajectory_colors["matisse"]["blue"],
    "left_hand": trajectory_colors["matisse"]["orange"],
    "right_hand": trajectory_colors["matisse"]["yellow"],
    "left_foot": trajectory_colors["matisse"]["magenta"],
    "right_foot": trajectory_colors["matisse"]["purple"],
}


def save_trajectories_as_png(
    trajectories, width, height, output_path, colors=None, thickness=2
):
    """
    Save a PNG image with just the trajectories drawn.
    Args:
        trajectories: dict of {track_point: list of (x, y) tuples}
        width: image width
        height: image height
        output_path: path to save the PNG
        colors: dict of {track_point: (B, G, R)}
        thickness: line thickness
    """
    # Create a blank black image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if colors is None:
        colors = {tp: (0, 255, 255) for tp in trajectories}
    for tp, traj in trajectories.items():
        color = colors.get(tp, (0, 255, 255))
        for i in range(1, len(traj)):
            cv2.line(img, traj[i - 1], traj[i], color, thickness)
    cv2.imwrite(output_path, img)


def extract_pose_and_draw_trajectory(
    video_path,
    output_path=None,  # optional, if not provided, the output video will be saved in the `output` folder
    track_point=["hip_mid"],  # a list of track points to draw trajectory for
    overlay_trajectory=False,  # if `True``, we draw trajectory on a semi-transparent black overlay
    overlay_opacity=0.8,  # opacity for the overlay, value should between [0.0, 1.0]
    show_gauges=False,  # whether to show gauges and related text
    draw_pose=True,  # whether to draw the body pose skeleton
    kalman_settings=[True, 1e-1],  # [use_kalman, measurement_variance]
    trajectory_png_path=None,  # NEW: optional PNG output path
):
    use_kalman = kalman_settings[0]  # whether to use Kalman filter
    measurement_variance = kalman_settings[1]  # variance for the Kalman filter

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Store trajectories and 3D trajectories for each track point
    trajectories = {tp: [] for tp in track_point}
    trajectories_3d = {tp: [] for tp in track_point}
    max_observed_velocity = {tp: 0 for tp in track_point}

    # Initialize Kalman filters for each track point if enabled
    kalman_filters = (
        # measurement_variance=1e0 for high noise, 1e-2 for low noise
        # default is 1e-1
        {
            tp: SimpleKalmanFilter(measurement_variance=measurement_variance)
            for tp in track_point
        }
        if use_kalman
        else None
    )

    # Get video properties
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set output path if not provided
    output_path = get_output_path(
        video_path,
        output_path,
        output_prefix="pose_trajectory",
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc if fourcc != 0 else cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Initialize overlay canvas if needed
    overlay_canvas = None

    # Initialize gauge configuration
    gauge_config = GaugeConfig()
    gauge_centers = get_gauge_centers(track_point, gauge_config)

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Prepare overlay canvas if needed
        if overlay_trajectory:
            if overlay_canvas is None:
                overlay_canvas = np.zeros_like(frame)
                overlay_canvas[:] = (0, 0, 0)

        if results.pose_landmarks:
            # Draw skeleton if enabled
            if draw_pose:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    # landmark_drawing_spec
                    mp_drawing.DrawingSpec(
                        color=(0, 0, 0), thickness=1, circle_radius=3
                    ),
                    # connection_drawing_spec
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                    is_drawing_landmarks=False,
                )

            # Get relevant landmarks
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # For each track point, calculate and store trajectory
            for tp in track_point:
                # Set a confidence threshold
                confidence_threshold = 0.6

                # Map track points to their relevant landmarks
                if tp == "hip_mid":
                    if (
                        left_hip.visibility < confidence_threshold
                        or right_hip.visibility < confidence_threshold
                    ):
                        continue
                    left_point = (int(left_hip.x * w), int(left_hip.y * h))
                    right_point = (int(right_hip.x * w), int(right_hip.y * h))
                    mid_point = (
                        int((left_point[0] + right_point[0]) / 2),
                        int((left_point[1] + right_point[1]) / 2),
                    )
                    mid_point_3d = (
                        (left_hip.x + right_hip.x) / 2,
                        (left_hip.y + right_hip.y) / 2,
                        (left_hip.z + right_hip.z) / 2,
                    )
                elif tp == "upper_body_center":
                    if (
                        left_hip.visibility < confidence_threshold
                        or right_hip.visibility < confidence_threshold
                        or left_shoulder.visibility < confidence_threshold
                        or right_shoulder.visibility < confidence_threshold
                    ):
                        continue
                    pts_2d = [
                        (int(left_hip.x * w), int(left_hip.y * h)),
                        (int(right_hip.x * w), int(right_hip.y * h)),
                        (int(left_shoulder.x * w), int(left_shoulder.y * h)),
                        (int(right_shoulder.x * w), int(right_shoulder.y * h)),
                    ]
                    mid_point = (
                        int(sum(p[0] for p in pts_2d) / 4),
                        int(sum(p[1] for p in pts_2d) / 4),
                    )
                    mid_point_3d = (
                        (left_hip.x + right_hip.x + left_shoulder.x + right_shoulder.x)
                        / 4,
                        (left_hip.y + right_hip.y + left_shoulder.y + right_shoulder.y)
                        / 4,
                        (left_hip.z + right_hip.z + left_shoulder.z + right_shoulder.z)
                        / 4,
                    )
                elif tp == "head":
                    if nose.visibility < confidence_threshold:
                        continue
                    mid_point = (
                        int(nose.x * w),
                        int(nose.y * h),
                    )
                    mid_point_3d = (
                        nose.x,
                        nose.y,
                        nose.z,
                    )
                elif tp == "left_hand":
                    if left_hand.visibility < confidence_threshold:
                        continue
                    mid_point = (
                        int(left_hand.x * w),
                        int(left_hand.y * h),
                    )
                    mid_point_3d = (
                        left_hand.x,
                        left_hand.y,
                        left_hand.z,
                    )
                elif tp == "right_hand":
                    if right_hand.visibility < confidence_threshold:
                        continue
                    mid_point = (
                        int(right_hand.x * w),
                        int(right_hand.y * h),
                    )
                    mid_point_3d = (
                        right_hand.x,
                        right_hand.y,
                        right_hand.z,
                    )
                elif tp == "left_foot":
                    if left_foot.visibility < confidence_threshold:
                        continue
                    mid_point = (
                        int(left_foot.x * w),
                        int(left_foot.y * h),
                    )
                    mid_point_3d = (
                        left_foot.x,
                        left_foot.y,
                        left_foot.z,
                    )
                elif tp == "right_foot":
                    if right_foot.visibility < confidence_threshold:
                        continue
                    mid_point = (
                        int(right_foot.x * w),
                        int(right_foot.y * h),
                    )
                    mid_point_3d = (
                        right_foot.x,
                        right_foot.y,
                        right_foot.z,
                    )
                else:
                    raise ValueError(
                        "Invalid track_point option. "
                        "Use "
                        "'hip_mid', "
                        "'upper_body_center', "
                        "'head', "
                        "'left_hand', "
                        "'right_hand', '"
                        "left_foot',"
                        "'right_foot'."
                    )

                # Apply Kalman filter to smooth the 2D point if enabled
                if use_kalman:
                    smoothed_mid_point = kalman_filters[tp].update(mid_point)
                else:
                    smoothed_mid_point = mid_point
                trajectories[tp].append(smoothed_mid_point)
                trajectories_3d[tp].append(mid_point_3d)

            for idx, tp in enumerate(track_point):
                traj = trajectories[tp]
                traj_3d = trajectories_3d[tp]
                color = colors.get(tp, (0, 255, 255))

                # Draw trajectory
                if overlay_trajectory:
                    draw_trajectory(overlay_canvas, traj, color, thickness=2)
                else:
                    draw_trajectory(frame, traj, color, thickness=2)

                # Draw velocity vector ONLY if overlay_trajectory is False
                if len(traj) > 1:
                    prev_point = traj[-2]
                    curr_point = traj[-1]
                    if not overlay_trajectory:
                        draw_velocity_arrow(
                            frame, prev_point, curr_point, color, scale=5, thickness=3
                        )

                    prev_3d = traj_3d[-2]
                    curr_3d = traj_3d[-1]
                    velocity_3d = (
                        curr_3d[0] - prev_3d[0],
                        curr_3d[1] - prev_3d[1],
                        curr_3d[2] - prev_3d[2],
                    )
                    print(
                        f"{tp} 3D velocity vector (frame {len(traj_3d)}): {velocity_3d}"
                    )
                    abs_velocity = (
                        velocity_3d[0] ** 2 + velocity_3d[1] ** 2 + velocity_3d[2] ** 2
                    ) ** 0.5
                    abs_velocity *= (
                        1000  # Scale to mm, this is a rough estimate, not accurate
                    )

                    if abs_velocity > max_observed_velocity[tp]:
                        max_observed_velocity[tp] = abs_velocity

                    if show_gauges:
                        center = gauge_centers[idx]
                        max_velocity = gauge_config.max_velocity
                        velocity_clamped = min(abs_velocity, max_velocity)
                        angle = int((velocity_clamped / max_velocity) * 270)
                        start_angle = 135
                        end_angle = start_angle + angle
                        gauge_color = (
                            int(0 + 255 * (velocity_clamped / max_velocity)),  # R
                            int(255 - 255 * (velocity_clamped / max_velocity)),  # G
                            0,
                        )
                        gauge_canvas = overlay_canvas if overlay_trajectory else frame
                        velocity_text = f"{abs_velocity:.1f} mm/frame"
                        max_velocity_text = (
                            f"Max: {max_observed_velocity[tp]:.1f} mm/frame"
                        )
                        draw_gauge(
                            gauge_canvas,
                            center,
                            gauge_config.radius,
                            gauge_config.thickness,
                            start_angle,
                            end_angle,
                            gauge_color,
                            max_velocity,
                            abs_velocity,
                            velocity_text,
                            max_velocity_text,
                        )
                        draw_label(gauge_canvas, center, gauge_config.radius, tp, color)

        # Blend overlay if enabled
        if overlay_trajectory:
            # Blend overlay_canvas (with trajectory and gauges) onto frame
            blended = cv2.addWeighted(
                frame, 1 - overlay_opacity, overlay_canvas, overlay_opacity, 0
            )
            frame = blended

        # Draw information box on top right
        if show_gauges:
            info_lines = [
                f"{tp} Max velocity: {max_observed_velocity[tp]:.1f} mm/frame"
                for tp in track_point
            ]

        # cv2.imshow("Pose Estimation & Drawing Trajectories", frame) # Uncomment to show the frame in a window

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save PNG with just the trajectories if requested
    if trajectory_png_path is not None:
        # from utils.body_trajectory import save_trajectories_as_png
        save_trajectories_as_png(
            trajectories, width, height, trajectory_png_path, colors=colors
        )
