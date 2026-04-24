import json
import cv2
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm
import os

from .kamlan_filter import SimpleKalmanFilter
from .file_operations import (
    get_landmarks_json_path,
    get_output_path,
    get_trajectory_metadata_path,
    get_world_landmarks_json_path,
)
from .draw_helpers import (
    draw_colored_trajectory,
    draw_velocity_arrow,
    draw_telemetry_panel,
)
from .pose_helpers import get_track_point_coords
from .pose_backend import (
    PRESENCE_THRESHOLD,
    POSE_CONNECTIONS,
    PoseDetector,
    PoseLandmark,
    NormalizedPoseLandmark,
    VISIBILITY_THRESHOLD,
    WorldPoseLandmark,
    draw_pose_landmarks,
)

# Utils
from .utils.images import save_trajectories_as_png

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

"""
Color Schema
Sequential
Qlik Sense Diverging
Diverging RdYIBu
Diverging BuYiRd 5 values
Blues
Reds
YIGnBu
https://community.qlik.com/t5/Visualization-and-Usability/Heatmap-visualisation-colours/td-p/1783518
"""
# BGR stops for a simple 3-color speed ramp: ice blue -> white -> candle flame.
SPEED_COLOR_SLOW = (227, 255, 255)
SPEED_COLOR_MID = (137, 221, 253)
SPEED_COLOR_FAST = (29, 96, 231)

TRAJECTORY_THICKNESS = 5
VELOCITY_ARROW_LENGTH = 40
VELOCITY_ARROW_THICKNESS = 5
TRAJECTORY_METADATA_SCHEMA_VERSION = "1.0"
WORLD_LANDMARKS_SCHEMA_VERSION = "1.0"
DEFAULT_VELOCITY_COLOR_PRESET = "ice_blue_candle"
DEFAULT_TRACK_POINT_VISIBILITY_THRESHOLD = 0.6
ROOT_TRANSLATION_CONFIDENCE_THRESHOLD = 0.3


def _serialize_landmarks(landmarks):
    if landmarks is None:
        return None

    return [
        {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
            "presence": landmark.presence,
        }
        for landmark in landmarks
    ]


def _deserialize_landmarks(serialized_landmarks, landmark_cls=NormalizedPoseLandmark):
    if serialized_landmarks is None:
        return None

    return [
        landmark_cls(
            x=landmark["x"],
            y=landmark["y"],
            z=landmark["z"],
            visibility=landmark.get("visibility"),
            presence=landmark.get("presence"),
        )
        for landmark in serialized_landmarks
    ]


def _append_track_points(
    trajectories,
    trajectories_3d,
    track_point,
    landmarks,
    frame_shape,
    track_point_visibility_threshold,
    use_kalman,
    kalman_filters,
):
    if landmarks:
        h, w = frame_shape[:2]
        for tp in track_point:
            result = get_track_point_coords(
                tp,
                landmarks,
                w,
                h,
                track_point_visibility_threshold,
            )
            if result is None:
                trajectories[tp].append(None)
                trajectories_3d[tp].append(None)
            else:
                mid_point, mid_point_3d = result
                if use_kalman:
                    smoothed_mid_point = kalman_filters[tp].update(mid_point)
                else:
                    smoothed_mid_point = mid_point
                trajectories[tp].append(smoothed_mid_point)
                trajectories_3d[tp].append(mid_point_3d)
        return

    for tp in track_point:
        trajectories[tp].append(None)
        trajectories_3d[tp].append(None)


def _build_landmarks_cache_metadata(
    video_path,
    total_frames,
    effective_fps,
    width,
    height,
):
    video_stats = os.stat(video_path)
    return {
        "cache_version": 1,
        "video": {
            "source_path": os.path.abspath(video_path),
            "file_name": os.path.basename(video_path),
            "file_size_bytes": video_stats.st_size,
            "file_mtime_ns": video_stats.st_mtime_ns,
            "frame_count": total_frames,
            "fps": effective_fps,
            "width": width,
            "height": height,
        },
    }


def _serialize_point_2d(point):
    if point is None:
        return None
    return {
        "x": int(point[0]),
        "y": int(point[1]),
    }


def _serialize_point_3d(point):
    if point is None:
        return None
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "z": float(point[2]),
    }


def _serialize_vector_2d(vector):
    if vector is None:
        return None
    return {
        "dx": float(vector[0]),
        "dy": float(vector[1]),
    }


def _serialize_vector_3d(vector):
    if vector is None:
        return None
    return {
        "dx": float(vector[0]),
        "dy": float(vector[1]),
        "dz": float(vector[2]),
    }


def _scale_vector(vector, scalar):
    if vector is None:
        return None
    return tuple(component * scalar for component in vector)


def _build_velocity_color_presets():
    return {
        DEFAULT_VELOCITY_COLOR_PRESET: {
            "color_space": "bgr",
            "interpolation": "three_stop_linear_bgr",
            "slow_bgr": list(SPEED_COLOR_SLOW),
            "mid_bgr": list(SPEED_COLOR_MID),
            "fast_bgr": list(SPEED_COLOR_FAST),
        }
    }


def _copy_pose_landmarks(landmarks):
    if landmarks is None:
        return None

    return [
        NormalizedPoseLandmark(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=landmark.visibility,
            presence=landmark.presence,
        )
        for landmark in landmarks
    ]


def _build_render_pose_landmarks(
    all_pose_landmarks,
    smoothed_pose_landmarks,
    use_smoothing,
    num_landmarks,
):
    rendered_pose_landmarks = []

    for frame_idx, pose_landmarks in enumerate(all_pose_landmarks):
        if pose_landmarks is None:
            rendered_pose_landmarks.append(None)
            continue

        frame_landmarks = _copy_pose_landmarks(pose_landmarks)
        if use_smoothing and frame_idx < len(smoothed_pose_landmarks):
            for lm_idx in range(num_landmarks):
                if lm_idx in smoothed_pose_landmarks[frame_idx]:
                    lm_data = smoothed_pose_landmarks[frame_idx][lm_idx]
                    frame_landmarks[lm_idx].x = lm_data["x"]
                    frame_landmarks[lm_idx].y = lm_data["y"]
                    frame_landmarks[lm_idx].z = lm_data["z"]
                    frame_landmarks[lm_idx].visibility = lm_data["visibility"]
                    frame_landmarks[lm_idx].presence = lm_data["presence"]

        rendered_pose_landmarks.append(frame_landmarks)

    return rendered_pose_landmarks


def _serialize_pose_landmarks_for_metadata(landmarks, width, height):
    if landmarks is None:
        return None

    serialized_landmarks = []
    for landmark in landmarks:
        serialized_landmarks.append(
            {
                "x": float(landmark.x * width),
                "y": float(landmark.y * height),
                "z": float(landmark.z),
                "visibility": (
                    None if landmark.visibility is None else float(landmark.visibility)
                ),
                "presence": (
                    None if landmark.presence is None else float(landmark.presence)
                ),
            }
        )

    return serialized_landmarks


def _build_pose_metadata(
    rendered_pose_landmarks, effective_fps, width, height, smoothing
):
    return {
        "landmark_model": "mediapipe_pose_33",
        "landmark_count": len(PoseLandmark),
        "landmark_names": [landmark.name.lower() for landmark in PoseLandmark],
        "render_landmarks_source": f"{smoothing}_smoothed" if smoothing else "raw",
        "coordinate_space": {
            "type": "pixel",
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "width": int(width),
            "height": int(height),
        },
        "skeleton_connections": [list(connection) for connection in POSE_CONNECTIONS],
        "frames": [
            {
                "frame_index": frame_idx,
                "timestamp_seconds": float(frame_idx / effective_fps),
                "landmarks": _serialize_pose_landmarks_for_metadata(
                    landmarks,
                    width,
                    height,
                ),
            }
            for frame_idx, landmarks in enumerate(rendered_pose_landmarks)
        ],
    }


def _serialize_world_landmarks_for_export(landmarks):
    if landmarks is None:
        return None

    positions = []
    visibility = []
    presence = []
    for landmark in landmarks:
        positions.extend([float(landmark.x), float(landmark.y), float(landmark.z)])
        visibility.append(
            None if landmark.visibility is None else float(landmark.visibility)
        )
        presence.append(None if landmark.presence is None else float(landmark.presence))

    return {
        "positions": positions,
        "visibility": visibility,
        "presence": presence,
    }


def _build_world_landmarks_export(
    video_path,
    source_video_metadata,
    rendered_pose_landmarks,
    world_pose_landmarks,
    effective_fps,
    width,
    height,
):
    root_translation_estimate = _build_root_translation_estimate(
        rendered_pose_landmarks,
        world_pose_landmarks,
        effective_fps,
        width,
        height,
    )

    frames = []
    for frame_idx, landmarks in enumerate(world_pose_landmarks):
        frame_payload = {
            "frame_index": frame_idx,
            "timestamp_seconds": float(frame_idx / effective_fps),
            "landmarks": _serialize_world_landmarks_for_export(landmarks),
        }
        if frame_idx < len(root_translation_estimate["frames"]):
            frame_payload["root_translation"] = root_translation_estimate["frames"][
                frame_idx
            ]["translation"]
        frames.append(frame_payload)

    return {
        "schema_version": WORLD_LANDMARKS_SCHEMA_VERSION,
        "format": "cruxes_pose_world_landmarks_webgpu",
        "source_video": {
            **source_video_metadata,
            "source_path": os.path.abspath(video_path),
        },
        "landmark_model": "mediapipe_pose_33",
        "landmark_count": len(PoseLandmark),
        "landmark_names": [landmark.name.lower() for landmark in PoseLandmark],
        "coordinate_space": {
            "type": "mediapipe_world_landmarks",
            "unit": "meters",
            "origin": "midpoint_of_hips",
            "notes": "Raw MediaPipe pose world landmarks.",
        },
        "root_translation_estimate": root_translation_estimate,
        "skeleton_connections": [list(connection) for connection in POSE_CONNECTIONS],
        "frames": frames,
    }


def _save_world_landmarks_export(export_path, world_landmarks_payload):
    with open(export_path, "w", encoding="utf-8") as file_obj:
        json.dump(world_landmarks_payload, file_obj, indent=2)


def _is_landmark_confident(
    landmark,
    confidence_threshold=ROOT_TRANSLATION_CONFIDENCE_THRESHOLD,
):
    if landmark is None:
        return False

    visibility = getattr(landmark, "visibility", None)
    presence = getattr(landmark, "presence", None)

    if visibility is not None and visibility < confidence_threshold:
        return False
    if presence is not None and presence < confidence_threshold:
        return False

    return True


def _get_pose_landmark_pixel_point(landmarks, landmark_index, width, height):
    if landmarks is None or landmark_index >= len(landmarks):
        return None

    landmark = landmarks[landmark_index]
    if not _is_landmark_confident(landmark):
        return None

    return (float(landmark.x * width), float(landmark.y * height))


def _get_world_landmark_point(landmarks, landmark_index):
    if landmarks is None or landmark_index >= len(landmarks):
        return None

    landmark = landmarks[landmark_index]
    if not _is_landmark_confident(landmark):
        return None

    return (float(landmark.x), float(landmark.y), float(landmark.z))


def _midpoint(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    return tuple((point_a[idx] + point_b[idx]) / 2.0 for idx in range(len(point_a)))


def _distance(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    return float(
        np.sqrt(sum((point_a[idx] - point_b[idx]) ** 2 for idx in range(len(point_a))))
    )


def _estimate_frame_meters_per_pixel(
    pose_landmarks,
    world_pose_landmarks,
    width,
    height,
):
    left_shoulder_px = _get_pose_landmark_pixel_point(
        pose_landmarks,
        PoseLandmark.LEFT_SHOULDER,
        width,
        height,
    )
    right_shoulder_px = _get_pose_landmark_pixel_point(
        pose_landmarks,
        PoseLandmark.RIGHT_SHOULDER,
        width,
        height,
    )
    left_hip_px = _get_pose_landmark_pixel_point(
        pose_landmarks,
        PoseLandmark.LEFT_HIP,
        width,
        height,
    )
    right_hip_px = _get_pose_landmark_pixel_point(
        pose_landmarks,
        PoseLandmark.RIGHT_HIP,
        width,
        height,
    )

    left_shoulder_world = _get_world_landmark_point(
        world_pose_landmarks,
        PoseLandmark.LEFT_SHOULDER,
    )
    right_shoulder_world = _get_world_landmark_point(
        world_pose_landmarks,
        PoseLandmark.RIGHT_SHOULDER,
    )
    left_hip_world = _get_world_landmark_point(
        world_pose_landmarks,
        PoseLandmark.LEFT_HIP,
    )
    right_hip_world = _get_world_landmark_point(
        world_pose_landmarks,
        PoseLandmark.RIGHT_HIP,
    )

    shoulder_mid_px = _midpoint(left_shoulder_px, right_shoulder_px)
    hip_mid_px = _midpoint(left_hip_px, right_hip_px)
    shoulder_mid_world = _midpoint(left_shoulder_world, right_shoulder_world)
    hip_mid_world = _midpoint(left_hip_world, right_hip_world)

    scale_ratios = []
    measurement_pairs = [
        (
            _distance(left_shoulder_px, right_shoulder_px),
            _distance(left_shoulder_world, right_shoulder_world),
        ),
        (
            _distance(left_hip_px, right_hip_px),
            _distance(left_hip_world, right_hip_world),
        ),
        (
            _distance(shoulder_mid_px, hip_mid_px),
            _distance(shoulder_mid_world, hip_mid_world),
        ),
    ]

    for pixel_distance, world_distance in measurement_pairs:
        if (
            pixel_distance is not None
            and pixel_distance > 1e-6
            and world_distance is not None
            and world_distance > 1e-6
        ):
            scale_ratios.append(world_distance / pixel_distance)

    if not scale_ratios:
        return hip_mid_px, None

    return hip_mid_px, float(np.median(scale_ratios))


def _estimate_root_translation_axis_signs(world_pose_landmarks):
    x_directions = []
    y_directions = []

    for frame_landmarks in world_pose_landmarks:
        left_shoulder = _get_world_landmark_point(
            frame_landmarks,
            PoseLandmark.LEFT_SHOULDER,
        )
        right_shoulder = _get_world_landmark_point(
            frame_landmarks,
            PoseLandmark.RIGHT_SHOULDER,
        )
        left_hip = _get_world_landmark_point(frame_landmarks, PoseLandmark.LEFT_HIP)
        right_hip = _get_world_landmark_point(
            frame_landmarks,
            PoseLandmark.RIGHT_HIP,
        )

        if left_shoulder is not None and right_shoulder is not None:
            x_delta = right_shoulder[0] - left_shoulder[0]
            if abs(x_delta) > 1e-6:
                x_directions.append(np.sign(x_delta))

        shoulder_mid = _midpoint(left_shoulder, right_shoulder)
        hip_mid = _midpoint(left_hip, right_hip)
        if shoulder_mid is not None and hip_mid is not None:
            y_delta = hip_mid[1] - shoulder_mid[1]
            if abs(y_delta) > 1e-6:
                y_directions.append(np.sign(y_delta))

    x_sign = 1.0 if not x_directions or np.mean(x_directions) >= 0 else -1.0
    y_sign = 1.0 if not y_directions or np.mean(y_directions) >= 0 else -1.0
    return x_sign, y_sign


def _build_root_translation_estimate(
    rendered_pose_landmarks,
    world_pose_landmarks,
    effective_fps,
    width,
    height,
):
    x_sign, y_sign = _estimate_root_translation_axis_signs(world_pose_landmarks)

    previous_root_px = None
    previous_meters_per_pixel = None
    cumulative_translation_x = 0.0
    cumulative_translation_y = 0.0
    frame_estimates = []

    for frame_idx, (pose_landmarks, world_landmarks) in enumerate(
        zip(rendered_pose_landmarks, world_pose_landmarks)
    ):
        hip_mid_px, meters_per_pixel = _estimate_frame_meters_per_pixel(
            pose_landmarks,
            world_landmarks,
            width,
            height,
        )
        updated = False

        if hip_mid_px is not None and meters_per_pixel is not None:
            if previous_root_px is not None and previous_meters_per_pixel is not None:
                average_scale = (previous_meters_per_pixel + meters_per_pixel) / 2.0
                cumulative_translation_x += (
                    (hip_mid_px[0] - previous_root_px[0]) * average_scale * x_sign
                )
                cumulative_translation_y += (
                    (hip_mid_px[1] - previous_root_px[1]) * average_scale * y_sign
                )
            previous_root_px = hip_mid_px
            previous_meters_per_pixel = meters_per_pixel
            updated = True

        frame_estimates.append(
            {
                "frame_index": frame_idx,
                "timestamp_seconds": float(frame_idx / effective_fps),
                "translation": {
                    "x": float(cumulative_translation_x),
                    "y": float(cumulative_translation_y),
                    "z": 0.0,
                },
                "hip_mid_pixel": _serialize_point_2d(hip_mid_px),
                "meters_per_pixel": (
                    None if meters_per_pixel is None else float(meters_per_pixel)
                ),
                "updated": updated,
            }
        )

    return {
        "method": "weak_perspective_xy_from_hip_mid",
        "notes": (
            "Cumulative x/y translation estimated from hip midpoint image motion "
            "scaled by observed torso size. z remains zero."
        ),
        "coordinate_space": {
            "unit": "meters",
            "origin": "first_valid_frame",
            "x_axis_sign": int(x_sign),
            "y_axis_sign": int(y_sign),
            "z_axis": "zero_only",
        },
        "frames": frame_estimates,
    }


def _build_trajectory_samples(
    trajectory_2d,
    trajectory_3d,
    effective_fps,
    velocity_percentiles,
):
    samples = []
    previous_valid_point_3d = None
    previous_valid_frame_index = None

    for frame_index, (point_2d, point_3d) in enumerate(
        zip(trajectory_2d, trajectory_3d)
    ):
        abs_velocity = None
        velocity_ratio = None
        frames_since_previous_valid_sample = None
        velocity_vector_2d = None
        velocity_vector_2d_per_second = None
        velocity_vector_3d = None
        velocity_vector_3d_per_second = None

        if (
            point_2d is not None
            and previous_valid_frame_index is not None
            and trajectory_2d[previous_valid_frame_index] is not None
        ):
            previous_valid_point_2d = trajectory_2d[previous_valid_frame_index]
            velocity_vector_2d = (
                point_2d[0] - previous_valid_point_2d[0],
                point_2d[1] - previous_valid_point_2d[1],
            )

        if point_3d is not None and previous_valid_point_3d is not None:
            velocity_vector_3d = (
                point_3d[0] - previous_valid_point_3d[0],
                point_3d[1] - previous_valid_point_3d[1],
                point_3d[2] - previous_valid_point_3d[2],
            )
            abs_velocity = _compute_abs_velocity(previous_valid_point_3d, point_3d)
            velocity_ratio = _normalize_speed(abs_velocity, velocity_percentiles)
            frames_since_previous_valid_sample = (
                frame_index - previous_valid_frame_index
            )

        if frames_since_previous_valid_sample is not None:
            seconds_since_previous_valid_sample = (
                frames_since_previous_valid_sample / effective_fps
            )
            if seconds_since_previous_valid_sample > 0:
                scale_to_per_second = 1.0 / seconds_since_previous_valid_sample
                velocity_vector_2d_per_second = _scale_vector(
                    velocity_vector_2d,
                    scale_to_per_second,
                )
                velocity_vector_3d_per_second = _scale_vector(
                    velocity_vector_3d,
                    scale_to_per_second,
                )

        sample = {
            "frame_index": frame_index,
            "timestamp_seconds": float(frame_index / effective_fps),
            "point": _serialize_point_2d(point_2d),
            "point_3d": _serialize_point_3d(point_3d),
            "velocity_vector_2d": _serialize_vector_2d(velocity_vector_2d),
            "velocity_vector_2d_per_second": _serialize_vector_2d(
                velocity_vector_2d_per_second
            ),
            "velocity_vector_3d": _serialize_vector_3d(velocity_vector_3d),
            "velocity_vector_3d_per_second": _serialize_vector_3d(
                velocity_vector_3d_per_second
            ),
            "abs_velocity": abs_velocity,
            "velocity_ratio": velocity_ratio,
        }

        if frames_since_previous_valid_sample is not None:
            sample["frames_since_previous_valid_sample"] = (
                frames_since_previous_valid_sample
            )

        samples.append(sample)

        if point_3d is not None:
            previous_valid_point_3d = point_3d
            previous_valid_frame_index = frame_index

    return samples


def _build_trajectory_metadata(
    video_path,
    source_video_metadata,
    track_point,
    trajectories,
    trajectories_3d,
    speed_percentiles,
    effective_fps,
    use_kalman,
    measurement_variance,
    smoothing,
    savgol_window,
    savgol_order,
    pose_metadata=None,
):
    velocity_color_presets = _build_velocity_color_presets()
    tracks = {}

    for tp in track_point:
        tracks[tp] = {
            "velocity_color_preset": DEFAULT_VELOCITY_COLOR_PRESET,
            "velocity_percentiles": {
                "low": float(speed_percentiles[tp]["low"]),
                "high": float(speed_percentiles[tp]["high"]),
            },
            "samples": _build_trajectory_samples(
                trajectories[tp],
                trajectories_3d[tp],
                effective_fps,
                speed_percentiles[tp],
            ),
        }

    trajectory_metadata = {
        "schema_version": TRAJECTORY_METADATA_SCHEMA_VERSION,
        "source_video": {
            **source_video_metadata,
            "source_path": os.path.abspath(video_path),
        },
        "coordinate_space": {
            "type": "pixel",
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "width": int(source_video_metadata["width"]),
            "height": int(source_video_metadata["height"]),
        },
        "velocity_measurement": {
            "source": "mediapipe_normalized_3d_delta",
            "scale_factor": 1000.0,
            "unit": "normalized_units_per_valid_sample",
        },
        "processing": {
            "kalman": {
                "enabled": bool(use_kalman),
                "measurement_variance": (
                    float(measurement_variance) if use_kalman else None
                ),
            },
            "smoothing": {
                "method": smoothing or "none",
                "savgol_window": int(savgol_window) if smoothing == "savgol" else None,
                "savgol_order": int(savgol_order) if smoothing == "savgol" else None,
                "gaussian_sigma": (
                    float(gaussian_sigma) if smoothing == "gaussian" else None
                ),
            },
        },
        "style": {
            "default_velocity_color_preset": DEFAULT_VELOCITY_COLOR_PRESET,
            "velocity_color_presets": velocity_color_presets,
        },
        "tracks": tracks,
    }

    if pose_metadata is not None:
        trajectory_metadata["pose"] = pose_metadata

    return trajectory_metadata


def _save_trajectory_metadata(metadata_path, trajectory_metadata):
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(trajectory_metadata, file_obj, indent=2)


def _deserialize_pose_landmarks_from_metadata(
    pose_payload, total_frames, width, height
):
    frames_payload = pose_payload.get("frames")
    if not isinstance(frames_payload, list) or len(frames_payload) != total_frames:
        return None, "trajectory metadata has invalid pose frames"

    pose_landmarks = []
    for frame_payload in frames_payload:
        frame_landmarks = None
        if isinstance(frame_payload, dict):
            landmarks_payload = frame_payload.get("landmarks")
            if landmarks_payload is not None:
                frame_landmarks = []
                for landmark in landmarks_payload:
                    frame_landmarks.append(
                        NormalizedPoseLandmark(
                            x=float(landmark["x"]) / width,
                            y=float(landmark["y"]) / height,
                            z=float(landmark.get("z", 0.0)),
                            visibility=landmark.get("visibility"),
                            presence=landmark.get("presence"),
                        )
                    )
        pose_landmarks.append(frame_landmarks)

    return pose_landmarks, None


def _load_trajectory_metadata(
    metadata_path,
    video_path,
    total_frames,
    effective_fps,
    width,
    height,
    track_point,
):
    try:
        with open(metadata_path, "r", encoding="utf-8") as file_obj:
            metadata_payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"failed to read trajectory metadata: {exc}"

    if not isinstance(metadata_payload, dict):
        return None, "trajectory metadata format is unsupported"

    source_video = metadata_payload.get("source_video")
    tracks_payload = metadata_payload.get("tracks")
    if not isinstance(source_video, dict) or not isinstance(tracks_payload, dict):
        return None, "trajectory metadata is missing source_video or tracks"

    current_stats = os.stat(video_path)
    expected_metadata = {
        "source_path": os.path.abspath(video_path),
        "file_size_bytes": current_stats.st_size,
        "file_mtime_ns": current_stats.st_mtime_ns,
        "frame_count": total_frames,
        "width": width,
        "height": height,
    }
    for key, expected_value in expected_metadata.items():
        actual_value = source_video.get(key)
        if actual_value != expected_value:
            return None, f"trajectory metadata mismatch for {key}"

    cached_fps = source_video.get("fps")
    if cached_fps is None or abs(cached_fps - effective_fps) > 1e-6:
        return None, "trajectory metadata mismatch for fps"

    trajectories = {}
    trajectories_3d = {}
    speed_percentiles = {}
    pose_landmarks = None

    for tp in track_point:
        track_data = tracks_payload.get(tp)
        if not isinstance(track_data, dict):
            return None, f"trajectory metadata does not contain track '{tp}'"

        samples = track_data.get("samples")
        if not isinstance(samples, list) or len(samples) != total_frames:
            return None, f"trajectory metadata has invalid samples for '{tp}'"

        velocity_percentiles = track_data.get("velocity_percentiles")
        if not isinstance(velocity_percentiles, dict):
            return (
                None,
                f"trajectory metadata is missing velocity percentiles for '{tp}'",
            )

        trajectories[tp] = []
        trajectories_3d[tp] = []
        for sample in samples:
            point = sample.get("point") if isinstance(sample, dict) else None
            point_3d = sample.get("point_3d") if isinstance(sample, dict) else None

            trajectories[tp].append(
                None if point is None else (int(point["x"]), int(point["y"]))
            )
            trajectories_3d[tp].append(
                None
                if point_3d is None
                else (
                    float(point_3d["x"]),
                    float(point_3d["y"]),
                    float(point_3d["z"]),
                )
            )

        speed_percentiles[tp] = {
            "low": float(velocity_percentiles.get("low", 0.0)),
            "high": float(velocity_percentiles.get("high", 1.0)),
        }

    pose_payload = metadata_payload.get("pose")
    if isinstance(pose_payload, dict):
        pose_landmarks, pose_error = _deserialize_pose_landmarks_from_metadata(
            pose_payload,
            total_frames,
            width,
            height,
        )
        if pose_error is not None:
            return None, pose_error

    return {
        "trajectories": trajectories,
        "trajectories_3d": trajectories_3d,
        "speed_percentiles": speed_percentiles,
        "pose_landmarks": pose_landmarks,
        "metadata": metadata_payload,
    }, None


def _save_landmarks_cache(
    cache_path, metadata, all_pose_landmarks, all_world_pose_landmarks=None
):
    cache_payload = {
        **metadata,
        "frames": [_serialize_landmarks(landmarks) for landmarks in all_pose_landmarks],
    }
    if all_world_pose_landmarks is not None:
        cache_payload["world_frames"] = [
            _serialize_landmarks(landmarks) for landmarks in all_world_pose_landmarks
        ]
    with open(cache_path, "w", encoding="utf-8") as file_obj:
        json.dump(cache_payload, file_obj)


def _load_landmarks_cache(
    cache_path,
    video_path,
    total_frames,
    effective_fps,
    width,
    height,
):
    try:
        with open(cache_path, "r", encoding="utf-8") as file_obj:
            cache_payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"failed to read landmarks cache: {exc}"

    if not isinstance(cache_payload, dict) or "video" not in cache_payload:
        return None, "cache format is unsupported"

    video_metadata = cache_payload.get("video", {})
    frames_payload = cache_payload.get("frames")
    world_frames_payload = cache_payload.get("world_frames")
    if not isinstance(frames_payload, list):
        return None, "cache does not contain frame landmarks"

    current_stats = os.stat(video_path)
    expected_metadata = {
        "source_path": os.path.abspath(video_path),
        "file_size_bytes": current_stats.st_size,
        "file_mtime_ns": current_stats.st_mtime_ns,
        "frame_count": total_frames,
        "width": width,
        "height": height,
    }

    for key, expected_value in expected_metadata.items():
        actual_value = video_metadata.get(key)
        if actual_value != expected_value:
            return None, f"cache metadata mismatch for {key}"

    cached_fps = video_metadata.get("fps")
    if cached_fps is None or abs(cached_fps - effective_fps) > 1e-6:
        return None, "cache metadata mismatch for fps"

    if len(frames_payload) != total_frames:
        return None, "cache frame count does not match the video"

    if world_frames_payload is not None:
        if not isinstance(world_frames_payload, list):
            return None, "cache world frame format is unsupported"
        if len(world_frames_payload) != total_frames:
            return None, "cache world frame count does not match the video"
        world_pose_landmarks = [
            _deserialize_landmarks(frame_landmarks, landmark_cls=WorldPoseLandmark)
            for frame_landmarks in world_frames_payload
        ]
    else:
        world_pose_landmarks = None

    return {
        "pose_landmarks": [
            _deserialize_landmarks(frame_landmarks)
            for frame_landmarks in frames_payload
        ],
        "world_pose_landmarks": world_pose_landmarks,
    }, None


def _compute_abs_velocity(prev_point_3d, curr_point_3d):
    velocity_3d = (
        curr_point_3d[0] - prev_point_3d[0],
        curr_point_3d[1] - prev_point_3d[1],
        curr_point_3d[2] - prev_point_3d[2],
    )
    abs_velocity = (
        velocity_3d[0] ** 2 + velocity_3d[1] ** 2 + velocity_3d[2] ** 2
    ) ** 0.5
    return abs_velocity * 1000


def _compute_speed_percentiles(trajectories_3d):
    speed_percentiles = {}
    for tp, joint_trajectory_3d in trajectories_3d.items():
        valid_points = [point for point in joint_trajectory_3d if point is not None]
        if len(valid_points) < 2:
            speed_percentiles[tp] = {"low": 0.0, "high": 1.0}
            continue

        speeds = [
            _compute_abs_velocity(valid_points[idx - 1], valid_points[idx])
            for idx in range(1, len(valid_points))
        ]
        low = float(np.percentile(speeds, 20))
        high = float(np.percentile(speeds, 80))
        if high <= low:
            high = low + 1.0

        speed_percentiles[tp] = {"low": low, "high": high}

    return speed_percentiles


def _normalize_speed(abs_velocity, speed_percentiles):
    low = speed_percentiles["low"]
    high = speed_percentiles["high"]
    if high <= low:
        return 0.0
    if abs_velocity <= low:
        return 0.0
    if abs_velocity >= high:
        return 1.0
    return (abs_velocity - low) / (high - low)


def _interpolate_bgr(color_a, color_b, ratio):
    return tuple(
        int(round(color_a[channel] + (color_b[channel] - color_a[channel]) * ratio))
        for channel in range(3)
    )


def _get_speed_color(normalized_speed):
    clamped_speed = min(max(normalized_speed, 0.0), 1.0)
    if clamped_speed <= 0.5:
        return _interpolate_bgr(
            SPEED_COLOR_SLOW,
            SPEED_COLOR_MID,
            clamped_speed / 0.5,
        )
    return _interpolate_bgr(
        SPEED_COLOR_MID,
        SPEED_COLOR_FAST,
        (clamped_speed - 0.5) / 0.5,
    )


def _validate_probability_threshold(name, value):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def extract_pose_and_draw_trajectory(
    video_path,
    output_path=None,  # optional, if not provided, the output video will be saved in the `output` folder
    track_point=["hip_mid"],  # a list of track points to draw trajectory for
    json_only=False,
    trajectory_only=False,
    hide_original_video=False,  # if True, the output video will have a black background instead of the original frames
    overlay_mask=False,  # if `True`, we draw trajectory on a semi-transparent black overlay
    overlay_trajectory=None,  # deprecated alias
    overlay_opacity=0.8,  # opacity for the overlay, value should between [0.0, 1.0]
    show_gauges=False,  # whether to show top-left telemetry text
    draw_pose=True,  # whether to draw the body pose skeleton
    pose_color=(
        255,
        255,
        255,
    ),  # Color for pose skeleton in BGR format (default: white)
    show_trajectory=True,  # whether to draw the trajectories
    trajectory_history_seconds=None,  # if set, only show the last N seconds of the trajectory
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
    kalman_settings=[True, 1e-1],  # [use_kalman, measurement_variance]
    trajectory_png_path=None,  # NEW: optional PNG output path
    smoothing=None,  # None | "savgol" | "gaussian" | "smoothnet"
    savgol_window=11,  # window length for Savgol filter (must be odd, > savgol_order)
    savgol_order=3,  # polynomial order for Savgol filter
    gaussian_sigma=3.0,  # sigma (std-dev in frames) for Gaussian filter
    smoothnet_window_size=32,  # temporal window for SmoothNet
    smoothnet_epochs=100,  # self-supervised training epochs
    smoothnet_lambda_accel=0.1,  # smoothness loss weight (higher = smoother); paper default = 0.1
    track_point_visibility_threshold=DEFAULT_TRACK_POINT_VISIBILITY_THRESHOLD,
    pose_visibility_threshold=VISIBILITY_THRESHOLD,
    pose_presence_threshold=PRESENCE_THRESHOLD,
    pose_backend="mediapipe",  # "mediapipe" or "vitpose"
    vitpose_device=None,  # device for ViTPose ("mps", "cpu", or None for auto)
    vitpose_det_frequency=3,  # how often YOLOX re-runs detection (every N frames)
    trajectory_thickness=None,  # thickness for trajectory lines and velocity arrows; defaults to TRAJECTORY_THICKNESS
):
    # Suppress MediaPipe warnings
    os.environ["GLOG_minloglevel"] = "2"

    _trajectory_thickness = (
        TRAJECTORY_THICKNESS
        if trajectory_thickness is None
        else int(trajectory_thickness)
    )

    if overlay_trajectory is not None:
        overlay_mask = overlay_trajectory

    if json_only:
        export_landmarks = True
        export_metadata = True
        export_world_landmarks = True

    if trajectory_only:
        hide_original_video = True
        overlay_mask = False
        draw_pose = False
        show_trajectory = True
        show_gauges = False
        use_cached_trajectory_metadata = True

    if export_trajectory_metadata is not None:
        export_metadata = export_metadata or export_trajectory_metadata
    if metadata_path is None and trajectory_metadata_path is not None:
        metadata_path = trajectory_metadata_path

    _validate_probability_threshold(
        "track_point_visibility_threshold",
        track_point_visibility_threshold,
    )
    _validate_probability_threshold(
        "pose_visibility_threshold",
        pose_visibility_threshold,
    )
    _validate_probability_threshold(
        "pose_presence_threshold",
        pose_presence_threshold,
    )

    landmarks_cache_path = None
    if use_cached_landmarks or export_landmarks:
        landmarks_cache_path = get_landmarks_json_path(
            video_path, landmarks_json_path=landmarks_json_path
        )

    world_landmarks_export_path = None
    if export_world_landmarks:
        world_landmarks_export_path = get_world_landmarks_json_path(
            video_path,
            world_landmarks_json_path=world_landmarks_json_path,
        )

    trajectory_export_path = None
    trajectory_cache_path = None
    if use_cached_trajectory_metadata or export_metadata:
        trajectory_cache_path = get_trajectory_metadata_path(
            video_path,
            trajectory_metadata_path=metadata_path,
        )

    if export_metadata:
        trajectory_export_path = get_trajectory_metadata_path(
            video_path,
            trajectory_metadata_path=metadata_path,
        )

    use_kalman = kalman_settings[0]  # whether to use Kalman filter
    measurement_variance = kalman_settings[1]  # variance for the Kalman filter

    # Smoothing is skipped when landmarks come from cache
    _apply_smoothing = smoothing is not None and not use_cached_landmarks

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Store trajectories and 3D trajectories for each track point
    trajectories = {tp: [] for tp in track_point}
    trajectories_3d = {tp: [] for tp in track_point}

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
    effective_fps = fps if fps and fps > 0 else 30.0
    trajectory_history_frames = None
    if trajectory_history_seconds is not None:
        if trajectory_history_seconds <= 0:
            raise ValueError("trajectory_history_seconds must be greater than 0")
        trajectory_history_frames = max(
            1, int(round(trajectory_history_seconds * effective_fps))
        )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_detector = None
    cached_trajectory_payload = None
    render_video = not json_only

    out = None
    if render_video:
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

    # Use a two-pass approach so we can optionally smooth pose landmarks over time
    # while keeping trajectory extraction separate.

    # First pass: read frames and either collect or load pose landmarks.
    frames_data = []  # Store frame data for second pass when video rendering is enabled
    all_pose_landmarks = []  # Store all pose landmarks for smoothing
    all_world_pose_landmarks = []

    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmarks_cache_metadata = _build_landmarks_cache_metadata(
        video_path,
        total_frames,
        effective_fps,
        width,
        height,
    )

    cached_pose_landmarks = None
    cached_world_pose_landmarks = None
    if (
        use_cached_trajectory_metadata
        and trajectory_cache_path
        and os.path.exists(trajectory_cache_path)
    ):
        cached_trajectory_payload, trajectory_cache_error = _load_trajectory_metadata(
            trajectory_cache_path,
            video_path,
            total_frames,
            effective_fps,
            width,
            height,
            track_point,
        )
        if cached_trajectory_payload is not None:
            trajectories = cached_trajectory_payload["trajectories"]
            trajectories_3d = cached_trajectory_payload["trajectories_3d"]
            cached_pose_landmarks = cached_trajectory_payload.get("pose_landmarks")
            print(f"Using cached trajectory metadata from {trajectory_cache_path}")
        else:
            print(
                f"Trajectory metadata at {trajectory_cache_path} could not be used: {trajectory_cache_error}. Recomputing trajectories..."
            )
    elif use_cached_trajectory_metadata and trajectory_cache_path:
        print(
            f"Trajectory metadata not found at {trajectory_cache_path}. Recomputing trajectories..."
        )

    if (
        use_cached_landmarks
        and landmarks_cache_path
        and os.path.exists(landmarks_cache_path)
    ):
        cached_landmarks_payload, cache_error = _load_landmarks_cache(
            landmarks_cache_path,
            video_path,
            total_frames,
            effective_fps,
            width,
            height,
        )
        if cached_landmarks_payload is not None:
            cached_pose_landmarks = cached_landmarks_payload["pose_landmarks"]
            cached_world_pose_landmarks = cached_landmarks_payload[
                "world_pose_landmarks"
            ]
            print(f"Using cached landmarks from {landmarks_cache_path}")
        else:
            print(
                f"Landmarks cache at {landmarks_cache_path} could not be used: {cache_error}. Recomputing landmarks..."
            )
    elif use_cached_landmarks and landmarks_cache_path:
        print(
            f"Landmarks cache not found at {landmarks_cache_path}. Recomputing landmarks..."
        )

    needs_world_pose_detection = (
        export_world_landmarks and cached_world_pose_landmarks is None
    )

    first_pass_desc = "Collecting landmarks"
    if cached_pose_landmarks is not None and cached_trajectory_payload is not None:
        print("First pass - reading frames with cached landmarks and trajectories...")
        first_pass_desc = "Loading cached pose/trajectory"
    elif cached_pose_landmarks is not None:
        print("First pass - reading frames with cached landmarks...")
        first_pass_desc = "Loading cached landmarks"
    elif cached_trajectory_payload is not None:
        print("First pass - reading frames with cached trajectories...")
        first_pass_desc = "Loading cached trajectories"
    else:
        print("First pass - collecting landmarks...")

    try:
        if (cached_pose_landmarks is None or needs_world_pose_detection) and (
            draw_pose
            or cached_trajectory_payload is None
            or export_landmarks
            or export_world_landmarks
            or export_metadata
        ):
            pose_detector = PoseDetector(
                backend=pose_backend,
                vitpose_device=vitpose_device,
                vitpose_det_frequency=vitpose_det_frequency,
            )
        with tqdm(total=total_frames, desc=first_pass_desc, unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number = len(all_pose_landmarks) + 1
                if render_video:
                    frames_data.append(frame.copy())

                detector_result = None
                if pose_detector is not None:
                    timestamp_ms = int(frame_number * 1000 / effective_fps)
                    detector_result = pose_detector.process(
                        frame, timestamp_ms=timestamp_ms
                    )

                if cached_pose_landmarks is not None:
                    landmarks = cached_pose_landmarks[frame_number - 1]
                elif detector_result is not None:
                    landmarks = detector_result.pose_landmarks
                else:
                    landmarks = None

                if cached_world_pose_landmarks is not None:
                    world_landmarks = cached_world_pose_landmarks[frame_number - 1]
                elif detector_result is not None:
                    world_landmarks = detector_result.world_pose_landmarks
                else:
                    world_landmarks = None

                all_pose_landmarks.append(landmarks)
                all_world_pose_landmarks.append(world_landmarks)
                if cached_trajectory_payload is None:
                    _append_track_points(
                        trajectories,
                        trajectories_3d,
                        track_point,
                        landmarks,
                        frame.shape,
                        track_point_visibility_threshold,
                        use_kalman,
                        kalman_filters,
                    )

                pbar.update(1)

        if (
            cached_pose_landmarks is None
            and export_landmarks
            and landmarks_cache_path is not None
        ):
            _save_landmarks_cache(
                landmarks_cache_path,
                landmarks_cache_metadata,
                all_pose_landmarks,
                all_world_pose_landmarks,
            )
            print(f"Saved landmarks cache to {landmarks_cache_path}")

        if cached_trajectory_payload is not None:
            speed_percentiles = cached_trajectory_payload["speed_percentiles"]
        else:
            speed_percentiles = _compute_speed_percentiles(trajectories_3d)

        smoothed_pose_landmarks = []

        num_landmarks = 33
        if _apply_smoothing and smoothing == "gaussian":
            from scipy.ndimage import gaussian_filter1d

            print(
                f"Applying Gaussian filter to pose skeleton (sigma={gaussian_sigma})..."
            )
            smoothed_pose_landmarks = [dict() for _ in all_pose_landmarks]

            for lm_idx in range(num_landmarks):
                valid_frames = []
                x_coords = []
                y_coords = []
                z_coords = []

                for frame_idx, pose_lm in enumerate(all_pose_landmarks):
                    if pose_lm is not None:
                        valid_frames.append(frame_idx)
                        x_coords.append(pose_lm[lm_idx].x)
                        y_coords.append(pose_lm[lm_idx].y)
                        z_coords.append(pose_lm[lm_idx].z)

                if valid_frames:
                    x_smooth = gaussian_filter1d(x_coords, sigma=gaussian_sigma)
                    y_smooth = gaussian_filter1d(y_coords, sigma=gaussian_sigma)
                    z_smooth = gaussian_filter1d(z_coords, sigma=gaussian_sigma)

                    for idx, frame_idx in enumerate(valid_frames):
                        smoothed_pose_landmarks[frame_idx][lm_idx] = {
                            "x": float(x_smooth[idx]),
                            "y": float(y_smooth[idx]),
                            "z": float(z_smooth[idx]),
                            "visibility": all_pose_landmarks[frame_idx][
                                lm_idx
                            ].visibility,
                            "presence": all_pose_landmarks[frame_idx][lm_idx].presence,
                        }

        elif _apply_smoothing and smoothing == "savgol":
            print(
                f"Applying Savgol filter to pose skeleton (window={savgol_window}, order={savgol_order})..."
            )
            smoothed_pose_landmarks = [dict() for _ in all_pose_landmarks]

            for lm_idx in range(num_landmarks):
                valid_frames = []
                x_coords = []
                y_coords = []
                z_coords = []

                for frame_idx, pose_lm in enumerate(all_pose_landmarks):
                    if pose_lm is not None:
                        valid_frames.append(frame_idx)
                        x_coords.append(pose_lm[lm_idx].x)
                        y_coords.append(pose_lm[lm_idx].y)
                        z_coords.append(pose_lm[lm_idx].z)

                if len(valid_frames) >= savgol_window:
                    x_smooth = savgol_filter(x_coords, savgol_window, savgol_order)
                    y_smooth = savgol_filter(y_coords, savgol_window, savgol_order)
                    z_smooth = savgol_filter(z_coords, savgol_window, savgol_order)

                    for idx, frame_idx in enumerate(valid_frames):
                        smoothed_pose_landmarks[frame_idx][lm_idx] = {
                            "x": x_smooth[idx],
                            "y": y_smooth[idx],
                            "z": z_smooth[idx],
                            "visibility": all_pose_landmarks[frame_idx][
                                lm_idx
                            ].visibility,
                            "presence": all_pose_landmarks[frame_idx][lm_idx].presence,
                        }

        elif _apply_smoothing and smoothing == "smoothnet":
            print(
                f"Applying SmoothNet to pose skeleton "
                f"(window={smoothnet_window_size}, epochs={smoothnet_epochs}, "
                f"lambda_accel={smoothnet_lambda_accel})..."
            )
            from .smoothnet import apply_smoothnet_to_landmarks

            smoothed_pose_landmarks = apply_smoothnet_to_landmarks(
                all_pose_landmarks,
                window_size=smoothnet_window_size,
                epochs=smoothnet_epochs,
                lambda_accel=smoothnet_lambda_accel,
            )

        rendered_pose_landmarks = _build_render_pose_landmarks(
            all_pose_landmarks,
            smoothed_pose_landmarks,
            _apply_smoothing,
            num_landmarks,
        )

        if export_metadata and trajectory_export_path is not None:
            pose_metadata = None
            if any(landmarks is not None for landmarks in rendered_pose_landmarks):
                pose_metadata = _build_pose_metadata(
                    rendered_pose_landmarks,
                    effective_fps,
                    width,
                    height,
                    smoothing if _apply_smoothing else None,
                )
            elif cached_trajectory_payload is not None:
                pose_metadata = cached_trajectory_payload["metadata"].get("pose")

            trajectory_metadata = _build_trajectory_metadata(
                video_path,
                landmarks_cache_metadata["video"],
                track_point,
                trajectories,
                trajectories_3d,
                speed_percentiles,
                effective_fps,
                use_kalman,
                measurement_variance,
                smoothing if _apply_smoothing else None,
                savgol_window,
                savgol_order,
                pose_metadata=pose_metadata,
            )
            _save_trajectory_metadata(trajectory_export_path, trajectory_metadata)
            print(f"Saved trajectory metadata to {trajectory_export_path}")

        if export_world_landmarks and world_landmarks_export_path is not None:
            if any(landmarks is not None for landmarks in all_world_pose_landmarks):
                world_landmarks_payload = _build_world_landmarks_export(
                    video_path,
                    landmarks_cache_metadata["video"],
                    rendered_pose_landmarks,
                    all_world_pose_landmarks,
                    effective_fps,
                    width,
                    height,
                )
                _save_world_landmarks_export(
                    world_landmarks_export_path,
                    world_landmarks_payload,
                )
                print(f"Saved pose world landmarks to {world_landmarks_export_path}")
            else:
                print(
                    "Pose world landmark export was requested, but MediaPipe did not provide world landmark data."
                )

        if render_video:
            print(
                "Second pass - rendering video with raw trajectories and "
                f"{f'{smoothing}-smoothed' if _apply_smoothing else 'raw'} skeleton..."
            )
            frame_idx = 0

            with tqdm(
                total=len(frames_data), desc="Rendering video", unit="frame"
            ) as pbar:
                for frame in frames_data:
                    if hide_original_video:
                        frame = np.zeros_like(frame)

                    velocity_arrows = []
                    telemetry_rows = []

                    if overlay_mask:
                        if (
                            overlay_canvas is None
                            or trajectory_history_frames is not None
                        ):
                            overlay_canvas = np.zeros_like(frame)
                            overlay_canvas[:] = (0, 0, 0)

                    pose_landmarks_for_drawing = None
                    if draw_pose and frame_idx < len(rendered_pose_landmarks):
                        pose_landmarks_for_drawing = rendered_pose_landmarks[frame_idx]

                    # Draw trajectories up to the current frame using the collected track points.
                    for idx, tp in enumerate(track_point):
                        history_start = 0
                        if trajectory_history_frames is not None:
                            history_start = max(
                                0, frame_idx + 1 - trajectory_history_frames
                            )

                        traj = [
                            p
                            for p in trajectories[tp][history_start : frame_idx + 1]
                            if p is not None
                        ]
                        traj_3d = [
                            p
                            for p in trajectories_3d[tp][history_start : frame_idx + 1]
                            if p is not None
                        ]
                        trajectory_segment_colors = []

                        if len(traj_3d) > 1:
                            for traj_idx in range(1, len(traj_3d)):
                                abs_velocity = _compute_abs_velocity(
                                    traj_3d[traj_idx - 1],
                                    traj_3d[traj_idx],
                                )
                                velocity_ratio = _normalize_speed(
                                    abs_velocity,
                                    speed_percentiles[tp],
                                )
                                trajectory_segment_colors.append(
                                    _get_speed_color(velocity_ratio)
                                )

                        # Draw trajectory if enabled
                        if show_trajectory:
                            if overlay_mask:
                                draw_colored_trajectory(
                                    overlay_canvas,
                                    traj,
                                    trajectory_segment_colors,
                                    thickness=_trajectory_thickness,
                                )
                            else:
                                draw_colored_trajectory(
                                    frame,
                                    traj,
                                    trajectory_segment_colors,
                                    thickness=_trajectory_thickness,
                                )

                        # Draw velocity arrows and optional telemetry.
                        if len(traj) > 1 and len(traj_3d) > 1:
                            prev_point = traj[-2]
                            curr_point = traj[-1]
                            prev_3d = traj_3d[-2]
                            curr_3d = traj_3d[-1]
                            abs_velocity = _compute_abs_velocity(prev_3d, curr_3d)

                            velocity_ratio = _normalize_speed(
                                abs_velocity,
                                speed_percentiles[tp],
                            )
                            arrow_color = _get_speed_color(velocity_ratio)

                            if show_trajectory:
                                velocity_arrows.append(
                                    (
                                        prev_point,
                                        curr_point,
                                        arrow_color,
                                    )
                                )

                            if show_gauges:
                                telemetry_rows.append(
                                    f"{tp:<16} {abs_velocity:>6.1f} {velocity_ratio:>10.2f}"
                                )
                        elif show_gauges:
                            telemetry_rows.append(f"{tp:<16} {'--':>6} {'--':>10}")

                    if overlay_mask and overlay_canvas is not None:
                        blended = cv2.addWeighted(
                            frame,
                            1 - overlay_opacity,
                            overlay_canvas,
                            overlay_opacity,
                            0,
                        )
                        frame = blended

                    for prev_point, curr_point, color in velocity_arrows:
                        draw_velocity_arrow(
                            frame,
                            prev_point,
                            curr_point,
                            color,
                            scale=VELOCITY_ARROW_LENGTH,
                            thickness=_trajectory_thickness,
                        )

                    if show_gauges:
                        draw_telemetry_panel(frame, telemetry_rows)

                    if draw_pose and pose_landmarks_for_drawing:
                        draw_pose_landmarks(
                            frame,
                            pose_landmarks_for_drawing,
                            color=pose_color,
                            thickness=2,
                            visibility_threshold=pose_visibility_threshold,
                            presence_threshold=pose_presence_threshold,
                        )

                    out.write(frame)
                    frame_idx += 1
                    pbar.update(1)
        else:
            print("Skipping video rendering because json_only=True")
    finally:
        if pose_detector is not None:
            pose_detector.close()
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    # Save PNG with just the trajectories if requested
    if trajectory_png_path is not None and not json_only:
        # from utils.body_trajectory import save_trajectories_as_png
        save_trajectories_as_png(
            trajectories, width, height, trajectory_png_path, colors=colors
        )
