from dataclasses import dataclass
from enum import IntEnum
import os
from pathlib import Path
import shutil
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import mediapipe as mp

POSE_MODEL_ENV_VAR = "CRUXES_POSE_MODEL_PATH"
DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_POSE_MODEL_PATH = (
    Path.home() / ".cache" / "cruxes" / "mediapipe" / "pose_landmarker_full.task"
)
VISIBILITY_THRESHOLD = 0.2
PRESENCE_THRESHOLD = 0.2


class PoseLandmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


@dataclass
class NormalizedPoseLandmark:
    x: float
    y: float
    z: float
    visibility: Optional[float] = None
    presence: Optional[float] = None

    @classmethod
    def from_landmark(cls, landmark):
        return cls(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=getattr(landmark, "visibility", None),
            presence=getattr(landmark, "presence", None),
        )


@dataclass
class WorldPoseLandmark:
    x: float
    y: float
    z: float
    visibility: Optional[float] = None
    presence: Optional[float] = None

    @classmethod
    def from_landmark(cls, landmark):
        return cls(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=getattr(landmark, "visibility", None),
            presence=getattr(landmark, "presence", None),
        )


@dataclass
class PoseResult:
    pose_landmarks: Optional[list[NormalizedPoseLandmark]]
    world_pose_landmarks: Optional[list[WorldPoseLandmark]] = None


class _MediaPipeDetector:
    """Internal MediaPipe-based pose detector."""

    def __init__(self):
        self._backend = None
        self._detector = None
        self._init_backend()

    def _init_backend(self):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._backend = "solutions"
            self._detector = mp.solutions.pose.Pose()
            return

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model_path = resolve_pose_model_path()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._backend = "tasks"
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr, timestamp_ms=None):
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            result = self._detector.process(image_rgb)
            if not result.pose_landmarks:
                return PoseResult(pose_landmarks=None, world_pose_landmarks=None)

            world_pose_landmarks = None
            if getattr(result, "pose_world_landmarks", None):
                world_pose_landmarks = [
                    WorldPoseLandmark.from_landmark(landmark)
                    for landmark in result.pose_world_landmarks.landmark
                ]

            return PoseResult(
                pose_landmarks=[
                    NormalizedPoseLandmark.from_landmark(landmark)
                    for landmark in result.pose_landmarks.landmark
                ],
                world_pose_landmarks=world_pose_landmarks,
            )

        if timestamp_ms is None:
            raise ValueError("timestamp_ms is required for the MediaPipe Tasks backend")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseResult(pose_landmarks=None, world_pose_landmarks=None)

        world_pose_landmarks = None
        if getattr(result, "pose_world_landmarks", None):
            world_pose_landmarks = [
                WorldPoseLandmark.from_landmark(landmark)
                for landmark in result.pose_world_landmarks[0]
            ]

        return PoseResult(
            pose_landmarks=[
                NormalizedPoseLandmark.from_landmark(landmark)
                for landmark in result.pose_landmarks[0]
            ],
            world_pose_landmarks=world_pose_landmarks,
        )

    def close(self):
        if self._detector is not None:
            self._detector.close()


def _auto_detect_vitpose_device():
    """Return 'mps' on Apple Silicon when CoreML is available, else 'cpu'."""
    import platform

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import onnxruntime

            if "CoreMLExecutionProvider" in onnxruntime.get_available_providers():
                return "mps"
        except ImportError:
            pass
    return "cpu"


class ViTPoseDetector:
    """ViTPose-based pose detector via rtmlib.

    Keypoints are returned in COCO-17 format and adapted into a 33-slot
    MediaPipe-compatible ``NormalizedPoseLandmark`` list so that all
    downstream code continues to work without modification.

    Unused MediaPipe slots receive ``visibility=0.0`` so they are skipped
    during rendering and trajectory extraction.
    """

    # COCO-17 keypoint index -> MediaPipe-33 index
    _COCO_TO_MP: dict = {
        0: 0,  # nose
        1: 2,  # left_eye
        2: 5,  # right_eye
        3: 7,  # left_ear
        4: 8,  # right_ear
        5: 11,  # left_shoulder
        6: 12,  # right_shoulder
        7: 13,  # left_elbow
        8: 14,  # right_elbow
        9: 15,  # left_wrist
        10: 16,  # right_wrist
        11: 23,  # left_hip
        12: 24,  # right_hip
        13: 25,  # left_knee
        14: 26,  # right_knee
        15: 27,  # left_ankle
        16: 28,  # right_ankle
    }
    _NUM_MP_LANDMARKS = 33

    _DET_URL = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_x_8xb8-300e_humanart-a39d44ed.zip"
    )
    _POSE_URL = (
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/"
        "vitpose-b-coco.onnx"
    )

    def __init__(self, device=None, backend="onnxruntime", det_frequency=3):
        if device is None:
            device = _auto_detect_vitpose_device()

        from functools import partial

        try:
            from rtmlib import PoseTracker, Custom
        except ImportError as exc:
            raise ImportError(
                "ViTPoseDetector requires the optional 'rtmlib' dependency. "
                "Install it with: pip install cruxes[vitpose]  "
                "(or directly: pip install rtmlib)"
            ) from exc

        vitpose_cls = partial(
            Custom,
            det_class="YOLOX",
            det=self._DET_URL,
            det_input_size=(640, 640),
            pose_class="ViTPose",
            pose=self._POSE_URL,
            pose_input_size=(192, 256),
        )
        self._tracker = PoseTracker(
            vitpose_cls,
            det_frequency=det_frequency,
            backend=backend,
            device=device,
            to_openpose=False,
        )
        # Pixel-space centroid of the last successfully tracked person.
        # Used to keep tracking the same individual across frames.
        self._last_center = None  # (x_px, y_px) or None

    def _person_center(self, kpts, confs):
        """Weighted centroid (pixel coords) of a single person's keypoints."""
        cx, cy, total = 0.0, 0.0, 0.0
        for i in range(len(kpts)):
            c = float(confs[i])
            if c > 0.1:
                cx += float(kpts[i][0]) * c
                cy += float(kpts[i][1]) * c
                total += c
        if total < 0.1:
            return None
        return (cx / total, cy / total)

    def _select_person(self, keypoints, scores):
        """Return the index of the person to track this frame.

        On the first detection: pick the person with the highest mean confidence
        (most visible / most complete detection — likely the climber on the wall).
        On every subsequent frame: pick whoever is spatially closest to the last
        known centroid, so the tracker sticks to the same individual even if
        another person walks into the scene.
        """
        if len(keypoints) == 1:
            return 0

        centers = [
            self._person_center(keypoints[i], scores[i]) for i in range(len(keypoints))
        ]

        if self._last_center is None:
            # First detection — pick highest average confidence
            best, best_conf = 0, -1.0
            for i in range(len(keypoints)):
                avg = float(scores[i].mean())
                if avg > best_conf:
                    best_conf = avg
                    best = i
            return best

        # Subsequent frames — pick person closest to last known position
        lx, ly = self._last_center
        best, best_dist = 0, float("inf")
        for i, center in enumerate(centers):
            if center is None:
                continue
            dist = (center[0] - lx) ** 2 + (center[1] - ly) ** 2
            if dist < best_dist:
                best_dist = dist
                best = i
        return best

    def process(self, frame_bgr, timestamp_ms=None):
        keypoints, scores = self._tracker(frame_bgr)
        # keypoints: (N, 17, 2) pixel coords; scores: (N, 17) confidence
        if keypoints is None or len(keypoints) == 0:
            return PoseResult(pose_landmarks=None, world_pose_landmarks=None)

        h, w = frame_bgr.shape[:2]

        person_idx = self._select_person(keypoints, scores)
        kpts = keypoints[person_idx]
        confs = scores[person_idx]

        # Update sticky centroid for next frame
        center = self._person_center(kpts, confs)
        if center is not None:
            self._last_center = center

        # Build 33-slot list; unused slots get visibility=0.0
        landmarks = [
            NormalizedPoseLandmark(x=0.0, y=0.0, z=0.0, visibility=0.0, presence=0.0)
            for _ in range(self._NUM_MP_LANDMARKS)
        ]
        for coco_idx, mp_idx in self._COCO_TO_MP.items():
            x_px = float(kpts[coco_idx][0])
            y_px = float(kpts[coco_idx][1])
            conf = float(confs[coco_idx])
            landmarks[mp_idx] = NormalizedPoseLandmark(
                x=x_px / w,
                y=y_px / h,
                z=0.0,
                visibility=conf,
                presence=conf,
            )

        return PoseResult(pose_landmarks=landmarks, world_pose_landmarks=None)

    def close(self):
        pass


class PoseDetector:
    """Pose detector that supports multiple backends.

    Args:
        backend: ``"mediapipe"`` (default) or ``"vitpose"``.
        vitpose_device: Device for ViTPose inference (``"mps"``, ``"cpu"``).
            Defaults to auto-detection (MPS on Apple Silicon when available).
        vitpose_det_frequency: How often YOLOX re-runs detection (every N frames).
    """

    def __init__(
        self,
        backend="mediapipe",
        vitpose_device=None,
        vitpose_det_frequency=3,
    ):
        _SUPPORTED_BACKENDS = {"mediapipe", "vitpose"}
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown pose backend: {backend!r}. "
                f"Supported backends are: {sorted(_SUPPORTED_BACKENDS)}"
            )
        if backend == "vitpose":
            self._impl = ViTPoseDetector(
                device=vitpose_device,
                det_frequency=vitpose_det_frequency,
            )
        else:
            self._impl = _MediaPipeDetector()

    def process(self, frame_bgr, timestamp_ms=None):
        return self._impl.process(frame_bgr, timestamp_ms=timestamp_ms)

    def close(self):
        self._impl.close()


def resolve_pose_model_path():
    env_model_path = os.environ.get(POSE_MODEL_ENV_VAR)
    if env_model_path:
        if os.path.isfile(env_model_path):
            return env_model_path
        raise FileNotFoundError(
            f"{POSE_MODEL_ENV_VAR} is set, but the file does not exist: {env_model_path}"
        )

    if DEFAULT_POSE_MODEL_PATH.is_file():
        return str(DEFAULT_POSE_MODEL_PATH)

    return download_default_pose_model(DEFAULT_POSE_MODEL_PATH)


def download_default_pose_model(destination_path):
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")

    print(f"Downloading MediaPipe pose model to {destination_path}...")
    try:
        with (
            urlopen(DEFAULT_POSE_MODEL_URL) as response,
            open(temporary_path, "wb") as file_obj,
        ):
            shutil.copyfileobj(response, file_obj)
        os.replace(temporary_path, destination_path)
    except (OSError, URLError) as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(
            "MediaPipe on this Python build requires a Pose Landmarker model. "
            f"Failed to download the default model from {DEFAULT_POSE_MODEL_URL}. "
            f"Set {POSE_MODEL_ENV_VAR} to a local .task file or rerun with network access."
        ) from exc

    return str(destination_path)


# Landmark indices that get a filled dot when drawing the pose skeleton.
PRIMARY_JOINT_INDICES = frozenset([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])


def draw_pose_landmarks(
    image,
    landmarks,
    color=(255, 255, 255),
    thickness=2,
    visibility_threshold=VISIBILITY_THRESHOLD,
    presence_threshold=PRESENCE_THRESHOLD,
):
    if not landmarks:
        return

    image_height, image_width = image.shape[:2]
    coordinates = {}
    for idx, landmark in enumerate(landmarks):
        visibility = getattr(landmark, "visibility", None)
        presence = getattr(landmark, "presence", None)
        if visibility is not None and visibility < visibility_threshold:
            continue
        if presence is not None and presence < presence_threshold:
            continue
        if not (0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0):
            continue
        coordinates[idx] = (
            min(int(landmark.x * image_width), image_width - 1),
            min(int(landmark.y * image_height), image_height - 1),
        )

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in coordinates and end_idx in coordinates:
            cv2.line(
                image, coordinates[start_idx], coordinates[end_idx], color, thickness
            )

    dot_radius = max(3, thickness * 2)
    for idx, pt in coordinates.items():
        if idx in PRIMARY_JOINT_INDICES:
            cv2.circle(image, pt, dot_radius, color, -1, lineType=cv2.LINE_AA)
