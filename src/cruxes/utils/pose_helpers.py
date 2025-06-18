import mediapipe as mp


def get_track_point_coords(tp, landmarks, w, h, confidence_threshold=0.6):
    """
    Map a track point name to its 2D and 3D coordinates using MediaPipe landmarks.
    Returns (mid_point, mid_point_3d) or None if confidence is too low or invalid tp.
    """
    mp_pose = mp.solutions.pose
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    if tp == "hip_mid":
        if (
            left_hip.visibility < confidence_threshold
            or right_hip.visibility < confidence_threshold
        ):
            return None
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
            return None
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
            (left_hip.x + right_hip.x + left_shoulder.x + right_shoulder.x) / 4,
            (left_hip.y + right_hip.y + left_shoulder.y + right_shoulder.y) / 4,
            (left_hip.z + right_hip.z + left_shoulder.z + right_shoulder.z) / 4,
        )
    elif tp == "head":
        if nose.visibility < confidence_threshold:
            return None
        mid_point = (int(nose.x * w), int(nose.y * h))
        mid_point_3d = (nose.x, nose.y, nose.z)
    elif tp == "left_hand":
        if left_hand.visibility < confidence_threshold:
            return None
        mid_point = (int(left_hand.x * w), int(left_hand.y * h))
        mid_point_3d = (left_hand.x, left_hand.y, left_hand.z)
    elif tp == "right_hand":
        if right_hand.visibility < confidence_threshold:
            return None
        mid_point = (int(right_hand.x * w), int(right_hand.y * h))
        mid_point_3d = (right_hand.x, right_hand.y, right_hand.z)
    elif tp == "left_foot":
        if left_foot.visibility < confidence_threshold:
            return None
        mid_point = (int(left_foot.x * w), int(left_foot.y * h))
        mid_point_3d = (left_foot.x, left_foot.y, left_foot.z)
    elif tp == "right_foot":
        if right_foot.visibility < confidence_threshold:
            return None
        mid_point = (int(right_foot.x * w), int(right_foot.y * h))
        mid_point_3d = (right_foot.x, right_foot.y, right_foot.z)
    else:
        raise ValueError(
            "Invalid track_point option. "
            "Use 'hip_mid', 'upper_body_center', 'head', 'left_hand', 'right_hand', 'left_foot', 'right_foot'."
        )
    return mid_point, mid_point_3d
