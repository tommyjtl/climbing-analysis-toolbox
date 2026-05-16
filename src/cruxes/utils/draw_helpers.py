import cv2
import numpy as np

# Each tuple is (centroid_landmark_index, radius_endpoint_landmark_index).
# The circle is drawn at the centroid with radius = pixel distance to the endpoint,
# visualising the full reach envelope — every position the endpoint could occupy
# if the centroid joint stayed fixed at its current location.
#
# Groups allow callers to opt in to specific body regions, mirroring the track_point API.
# Pass ["all"] to draw every group, or a subset like ["left_upper", "right_lower"].
LIMB_REACH_CIRCLE_GROUPS = {
    "left_upper": [
        (15, 13),  # left wrist — left elbow
        (13, 11),  # left elbow — left shoulder
    ],
    "right_upper": [
        (16, 14),  # right wrist — right elbow
        (14, 12),  # right elbow — right shoulder
    ],
    "left_lower": [
        (23, 25),  # left hip — left knee
        (23, 27),  # left hip — left ankle
    ],
    "right_lower": [
        (24, 26),  # right hip — right knee
        (24, 28),  # right hip — right ankle
    ],
}


def draw_joint_circles(
    frame,
    landmarks,
    color,
    width,
    height,
    groups,
    visibility_threshold=0.5,
    presence_threshold=0.5,
    thickness=2,
    opacity=0.25,
):
    """Draw reach-envelope circles for the requested limb groups.

    groups: list of region names — any of "left_upper", "right_upper",
            "left_lower", "right_lower", or "all" to draw every group.
    Each circle is centered at a joint with radius = pixel distance to its
    paired joint, blended onto the frame at the given opacity."""
    active_groups = list(LIMB_REACH_CIRCLE_GROUPS.keys()) if "all" in groups else groups
    pairs = []
    for g in active_groups:
        pairs.extend(LIMB_REACH_CIRCLE_GROUPS.get(g, []))
    if not pairs:
        return

    overlay = frame.copy()
    drew_any = False
    for centroid_idx, radius_idx in pairs:
        if centroid_idx >= len(landmarks) or radius_idx >= len(landmarks):
            continue
        c = landmarks[centroid_idx]
        r = landmarks[radius_idx]
        c_vis = c.visibility if c.visibility is not None else 1.0
        c_pres = c.presence if c.presence is not None else 1.0
        r_vis = r.visibility if r.visibility is not None else 1.0
        r_pres = r.presence if r.presence is not None else 1.0
        if c_vis < visibility_threshold or c_pres < presence_threshold:
            continue
        if r_vis < visibility_threshold or r_pres < presence_threshold:
            continue
        cx = int(c.x * width)
        cy = int(c.y * height)
        rx = int(r.x * width)
        ry = int(r.y * height)
        radius = int(round(np.hypot(cx - rx, cy - ry)))
        if radius <= 0:
            continue
        cv2.circle(overlay, (cx, cy), radius, color, thickness, lineType=cv2.LINE_AA)
        drew_any = True
    if drew_any:
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def draw_trajectory(canvas, traj, color, thickness=2):
    for i in range(1, len(traj)):
        cv2.line(canvas, traj[i - 1], traj[i], color, thickness)


def _interpolate_color(color_a, color_b, ratio):
    clamped_ratio = min(max(float(ratio), 0.0), 1.0)
    return tuple(
        int(
            round(
                color_a[channel] + (color_b[channel] - color_a[channel]) * clamped_ratio
            )
        )
        for channel in range(3)
    )


def _draw_gradient_line(
    canvas, start_point, end_point, start_color, end_color, thickness
):
    steps = int(
        max(abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1]))
    )
    if steps <= 1:
        cv2.line(canvas, start_point, end_point, end_color, thickness)
        return

    xs = np.linspace(start_point[0], end_point[0], steps + 1)
    ys = np.linspace(start_point[1], end_point[1], steps + 1)

    for step_idx in range(steps):
        segment_start = (int(round(xs[step_idx])), int(round(ys[step_idx])))
        segment_end = (int(round(xs[step_idx + 1])), int(round(ys[step_idx + 1])))
        color = _interpolate_color(start_color, end_color, step_idx / max(steps - 1, 1))
        cv2.line(canvas, segment_start, segment_end, color, thickness)


def draw_colored_trajectory(canvas, traj, segment_colors, thickness=2):
    if len(traj) < 2 or not segment_colors:
        return

    for idx in range(1, len(traj)):
        end_color = segment_colors[idx - 1]
        start_color = segment_colors[idx - 2] if idx > 1 else end_color
        _draw_gradient_line(
            canvas,
            traj[idx - 1],
            traj[idx],
            start_color,
            end_color,
            thickness,
        )


def draw_velocity_arrow(canvas, prev_point, curr_point, color, scale=5, thickness=3):
    dx = curr_point[0] - prev_point[0]
    dy = curr_point[1] - prev_point[1]
    direction_norm = np.hypot(dx, dy)
    if direction_norm == 0:
        return

    arrow_length = scale
    direction_x = dx / direction_norm
    direction_y = dy / direction_norm
    end_point = (
        curr_point[0] + int(direction_x * arrow_length),
        curr_point[1] + int(direction_y * arrow_length),
    )
    cv2.arrowedLine(canvas, curr_point, end_point, color, thickness, tipLength=0.3)


def draw_telemetry_panel(canvas, telemetry_rows, origin=(20, 20)):
    if not telemetry_rows:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1
    line_height = 24
    header = "joint | raw_v | vel_ratio"
    padding_x = 10
    padding_y = 10

    rendered_rows = [header] + telemetry_rows
    text_width = max(
        cv2.getTextSize(row, font, font_scale, text_thickness)[0][0]
        for row in rendered_rows
    )
    panel_width = text_width + padding_x * 2
    panel_height = padding_y * 2 + line_height * len(rendered_rows)
    x0, y0 = origin
    x1 = x0 + panel_width
    y1 = y0 + panel_height

    cv2.rectangle(canvas, (x0, y0), (x1, y1), (15, 15, 15), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (230, 230, 230), 1)

    for row_idx, row in enumerate(rendered_rows):
        text_y = y0 + padding_y + (row_idx + 1) * line_height - 6
        color = (230, 230, 230) if row_idx == 0 else (245, 245, 245)
        cv2.putText(
            canvas,
            row,
            (x0 + padding_x, text_y),
            font,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )
