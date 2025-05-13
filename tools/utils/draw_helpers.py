import cv2
import numpy as np


def draw_trajectory(canvas, traj, color, thickness=2):
    for i in range(1, len(traj)):
        cv2.line(canvas, traj[i - 1], traj[i], color, thickness)


def draw_velocity_arrow(canvas, prev_point, curr_point, color, scale=5, thickness=3):
    dx = curr_point[0] - prev_point[0]
    dy = curr_point[1] - prev_point[1]
    end_point = (
        curr_point[0] + int(dx * scale),
        curr_point[1] + int(dy * scale),
    )
    cv2.arrowedLine(canvas, curr_point, end_point, color, thickness, tipLength=0.3)


# Gauge configuration dataclass
class GaugeConfig:
    def __init__(
        self,
        radius=60,
        thickness=15,
        spacing=120,
        max_velocity=50,
        top=120,
        left_start=120,
    ):
        self.radius = radius
        self.thickness = thickness
        self.spacing = spacing
        self.max_velocity = max_velocity
        self.top = top
        self.left_start = left_start


def get_gauge_centers(track_point, config: GaugeConfig):
    return [
        (
            config.left_start + i * (2 * config.radius + config.spacing),
            config.top,
        )
        for i in range(len(track_point))
    ]


def draw_gauge(
    canvas,
    center,
    gauge_radius,
    gauge_thickness,
    start_angle,
    end_angle,
    gauge_color,
    max_velocity,
    abs_velocity,
    velocity_text,
    max_velocity_text,
):
    # Draw background arc (gray)
    cv2.ellipse(
        canvas,
        center,
        (gauge_radius, gauge_radius),
        0,
        start_angle,
        start_angle + 270,
        (220, 220, 220),
        gauge_thickness,
    )
    # Draw value arc
    cv2.ellipse(
        canvas,
        center,
        (gauge_radius, gauge_radius),
        0,
        start_angle,
        end_angle,
        gauge_color,
        gauge_thickness,
    )
    # Draw center circle
    cv2.circle(
        canvas,
        center,
        gauge_radius - gauge_thickness,
        (255, 255, 255),
        -1,
    )
    # Draw pointer
    pointer_angle_rad = np.deg2rad(end_angle)
    pointer_length = gauge_radius - gauge_thickness // 2
    pointer_x = int(center[0] + pointer_length * np.cos(pointer_angle_rad))
    pointer_y = int(center[1] + pointer_length * np.sin(pointer_angle_rad))
    cv2.line(canvas, center, (pointer_x, pointer_y), gauge_color, 3)
    # Draw velocity value below the gauge
    text_size = cv2.getTextSize(velocity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + gauge_radius + 30
    text_bg_x0 = text_x - 5
    text_bg_y0 = text_y - text_size[1] - 5
    text_bg_x1 = text_x + text_size[0] + 5
    text_bg_y1 = text_y + 5
    cv2.rectangle(
        canvas, (text_bg_x0, text_bg_y0), (text_bg_x1, text_bg_y1), (245, 245, 245), -1
    )
    cv2.putText(
        canvas,
        velocity_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    # Draw max velocity value below the current velocity
    max_text_size = cv2.getTextSize(
        max_velocity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )[0]
    max_text_x = center[0] - max_text_size[0] // 2
    max_text_y = text_y + text_size[1] + 10
    max_text_bg_x0 = max_text_x - 5
    max_text_bg_y0 = max_text_y - max_text_size[1] - 5
    max_text_bg_x1 = max_text_x + max_text_size[0] + 5
    max_text_bg_y1 = max_text_y + 5
    cv2.rectangle(
        canvas,
        (max_text_bg_x0, max_text_bg_y0),
        (max_text_bg_x1, max_text_bg_y1),
        (245, 245, 245),
        -1,
    )
    cv2.putText(
        canvas,
        max_velocity_text,
        (max_text_x, max_text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )


def draw_label(canvas, center, gauge_radius, label_text, color):
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    label_x = center[0] - label_size[0] // 2
    label_y = center[1] - gauge_radius - 25
    label_bg_x0 = label_x - 5
    label_bg_y0 = label_y - label_size[1] - 5
    label_bg_x1 = label_x + label_size[0] + 5
    label_bg_y1 = label_y + 5
    cv2.rectangle(
        canvas, (label_bg_x0, label_bg_y0), (label_bg_x1, label_bg_y1), (0, 0, 0), -1
    )
    cv2.putText(
        canvas,
        label_text,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
