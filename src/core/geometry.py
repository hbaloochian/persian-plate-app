"""توابع هندسی: چرخش تصویر سنسور دوربین اندروید و نگاشت مختصات کادر روی Canvas."""

from __future__ import annotations

from PIL import Image


def compute_android_rotation_degrees(
    sensor_orientation: int,
    device_orientation_degrees: int,
    is_front_camera: bool = False,
) -> int:
    """زاویه چرخش ساعتگرد لازم برای عمودی‌سازی فریم خام سنسور دوربین اندروید.

    در اندروید، سنسور دوربین عقب معمولاً 90 درجه نسبت به حالت عمودی گوشی چرخیده است.
    اگر فریم بدون چرخش به مدل YOLO داده شود، پلاک به‌صورت عمودی دیده شده و تشخیص افت می‌کند.
    """
    if is_front_camera:
        return (sensor_orientation + device_orientation_degrees) % 360
    return (sensor_orientation - device_orientation_degrees + 360) % 360


def rotate_pil_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    """تصویر PIL را به‌اندازه مضرب 90 درجه ساعتگرد می‌چرخاند (بسیار سریع با transpose)."""
    angle = degrees % 360
    if angle == 90:
        return image.transpose(Image.Transpose.ROTATE_270)
    if angle == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if angle == 270:
        return image.transpose(Image.Transpose.ROTATE_90)
    return image


def normalized_box_to_canvas(
    norm_box: tuple[float, float, float, float],
    canvas_width: float,
    canvas_height: float,
    mirror_x: bool = False,
) -> tuple[float, float, float, float]:
    """تبدیل مختصات نرمال‌شده [nx1, ny1, nx2, ny2] در بازه [0, 1] به (x, y, w, h) روی Canvas."""
    nx1, ny1, nx2, ny2 = norm_box
    nx1 = max(0.0, min(1.0, nx1))
    ny1 = max(0.0, min(1.0, ny1))
    nx2 = max(0.0, min(1.0, nx2))
    ny2 = max(0.0, min(1.0, ny2))

    if mirror_x:
        nx1, nx2 = 1.0 - nx2, 1.0 - nx1

    x = nx1 * canvas_width
    y = ny1 * canvas_height
    w = max(2.0, (nx2 - nx1) * canvas_width)
    h = max(2.0, (ny2 - ny1) * canvas_height)
    return x, y, w, h
