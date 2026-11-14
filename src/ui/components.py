"""کامپوننت‌های گرافیکی: کادرهای رنگی زنده روی Canvas و ویجت پلاک استاندارد ایران."""

from __future__ import annotations

import flet as ft
import flet.canvas as cv

from config import format_iranian_plate, to_persian_digits
from core.geometry import normalized_box_to_canvas
from core.plate_detector import PlateDetection
from services.api_client import PlateCheckStatus


def status_to_box_color(status: PlateCheckStatus) -> str:
    """تعیین رنگ دقیق کادر دور پلاک مطابق نیازمندی:
    - تشخیص اولیه / در حال استعلام: آبی
    - پاسخ NO از سرور: قرمز
    - سایر پاسخ‌های معتبر سرور: سبز
    """
    if status == PlateCheckStatus.REJECTED:
        return ft.Colors.RED_500
    if status == PlateCheckStatus.APPROVED:
        return ft.Colors.GREEN_500
    return ft.Colors.BLUE_500


def status_to_badge_info(status: PlateCheckStatus) -> tuple[str, str, str]:
    """برگرداندن (عنوان فارسی، رنگ پس‌زمینه، آیکون مناسب) برای وضعیت استعلام."""
    if status == PlateCheckStatus.REJECTED:
        return ("پاسخ سرور: NO (غیرمجاز)", ft.Colors.RED_600, ft.Icons.GAVEL)
    if status == PlateCheckStatus.APPROVED:
        return ("پاسخ سرور: مجاز", ft.Colors.GREEN_600, ft.Icons.CHECK_CIRCLE)
    if status == PlateCheckStatus.ERROR:
        return ("خطا در ارتباط با سرور", ft.Colors.ORANGE_700, ft.Icons.WIFI_OFF)
    return ("پلاک شناسایی شد (در حال استعلام...)", ft.Colors.BLUE_600, ft.Icons.SEARCH)


def build_overlay_shapes(
    plates: list[PlateDetection],
    plate_statuses: dict[str, PlateCheckStatus],
    canvas_width: float,
    canvas_height: float,
    mirror_x: bool = False,
) -> list[cv.Shape]:
    """اشکال برداری کادر دور پلاک و برچسب آن را برای رسم سریع روی Canvas می‌سازد."""
    if canvas_width <= 0 or canvas_height <= 0 or not plates:
        return []

    shapes: list[cv.Shape] = []
    for plate in plates:
        plate_key = plate.plate_latin or plate.plate_persian
        status = plate_statuses.get(plate_key, PlateCheckStatus.PENDING)
        color = status_to_box_color(status)

        x, y, w, h = normalized_box_to_canvas(
            plate.norm_box,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            mirror_x=mirror_x,
        )

        # پرشدگی بسیار کم‌رنگ داخل کادر برای خوانایی بهتر روی نور روز
        shapes.append(
            cv.Rect(
                x=x,
                y=y,
                width=w,
                height=h,
                border_radius=6,
                paint=ft.Paint(
                    color=ft.Colors.with_opacity(0.14, color),
                    style=ft.PaintingStyle.FILL,
                ),
            )
        )

        # کادر اصلی دور پلاک (آبی، قرمز یا سبز)
        shapes.append(
            cv.Rect(
                x=x,
                y=y,
                width=w,
                height=h,
                border_radius=6,
                paint=ft.Paint(
                    color=color,
                    stroke_width=3.5,
                    style=ft.PaintingStyle.STROKE,
                ),
            )
        )

        # برچسب متن پلاک و درصد اطمینان بالای کادر
        conf_pct = to_persian_digits(f"{int(round(plate.confidence * 100))}%")
        label_text = f"{plate.display_text} ({conf_pct})"
        label_h = 26.0
        label_w = max(w, min(canvas_width - 8.0, len(label_text) * 8.5 + 18.0))
        label_y = y - label_h - 4.0 if y > label_h + 6.0 else min(canvas_height - label_h - 2.0, y + h + 4.0)
        label_x = max(4.0, min(canvas_width - label_w - 4.0, x))

        shapes.append(
            cv.Rect(
                x=label_x,
                y=label_y,
                width=label_w,
                height=label_h,
                border_radius=5,
                paint=ft.Paint(
                    color=ft.Colors.with_opacity(0.88, color),
                    style=ft.PaintingStyle.FILL,
                ),
            )
        )
        shapes.append(
            cv.Text(
                x=label_x + label_w / 2.0,
                y=label_y + label_h / 2.0,
                value=label_text,
                alignment=ft.Alignment.CENTER,
                style=ft.TextStyle(
                    color=ft.Colors.WHITE,
                    size=12,
                    weight=ft.FontWeight.BOLD,
                ),
            )
        )

    return shapes


def build_iranian_plate_widget(
    chars_persian: list[str],
    status: PlateCheckStatus,
) -> ft.Container:
    """نمایش گرافیکی پلاک استاندارد خودروهای ایران با حاشیه رنگی متناسب با وضعیت سرور.

    چیدمان صحیح پلاک ایران (از راست به چپ در صفحه RTL، و از چپ به راست به‌صورت دیداری):
    - سمت راست: کادر «ایران» و کد ۲ رقمی شهر
    - خط جداکننده عمودی
    - وسط (راست به چپ): ۳ رقم ← حرف فارسی ← ۲ رقم
    - سمت چپ: نوار آبی پرچم ایران و عبارت I.R. IRAN

    نکته فنی Flet 1.0: چون `rtl=False` مقدار پیش‌فرض Control است، روی صفحه‌ای با
    `page.rtl = True` مقدار `rtl=False` جهت چیدمان را به LTR برنمی‌گرداند؛ بنابراین
    چیدمان عناصر صراحتاً با `rtl=True` و به ترتیب راست‌به‌چپ تنظیم شده است تا
    روی همه گوشی‌ها همیشه نوار آبی دقیقاً در سمت چپ و «ایران» در سمت راست قرار گیرد.
    """
    border_color = status_to_box_color(status)
    left2, letter, mid3, iran2 = format_iranian_plate(chars_persian)

    if len(chars_persian) == 8:
        plate_body = ft.Row(
            spacing=0,
            rtl=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                # ۱. سمت راست پلاک: کادر «ایران» و کد دو رقمی شهر
                ft.Container(
                    width=58,
                    height=54,
                    bgcolor=ft.Colors.WHITE,
                    alignment=ft.Alignment.CENTER,
                    content=ft.Column(
                        spacing=0,
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Text("ایران", size=10, color=ft.Colors.BLACK, weight=ft.FontWeight.W_600),
                            ft.Text(iran2, size=20, color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD),
                        ],
                    ),
                ),
                # ۲. خط جداکننده عمودی
                ft.Container(width=2, height=54, bgcolor=ft.Colors.BLACK),
                # ۳. بخش اصلی شماره پلاک (در RTL از راست به چپ: ۳ رقم ← حرف ← ۲ رقم)
                ft.Container(
                    expand=True,
                    height=54,
                    bgcolor=ft.Colors.WHITE,
                    alignment=ft.Alignment.CENTER,
                    padding=ft.Padding.symmetric(horizontal=8),
                    content=ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        rtl=True,
                        controls=[
                            ft.Text(mid3, size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK),
                            ft.Text(letter, size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK),
                            ft.Text(left2, size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK),
                        ],
                    ),
                ),
                # ۴. سمت چپ پلاک: نوار آبی پرچم ایران و I.R. IRAN
                ft.Container(
                    width=34,
                    height=54,
                    bgcolor=ft.Colors.BLUE_900,
                    alignment=ft.Alignment.CENTER,
                    padding=ft.Padding.symmetric(vertical=4),
                    content=ft.Column(
                        spacing=2,
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Container(width=18, height=3, bgcolor=ft.Colors.GREEN_500),
                            ft.Container(width=18, height=3, bgcolor=ft.Colors.WHITE),
                            ft.Container(width=18, height=3, bgcolor=ft.Colors.RED_500),
                            ft.Text("I.R.", size=8, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
                            ft.Text("IRAN", size=7, color=ft.Colors.WHITE),
                        ],
                    ),
                ),
            ],
        )
    else:
        placeholder = left2 if left2 else "در انتظار تشخیص پلاک..."
        plate_body = ft.Container(
            height=54,
            bgcolor=ft.Colors.WHITE,
            alignment=ft.Alignment.CENTER,
            padding=ft.Padding.symmetric(horizontal=12),
            content=ft.Text(
                placeholder,
                size=16,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.BLUE_GREY_800,
            ),
        )

    return ft.Container(
        border=ft.Border.all(3, border_color),
        border_radius=ft.BorderRadius.all(10),
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        content=plate_body,
    )
