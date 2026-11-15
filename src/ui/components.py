"""کامپوننت‌های گرافیکی: کادرهای رنگی زنده روی Canvas و ویجت پلاک استاندارد ایران."""

from __future__ import annotations

import flet as ft
import flet.canvas as cv

from config import (
    PLATE_LETTERS_PERSIAN,
    compose_iranian_plate,
    format_iranian_plate,
    parse_iranian_plate_text,
    to_latin_digits,
    to_persian_digits,
)
from core.geometry import normalized_box_to_canvas
from core.plate_detector import PlateDetection
from services.api_client import PlateCheckStatus


def _no_input_border():
    """سازگار با Flet 1.0+ (`NoInputBorder`) و نسخه‌های قبلی (`InputBorder.NONE`)."""
    if hasattr(ft, "NoInputBorder"):
        return ft.NoInputBorder()
    return ft.InputBorder.NONE


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


def status_to_badge_info(
    status: PlateCheckStatus,
    *,
    auto_query_enabled: bool = True,
) -> tuple[str, str, str]:
    """برگرداندن (عنوان فارسی، رنگ پس‌زمینه، آیکون مناسب) برای وضعیت استعلام."""
    if status == PlateCheckStatus.REJECTED:
        return ("پاسخ سرور: NO (غیرمجاز)", ft.Colors.RED_600, ft.Icons.GAVEL)
    if status == PlateCheckStatus.APPROVED:
        return ("پاسخ سرور: مجاز", ft.Colors.GREEN_600, ft.Icons.CHECK_CIRCLE)
    if status == PlateCheckStatus.ERROR:
        return ("خطا در ارتباط با سرور", ft.Colors.ORANGE_700, ft.Icons.WIFI_OFF)
    if not auto_query_enabled:
        return ("پلاک شناسایی شد (آماده استعلام دستی)", ft.Colors.BLUE_600, ft.Icons.TOUCH_APP)
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


class EditableIranianPlateInput(ft.Container):
    """ویجت ورود و ویرایش پلاک خودرو دقیقاً به شکل گرافیکی پلاک استاندارد ایران (مانند صفحه اول).

    چیدمان دیداری از چپ به راست:
    [نوار آبی I.R. IRAN]  [۲ رقم اول]  [حرف پلاک با انتخاب سریع]  [۳ رقم وسط]  |  [ایران + کد ۲ رقمی]
    """

    def __init__(self, page: ft.Page, border_color: str = ft.Colors.BLUE_700) -> None:
        super().__init__()
        self._app_page = page
        self._selected_letter: str = "ب"
        self._letter_dialog: ft.AlertDialog | None = None

        self.iran2_field = self._make_digit_field(hint="۱۱", max_digits=2, text_size=20)
        self.mid3_field = self._make_digit_field(hint="۳۴۵", max_digits=3, text_size=24)
        self.left2_field = self._make_digit_field(hint="۱۲", max_digits=2, text_size=24)

        self._letter_text = ft.Text(
            self._selected_letter,
            size=23,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.BLACK,
            text_align=ft.TextAlign.CENTER,
        )
        self._letter_button = ft.Container(
            expand=2,
            height=54,
            bgcolor=ft.Colors.BLUE_GREY_50,
            border_radius=6,
            border=ft.Border.all(1, ft.Colors.BLUE_GREY_200),
            alignment=ft.Alignment.CENTER,
            ink=True,
            tooltip="انتخاب حرف پلاک",
            on_click=self._open_letter_picker,
            content=ft.Row(
                spacing=0,
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    self._letter_text,
                    ft.Icon(ft.Icons.ARROW_DROP_DOWN, size=16, color=ft.Colors.BLUE_GREY_800),
                ],
            ),
        )

        # ساختار دقیقاً هم‌تراز با build_iranian_plate_widget (چیدمان rtl=True راست به چپ)
        plate_row = ft.Row(
            spacing=0,
            rtl=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                # ۱. سمت راست پلاک: کادر «ایران» و ورودی ۲ رقمی کد شهر
                ft.Container(
                    width=68,
                    height=66,
                    bgcolor=ft.Colors.WHITE,
                    alignment=ft.Alignment.CENTER,
                    padding=ft.Padding.symmetric(horizontal=2, vertical=2),
                    content=ft.Column(
                        spacing=0,
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Text(
                                "ایران",
                                size=10,
                                color=ft.Colors.BLACK,
                                weight=ft.FontWeight.W_700,
                            ),
                            self.iran2_field,
                        ],
                    ),
                ),
                # ۲. خط جداکننده عمودی مشکی
                ft.Container(width=2, height=66, bgcolor=ft.Colors.BLACK),
                # ۳. بخش اصلی پلاک (در RTL از راست به چپ: ۳ رقم ← حرف ← ۲ رقم)
                ft.Container(
                    expand=True,
                    height=66,
                    bgcolor=ft.Colors.WHITE,
                    alignment=ft.Alignment.CENTER,
                    padding=ft.Padding.symmetric(horizontal=6, vertical=4),
                    content=ft.Row(
                        spacing=4,
                        rtl=True,
                        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Container(
                                expand=3,
                                bgcolor=ft.Colors.GREY_50,
                                border_radius=6,
                                border=ft.Border.all(1, ft.Colors.BLUE_GREY_200),
                                alignment=ft.Alignment.CENTER,
                                content=self.mid3_field,
                            ),
                            self._letter_button,
                            ft.Container(
                                expand=2,
                                bgcolor=ft.Colors.GREY_50,
                                border_radius=6,
                                border=ft.Border.all(1, ft.Colors.BLUE_GREY_200),
                                alignment=ft.Alignment.CENTER,
                                content=self.left2_field,
                            ),
                        ],
                    ),
                ),
                # ۴. سمت چپ پلاک: نوار آبی پرچم ایران و I.R. IRAN
                ft.Container(
                    width=36,
                    height=66,
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

        self.border = ft.Border.all(3, border_color)
        self.border_radius = ft.BorderRadius.all(10)
        self.clip_behavior = ft.ClipBehavior.ANTI_ALIAS
        self.content = plate_row

    def _make_digit_field(self, hint: str, max_digits: int, text_size: int) -> ft.TextField:
        def _on_change(e: ft.ControlEvent) -> None:
            raw = e.control.value or ""
            digits = "".join(ch for ch in to_latin_digits(raw) if ch.isdigit())[:max_digits]
            formatted = to_persian_digits(digits)
            if e.control.value != formatted:
                e.control.value = formatted
                e.control.update()

        return ft.TextField(
            value="",
            hint_text=hint,
            text_align=ft.TextAlign.CENTER,
            keyboard_type=ft.KeyboardType.NUMBER,
            border=_no_input_border(),
            filled=False,
            dense=True,
            content_padding=ft.Padding.symmetric(horizontal=2, vertical=4),
            color=ft.Colors.BLACK,
            cursor_color=ft.Colors.BLUE_900,
            text_style=ft.TextStyle(
                size=text_size,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.BLACK,
            ),
            hint_style=ft.TextStyle(
                size=max(16, text_size - 4),
                color=ft.Colors.BLUE_GREY_300,
            ),
            on_change=_on_change,
        )

    def _open_letter_picker(self, _e: ft.ControlEvent) -> None:
        def _select_letter(letter: str) -> None:
            self._selected_letter = letter
            self._letter_text.value = letter
            self._letter_text.size = 14 if len(letter) > 3 else 23
            if self._letter_dialog is not None:
                self._app_page.pop_dialog()
            self._app_page.update()

        buttons: list[ft.Control] = []
        for ch in PLATE_LETTERS_PERSIAN:
            is_selected = ch == self._selected_letter
            buttons.append(
                ft.Container(
                    width=56 if len(ch) <= 3 else 92,
                    height=44,
                    border_radius=8,
                    bgcolor=ft.Colors.BLUE_600 if is_selected else "#1E293B",
                    border=ft.Border.all(
                        1.5,
                        ft.Colors.BLUE_300 if is_selected else ft.Colors.BLUE_GREY_700,
                    ),
                    alignment=ft.Alignment.CENTER,
                    ink=True,
                    on_click=lambda _ev, val=ch: _select_letter(val),
                    content=ft.Text(
                        ch,
                        size=16 if len(ch) <= 3 else 13,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.WHITE,
                    ),
                )
            )

        self._letter_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("انتخاب حرف پلاک", weight=ft.FontWeight.BOLD, size=17),
            content=ft.Container(
                width=310,
                height=310,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    controls=[
                        ft.Row(
                            wrap=True,
                            spacing=8,
                            run_spacing=8,
                            alignment=ft.MainAxisAlignment.CENTER,
                            controls=buttons,
                        )
                    ],
                ),
            ),
            actions=[
                ft.TextButton("بستن", on_click=lambda _ev: self._app_page.pop_dialog()),
            ],
        )
        self._app_page.show_dialog(self._letter_dialog)

    def set_plate(
        self,
        chars_persian: list[str] | None = None,
        plate_text: str = "",
    ) -> None:
        """ورودی پلاک را از لیست ۸ کاراکتر شناسایی‌شده یا رشته نمایشی پلاک پر می‌کند."""
        left2 = letter = mid3 = iran2 = ""
        if chars_persian and len(chars_persian) == 8:
            left2, letter, mid3, iran2 = format_iranian_plate(chars_persian)
        elif plate_text:
            left2, letter, mid3, iran2 = parse_iranian_plate_text(plate_text)

        self.left2_field.value = left2
        self.mid3_field.value = mid3
        self.iran2_field.value = iran2
        if letter:
            self._selected_letter = letter
            self._letter_text.value = letter
            self._letter_text.size = 14 if len(letter) > 3 else 23
        else:
            self._selected_letter = "ب"
            self._letter_text.value = "ب"
            self._letter_text.size = 23

    def get_plate_values(self) -> tuple[str, str, list[str]]:
        """مقادیر (رشته فارسی استاندارد، کد لاتین، لیست ۸ کاراکتر) را برمی‌گرداند."""
        return compose_iranian_plate(
            left2=self.left2_field.value or "",
            letter=self._selected_letter,
            mid3=self.mid3_field.value or "",
            iran2=self.iran2_field.value or "",
        )

    def is_complete(self) -> bool:
        """بررسی می‌کند آیا هر ۴ بخش پلاک ایران (۲ رقم + حرف + ۳ رقم + کد ایران) کامل پر شده‌اند."""
        _, _, chars_8 = self.get_plate_values()
        return len(chars_8) == 8

