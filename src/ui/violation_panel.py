"""صفحه تمام‌صفحه ثبت تخلف خودرو با ورودی گرافیکی پلاک استاندارد ایران.

امکانات:
- نمایش تمام‌صفحه با نوار عنوان و نمایش تاریخ شمسی و ساعت فعلی
- ورودی پلاک خودرو دقیقاً به شکل گرافیکی پلاک ایران (مانند صفحه اول) با قابلیت پر شدن خودکار و ویرایش
- انتخاب نوع تخلف و توضیحات تکمیلی
- پیوست تصویر برش‌خورده پلاک (با پیش‌نمایش)
- افزودن عکس مستندات تخلف (از طریق دوربین زنده یا انتخاب از گالری گوشی)
"""

from __future__ import annotations

from typing import Awaitable, Callable

import flet as ft

from config import VIOLATION_TYPES, to_persian_digits
from core.jalali import format_jalali_date, format_jalali_time, now_iran_datetime
from ui.components import EditableIranianPlateInput
from ui.theme import get_theme_palette


class ViolationRegistrationPanel(ft.Container):
    """صفحه تمام‌صفحه ثبت تخلف."""

    def __init__(
        self,
        page: ft.Page,
        on_back: Callable[[], None],
        on_submit: Callable[
            [str, str, str, str, bool, bytes | None, float | None, float | None, float | None],
            Awaitable[None],
        ],
        on_capture_evidence: Callable[[], Awaitable[bytes | None]],
        on_pick_evidence: Callable[[], Awaitable[bytes | None]],
        on_refresh_gps: Callable[[], Awaitable[None]] | None = None,
        dark_mode: bool = True,
    ) -> None:
        self._palette = get_theme_palette(dark_mode)
        super().__init__(
            expand=True,
            bgcolor=self._palette.page_bg,
            padding=ft.Padding.all(14),
        )
        self._on_back = on_back
        self._on_submit = on_submit
        self._on_capture_evidence = on_capture_evidence
        self._on_pick_evidence = on_pick_evidence
        self._on_refresh_gps = on_refresh_gps
        self._submitting = False
        self._capturing_evidence = False
        self._evidence_bytes: bytes | None = None
        self._gps_lat: float | None = None
        self._gps_lon: float | None = None
        self._gps_accuracy: float | None = None

        self.datetime_badge = ft.Text(
            "",
            size=12,
            color=self._palette.accent_text,
            weight=ft.FontWeight.W_600,
        )

        # ورودی گرافیکی پلاک ایران (دقیقاً به شکل پلاک صفحه اول)
        self.plate_input = EditableIranianPlateInput(page=page, border_color=ft.Colors.BLUE_500)

        self.violation_dropdown = ft.Dropdown(
            label="نوع تخلف",
            value=VIOLATION_TYPES[0],
            expand=True,
            options=[
                ft.DropdownOption(key=item, text=item)
                for item in VIOLATION_TYPES
            ],
        )

        self.notes_field = ft.TextField(
            label="توضیحات تکمیلی (اختیاری)",
            hint_text="محل دقیق تخلف یا یادداشت مأمور...",
            expand=True,
            multiline=True,
            min_lines=6,
            max_lines=10,
        )

        # بخش پیوست تصویر برش‌خورده پلاک
        self.attach_crop_checkbox = ft.Checkbox(
            label="پیوست تصویر برش‌خورده پلاک",
            value=True,
        )
        self.plate_crop_preview_holder = ft.Container(visible=False)

        # بخش افزودن عکس مستندات تخلف (دوربین زنده یا گالری)
        self.evidence_status_text = ft.Text(
            "عکسی انتخاب نشده (اختیاری)",
            size=12,
            color=self._palette.text_muted,
        )
        self.evidence_preview_holder = ft.Container(visible=False)

        self.capture_evidence_btn = ft.OutlinedButton(
            content=ft.Row(
                tight=True,
                spacing=6,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.CAMERA_ALT, size=18),
                    ft.Text("عکس با دوربین", size=13),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=10, vertical=10),
            ),
            expand=True,
            on_click=self._handle_capture_evidence,
        )

        self.pick_evidence_btn = ft.OutlinedButton(
            content=ft.Row(
                tight=True,
                spacing=6,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.PHOTO_LIBRARY, size=18),
                    ft.Text("انتخاب از گالری", size=13),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=10, vertical=10),
            ),
            expand=True,
            on_click=self._handle_pick_evidence,
        )

        self.remove_evidence_btn = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_color=ft.Colors.RED_400,
            icon_size=22,
            tooltip="حذف عکس مستندات",
            visible=False,
            on_click=self._handle_remove_evidence,
        )

        self._evidence_icon = ft.Icon(
            ft.Icons.ADD_A_PHOTO,
            size=19,
            color=self._palette.accent_text,
        )
        self.evidence_card = ft.Container(
            padding=ft.Padding.all(12),
            border_radius=ft.BorderRadius.all(12),
            bgcolor=self._palette.card_bg,
            border=ft.Border.all(1, self._palette.border),
            content=ft.Column(
                spacing=10,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Row(
                                spacing=6,
                                controls=[
                                    self._evidence_icon,
                                    ft.Text(
                                        "تصویر مستندات تخلف",
                                        size=14,
                                        weight=ft.FontWeight.W_600,
                                    ),
                                ],
                            ),
                            self.remove_evidence_btn,
                        ],
                    ),
                    ft.Row(
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            self.evidence_preview_holder,
                            ft.Column(
                                expand=True,
                                spacing=6,
                                controls=[
                                    self.evidence_status_text,
                                    ft.Row(
                                        spacing=8,
                                        controls=[
                                            self.capture_evidence_btn,
                                            self.pick_evidence_btn,
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )

        self.submit_progress = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        self.submit_button = ft.FilledButton(
            content=ft.Row(
                tight=True,
                spacing=8,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    self.submit_progress,
                    ft.Icon(ft.Icons.SEND, size=18),
                    ft.Text("ثبت نهایی تخلف", weight=ft.FontWeight.BOLD, size=14),
                ],
            ),
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.RED_600,
                color=ft.Colors.WHITE,
                shape=ft.RoundedRectangleBorder(radius=12),
                padding=ft.Padding.symmetric(horizontal=16, vertical=14),
            ),
            expand=True,
            on_click=self._handle_submit_click,
        )

        self.cancel_button = ft.OutlinedButton(
            content=ft.Row(
                tight=True,
                spacing=6,
                controls=[
                    ft.Icon(ft.Icons.ARROW_FORWARD, size=18),
                    ft.Text("بازگشت", size=14),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12),
                padding=ft.Padding.symmetric(horizontal=16, vertical=14),
            ),
            on_click=self._handle_back_click,
        )

        self.status_banner = ft.Text("", size=12, color=ft.Colors.RED_400, visible=False)

        self._header_bar = ft.Container(
            padding=ft.Padding.only(bottom=10),
            border=ft.Border.only(bottom=ft.BorderSide(1, self._palette.border)),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Row(
                        spacing=8,
                        controls=[
                            ft.Icon(ft.Icons.REPORT_PROBLEM, color=ft.Colors.RED_500, size=24),
                            ft.Column(
                                spacing=2,
                                controls=[
                                    ft.Text(
                                        "ثبت تخلف خودرو",
                                        size=17,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    self.datetime_badge,
                                ],
                            ),
                        ],
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        tooltip="بازگشت به صفحه اصلی",
                        on_click=self._handle_back_click,
                    ),
                ],
            ),
        )

        self._plate_label = ft.Text(
            "شماره پلاک خودرو (قابل ویرایش مستقیم روی پلاک):",
            size=13,
            color=self._palette.text_secondary,
            weight=ft.FontWeight.W_600,
        )
        self._crop_card = ft.Container(
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            border_radius=ft.BorderRadius.all(10),
            bgcolor=self._palette.card_bg,
            border=ft.Border.all(1, self._palette.border),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    self.attach_crop_checkbox,
                    self.plate_crop_preview_holder,
                ],
            ),
        )

        # کارت موقعیت مکانی خودکار (GPS)
        self._gps_icon = ft.Icon(ft.Icons.LOCATION_ON, size=20, color=self._palette.accent_text)
        self._gps_spinner = ft.ProgressRing(width=16, height=16, stroke_width=2, visible=False)
        self.gps_status_text = ft.Text(
            "در حال دریافت موقعیت مکانی (GPS)...",
            size=12,
            color=self._palette.text_secondary,
            expand=True,
        )
        self.refresh_gps_btn = ft.IconButton(
            icon=ft.Icons.MY_LOCATION,
            icon_size=20,
            tooltip="دریافت مجدد موقعیت GPS",
            on_click=self._handle_refresh_gps_click,
        )
        self._gps_card = ft.Container(
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            border_radius=ft.BorderRadius.all(10),
            bgcolor=self._palette.card_bg,
            border=ft.Border.all(1, self._palette.border),
            content=ft.Row(
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    self._gps_icon,
                    self._gps_spinner,
                    self.gps_status_text,
                    self.refresh_gps_btn,
                ],
            ),
        )

        form_body = ft.Column(
            spacing=14,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                ft.Column(
                    spacing=6,
                    controls=[
                        self._plate_label,
                        self.plate_input,
                    ],
                ),
                self._gps_card,
                ft.Row(controls=[self.violation_dropdown]),
                ft.Row(controls=[self.notes_field]),
                self._crop_card,
                self.evidence_card,
                self.status_banner,
            ],
        )

        self._footer_actions = ft.Container(
            padding=ft.Padding.only(top=10),
            border=ft.Border.only(top=ft.BorderSide(1, self._palette.border)),
            content=ft.Row(
                spacing=10,
                controls=[
                    self.submit_button,
                    self.cancel_button,
                ],
            ),
        )

        self.content = ft.Column(
            spacing=12,
            expand=True,
            controls=[
                self._header_bar,
                form_body,
                self._footer_actions,
            ],
        )

    def apply_theme(self, dark_mode: bool) -> None:
        """رنگ‌های صفحه ثبت تخلف را با تم شب یا روز هماهنگ می‌کند."""
        self._palette = get_theme_palette(dark_mode)
        self.bgcolor = self._palette.page_bg
        self.datetime_badge.color = self._palette.accent_text
        self._evidence_icon.color = self._palette.accent_text
        self._gps_icon.color = self._palette.accent_text
        self._plate_label.color = self._palette.text_secondary
        self.evidence_card.bgcolor = self._palette.card_bg
        self.evidence_card.border = ft.Border.all(1, self._palette.border)
        self._crop_card.bgcolor = self._palette.card_bg
        self._crop_card.border = ft.Border.all(1, self._palette.border)
        self._gps_card.bgcolor = self._palette.card_bg
        self._gps_card.border = ft.Border.all(1, self._palette.border)
        self._header_bar.border = ft.Border.only(bottom=ft.BorderSide(1, self._palette.border))
        self._footer_actions.border = ft.Border.only(top=ft.BorderSide(1, self._palette.border))
        if not self._evidence_bytes:
            self.evidence_status_text.color = self._palette.text_muted
        if self._gps_lat is None or self._gps_lon is None:
            self.gps_status_text.color = self._palette.text_secondary
        else:
            self.gps_status_text.color = (
                ft.Colors.GREEN_300 if dark_mode else ft.Colors.GREEN_700
            )

    def set_gps_loading(self, message: str = "در حال دریافت موقعیت مکانی (GPS)...") -> None:
        self._gps_spinner.visible = True
        self.gps_status_text.value = message
        self.gps_status_text.color = self._palette.text_secondary

    def set_gps_location(
        self,
        latitude: float | None,
        longitude: float | None,
        accuracy_m: float | None = None,
        error_note: str = "",
    ) -> None:
        """مختصات دریافتی GPS را در کارت موقعیت مکانی ثبت و نمایش می‌دهد."""
        self._gps_spinner.visible = False
        self._gps_lat = latitude
        self._gps_lon = longitude
        self._gps_accuracy = accuracy_m

        if latitude is not None and longitude is not None:
            coords = f"{latitude:.6f}, {longitude:.6f}"
            if accuracy_m is not None and accuracy_m > 0:
                acc_fa = to_persian_digits(str(int(round(accuracy_m))))
                self.gps_status_text.value = f"موقعیت GPS ثبت شد: {coords} (دقت ±{acc_fa} متر)"
            else:
                self.gps_status_text.value = f"موقعیت GPS ثبت شد: {coords}"
            self.gps_status_text.color = (
                ft.Colors.GREEN_300 if self._palette.dark_mode else ft.Colors.GREEN_700
            )
            self._gps_icon.color = ft.Colors.GREEN_400 if self._palette.dark_mode else ft.Colors.GREEN_700
        else:
            self.gps_status_text.value = error_note or "موقعیت GPS در دسترس نیست (بدون GPS نیز قابل ثبت است)"
            self.gps_status_text.color = ft.Colors.AMBER_400 if self._palette.dark_mode else ft.Colors.ORANGE_800
            self._gps_icon.color = self._palette.accent_text

    async def _handle_refresh_gps_click(self, _e=None) -> None:
        if self._on_refresh_gps is None or self._submitting:
            return
        self.set_gps_loading("در حال بروزرسانی مختصات GPS...")
        self.update()
        await self._on_refresh_gps()

    def prepare(
        self,
        plate_display: str,
        has_crop: bool,
        plate_crop_bytes: bytes | None = None,
        chars_persian: list[str] | None = None,
        gps_coords: tuple[float, float, float | None] | None = None,
    ) -> None:
        """فرم را با اطلاعات آخرین پلاک، تاریخ شمسی و موقعیت GPS آماده و عکس مستندات قبلی را پاک می‌کند."""
        now_ir = now_iran_datetime()
        j_date = format_jalali_date(now_ir, persian_digits=True)
        j_time = format_jalali_time(now_ir, persian_digits=True)
        self.datetime_badge.value = f"تاریخ شمسی: {j_date} | ساعت: {j_time}"

        self.plate_input.set_plate(chars_persian=chars_persian, plate_text=plate_display)
        self.notes_field.value = ""
        self.attach_crop_checkbox.value = has_crop
        self.attach_crop_checkbox.disabled = not has_crop

        if gps_coords is not None:
            self.set_gps_location(gps_coords[0], gps_coords[1], gps_coords[2])
        else:
            self._gps_lat = None
            self._gps_lon = None
            self._gps_accuracy = None
            self.set_gps_loading()

        if has_crop and plate_crop_bytes:
            self.plate_crop_preview_holder.content = ft.Image(
                src=plate_crop_bytes,
                width=104,
                height=38,
                fit=ft.BoxFit.CONTAIN,
                border_radius=ft.BorderRadius.all(6),
            )
            self.plate_crop_preview_holder.visible = True
        else:
            self.plate_crop_preview_holder.content = None
            self.plate_crop_preview_holder.visible = False

        self.set_evidence_bytes(None)
        self.status_banner.visible = False
        self.set_busy(False)

    def set_evidence_bytes(self, image_bytes: bytes | None) -> None:
        """تصویر مستندات انتخاب یا ثبت‌شده را در فرم به‌روزرسانی می‌کند."""
        self._evidence_bytes = image_bytes
        if image_bytes:
            kb = max(1, int(round(len(image_bytes) / 1024)))
            self.evidence_status_text.value = (
                f"عکس مستندات پیوست شد ({to_persian_digits(str(kb))} کیلوبایت)"
            )
            self.evidence_status_text.color = (
                ft.Colors.GREEN_300 if self._palette.dark_mode else ft.Colors.GREEN_700
            )
            self.evidence_preview_holder.content = ft.Image(
                src=image_bytes,
                width=96,
                height=72,
                fit=ft.BoxFit.COVER,
                border_radius=ft.BorderRadius.all(8),
            )
            self.evidence_preview_holder.visible = True
            self.remove_evidence_btn.visible = True
        else:
            self.evidence_status_text.value = "عکسی انتخاب نشده (اختیاری)"
            self.evidence_status_text.color = self._palette.text_muted
            self.evidence_preview_holder.content = None
            self.evidence_preview_holder.visible = False
            self.remove_evidence_btn.visible = False

    def set_busy(self, busy: bool) -> None:
        self._submitting = busy
        self.submit_progress.visible = busy
        self.submit_button.disabled = busy
        self.cancel_button.disabled = busy
        self.capture_evidence_btn.disabled = busy or self._capturing_evidence
        self.pick_evidence_btn.disabled = busy or self._capturing_evidence

    def _set_capturing(self, capturing: bool) -> None:
        self._capturing_evidence = capturing
        self.capture_evidence_btn.disabled = capturing or self._submitting
        self.pick_evidence_btn.disabled = capturing or self._submitting

    def _handle_back_click(self, _e=None) -> None:
        if not self._submitting and not self._capturing_evidence:
            self._on_back()

    def _handle_remove_evidence(self, _e=None) -> None:
        if self._submitting or self._capturing_evidence:
            return
        self.set_evidence_bytes(None)
        self.update()

    async def _handle_capture_evidence(self, _e=None) -> None:
        if self._submitting or self._capturing_evidence:
            return
        self._set_capturing(True)
        self.evidence_status_text.value = "در حال باز کردن نمای دوربین برای ثبت مستندات..."
        self.evidence_status_text.color = ft.Colors.AMBER_300
        self.update()
        try:
            captured = await self._on_capture_evidence()
            if captured:
                self.set_evidence_bytes(captured)
            else:
                self.set_evidence_bytes(self._evidence_bytes)
        finally:
            self._set_capturing(False)
            self.update()

    async def _handle_pick_evidence(self, _e=None) -> None:
        if self._submitting or self._capturing_evidence:
            return
        self._set_capturing(True)
        self.evidence_status_text.value = "در حال انتخاب تصویر از گالری..."
        self.evidence_status_text.color = ft.Colors.AMBER_300
        self.update()
        try:
            picked = await self._on_pick_evidence()
            if picked:
                self.set_evidence_bytes(picked)
            else:
                self.set_evidence_bytes(self._evidence_bytes)
        finally:
            self._set_capturing(False)
            self.update()

    async def _handle_submit_click(self, _e=None) -> None:
        if self._submitting or self._capturing_evidence:
            return
        plate_display, plate_latin, _ = self.plate_input.get_plate_values()
        if not self.plate_input.is_complete():
            self.status_banner.value = (
                "لطفاً تمامی بخش‌های پلاک ایران (۲ رقم، حرف، ۳ رقم و کد ۲ رقمی ایران) را کامل وارد کنید."
            )
            self.status_banner.visible = True
            self.update()
            return

        self.status_banner.visible = False
        self.set_busy(True)
        self.update()
        try:
            await self._on_submit(
                plate_display,
                plate_latin,
                self.violation_dropdown.value or VIOLATION_TYPES[0],
                self.notes_field.value or "",
                bool(self.attach_crop_checkbox.value),
                self._evidence_bytes,
                self._gps_lat,
                self._gps_lon,
                self._gps_accuracy,
            )
        finally:
            self.set_busy(False)
            self.update()
