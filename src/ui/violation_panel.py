"""بخش پایینی صفحه برای ثبت تخلف خودرو (با فشردن دکمه «ثبت تخلف» جایگزین بخش اطلاعات سرور می‌شود)."""

from __future__ import annotations

from typing import Awaitable, Callable

import flet as ft

from config import VIOLATION_TYPES


class ViolationRegistrationPanel(ft.Container):
    """فرم ثبت تخلف در نیمه پایینی صفحه اصلی."""

    def __init__(
        self,
        on_back: Callable[[], None],
        on_submit: Callable[[str, str, str, bool], Awaitable[None]],
    ) -> None:
        super().__init__(expand=True, padding=ft.Padding.all(12))
        self._on_back = on_back
        self._on_submit = on_submit
        self._submitting = False

        self.plate_field = ft.TextField(
            label="شماره پلاک خودرو",
            hint_text="مثال: ۱۲ ب ۳۴۵ | ایران ۶۷",
            text_size=15,
            dense=True,
            prefix_icon=ft.Icons.DIRECTIONS_CAR,
        )

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
            hint_text="محل تخلف یا یادداشت مأمور...",
            multiline=True,
            min_lines=2,
            max_lines=3,
            dense=True,
        )

        self.attach_crop_checkbox = ft.Checkbox(
            label="پیوست تصویر برش‌خورده پلاک در گزارش تخلف",
            value=True,
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
                    ft.Text("ثبت نهایی تخلف", weight=ft.FontWeight.BOLD),
                ],
            ),
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.RED_600,
                color=ft.Colors.WHITE,
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=16, vertical=12),
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
                    ft.Text("بازگشت"),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=14, vertical=12),
            ),
            on_click=self._handle_back_click,
        )

        self.status_banner = ft.Text("", size=12, color=ft.Colors.RED_400, visible=False)

        self.content = ft.Column(
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Row(
                            spacing=8,
                            controls=[
                                ft.Icon(ft.Icons.REPORT_PROBLEM, color=ft.Colors.RED_500, size=22),
                                ft.Text(
                                    "ثبت تخلف خودرو",
                                    size=16,
                                    weight=ft.FontWeight.BOLD,
                                ),
                            ],
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            tooltip="بازگشت به اطلاعات سرور",
                            on_click=self._handle_back_click,
                        ),
                    ],
                ),
                self.plate_field,
                ft.Row(controls=[self.violation_dropdown]),
                self.notes_field,
                self.attach_crop_checkbox,
                self.status_banner,
                ft.Row(
                    spacing=10,
                    controls=[
                        self.submit_button,
                        self.cancel_button,
                    ],
                ),
            ],
        )

    def prepare(self, plate_display: str, has_crop: bool) -> None:
        """فرم را با اطلاعات آخرین پلاک تشخیص‌داده‌شده آماده می‌کند."""
        self.plate_field.value = plate_display
        self.notes_field.value = ""
        self.attach_crop_checkbox.value = has_crop
        self.attach_crop_checkbox.disabled = not has_crop
        self.status_banner.visible = False
        self.set_busy(False)

    def set_busy(self, busy: bool) -> None:
        self._submitting = busy
        self.submit_progress.visible = busy
        self.submit_button.disabled = busy
        self.cancel_button.disabled = busy

    def _handle_back_click(self, _e=None) -> None:
        if not self._submitting:
            self._on_back()

    async def _handle_submit_click(self, _e=None) -> None:
        if self._submitting:
            return
        plate_value = (self.plate_field.value or "").strip()
        if not plate_value:
            self.status_banner.value = "لطفاً شماره پلاک خودرو را وارد یا اسکن کنید."
            self.status_banner.visible = True
            self.update()
            return

        self.status_banner.visible = False
        self.set_busy(True)
        self.update()
        try:
            await self._on_submit(
                plate_value,
                self.violation_dropdown.value or VIOLATION_TYPES[0],
                self.notes_field.value or "",
                bool(self.attach_crop_checkbox.value),
            )
        finally:
            self.set_busy(False)
            self.update()
