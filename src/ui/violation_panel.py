"""بخش پایینی صفحه برای ثبت تخلف خودرو (با فشردن دکمه «ثبت تخلف» جایگزین بخش اطلاعات سرور می‌شود).

امکانات:
- شماره پلاک (پُر شده به‌صورت خودکار از آخرین تشخیص + قابل ویرایش)
- انتخاب نوع تخلف و توضیحات تکمیلی
- پیوست تصویر برش‌خورده پلاک (با پیش‌نمایش)
- افزودن عکس مستندات تخلف (از طریق عکس‌برداری مستقیم با دوربین یا انتخاب از گالری گوشی)
"""

from __future__ import annotations

from typing import Awaitable, Callable

import flet as ft

from config import VIOLATION_TYPES, to_persian_digits


class ViolationRegistrationPanel(ft.Container):
    """فرم ثبت تخلف در نیمه پایینی صفحه اصلی."""

    def __init__(
        self,
        on_back: Callable[[], None],
        on_submit: Callable[[str, str, str, bool, bytes | None], Awaitable[None]],
        on_capture_evidence: Callable[[], Awaitable[bytes | None]],
        on_pick_evidence: Callable[[], Awaitable[bytes | None]],
    ) -> None:
        super().__init__(expand=True, padding=ft.Padding.all(12))
        self._on_back = on_back
        self._on_submit = on_submit
        self._on_capture_evidence = on_capture_evidence
        self._on_pick_evidence = on_pick_evidence
        self._submitting = False
        self._capturing_evidence = False
        self._evidence_bytes: bytes | None = None

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

        # بخش پیوست تصویر برش‌خورده پلاک
        self.attach_crop_checkbox = ft.Checkbox(
            label="پیوست تصویر برش‌خورده پلاک",
            value=True,
        )
        self.plate_crop_preview_holder = ft.Container(visible=False)

        # بخش افزودن عکس مستندات تخلف (دوربین زنده یا گالری)
        self.evidence_status_text = ft.Text(
            "عکسی انتخاب نشده (اختیاری)",
            size=11,
            color=ft.Colors.WHITE_60,
        )
        self.evidence_preview_holder = ft.Container(visible=False)

        self.capture_evidence_btn = ft.OutlinedButton(
            content=ft.Row(
                tight=True,
                spacing=6,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.CAMERA_ALT, size=16),
                    ft.Text("عکس با دوربین", size=12),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=ft.Padding.symmetric(horizontal=10, vertical=8),
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
                    ft.Icon(ft.Icons.PHOTO_LIBRARY, size=16),
                    ft.Text("انتخاب از گالری", size=12),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            ),
            expand=True,
            on_click=self._handle_pick_evidence,
        )

        self.remove_evidence_btn = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_color=ft.Colors.RED_400,
            icon_size=20,
            tooltip="حذف عکس مستندات",
            visible=False,
            on_click=self._handle_remove_evidence,
        )

        self.evidence_card = ft.Container(
            padding=ft.Padding.all(10),
            border_radius=ft.BorderRadius.all(10),
            bgcolor="#161B22",
            border=ft.Border.all(1, ft.Colors.BLUE_GREY_800),
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Row(
                                spacing=6,
                                controls=[
                                    ft.Icon(
                                        ft.Icons.ADD_A_PHOTO,
                                        size=17,
                                        color=ft.Colors.LIGHT_BLUE_300,
                                    ),
                                    ft.Text(
                                        "تصویر مستندات تخلف",
                                        size=13,
                                        weight=ft.FontWeight.W_600,
                                    ),
                                ],
                            ),
                            self.remove_evidence_btn,
                        ],
                    ),
                    ft.Row(
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            self.evidence_preview_holder,
                            ft.Column(
                                expand=True,
                                spacing=4,
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
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        self.attach_crop_checkbox,
                        self.plate_crop_preview_holder,
                    ],
                ),
                self.evidence_card,
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

    def prepare(
        self,
        plate_display: str,
        has_crop: bool,
        plate_crop_bytes: bytes | None = None,
    ) -> None:
        """فرم را با اطلاعات آخرین پلاک تشخیص‌داده‌شده آماده و عکس مستندات قبلی را پاک می‌کند."""
        self.plate_field.value = plate_display
        self.notes_field.value = ""
        self.attach_crop_checkbox.value = has_crop
        self.attach_crop_checkbox.disabled = not has_crop

        if has_crop and plate_crop_bytes:
            self.plate_crop_preview_holder.content = ft.Image(
                src=plate_crop_bytes,
                width=92,
                height=34,
                fit=ft.BoxFit.CONTAIN,
                border_radius=ft.BorderRadius.all(6),
            )
            self.plate_crop_preview_holder.visible = True
        else:
            self.plate_crop_preview_holder.content = None
            self.plate_crop_preview_holder.visible = False

        self._set_evidence_bytes(None)
        self.status_banner.visible = False
        self.set_busy(False)

    def _set_evidence_bytes(self, image_bytes: bytes | None) -> None:
        self._evidence_bytes = image_bytes
        if image_bytes:
            kb = max(1, int(round(len(image_bytes) / 1024)))
            self.evidence_status_text.value = (
                f"عکس مستندات پیوست شد ({to_persian_digits(str(kb))} کیلوبایت)"
            )
            self.evidence_status_text.color = ft.Colors.GREEN_300
            self.evidence_preview_holder.content = ft.Image(
                src=image_bytes,
                width=84,
                height=62,
                fit=ft.BoxFit.COVER,
                border_radius=ft.BorderRadius.all(8),
            )
            self.evidence_preview_holder.visible = True
            self.remove_evidence_btn.visible = True
        else:
            self.evidence_status_text.value = "عکسی انتخاب نشده (اختیاری)"
            self.evidence_status_text.color = ft.Colors.WHITE_60
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
        self._set_evidence_bytes(None)
        self.update()

    async def _handle_capture_evidence(self, _e=None) -> None:
        if self._submitting or self._capturing_evidence:
            return
        self._set_capturing(True)
        self.evidence_status_text.value = "در حال ثبت تصویر از دوربین..."
        self.evidence_status_text.color = ft.Colors.AMBER_300
        self.update()
        try:
            captured = await self._on_capture_evidence()
            if captured:
                self._set_evidence_bytes(captured)
            else:
                self._set_evidence_bytes(self._evidence_bytes)
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
                self._set_evidence_bytes(picked)
            else:
                self._set_evidence_bytes(self._evidence_bytes)
        finally:
            self._set_capturing(False)
            self.update()

    async def _handle_submit_click(self, _e=None) -> None:
        if self._submitting or self._capturing_evidence:
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
                self._evidence_bytes,
            )
        finally:
            self.set_busy(False)
            self.update()
