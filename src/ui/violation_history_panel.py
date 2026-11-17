"""صفحه تمام‌صفحه بازبینی تخلفات ثبت‌شده (نمایش لیست کامل تخلفات با تاریخ شمسی، تصاویر و خروجی PDF)."""

from __future__ import annotations

from typing import Awaitable, Callable

import flet as ft

from config import to_persian_digits
from services.api_client import ViolationRecord
from ui.theme import get_theme_palette


class ViolationHistoryPanel(ft.Container):
    """صفحه تمام‌صفحه نمایش و بازبینی تخلفات ثبت‌شده با پشتیبانی از تم روز و شب."""

    def __init__(
        self,
        on_back: Callable[[], None],
        on_refresh: Callable[[], Awaitable[None]],
        on_select_record: Callable[[ViolationRecord], Awaitable[None]],
        dark_mode: bool = True,
    ) -> None:
        self._palette = get_theme_palette(dark_mode)
        self._last_records: list[ViolationRecord] = []
        super().__init__(
            expand=True,
            bgcolor=self._palette.page_bg,
            padding=ft.Padding.all(14),
        )
        self._on_back = on_back
        self._on_refresh = on_refresh
        self._on_select_record = on_select_record

        self.title_text = ft.Text(
            "بازبینی تخلفات ثبت‌شده",
            size=17,
            weight=ft.FontWeight.BOLD,
        )
        self.count_badge = ft.Container(
            padding=ft.Padding.symmetric(horizontal=9, vertical=3),
            border_radius=ft.BorderRadius.all(12),
            bgcolor=self._palette.accent,
            content=ft.Text("۰", size=12, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        )

        self.records_column = ft.Column(
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[],
        )

        self.back_button = ft.OutlinedButton(
            content=ft.Row(
                tight=True,
                spacing=6,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.ARROW_FORWARD, size=18),
                    ft.Text("بازگشت به صفحه اصلی", size=14),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12),
                padding=ft.Padding.symmetric(horizontal=16, vertical=13),
            ),
            width=float("inf"),
            on_click=lambda _: self._on_back(),
        )

        self._history_icon = ft.Icon(ft.Icons.HISTORY, color=self._palette.accent_text, size=24)
        self._header_bar = ft.Container(
            padding=ft.Padding.only(bottom=10),
            border=ft.Border.only(bottom=ft.BorderSide(1, self._palette.border)),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Row(
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            self._history_icon,
                            self.title_text,
                            self.count_badge,
                        ],
                    ),
                    ft.Row(
                        spacing=0,
                        controls=[
                            ft.IconButton(
                                icon=ft.Icons.REFRESH,
                                tooltip="بروزرسانی فهرست",
                                on_click=self._handle_refresh,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.CLOSE,
                                tooltip="بازگشت",
                                on_click=lambda _: self._on_back(),
                            ),
                        ],
                    ),
                ],
            ),
        )

        self._footer_bar = ft.Container(
            padding=ft.Padding.only(top=10),
            border=ft.Border.only(top=ft.BorderSide(1, self._palette.border)),
            content=self.back_button,
        )

        self.content = ft.Column(
            spacing=12,
            expand=True,
            controls=[
                self._header_bar,
                ft.Container(
                    expand=True,
                    content=self.records_column,
                ),
                self._footer_bar,
            ],
        )

    def apply_theme(self, dark_mode: bool) -> None:
        """رنگ‌های صفحه بازبینی تخلفات و کارت‌ها را با تم شب یا روز هماهنگ می‌کند."""
        self._palette = get_theme_palette(dark_mode)
        self.bgcolor = self._palette.page_bg
        self._history_icon.color = self._palette.accent_text
        self._header_bar.border = ft.Border.only(bottom=ft.BorderSide(1, self._palette.border))
        self._footer_bar.border = ft.Border.only(top=ft.BorderSide(1, self._palette.border))
        self.set_records(self._last_records)

    async def _handle_refresh(self, _e=None) -> None:
        await self._on_refresh()

    def set_records(self, records: list[ViolationRecord]) -> None:
        """فهرست کارت‌های تخلفات ثبت‌شده را بروزرسانی می‌کند."""
        self._last_records = list(records)
        p = self._palette
        self.count_badge.content = ft.Text(
            to_persian_digits(str(len(records))),
            size=11,
            color=ft.Colors.WHITE,
            weight=ft.FontWeight.BOLD,
        )

        if not records:
            self.records_column.controls = [
                ft.Container(
                    expand=True,
                    alignment=ft.Alignment.CENTER,
                    padding=ft.Padding.all(24),
                    content=ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                        controls=[
                            ft.Icon(
                                ft.Icons.INVENTORY_2_OUTLINED,
                                size=38,
                                color=p.text_muted,
                            ),
                            ft.Text(
                                "هنوز هیچ تخلفی ثبت نشده است.",
                                size=13,
                                color=p.text_secondary,
                                weight=ft.FontWeight.W_500,
                            ),
                            ft.Text(
                                "پس از ثبت تخلف، گزارش‌ها به همراه تصاویر پلاک و مستندات اینجا نمایش داده می‌شوند.",
                                size=11,
                                color=p.text_muted,
                                text_align=ft.TextAlign.CENTER,
                            ),
                        ],
                    ),
                )
            ]
            return

        cards: list[ft.Control] = []
        for rec in records:
            cards.append(self._build_record_card(rec))
        self.records_column.controls = cards

    def _build_record_card(self, rec: ViolationRecord) -> ft.Container:
        p = self._palette
        sync_label = "ارسال‌شده به سرور" if rec.synced else "ذخیره محلی"
        sync_color = ft.Colors.GREEN_700 if rec.synced else ft.Colors.ORANGE_800
        sync_icon = ft.Icons.CLOUD_DONE if rec.synced else ft.Icons.CLOUD_OFF

        media_chips: list[ft.Control] = []
        if rec.has_image:
            media_chips.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    border_radius=ft.BorderRadius.all(6),
                    bgcolor=p.plate_chip_bg,
                    content=ft.Row(
                        tight=True,
                        spacing=4,
                        controls=[
                            ft.Icon(ft.Icons.CROP_ORIGINAL, size=12, color=p.accent_text),
                            ft.Text("تصویر پلاک", size=10, color=p.chip_text),
                        ],
                    ),
                )
            )
        if rec.has_evidence_image:
            media_chips.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    border_radius=ft.BorderRadius.all(6),
                    bgcolor=p.evidence_chip_bg,
                    content=ft.Row(
                        tight=True,
                        spacing=4,
                        controls=[
                            ft.Icon(
                                ft.Icons.IMAGE,
                                size=12,
                                color=ft.Colors.TEAL_200 if p.dark_mode else ft.Colors.TEAL_800,
                            ),
                            ft.Text("عکس مستندات", size=10, color=p.chip_text),
                        ],
                    ),
                )
            )

        body_controls: list[ft.Control] = [
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Row(
                        spacing=6,
                        controls=[
                            ft.Icon(ft.Icons.DIRECTIONS_CAR, size=16, color=p.accent_text),
                            ft.Text(
                                rec.plate,
                                size=14,
                                weight=ft.FontWeight.BOLD,
                                color=p.text_primary,
                            ),
                        ],
                    ),
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=7, vertical=3),
                        border_radius=ft.BorderRadius.all(8),
                        bgcolor=sync_color,
                        content=ft.Row(
                            tight=True,
                            spacing=4,
                            controls=[
                                ft.Icon(sync_icon, size=12, color=ft.Colors.WHITE),
                                ft.Text(
                                    sync_label,
                                    size=10,
                                    color=ft.Colors.WHITE,
                                    weight=ft.FontWeight.BOLD,
                                ),
                            ],
                        ),
                    ),
                ],
            ),
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text(
                        rec.violation_type,
                        size=13,
                        color=p.violation_text,
                        weight=ft.FontWeight.W_600,
                    ),
                    ft.Row(
                        tight=True,
                        spacing=4,
                        controls=[
                            ft.Icon(ft.Icons.CALENDAR_TODAY, size=12, color=p.accent_text),
                            ft.Text(
                                rec.display_jalali_datetime or "---",
                                size=11,
                                color=p.text_secondary,
                            ),
                        ],
                    ),
                ],
            ),
        ]

        if rec.description:
            snippet = rec.description if len(rec.description) <= 75 else f"{rec.description[:72]}..."
            body_controls.append(ft.Text(snippet, size=11, color=p.text_secondary))

        footer_row = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Row(spacing=6, controls=media_chips),
                ft.Row(
                    tight=True,
                    spacing=2,
                    controls=[
                        ft.Text("مشاهده جزئیات", size=11, color=p.accent_text),
                        ft.Icon(ft.Icons.CHEVRON_LEFT, size=16, color=p.accent_text),
                    ],
                ),
            ],
        )
        body_controls.append(footer_row)

        async def _on_card_click(_e, record=rec) -> None:
            await self._on_select_record(record)

        return ft.Container(
            padding=ft.Padding.all(10),
            border_radius=ft.BorderRadius.all(10),
            bgcolor=p.card_bg,
            border=ft.Border.all(1, p.border),
            ink=True,
            on_click=_on_card_click,
            content=ft.Column(spacing=5, controls=body_controls),
        )
