"""بخش بازبینی تخلفات ثبت‌شده (نمایش لیست کامل تخلفات با امکان مشاهده تصاویر پلاک و مستندات)."""

from __future__ import annotations

from typing import Awaitable, Callable

import flet as ft

from config import to_persian_digits
from services.api_client import ViolationRecord


class ViolationHistoryPanel(ft.Container):
    """پنل نمایش و بازبینی تخلفات ثبت‌شده در بخش پایینی صفحه اصلی."""

    def __init__(
        self,
        on_back: Callable[[], None],
        on_refresh: Callable[[], Awaitable[None]],
        on_select_record: Callable[[ViolationRecord], Awaitable[None]],
    ) -> None:
        super().__init__(expand=True, padding=ft.Padding.all(12))
        self._on_back = on_back
        self._on_refresh = on_refresh
        self._on_select_record = on_select_record

        self.title_text = ft.Text(
            "بازبینی تخلفات ثبت‌شده",
            size=15,
            weight=ft.FontWeight.BOLD,
        )
        self.count_badge = ft.Container(
            padding=ft.Padding.symmetric(horizontal=8, vertical=2),
            border_radius=ft.BorderRadius.all(12),
            bgcolor=ft.Colors.BLUE_800,
            content=ft.Text("۰", size=11, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        )

        self.records_column = ft.Column(
            spacing=8,
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
                    ft.Text("بازگشت به صفحه اصلی"),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=14, vertical=10),
            ),
            width=float("inf"),
            on_click=lambda _: self._on_back(),
        )

        self.content = ft.Column(
            spacing=10,
            expand=True,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Row(
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[
                                ft.Icon(ft.Icons.HISTORY, color=ft.Colors.LIGHT_BLUE_300, size=22),
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
                ft.Container(
                    expand=True,
                    content=self.records_column,
                ),
                self.back_button,
            ],
        )

    async def _handle_refresh(self, _e=None) -> None:
        await self._on_refresh()

    def set_records(self, records: list[ViolationRecord]) -> None:
        """فهرست کارت‌های تخلفات ثبت‌شده را بروزرسانی می‌کند."""
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
                                color=ft.Colors.BLUE_GREY_400,
                            ),
                            ft.Text(
                                "هنوز هیچ تخلفی ثبت نشده است.",
                                size=13,
                                color=ft.Colors.WHITE_70,
                                weight=ft.FontWeight.W_500,
                            ),
                            ft.Text(
                                "پس از ثبت تخلف، گزارش‌ها به همراه تصاویر پلاک و مستندات اینجا نمایش داده می‌شوند.",
                                size=11,
                                color=ft.Colors.WHITE_60,
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
        sync_label = "ارسال‌شده به سرور" if rec.synced else "ذخیره محلی"
        sync_color = ft.Colors.GREEN_700 if rec.synced else ft.Colors.ORANGE_800
        sync_icon = ft.Icons.CLOUD_DONE if rec.synced else ft.Icons.CLOUD_OFF

        media_chips: list[ft.Control] = []
        if rec.has_image:
            media_chips.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    border_radius=ft.BorderRadius.all(6),
                    bgcolor=ft.Colors.BLUE_GREY_800,
                    content=ft.Row(
                        tight=True,
                        spacing=4,
                        controls=[
                            ft.Icon(ft.Icons.CROP_ORIGINAL, size=12, color=ft.Colors.LIGHT_BLUE_200),
                            ft.Text("تصویر پلاک", size=10, color=ft.Colors.WHITE_70),
                        ],
                    ),
                )
            )
        if rec.has_evidence_image:
            media_chips.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    border_radius=ft.BorderRadius.all(6),
                    bgcolor=ft.Colors.TEAL_900,
                    content=ft.Row(
                        tight=True,
                        spacing=4,
                        controls=[
                            ft.Icon(ft.Icons.IMAGE, size=12, color=ft.Colors.TEAL_200),
                            ft.Text("عکس مستندات", size=10, color=ft.Colors.WHITE_70),
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
                            ft.Icon(ft.Icons.DIRECTIONS_CAR, size=16, color=ft.Colors.LIGHT_BLUE_300),
                            ft.Text(
                                rec.plate,
                                size=14,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE,
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
                                ft.Text(sync_label, size=10, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
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
                        size=12,
                        color=ft.Colors.AMBER_300,
                        weight=ft.FontWeight.W_600,
                    ),
                    ft.Text(
                        to_persian_digits(rec.local_time or "---"),
                        size=10,
                        color=ft.Colors.WHITE_60,
                    ),
                ],
            ),
        ]

        if rec.description:
            snippet = rec.description if len(rec.description) <= 75 else f"{rec.description[:72]}..."
            body_controls.append(
                ft.Text(snippet, size=11, color=ft.Colors.WHITE_70)
            )

        footer_row = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Row(spacing=6, controls=media_chips),
                ft.Row(
                    tight=True,
                    spacing=2,
                    controls=[
                        ft.Text("مشاهده جزئیات", size=11, color=ft.Colors.LIGHT_BLUE_300),
                        ft.Icon(ft.Icons.CHEVRON_LEFT, size=16, color=ft.Colors.LIGHT_BLUE_300),
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
            bgcolor="#161B22",
            border=ft.Border.all(1, ft.Colors.BLUE_GREY_800),
            ink=True,
            on_click=_on_card_click,
            content=ft.Column(spacing=5, controls=body_controls),
        )
