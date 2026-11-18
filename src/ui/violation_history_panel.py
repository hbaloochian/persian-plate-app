"""صفحه تمام‌صفحه بازبینی تخلفات ثبت‌شده با جستجو، فیلتر پیشرفته، ارسال یکجا و خروجی گروهی PDF/CSV."""

from __future__ import annotations

from typing import Awaitable, Callable

import flet as ft

from config import VIOLATION_TYPES, to_persian_digits
from services.api_client import ViolationRecord, filter_violation_records
from ui.theme import get_theme_palette


class ViolationHistoryPanel(ft.Container):
    """صفحه تمام‌صفحه نمایش، جستجو، فیلتر و خروجی گروهی تخلفات ثبت‌شده."""

    def __init__(
        self,
        on_back: Callable[[], None],
        on_refresh: Callable[[], Awaitable[None]],
        on_select_record: Callable[[ViolationRecord], Awaitable[None]],
        on_sync_all: Callable[[list[ViolationRecord]], Awaitable[None]] | None = None,
        on_export_batch_pdf: Callable[[list[ViolationRecord], str], Awaitable[None]] | None = None,
        on_export_batch_csv: Callable[[list[ViolationRecord]], Awaitable[None]] | None = None,
        dark_mode: bool = True,
    ) -> None:
        self._palette = get_theme_palette(dark_mode)
        self._all_records: list[ViolationRecord] = []
        self._filtered_records: list[ViolationRecord] = []
        self._sync_filter: str = "all"
        self._violation_type_filter: str = "همه"
        self._search_query: str = ""
        self._busy: bool = False

        super().__init__(
            expand=True,
            bgcolor=self._palette.page_bg,
            padding=ft.Padding.all(14),
        )
        self._on_back = on_back
        self._on_refresh = on_refresh
        self._on_select_record = on_select_record
        self._on_sync_all = on_sync_all
        self._on_export_batch_pdf = on_export_batch_pdf
        self._on_export_batch_csv = on_export_batch_csv

        self.title_text = ft.Text(
            "بازبینی تخلفات ثبت‌شده",
            size=16,
            weight=ft.FontWeight.BOLD,
        )
        self.count_badge = ft.Container(
            padding=ft.Padding.symmetric(horizontal=9, vertical=3),
            border_radius=ft.BorderRadius.all(12),
            bgcolor=self._palette.accent,
            content=ft.Text("۰", size=12, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        )

        # -------------------------------------------------------------------
        # نوار جستجو و فیلتر پیشرفته
        # -------------------------------------------------------------------
        self.clear_search_btn = ft.IconButton(
            icon=ft.Icons.CLEAR,
            icon_size=18,
            tooltip="پاک کردن جستجو",
            visible=False,
            on_click=self._handle_clear_search,
        )
        self.search_field = ft.TextField(
            hint_text="جستجو بر اساس پلاک، نوع تخلف، تاریخ یا توضیحات...",
            prefix_icon=ft.Icons.SEARCH,
            suffix=self.clear_search_btn,
            dense=True,
            content_padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            border_radius=ft.BorderRadius.all(10),
            on_change=self._handle_search_change,
        )

        self.filter_chips_row = ft.Row(
            spacing=6,
            scroll=ft.ScrollMode.AUTO,
            controls=[],
        )

        self.violation_type_dropdown = ft.Dropdown(
            label="فیلتر نوع تخلف",
            value="همه",
            dense=True,
            expand=True,
            options=[
                ft.DropdownOption(key="همه", text="همه انواع تخلف"),
                *[ft.DropdownOption(key=item, text=item) for item in VIOLATION_TYPES],
            ],
            on_select=self._handle_type_filter_change,
        )

        self.filter_panel = ft.Container(
            padding=ft.Padding.all(10),
            border_radius=ft.BorderRadius.all(10),
            bgcolor=self._palette.card_bg,
            border=ft.Border.all(1, self._palette.border),
            content=ft.Column(
                spacing=8,
                controls=[
                    self.search_field,
                    self.filter_chips_row,
                    ft.Row(controls=[self.violation_type_dropdown]),
                ],
            ),
        )

        # -------------------------------------------------------------------
        # نوار عملیات گروهی (ارسال یکجا + PDF گروهی + اکسل CSV)
        # -------------------------------------------------------------------
        self.batch_progress = ft.ProgressRing(width=15, height=15, stroke_width=2, visible=False)
        self.batch_status_text = ft.Text("", size=11, color=self._palette.accent_text, visible=False)

        self.sync_all_label = ft.Text("ارسال یکجا", size=11, weight=ft.FontWeight.BOLD)
        self.sync_all_btn = ft.FilledButton(
            content=ft.Row(
                tight=True,
                spacing=4,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.CLOUD_UPLOAD, size=15),
                    self.sync_all_label,
                ],
            ),
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.ORANGE_700,
                color=ft.Colors.WHITE,
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=8, vertical=9),
            ),
            expand=True,
            on_click=self._handle_sync_all_click,
        )

        self.batch_pdf_btn = ft.FilledButton(
            content=ft.Row(
                tight=True,
                spacing=4,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.PICTURE_AS_PDF, size=15),
                    ft.Text("PDF گروهی", size=11, weight=ft.FontWeight.BOLD),
                ],
            ),
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.BLUE_700,
                color=ft.Colors.WHITE,
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=8, vertical=9),
            ),
            expand=True,
            on_click=self._handle_batch_pdf_click,
        )

        self.batch_csv_btn = ft.OutlinedButton(
            content=ft.Row(
                tight=True,
                spacing=4,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.TABLE_CHART, size=15),
                    ft.Text("اکسل CSV", size=11, weight=ft.FontWeight.BOLD),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=8, vertical=9),
            ),
            expand=True,
            on_click=self._handle_batch_csv_click,
        )

        self.batch_bar = ft.Column(
            spacing=6,
            controls=[
                ft.Row(
                    spacing=6,
                    controls=[
                        self.sync_all_btn,
                        self.batch_pdf_btn,
                        self.batch_csv_btn,
                    ],
                ),
                ft.Row(
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[self.batch_progress, self.batch_status_text],
                ),
            ],
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
                padding=ft.Padding.symmetric(horizontal=16, vertical=12),
            ),
            width=float("inf"),
            on_click=lambda _: self._on_back(),
        )

        self._history_icon = ft.Icon(ft.Icons.HISTORY, color=self._palette.accent_text, size=22)
        self._header_bar = ft.Container(
            padding=ft.Padding.only(bottom=8),
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
            padding=ft.Padding.only(top=8),
            border=ft.Border.only(top=ft.BorderSide(1, self._palette.border)),
            content=self.back_button,
        )

        self.content = ft.Column(
            spacing=10,
            expand=True,
            controls=[
                self._header_bar,
                self.filter_panel,
                self.batch_bar,
                ft.Container(
                    expand=True,
                    content=self.records_column,
                ),
                self._footer_bar,
            ],
        )

    def apply_theme(self, dark_mode: bool) -> None:
        """رنگ‌های صفحه بازبینی تخلفات، فیلترها و کارت‌ها را با تم شب یا روز هماهنگ می‌کند."""
        self._palette = get_theme_palette(dark_mode)
        self.bgcolor = self._palette.page_bg
        self._history_icon.color = self._palette.accent_text
        self.batch_status_text.color = self._palette.accent_text
        self.filter_panel.bgcolor = self._palette.card_bg
        self.filter_panel.border = ft.Border.all(1, self._palette.border)
        self._header_bar.border = ft.Border.only(bottom=ft.BorderSide(1, self._palette.border))
        self._footer_bar.border = ft.Border.only(top=ft.BorderSide(1, self._palette.border))
        self._apply_filters_and_render()

    def set_batch_busy(self, busy: bool, status_message: str = "") -> None:
        """وضعیت در حال اجرا برای عملیات ارسال یکجا یا خروجی گروهی."""
        self._busy = busy
        self.batch_progress.visible = busy
        self.batch_status_text.value = status_message
        self.batch_status_text.visible = bool(status_message)
        self._update_batch_buttons_state()

    def _update_batch_buttons_state(self) -> None:
        pending_count = sum(1 for r in self._filtered_records if not r.synced)
        has_filtered = len(self._filtered_records) > 0
        if pending_count > 0:
            self.sync_all_label.value = f"ارسال یکجا ({to_persian_digits(str(pending_count))})"
        else:
            self.sync_all_label.value = "ارسال یکجا"
        self.sync_all_btn.disabled = self._busy or pending_count == 0 or self._on_sync_all is None
        self.batch_pdf_btn.disabled = self._busy or not has_filtered or self._on_export_batch_pdf is None
        self.batch_csv_btn.disabled = self._busy or not has_filtered or self._on_export_batch_csv is None

    def _build_filter_summary_text(self) -> str:
        parts: list[str] = []
        filter_names = {
            "all": "همه تخلفات",
            "pending": "فقط در انتظار ارسال",
            "synced": "فقط ارسال‌شده به سرور",
            "gps": "فقط دارای GPS",
        }
        parts.append(f"فیلتر: {filter_names.get(self._sync_filter, 'همه')}")
        if self._violation_type_filter and self._violation_type_filter != "همه":
            parts.append(f"نوع: {self._violation_type_filter}")
        if self._search_query.strip():
            parts.append(f"جستجو: «{self._search_query.strip()}»")
        return " | ".join(parts)

    async def _handle_refresh(self, _e=None) -> None:
        await self._on_refresh()

    def _handle_search_change(self, e: ft.ControlEvent) -> None:
        self._search_query = (e.control.value or "").strip()
        self.clear_search_btn.visible = bool(self._search_query)
        self._apply_filters_and_render()
        self.update()

    def _handle_clear_search(self, _e=None) -> None:
        self._search_query = ""
        self.search_field.value = ""
        self.clear_search_btn.visible = False
        self._apply_filters_and_render()
        self.update()

    def _handle_type_filter_change(self, e: ft.ControlEvent) -> None:
        self._violation_type_filter = str(e.control.value or "همه")
        self._apply_filters_and_render()
        self.update()

    def _select_sync_filter(self, filter_key: str) -> None:
        self._sync_filter = filter_key
        self._apply_filters_and_render()
        self.update()

    async def _handle_sync_all_click(self, _e=None) -> None:
        if self._busy or self._on_sync_all is None:
            return
        pending = [r for r in self._filtered_records if not r.synced]
        if not pending:
            return
        await self._on_sync_all(pending)

    async def _handle_batch_pdf_click(self, _e=None) -> None:
        if self._busy or self._on_export_batch_pdf is None or not self._filtered_records:
            return
        await self._on_export_batch_pdf(list(self._filtered_records), self._build_filter_summary_text())

    async def _handle_batch_csv_click(self, _e=None) -> None:
        if self._busy or self._on_export_batch_csv is None or not self._filtered_records:
            return
        await self._on_export_batch_csv(list(self._filtered_records))

    def set_records(self, records: list[ViolationRecord]) -> None:
        """فهرست کامل تخلفات را دریافت کرده و فیلترها و کارت‌ها را بروزرسانی می‌کند."""
        self._all_records = list(records)
        self._apply_filters_and_render()

    def _rebuild_filter_chips(self) -> None:
        p = self._palette
        total_n = len(self._all_records)
        pending_n = sum(1 for r in self._all_records if not r.synced)
        synced_n = total_n - pending_n
        gps_n = sum(1 for r in self._all_records if r.has_gps)

        specs = [
            ("all", f"همه ({to_persian_digits(str(total_n))})", ft.Icons.LIST_ALT),
            ("pending", f"در انتظار ({to_persian_digits(str(pending_n))})", ft.Icons.CLOUD_OFF),
            ("synced", f"ارسال‌شده ({to_persian_digits(str(synced_n))})", ft.Icons.CLOUD_DONE),
            ("gps", f"دارای GPS ({to_persian_digits(str(gps_n))})", ft.Icons.LOCATION_ON),
        ]

        chips: list[ft.Control] = []
        for key, label, icon in specs:
            selected = self._sync_filter == key
            bg = p.accent if selected else p.surface_alt
            fg = ft.Colors.WHITE if selected else p.text_primary
            border_col = p.accent if selected else p.border

            def _make_click(k=key):
                return lambda _ev: self._select_sync_filter(k)

            chips.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=9, vertical=5),
                    border_radius=ft.BorderRadius.all(16),
                    bgcolor=bg,
                    border=ft.Border.all(1, border_col),
                    ink=True,
                    on_click=_make_click(),
                    content=ft.Row(
                        tight=True,
                        spacing=4,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(icon, size=13, color=fg),
                            ft.Text(
                                label,
                                size=11,
                                color=fg,
                                weight=ft.FontWeight.BOLD if selected else ft.FontWeight.NORMAL,
                            ),
                        ],
                    ),
                )
            )
        self.filter_chips_row.controls = chips

    def _apply_filters_and_render(self) -> None:
        p = self._palette
        self._filtered_records = filter_violation_records(
            self._all_records,
            search_query=self._search_query,
            sync_filter=self._sync_filter,
            violation_type=self._violation_type_filter,
        )
        self._rebuild_filter_chips()
        self._update_batch_buttons_state()

        if len(self._filtered_records) == len(self._all_records):
            badge_str = to_persian_digits(str(len(self._all_records)))
        else:
            badge_str = to_persian_digits(
                f"{len(self._filtered_records)}/{len(self._all_records)}"
            )
        self.count_badge.content = ft.Text(
            badge_str,
            size=11,
            color=ft.Colors.WHITE,
            weight=ft.FontWeight.BOLD,
        )

        if not self._all_records:
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
                                "پس از ثبت تخلف، گزارش‌ها به همراه تاریخ شمسی، موقعیت GPS و تصاویر اینجا نمایش داده می‌شوند.",
                                size=11,
                                color=p.text_muted,
                                text_align=ft.TextAlign.CENTER,
                            ),
                        ],
                    ),
                )
            ]
            return

        if not self._filtered_records:
            self.records_column.controls = [
                ft.Container(
                    expand=True,
                    alignment=ft.Alignment.CENTER,
                    padding=ft.Padding.all(24),
                    content=ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                        controls=[
                            ft.Icon(ft.Icons.SEARCH_OFF, size=36, color=p.text_muted),
                            ft.Text(
                                "تخلفی مطابق جستجو یا فیلتر انتخاب‌شده یافت نشد.",
                                size=13,
                                color=p.text_secondary,
                                weight=ft.FontWeight.W_500,
                            ),
                        ],
                    ),
                )
            ]
            return

        self.records_column.controls = [
            self._build_record_card(rec) for rec in self._filtered_records
        ]

    def _build_record_card(self, rec: ViolationRecord) -> ft.Container:
        p = self._palette
        sync_label = "ارسال‌شده به سرور" if rec.synced else "ذخیره محلی"
        sync_color = ft.Colors.GREEN_700 if rec.synced else ft.Colors.ORANGE_800
        sync_icon = ft.Icons.CLOUD_DONE if rec.synced else ft.Icons.CLOUD_OFF

        media_chips: list[ft.Control] = []
        if rec.has_gps:
            media_chips.append(
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                    border_radius=ft.BorderRadius.all(6),
                    bgcolor=ft.Colors.GREEN_900 if p.dark_mode else ft.Colors.GREEN_100,
                    content=ft.Row(
                        tight=True,
                        spacing=3,
                        controls=[
                            ft.Icon(
                                ft.Icons.LOCATION_ON,
                                size=12,
                                color=ft.Colors.GREEN_200 if p.dark_mode else ft.Colors.GREEN_800,
                            ),
                            ft.Text(
                                "GPS",
                                size=10,
                                color=ft.Colors.GREEN_100 if p.dark_mode else ft.Colors.GREEN_900,
                                weight=ft.FontWeight.BOLD,
                            ),
                        ],
                    ),
                )
            )
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
                            ft.Text("پلاک", size=10, color=p.chip_text),
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
                            ft.Text("مستندات", size=10, color=p.chip_text),
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
                        expand=True,
                        max_lines=1,
                        overflow=ft.TextOverflow.ELLIPSIS,
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
                ft.Row(spacing=5, wrap=True, controls=media_chips),
                ft.Row(
                    tight=True,
                    spacing=2,
                    controls=[
                        ft.Text("جزئیات", size=11, color=p.accent_text),
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
