"""صفحه ورود (لاگین) اپلیکیشن با الهام از هویت بصری و آیکون برنامه (#0065A0).

ویژگی‌ها:
- الهام‌گرفته از آیکون رسمی برنامه (`src/assets/icon.png`) با طیف آبی برند `#0065A0`
- پشتیبانی کامل از تم شب (تیره) و تم روز (روشن)
- احراز هویت با نام کاربری `admin` و رمز عبور `1`
- پشتیبانی از ورود با اثر انگشت (احراز هویت بیومتریک گوشی) در صورت فعال بودن در تنظیمات
"""

from __future__ import annotations

from typing import Awaitable, Callable

import flet as ft

from config import APP_DIR
from services.auth_service import authenticate_user
from ui.theme import BRAND_BLUE, BRAND_BLUE_DARK, get_theme_palette


def _load_app_icon_bytes() -> bytes | None:
    for name in ("icon.png", "icon_android.png"):
        path = APP_DIR / "assets" / name
        if path.is_file():
            try:
                return path.read_bytes()
            except OSError:
                continue
    return None


class LoginScreen(ft.Container):
    """صفحه ورود تمام‌صفحه با الهام از آیکون پلاک‌خوان هوشمند."""

    def __init__(
        self,
        on_login_success: Callable[[str], Awaitable[None]],
        on_biometric_login: Callable[[], Awaitable[tuple[bool, str]]] | None = None,
        biometric_enabled: bool = False,
        dark_mode: bool = True,
    ) -> None:
        super().__init__(expand=True, padding=0)
        self._on_login_success = on_login_success
        self._on_biometric_login = on_biometric_login
        self._biometric_enabled = biometric_enabled
        self._busy = False

        palette = get_theme_palette(dark_mode)

        # لوگوی دایره‌ای الهام‌گرفته از آیکون 2.png
        icon_bytes = _load_app_icon_bytes()
        if icon_bytes:
            emblem_inner: ft.Control = ft.Image(
                src=icon_bytes,
                width=106,
                height=106,
                fit=ft.BoxFit.CONTAIN,
                border_radius=ft.BorderRadius.all(53),
            )
        else:
            emblem_inner = ft.Icon(ft.Icons.DIRECTIONS_CAR, size=56, color=ft.Colors.WHITE)

        self._emblem_badge = ft.Container(
            width=118,
            height=118,
            border_radius=ft.BorderRadius.all(59),
            bgcolor=ft.Colors.WHITE,
            padding=ft.Padding.all(4),
            alignment=ft.Alignment.CENTER,
            shadow=ft.BoxShadow(
                blur_radius=20,
                spread_radius=2,
                color=ft.Colors.with_opacity(0.32, ft.Colors.BLACK),
                offset=ft.Offset(0, 6),
            ),
            content=emblem_inner,
        )

        # سربرگ آبی برند با قوس پایین الهام‌گرفته از آیکون خودرو و پلاک
        self._hero_header = ft.Container(
            width=float("inf"),
            padding=ft.Padding.only(left=20, right=20, top=36, bottom=32),
            border_radius=ft.BorderRadius.only(bottom_left=32, bottom_right=32),
            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_CENTER,
                end=ft.Alignment.BOTTOM_CENTER,
                colors=[BRAND_BLUE, BRAND_BLUE_DARK],
            ),
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                controls=[
                    self._emblem_badge,
                    ft.Text(
                        "پلاک‌خوان هوشمند",
                        size=23,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.WHITE,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        "سامانه هوشمند تشخیص پلاک و ثبت تخلفات ناوگان",
                        size=12,
                        color=ft.Colors.LIGHT_BLUE_100,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            ),
        )

        self._card_title = ft.Text(
            "ورود به حساب کاربری",
            size=16,
            weight=ft.FontWeight.BOLD,
            color=palette.text_primary,
        )
        self._card_subtitle = ft.Text(
            "برای دسترسی به دوربین پلاک‌خوان و مدیریت تخلفات وارد شوید.",
            size=12,
            color=palette.text_secondary,
        )

        self.username_field = ft.TextField(
            label="نام کاربری",
            hint_text="نام کاربری خود را وارد کنید",
            prefix_icon=ft.Icons.PERSON_OUTLINE,
            expand=True,
            on_submit=self._focus_password,
        )

        self.password_field = ft.TextField(
            label="رمز عبور",
            hint_text="رمز عبور خود را وارد کنید",
            prefix_icon=ft.Icons.LOCK_OUTLINE,
            password=True,
            can_reveal_password=True,
            expand=True,
            on_submit=self._handle_login_click,
        )

        self._error_text = ft.Text(
            "",
            size=12,
            color=ft.Colors.RED_400,
            weight=ft.FontWeight.W_600,
            expand=True,
        )
        self._error_box = ft.Container(
            visible=False,
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            border_radius=ft.BorderRadius.all(8),
            bgcolor=ft.Colors.with_opacity(0.14, ft.Colors.RED_500),
            border=ft.Border.all(1, ft.Colors.RED_400),
            content=ft.Row(
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.ERROR_OUTLINE, size=18, color=ft.Colors.RED_400),
                    self._error_text,
                ],
            ),
        )

        self._login_progress = ft.ProgressRing(
            width=18,
            height=18,
            stroke_width=2,
            color=ft.Colors.WHITE,
            visible=False,
        )
        self.login_button = ft.FilledButton(
            content=ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
                controls=[
                    self._login_progress,
                    ft.Icon(ft.Icons.LOGIN, size=20),
                    ft.Text("ورود", size=15, weight=ft.FontWeight.BOLD),
                ],
            ),
            style=ft.ButtonStyle(
                bgcolor=BRAND_BLUE,
                color=ft.Colors.WHITE,
                shape=ft.RoundedRectangleBorder(radius=12),
                padding=ft.Padding.symmetric(horizontal=20, vertical=15),
            ),
            width=float("inf"),
            on_click=self._handle_login_click,
        )

        self._biometric_progress = ft.ProgressRing(
            width=18,
            height=18,
            stroke_width=2,
            visible=False,
        )
        self.biometric_button = ft.OutlinedButton(
            visible=bool(biometric_enabled and on_biometric_login is not None),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
                controls=[
                    self._biometric_progress,
                    ft.Icon(ft.Icons.FINGERPRINT, size=22, color=BRAND_BLUE),
                    ft.Text("ورود با اثر انگشت", size=14, weight=ft.FontWeight.BOLD),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12),
                side=ft.BorderSide(1.5, BRAND_BLUE),
                padding=ft.Padding.symmetric(horizontal=20, vertical=13),
            ),
            width=float("inf"),
            on_click=self._handle_biometric_click,
        )

        self._login_card = ft.Container(
            width=380,
            padding=ft.Padding.all(20),
            border_radius=ft.BorderRadius.all(18),
            bgcolor=palette.card_bg,
            border=ft.Border.all(1, palette.border),
            shadow=ft.BoxShadow(
                blur_radius=16,
                spread_radius=0,
                color=ft.Colors.with_opacity(0.18 if dark_mode else 0.08, ft.Colors.BLACK),
                offset=ft.Offset(0, 6),
            ),
            content=ft.Column(
                spacing=14,
                controls=[
                    ft.Row(
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.Icons.VERIFIED_USER, size=22, color=BRAND_BLUE),
                            self._card_title,
                        ],
                    ),
                    self._card_subtitle,
                    ft.Row(controls=[self.username_field]),
                    ft.Row(controls=[self.password_field]),
                    self._error_box,
                    self.login_button,
                    self.biometric_button,
                ],
            ),
        )

        self._footer_text = ft.Text(
            "نسخه اندروید | هوش مصنوعی محلی ONNX",
            size=11,
            color=palette.text_muted,
            text_align=ft.TextAlign.CENTER,
        )

        self.bgcolor = palette.page_bg
        self.content = ft.Column(
            expand=True,
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                self._hero_header,
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=20, vertical=24),
                    alignment=ft.Alignment.CENTER,
                    content=self._login_card,
                ),
                ft.Container(
                    padding=ft.Padding.only(bottom=20),
                    content=self._footer_text,
                ),
            ],
        )

    def apply_theme(self, dark_mode: bool) -> None:
        """تم شب یا روز را روی تمامی اجزای صفحه ورود اعمال می‌کند."""
        palette = get_theme_palette(dark_mode)
        self.bgcolor = palette.page_bg
        self._login_card.bgcolor = palette.card_bg
        self._login_card.border = ft.Border.all(1, palette.border)
        self._login_card.shadow = ft.BoxShadow(
            blur_radius=16,
            spread_radius=0,
            color=ft.Colors.with_opacity(0.18 if dark_mode else 0.08, ft.Colors.BLACK),
            offset=ft.Offset(0, 6),
        )
        self._card_title.color = palette.text_primary
        self._card_subtitle.color = palette.text_secondary
        self._footer_text.color = palette.text_muted

    def set_biometric_enabled(self, enabled: bool) -> None:
        """نمایش یا پنهان‌کردن دکمه ورود با اثر انگشت بر اساس تنظیمات کاربر."""
        self._biometric_enabled = bool(enabled)
        self.biometric_button.visible = bool(
            self._biometric_enabled and self._on_biometric_login is not None
        )

    async def trigger_biometric_login(self) -> None:
        """درخواست خودکار یا دستی ورود بیومتریک با اثر انگشت گوشی."""
        if self._biometric_enabled and self._on_biometric_login is not None:
            await self._handle_biometric_click()

    async def _focus_password(self, _e=None) -> None:
        try:
            await self.password_field.focus()
        except Exception:
            pass

    def _set_busy(self, busy: bool, *, biometric: bool = False) -> None:
        self._busy = busy
        self._login_progress.visible = busy and not biometric
        self._biometric_progress.visible = busy and biometric
        self.login_button.disabled = busy
        self.biometric_button.disabled = busy
        self.username_field.disabled = busy
        self.password_field.disabled = busy

    async def _handle_login_click(self, _e=None) -> None:
        if self._busy:
            return
        self._error_box.visible = False
        self._set_busy(True, biometric=False)
        self.update()
        try:
            ok, message = authenticate_user(
                self.username_field.value or "",
                self.password_field.value or "",
            )
            if ok:
                username = (self.username_field.value or "").strip()
                self.password_field.value = ""
                await self._on_login_success(username)
                return

            self._error_text.value = message
            self._error_box.visible = True
        finally:
            self._set_busy(False)
            try:
                self.update()
            except Exception:
                pass

    async def _handle_biometric_click(self, _e=None) -> None:
        if self._busy or self._on_biometric_login is None:
            return
        self._error_box.visible = False
        self._set_busy(True, biometric=True)
        try:
            self.update()
        except Exception:
            pass
        try:
            ok, message = await self._on_biometric_login()
            if ok:
                await self._on_login_success("biometric")
                return
            if message:
                self._error_text.value = message
                self._error_box.visible = True
        finally:
            self._set_busy(False)
            try:
                self.update()
            except Exception:
                pass
