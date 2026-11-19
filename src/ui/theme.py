"""پالت رنگی متمرکز برای تم شب (تیره) و تم روز (روشن) با الهام از رنگ برند آیکون برنامه (#0065A0)."""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft

BRAND_BLUE = "#0065A0"
BRAND_BLUE_DARK = "#004B78"
BRAND_BLUE_LIGHT = "#0284C7"


@dataclass(frozen=True, slots=True)
class ThemePalette:
    """مجموعه رنگ‌های سازگار با تم فعلی (شب یا روز)."""

    dark_mode: bool
    page_bg: str
    panel_bg: str
    card_bg: str
    surface_alt: str
    border: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent: str
    accent_text: str
    violation_text: str
    plate_chip_bg: str
    evidence_chip_bg: str
    chip_text: str
    overlay_scrim: str


DARK_PALETTE = ThemePalette(
    dark_mode=True,
    page_bg="#0D1117",
    panel_bg="#0F141B",
    card_bg="#161B22",
    surface_alt="#1E293B",
    border="#334155",
    text_primary=ft.Colors.WHITE,
    text_secondary=ft.Colors.WHITE_70,
    text_muted=ft.Colors.WHITE_60,
    accent=BRAND_BLUE,
    accent_text=ft.Colors.LIGHT_BLUE_200,
    violation_text=ft.Colors.AMBER_300,
    plate_chip_bg="#1E293B",
    evidence_chip_bg="#134E4A",
    chip_text=ft.Colors.WHITE_70,
    overlay_scrim="#EB0F141B",
)

LIGHT_PALETTE = ThemePalette(
    dark_mode=False,
    page_bg="#F1F5F9",
    panel_bg="#FFFFFF",
    card_bg="#F8FAFC",
    surface_alt="#E2E8F0",
    border="#CBD5E1",
    text_primary="#0F172A",
    text_secondary="#334155",
    text_muted="#64748B",
    accent=BRAND_BLUE,
    accent_text=BRAND_BLUE,
    violation_text="#B45309",
    plate_chip_bg="#E0F2FE",
    evidence_chip_bg="#CCFBF1",
    chip_text="#0F172A",
    overlay_scrim="#F2FFFFFF",
)


def get_theme_palette(dark_mode: bool) -> ThemePalette:
    """پالت رنگی متناظر با تم شب یا روز را برمی‌گرداند."""
    return DARK_PALETTE if dark_mode else LIGHT_PALETTE
