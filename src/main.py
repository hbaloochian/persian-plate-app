"""نقطه ورود اصلی اپلیکیشن اندروید «پلاک‌خوان هوشمند» با Flet 1.0 + ONNX Runtime.

ساختار رابط کاربری:
- صفحه اصلی (دوبخشی):
    • بخش بالا (55% صفحه): نمایش زنده دوربین گوشی + رسم کادرهای رنگی (آبی -> قرمز/سبز)
    • بخش پایین (45% صفحه): نمایش گرافیکی استاندارد پلاک ایران، اطلاعات دریافتی از سرور myweb.web.com
      و دکمه‌های «استعلام پلاک»، «ثبت تخلف» و «بازبینی تخلفات ثبت‌شده»
- صفحه تمام‌صفحه «ثبت تخلف»: ورودی پلاک به شکل گرافیکی پلاک ایران + تاریخ شمسی و ساعت + پیوست تصویر پلاک و مستندات
- صفحه تمام‌صفحه «بازبینی تخلفات»: لیست کامل تخلفات با تاریخ شمسی، پنجره جزئیات تک‌ردیفه و تولید گزارش PDF در پوشه دانلود
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps
import flet as ft
import flet.canvas as cv
import flet_camera as fc
import flet_permission_handler as fph

try:
    import flet_local_auth as fla
except ImportError:  # pragma: no cover - در محیط‌های تست بدون بسته محلی
    fla = None

try:
    import flet_geolocator as ftg
except ImportError:  # pragma: no cover - در محیط‌های تست بدون بسته محلی
    ftg = None

# اطمینان از در دسترس بودن ماژول‌های داخل پوشه src در همه محیط‌های اجرا
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_settings, save_settings, should_query_plate, to_persian_digits
from core.geometry import compute_android_rotation_degrees, rotate_pil_clockwise
from core.plate_detector import FrameInferenceResult, PersianPlateEngine, PlateDetection
from core.plate_stabilizer import MultiFramePlateStabilizer, StabilizedPlateResult
from services.api_client import (
    ApiCheckResult,
    PlateApiClient,
    PlateCheckStatus,
    ViolationRecord,
    delete_violation_record,
    load_violation_records,
    read_violation_image_bytes,
)
from services.pdf_report import (
    build_batch_export_filename,
    build_violation_pdf_filename,
    generate_batch_violations_csv_bytes,
    generate_batch_violations_pdf_bytes,
    generate_violation_pdf_bytes,
    save_pdf_backup_in_app_storage,
    try_write_pdf_to_downloads,
)
from services.gps_service import GpsService
from ui.components import (
    build_iranian_plate_widget,
    build_overlay_shapes,
    status_to_badge_info,
)
from ui.login_screen import LoginScreen
from ui.theme import BRAND_BLUE, get_theme_palette
from ui.violation_history_panel import ViolationHistoryPanel
from ui.violation_panel import ViolationRegistrationPanel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("plate_app")


@dataclass
class AppRuntimeState:
    """وضعیت لحظه‌ای دوربین، استنتاج هوش مصنوعی و آخرین پلاک استعلام‌شده."""

    cameras: list[fc.CameraDescription] = field(default_factory=list)
    selected_camera: fc.CameraDescription | None = None
    device_orientation: ft.DeviceOrientation | None = ft.DeviceOrientation.PORTRAIT_UP
    is_camera_ready: bool = False
    is_streaming: bool = False
    is_paused: bool = False
    torch_on: bool = False

    # ابعاد فعلی Canvas روی پیش‌نمایش دوربین
    canvas_width: float = 0.0
    canvas_height: float = 0.0

    # کنترل همزمانی (Drop-frame backpressure برای جلوگیری از تأخیر و لگ)
    inference_busy: bool = False
    last_inference_ts: float = 0.0
    fps: float = 0.0
    last_inference_ms: float = 0.0

    # آخرین فریم خام دوربین (جهت پشتیبان گرفتن عکس مستندات در صورت نیاز)
    last_raw_frame_bytes: bytes | None = None
    last_raw_frame_rotation: int = 0

    # آخرین تشخیص‌ها، نتیجه تثبیت چندفریمی و وضعیت سرور برای هر پلاک
    active_plates: list[PlateDetection] = field(default_factory=list)
    plate_statuses: dict[str, PlateCheckStatus] = field(default_factory=dict)
    inflight_queries: set[str] = field(default_factory=set)
    selected_plate: PlateDetection | None = None
    last_stabilized: StabilizedPlateResult | None = None
    last_api_result: ApiCheckResult | None = None

    # آخرین موقعیت مکانی دریافتی GPS: (lat, lon, accuracy_m)
    last_gps: tuple[float, float, float | None] | None = None
    last_gps_ts: float = 0.0

    # حالت فعلی نمایش برنامه: "info" | "violation" | "history" | "evidence_camera"
    screen_mode: str = "info"
    is_authenticated: bool = False
    current_username: str = ""
    camera_started: bool = False
    manual_query_busy: bool = False
    evidence_capture_future: asyncio.Future[bytes | None] | None = None


def _device_orientation_to_degrees(orientation: ft.DeviceOrientation | None) -> int:
    if orientation == ft.DeviceOrientation.LANDSCAPE_RIGHT:
        return 90
    if orientation == ft.DeviceOrientation.PORTRAIT_DOWN:
        return 180
    if orientation == ft.DeviceOrientation.LANDSCAPE_LEFT:
        return 270
    return 0


def _normalize_evidence_jpeg(
    raw_bytes: bytes,
    rotation_degrees: int = 0,
    max_side: int = 1600,
    quality: int = 85,
) -> bytes:
    """تصویر مستندات را با اصلاح جهت EXIF و بهینه‌سازی ابعاد به JPEG استاندارد تبدیل می‌کند."""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        oriented = ImageOps.exif_transpose(img)
        rgb = oriented.convert("RGB")
    if rotation_degrees:
        rgb = rotate_pil_clockwise(rgb, rotation_degrees)
    rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _create_demo_jpeg(color: tuple[int, int, int], size: tuple[int, int] = (320, 120)) -> bytes:
    """تصویر نمونه سبک برای آزمایش رابط کاربری در حالت پیش‌نمایش دسکتاپ."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def main(page: ft.Page) -> None:
    # بارگذاری تنظیمات ذخیره‌شده (استعلام خودکار و استعلام تمام پلاکها پیش‌فرض غیرفعال هستند)
    settings = load_settings()
    palette = get_theme_palette(settings.dark_mode)

    page.title = "پلاک‌خوان هوشمند"
    page.rtl = True
    page.theme_mode = ft.ThemeMode.DARK if settings.dark_mode else ft.ThemeMode.LIGHT
    page.padding = 0
    page.spacing = 0
    page.bgcolor = palette.page_bg
    page.theme = ft.Theme(color_scheme_seed=BRAND_BLUE)

    state = AppRuntimeState()
    engine = PersianPlateEngine(settings=settings)
    plate_stabilizer = MultiFramePlateStabilizer(
        window_size=5,
        min_confirmations=2,
        max_age_sec=3.5,
    )
    api_client = PlateApiClient(settings=settings)
    permission_handler = fph.PermissionHandler()
    file_picker = ft.FilePicker()
    local_auth = fla.LocalAuthentication() if fla is not None else None
    geolocator = (
        ftg.Geolocator(
            configuration=ftg.GeolocatorConfiguration(
                accuracy=ftg.GeolocatorPositionAccuracy.HIGH,
                time_limit=ft.Duration(seconds=20),
            )
        )
        if ftg is not None
        else None
    )
    if hasattr(page, "services"):
        page.services.append(file_picker)
        if local_auth is not None:
            page.services.append(local_auth)
        if geolocator is not None:
            page.services.append(geolocator)

    # سرویس متمرکز دریافت موقعیت مکانی (GPS) — منطق آن در modul gps_service.py نگهداری می‌شود
    gps_service = GpsService(page=page, geolocator=geolocator, ftg=ftg, state=state)

    # -----------------------------------------------------------------------
    # کنترل‌های بخش بالای صفحه (دوربین زنده + کادرهای Canvas + نوار HUD)
    # -----------------------------------------------------------------------
    overlay_canvas = cv.Canvas(
        expand=True,
        resize_interval=50,
        shapes=[],
    )

    def handle_canvas_resize(e: cv.CanvasResizeEvent) -> None:
        state.canvas_width = float(e.width)
        state.canvas_height = float(e.height)
        refresh_overlay_boxes()

    overlay_canvas.on_resize = handle_canvas_resize

    camera_control = fc.Camera(
        expand=True,
        preview_enabled=True,
        content=overlay_canvas,
    )

    hud_status_text = ft.Text(
        "در حال بارگذاری مدل ONNX...",
        size=11,
        color=ft.Colors.WHITE,
        weight=ft.FontWeight.W_500,
        max_lines=1,
        overflow=ft.TextOverflow.ELLIPSIS,
        expand=True,
    )
    hud_fps_text = ft.Text(
        "FPS: --",
        size=11,
        color=ft.Colors.LIGHT_BLUE_200,
        weight=ft.FontWeight.BOLD,
    )

    pause_btn = ft.IconButton(
        icon=ft.Icons.PAUSE_CIRCLE_OUTLINE,
        icon_color=ft.Colors.WHITE,
        tooltip="توقف / ادامه اسکن",
        icon_size=22,
    )
    flash_btn = ft.IconButton(
        icon=ft.Icons.FLASH_OFF,
        icon_color=ft.Colors.WHITE,
        tooltip="چراغ قوه دوربین",
        icon_size=22,
    )
    settings_btn = ft.IconButton(
        icon=ft.Icons.SETTINGS,
        icon_color=ft.Colors.WHITE,
        tooltip="تنظیمات سرور و استعلام خودکار",
        icon_size=22,
    )

    camera_placeholder_text = ft.Text(
        "در حال راه‌اندازی دوربین...",
        size=14,
        color=ft.Colors.WHITE_70,
        text_align=ft.TextAlign.CENTER,
    )
    retry_camera_btn = ft.FilledButton(
        content="تلاش مجدد برای دسترسی به دوربین",
        visible=False,
    )

    desktop_demo_controls = ft.Row(
        visible=False,
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=8,
        controls=[
            ft.FilledButton(
                content="شبیه‌سازی پاسخ مجاز (سبز)",
                style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE),
                on_click=lambda _: asyncio.create_task(run_demo_detection(is_no=False)),
            ),
            ft.FilledButton(
                content="شبیه‌سازی پاسخ NO (قرمز)",
                style=ft.ButtonStyle(bgcolor=ft.Colors.RED_700, color=ft.Colors.WHITE),
                on_click=lambda _: asyncio.create_task(run_demo_detection(is_no=True)),
            ),
        ],
    )

    camera_fallback_overlay = ft.Container(
        expand=True,
        alignment=ft.Alignment.CENTER,
        padding=ft.Padding.all(16),
        bgcolor=ft.Colors.with_opacity(0.65, ft.Colors.BLACK),
        visible=True,
        content=ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
            controls=[
                ft.Icon(ft.Icons.VIDEOCAM, size=42, color=ft.Colors.BLUE_300),
                camera_placeholder_text,
                retry_camera_btn,
                desktop_demo_controls,
            ],
        ),
    )

    top_hud_bar = ft.Container(
        left=8,
        right=8,
        top=8,
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        border_radius=ft.BorderRadius.all(12),
        bgcolor=ft.Colors.with_opacity(0.62, ft.Colors.BLACK),
        content=ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
            controls=[
                ft.Row(
                    expand=True,
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Icon(ft.Icons.MEMORY, size=16, color=ft.Colors.LIGHT_BLUE_300),
                        hud_status_text,
                        ft.Text("|", color=ft.Colors.WHITE_38),
                        hud_fps_text,
                    ],
                ),
                ft.Row(
                    spacing=0,
                    tight=True,
                    controls=[flash_btn, pause_btn, settings_btn],
                ),
            ],
        ),
    )

    centered_camera_holder = ft.Container(
        expand=True,
        width=float("inf"),
        bgcolor=ft.Colors.BLACK,
        alignment=ft.Alignment.CENTER,
        content=camera_control,
    )

    top_camera_section = ft.Container(
        expand=5,
        width=float("inf"),
        bgcolor=ft.Colors.BLACK,
        alignment=ft.Alignment.CENTER,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
        content=ft.Stack(
            expand=True,
            alignment=ft.Alignment.CENTER,
            fit=ft.StackFit.EXPAND,
            controls=[
                centered_camera_holder,
                camera_fallback_overlay,
                top_hud_bar,
            ],
        ),
    )

    # -----------------------------------------------------------------------
    # کنترل‌های بخش پایین صفحه اصلی (اطلاعات سرور + دکمه‌های عملیاتی)
    # -----------------------------------------------------------------------
    plate_widget_holder = ft.Container(
        content=build_iranian_plate_widget([], PlateCheckStatus.PENDING)
    )

    status_chip_icon = ft.Icon(ft.Icons.SEARCH, size=18, color=ft.Colors.WHITE)
    status_chip_label = ft.Text(
        "آماده اسکن زنده پلاک خودرو",
        size=13,
        color=ft.Colors.WHITE,
        weight=ft.FontWeight.BOLD,
    )
    status_chip = ft.Container(
        padding=ft.Padding.symmetric(horizontal=12, vertical=7),
        border_radius=ft.BorderRadius.all(10),
        bgcolor=ft.Colors.BLUE_GREY_800,
        content=ft.Row(
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[status_chip_icon, status_chip_label],
        ),
    )

    initial_guide_text = ft.Text(
        "دوربین گوشی را به سمت پلاک خودرو بگیرید. پس از تشخیص محلی (کادر آبی)، "
        "برای استعلام از سرور دکمه «استعلام پلاک» را بزنید یا حالت استعلام خودکار را از تنظیمات فعال کنید.",
        size=12,
        color=palette.text_secondary,
    )
    server_details_column = ft.Column(
        spacing=5,
        controls=[initial_guide_text],
    )

    server_info_card = ft.Container(
        expand=True,
        padding=ft.Padding.all(10),
        border_radius=ft.BorderRadius.all(10),
        bgcolor=palette.card_bg,
        border=ft.Border.all(1, palette.border),
        content=ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[server_details_column],
        ),
    )

    # دکمه جدید «استعلام پلاک» در صفحه اصلی برای ارسال دستی پلاک به سرور
    query_plate_progress = ft.ProgressRing(width=15, height=15, stroke_width=2, visible=False)
    query_plate_btn = ft.FilledButton(
        content=ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=6,
            controls=[
                query_plate_progress,
                ft.Icon(ft.Icons.CLOUD_SYNC, size=18),
                ft.Text("استعلام پلاک", size=13, weight=ft.FontWeight.BOLD),
            ],
        ),
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.BLUE_600,
            color=ft.Colors.WHITE,
            shape=ft.RoundedRectangleBorder(radius=11),
            padding=ft.Padding.symmetric(horizontal=8, vertical=11),
        ),
        expand=1,
    )

    register_violation_btn = ft.FilledButton(
        content=ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=6,
            controls=[
                ft.Icon(ft.Icons.REPORT_PROBLEM, size=18),
                ft.Text("ثبت تخلف", size=13, weight=ft.FontWeight.BOLD),
            ],
        ),
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.RED_600,
            color=ft.Colors.WHITE,
            shape=ft.RoundedRectangleBorder(radius=11),
            padding=ft.Padding.symmetric(horizontal=8, vertical=11),
        ),
        expand=1,
    )

    review_violations_label = ft.Text("بازبینی تخلفات ثبت‌شده", size=13, weight=ft.FontWeight.BOLD)
    review_violations_btn = ft.FilledButton(
        content=ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=6,
            controls=[
                ft.Icon(ft.Icons.HISTORY, size=18),
                review_violations_label,
            ],
        ),
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.BLUE_GREY_700,
            color=ft.Colors.WHITE,
            shape=ft.RoundedRectangleBorder(radius=11),
            padding=ft.Padding.symmetric(horizontal=10, vertical=10),
        ),
        width=float("inf"),
    )

    server_info_view = ft.Container(
        expand=True,
        padding=ft.Padding.all(10),
        content=ft.Column(
            spacing=8,
            expand=True,
            controls=[
                plate_widget_holder,
                status_chip,
                server_info_card,
                ft.Row(
                    spacing=8,
                    controls=[
                        query_plate_btn,
                        register_violation_btn,
                    ],
                ),
                review_violations_btn,
            ],
        ),
    )

    bottom_section_container = ft.Container(
        expand=4,
        bgcolor=palette.panel_bg,
        border=ft.Border(top=ft.BorderSide(width=1, color=palette.border)),
        content=server_info_view,
    )

    # لایه تمام‌صفحه برای «ثبت تخلف» و «بازبینی تخلفات»
    # (دوربین زنده در لایه زیرین فعال می‌ماند تا بازگشت فوری و عکس‌برداری مستندات بی‌وقفه باشد)
    fullscreen_overlay = ft.Container(
        expand=True,
        bgcolor=palette.page_bg,
        visible=False,
    )

    # نوار کمکی هنگام گرفتن عکس مستندات از دوربین زنده (در حالت تمام‌صفحه ثبت تخلف)
    evidence_shutter_progress = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
    evidence_shutter_btn = ft.FilledButton(
        content=ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8,
            controls=[
                evidence_shutter_progress,
                ft.Icon(ft.Icons.CAMERA_ALT, size=20),
                ft.Text("گرفتن عکس مستندات", size=14, weight=ft.FontWeight.BOLD),
            ],
        ),
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.GREEN_600,
            color=ft.Colors.WHITE,
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding.symmetric(horizontal=16, vertical=14),
        ),
        expand=True,
    )
    evidence_cancel_btn = ft.OutlinedButton(
        content="انصراف",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding.symmetric(horizontal=16, vertical=14),
        ),
    )
    evidence_prompt_text = ft.Text(
        "دوربین را به سمت خودرو یا صحنه تخلف بگیرید و دکمه ثبت عکس را بزنید.",
        size=13,
        color=palette.text_primary,
        expand=True,
    )
    evidence_capture_overlay = ft.Container(
        left=0,
        right=0,
        bottom=0,
        padding=ft.Padding.all(14),
        bgcolor=palette.overlay_scrim,
        border=ft.Border(top=ft.BorderSide(1, palette.border)),
        visible=False,
        content=ft.Column(
            tight=True,
            spacing=10,
            controls=[
                ft.Row(
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Icon(ft.Icons.ADD_A_PHOTO, color=ft.Colors.GREEN_500, size=20),
                        evidence_prompt_text,
                    ],
                ),
                ft.Row(
                    spacing=10,
                    controls=[
                        evidence_shutter_btn,
                        evidence_cancel_btn,
                    ],
                ),
            ],
        ),
    )

    # -----------------------------------------------------------------------
    # توابع کمکی رابط کاربری و جابجایی بین صفحه اصلی و صفحات تمام‌صفحه
    # -----------------------------------------------------------------------
    def show_snackbar(message: str, is_error: bool = False) -> None:
        page.show_dialog(
            ft.SnackBar(
                content=ft.Text(message, color=ft.Colors.WHITE),
                bgcolor=ft.Colors.RED_700 if is_error else ft.Colors.GREEN_700,
            )
        )

    def show_server_response_dialog(api_res: ApiCheckResult) -> None:
        """پاسخ سرور را در یک پنجره جدید (با دکمه بستن) نمایش می‌دهد."""
        p = get_theme_palette(settings.dark_mode)
        rows: list[ft.Control] = [
            _info_row("وضعیت (status)", api_res.details.get("status", "---")),
        ]
        passenger_val = api_res.details.get("passenger")
        rows.append(
            _info_row("تعداد مسافر (passenger)", "---" if passenger_val is None else str(passenger_val))
        )
        description_val = api_res.details.get("description")
        rows.append(
            _info_row("توضیحات (description)", "---" if description_val is None else str(description_val))
        )
        rows.append(ft.Divider(height=6, color=p.border))
        rows.append(
            ft.Container(
                padding=ft.Padding.all(6),
                border_radius=ft.BorderRadius.all(8),
                bgcolor=p.card_bg,
                content=ft.Text(
                    "پاسخ کامل سرور:",
                    size=12,
                    color=p.text_muted,
                    weight=ft.FontWeight.W_600,
                ),
            )
        )
        rows.append(
            ft.Container(
                padding=ft.Padding.all(8),
                border_radius=ft.BorderRadius.all(8),
                bgcolor=p.card_bg,
                content=ft.Text(
                    api_res.raw_response or "(پاسخ خالی)",
                    size=12,
                    color=p.text_primary,
                ),
            )
        )

        def _close_server_response_dialog(_e=None) -> None:
            page.close_dialog()

        page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("پاسخ سرور برای این پلاک"),
                content=ft.Container(
                    width=360,
                    content=ft.Column(tight=True, spacing=8, controls=rows),
                ),
                actions=[
                    ft.FilledButton("بستن", on_click=_close_server_response_dialog),
                ],
            )
        )

    async def refresh_violation_counter(update_ui: bool = True) -> list[ViolationRecord]:
        records = await asyncio.to_thread(load_violation_records)
        count = len(records)
        review_violations_label.value = (
            f"بازبینی تخلفات ثبت‌شده ({to_persian_digits(str(count))})"
            if count > 0
            else "بازبینی تخلفات ثبت‌شده"
        )
        if update_ui and state.screen_mode == "info":
            review_violations_label.update()
        return records

    def refresh_overlay_boxes() -> None:
        mirror_x = (
            state.selected_camera is not None
            and state.selected_camera.lens_direction == fc.CameraLensDirection.FRONT
        )
        overlay_canvas.shapes = build_overlay_shapes(
            plates=state.active_plates if state.screen_mode == "info" else [],
            plate_statuses=state.plate_statuses,
            canvas_width=state.canvas_width,
            canvas_height=state.canvas_height,
            mirror_x=mirror_x,
        )
        overlay_canvas.update()

    def _info_row(
        label: str,
        value: str,
        value_color: str | None = None,
    ) -> ft.Row:
        p = get_theme_palette(settings.dark_mode)
        return ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.START,
            controls=[
                ft.Text(f"{label}:", size=12, color=p.text_muted),
                ft.Text(
                    value,
                    size=12,
                    color=value_color or p.text_primary,
                    weight=ft.FontWeight.W_600,
                    text_align=ft.TextAlign.LEFT,
                    expand=True,
                ),
            ],
        )

    def render_server_info_panel() -> None:
        p = get_theme_palette(settings.dark_mode)
        plate = state.selected_plate
        api_res = state.last_api_result

        if plate is None:
            plate_widget_holder.content = build_iranian_plate_widget([], PlateCheckStatus.PENDING)
            status_chip.bgcolor = ft.Colors.BLUE_GREY_700 if p.dark_mode else BRAND_BLUE
            status_chip_icon.name = ft.Icons.SEARCH
            mode_note = "خودکار" if settings.auto_query_enabled else "دستی"
            scope_note = "همه پلاک‌ها" if settings.query_all_plates else "فقط پلاک‌های «ع»"
            status_chip_label.value = f"در انتظار تشخیص پلاک ({mode_note} | {scope_note})"
            initial_guide_text.color = p.text_secondary
            return

        plate_key = plate.plate_latin or plate.plate_persian
        current_status = state.plate_statuses.get(plate_key, PlateCheckStatus.PENDING)
        plate_widget_holder.content = build_iranian_plate_widget(plate.chars_persian, current_status)

        plate_allowed_for_query = should_query_plate(
            settings,
            chars_persian=plate.chars_persian,
            plate_persian=plate.plate_persian,
            plate_latin=plate.plate_latin,
        )

        title, bg_color, icon_name = status_to_badge_info(
            current_status,
            auto_query_enabled=(
                (settings.auto_query_enabled and plate_allowed_for_query)
                or state.manual_query_busy
            ),
        )
        if current_status == PlateCheckStatus.PENDING and not plate_allowed_for_query:
            title = "پلاک شناسایی شد (فقط پلاک‌های «ع» استعلام می‌شوند)"
        status_chip.bgcolor = bg_color
        status_chip_icon.name = icon_name
        status_chip_label.value = title

        rows: list[ft.Control] = [
            _info_row("پلاک تشخیص‌داده‌شده", plate.display_text),
            _info_row("کد لاتین ارسالی به سرور", plate.plate_latin or "---"),
            _info_row("دقت مدل محلی", to_persian_digits(f"{int(plate.confidence * 100)}٪")),
        ]
        if state.last_stabilized is not None:
            stab_color = (
                (ft.Colors.GREEN_400 if p.dark_mode else ft.Colors.GREEN_700)
                if state.last_stabilized.is_confirmed
                else (ft.Colors.AMBER_300 if p.dark_mode else ft.Colors.ORANGE_800)
            )
            rows.append(
                _info_row(
                    "تثبیت چندفریمی",
                    state.last_stabilized.status_badge_text(),
                    value_color=stab_color,
                )
            )
        rows.extend(
            [
                _info_row(
                    "حالت استعلام",
                    "خودکار" if settings.auto_query_enabled else "دستی (با دکمه استعلام پلاک)",
                    value_color=p.accent_text,
                ),
                _info_row(
                    "نوع پلاک مجاز استعلام",
                    "تمام پلاک‌ها" if settings.query_all_plates else "فقط پلاک‌های عمومی (ع)",
                    value_color=p.accent_text,
                ),
            ]
        )

        if api_res is not None:
            rows.append(ft.Divider(height=8, color=p.border))
            if api_res.error_message:
                rows.append(
                    _info_row(
                        "وضعیت ارتباط",
                        api_res.error_message,
                        value_color=ft.Colors.ORANGE_400 if p.dark_mode else ft.Colors.ORANGE_800,
                    )
                )
            else:
                rows.append(_info_row("زمان استعلام", to_persian_digits(api_res.checked_at)))
                rows.append(_info_row("سرعت پاسخ سرور", to_persian_digits(f"{int(api_res.latency_ms)} میلی‌ثانیه")))
                label_map = {
                    "status": "وضعیت (status)",
                    "passenger": "تعداد مسافر (passenger)",
                    "description": "توضیحات (description)",
                }
                for key, val in api_res.details.items():
                    display_val = "---" if val is None else str(val)
                    rows.append(_info_row(label_map.get(str(key), str(key)), display_val))
        else:
            if not engine.single_stage_mode and not plate.complete:
                rows.append(_info_row("وضعیت", "در حال خوانش کامل ۸ کاراکتر پلاک (کادر آبی)..."))
            elif not plate_allowed_for_query:
                rows.append(
                    _info_row(
                        "وضعیت فیلتر",
                        "این پلاک حرف «ع» ندارد و ارسال نمی‌شود (برای استعلام همه، سوییچ «استعلام تمام پلاکها» را در تنظیمات فعال کنید).",
                        value_color=p.violation_text,
                    )
                )
            elif settings.auto_query_enabled or state.manual_query_busy:
                rows.append(_info_row("آدرس API", settings.check_url))
                rows.append(_info_row("وضعیت", "در حال ارسال پلاک به سرور..."))
            else:
                rows.append(
                    _info_row(
                        "راهنما",
                        "برای ارسال به سرور، دکمه «استعلام پلاک» را فشار دهید.",
                        value_color=p.accent_text,
                    )
                )

        server_details_column.controls = rows

    def switch_to_info_view() -> None:
        state.screen_mode = "info"
        fullscreen_overlay.visible = False
        fullscreen_overlay.content = None
        evidence_capture_overlay.visible = False
        render_server_info_panel()
        refresh_overlay_boxes()
        page.update()

    # -----------------------------------------------------------------------
    # دریافت خودکار موقعیت مکانی (GPS) گوشی با flet-geolocator
    # -----------------------------------------------------------------------
    async def fetch_current_gps(*, request_if_needed: bool = True) -> tuple[tuple[float, float, float | None] | None, str]:
        """مختصات جغرافیایی فعلی گوشی را از سرویس متمرکز GPS دریافت می‌کند."""
        return await gps_service.fetch_current_gps(request_if_needed=request_if_needed)

    async def refresh_violation_panel_gps() -> None:
        """مختصات جدید GPS را در کارت موقعیت مکانی صفحه ثبت تخلف به‌روزرسانی می‌کند."""
        await gps_service.refresh_panel_gps(violation_panel)

    def switch_to_violation_view(_e=None) -> None:
        state.screen_mode = "violation"
        plate_text = state.selected_plate.display_text if state.selected_plate else ""
        plate_chars = (
            state.selected_plate.chars_persian
            if state.selected_plate is not None
            else None
        )
        crop_bytes = (
            state.selected_plate.plate_crop_jpeg
            if state.selected_plate is not None
            else None
        )
        # استفاده فوری از آخرین GPS کش‌شده (در صورت تازگی زیر ۵ دقیقه) و بروزرسانی همزمان در پس‌زمینه
        fresh_cached_gps = (
            state.last_gps
            if (state.last_gps is not None and (time.monotonic() - state.last_gps_ts) < 300.0)
            else None
        )
        violation_panel.prepare(
            plate_display=plate_text,
            has_crop=crop_bytes is not None,
            plate_crop_bytes=crop_bytes,
            chars_persian=plate_chars,
            gps_coords=fresh_cached_gps,
        )
        fullscreen_overlay.content = violation_panel
        fullscreen_overlay.visible = True
        evidence_capture_overlay.visible = False
        page.update()
        asyncio.create_task(refresh_violation_panel_gps())

    async def switch_to_history_view(_e=None) -> None:
        state.screen_mode = "history"
        records = await refresh_violation_counter(update_ui=False)
        violation_history_panel.set_records(records)
        fullscreen_overlay.content = violation_history_panel
        fullscreen_overlay.visible = True
        evidence_capture_overlay.visible = False
        page.update()

    async def reload_history_panel() -> None:
        records = await refresh_violation_counter(update_ui=False)
        violation_history_panel.set_records(records)
        if state.screen_mode == "history":
            fullscreen_overlay.update()

    # -----------------------------------------------------------------------
    # ثبت عکس مستندات از دوربین زنده (با نمایش نمای دوربین برای کادربندی دقیق)
    # -----------------------------------------------------------------------
    async def _perform_camera_snapshot() -> bytes | None:
        was_streaming = state.is_streaming
        prev_paused = state.is_paused
        state.is_paused = True
        captured_bytes: bytes | None = None
        try:
            if was_streaming:
                try:
                    await camera_control.stop_image_stream()
                    state.is_streaming = False
                except Exception as exc:
                    logger.warning("Could not pause stream before take_picture: %s", exc)

            raw_photo = await camera_control.take_picture()
            if raw_photo:
                captured_bytes = await asyncio.to_thread(_normalize_evidence_jpeg, raw_photo, 0)
        except Exception as exc:
            logger.warning("take_picture failed, falling back to latest stream frame: %s", exc)
            if state.last_raw_frame_bytes:
                try:
                    captured_bytes = await asyncio.to_thread(
                        _normalize_evidence_jpeg,
                        state.last_raw_frame_bytes,
                        state.last_raw_frame_rotation,
                    )
                except Exception as fallback_exc:
                    logger.warning("Fallback frame encoding failed: %s", fallback_exc)
        finally:
            if was_streaming and not state.is_streaming:
                try:
                    await camera_control.start_image_stream()
                    state.is_streaming = True
                except Exception as exc:
                    logger.warning("Could not resume image stream after take_picture: %s", exc)
            state.is_paused = prev_paused
        return captured_bytes

    async def _on_evidence_shutter_click(_e=None) -> None:
        fut = state.evidence_capture_future
        if fut is None or fut.done():
            return
        evidence_shutter_progress.visible = True
        evidence_shutter_btn.disabled = True
        evidence_cancel_btn.disabled = True
        evidence_capture_overlay.update()
        try:
            photo_bytes = await _perform_camera_snapshot()
            if not fut.done():
                fut.set_result(photo_bytes)
        finally:
            evidence_shutter_progress.visible = False
            evidence_shutter_btn.disabled = False
            evidence_cancel_btn.disabled = False

    def _on_evidence_cancel_click(_e=None) -> None:
        fut = state.evidence_capture_future
        if fut is not None and not fut.done():
            fut.set_result(None)

    evidence_shutter_btn.on_click = _on_evidence_shutter_click
    evidence_cancel_btn.on_click = _on_evidence_cancel_click

    async def handle_capture_evidence_photo() -> bytes | None:
        """نمای زنده دوربین را برای کادربندی عکس مستندات نمایش می‌دهد و تصویر گرفته‌شده را برمی‌گرداند."""
        if state.is_camera_ready:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[bytes | None] = loop.create_future()
            state.evidence_capture_future = fut
            state.screen_mode = "evidence_camera"

            # مخفی کردن موقت فرم تمام‌صفحه تا کاربر بتواند صحنه تخلف را در دوربین ببیند
            fullscreen_overlay.visible = False
            evidence_capture_overlay.visible = True
            refresh_overlay_boxes()
            page.update()

            try:
                captured_bytes = await fut
            finally:
                state.evidence_capture_future = None
                state.screen_mode = "violation"
                evidence_capture_overlay.visible = False
                fullscreen_overlay.content = violation_panel
                fullscreen_overlay.visible = True
                page.update()

            if captured_bytes:
                show_snackbar("عکس مستندات با دوربین ثبت شد.")
                return captured_bytes
            return None

        # در حالت پیش‌نمایش دسکتاپ یک تصویر مستندات نمونه تولید می‌شود
        demo_bytes = await asyncio.to_thread(_create_demo_jpeg, (36, 72, 118), (640, 360))
        show_snackbar("تصویر نمونه مستندات در حالت شبیه‌ساز دسکتاپ ثبت شد.")
        return demo_bytes

    async def handle_pick_evidence_photo() -> bytes | None:
        """انتخاب تصویر مستندات از گالری گوشی یا فایل‌های سیستم."""
        try:
            files = await file_picker.pick_files(
                dialog_title="انتخاب تصویر مستندات تخلف",
                file_type=ft.FilePickerFileType.IMAGE,
                allow_multiple=False,
                with_data=True,
            )
            if not files:
                return None

            selected = files[0]
            raw_bytes: bytes | None = selected.bytes
            if not raw_bytes and selected.path:
                raw_bytes = await asyncio.to_thread(Path(selected.path).read_bytes)

            if not raw_bytes:
                show_snackbar("خواندن فایل تصویر انتخاب‌شده ممکن نشد.", is_error=True)
                return None

            normalized = await asyncio.to_thread(_normalize_evidence_jpeg, raw_bytes, 0)
            show_snackbar("تصویر مستندات از گالری انتخاب شد.")
            return normalized
        except Exception as exc:
            logger.warning("Gallery image picker error: %s", exc)
            show_snackbar(f"خطا در انتخاب تصویر از گالری: {exc}", is_error=True)
            return None

    async def handle_violation_submit(
        plate_text: str,
        plate_latin: str,
        violation_type: str,
        notes: str,
        attach_crop: bool,
        evidence_bytes: bytes | None,
        latitude: float | None = None,
        longitude: float | None = None,
        location_accuracy_m: float | None = None,
    ) -> None:
        final_latin = plate_latin or (state.selected_plate.plate_latin if state.selected_plate else "")
        crop_bytes = (
            state.selected_plate.plate_crop_jpeg
            if (attach_crop and state.selected_plate is not None)
            else None
        )
        # اگر در لحظه ثبت GPS هنوز روی فرم نیامده ولی آخرین موقعیت کش‌شده موجود است
        if latitude is None and longitude is None and state.last_gps is not None:
            latitude, longitude, location_accuracy_m = state.last_gps

        result = await api_client.submit_violation(
            plate_persian=plate_text,
            plate_latin=final_latin,
            violation_type=violation_type,
            description=notes,
            plate_crop_jpeg=crop_bytes,
            evidence_image_jpeg=evidence_bytes,
            latitude=latitude,
            longitude=longitude,
            location_accuracy_m=location_accuracy_m,
        )
        await refresh_violation_counter(update_ui=False)
        show_snackbar(result.message, is_error=not result.success)
        if result.success or result.saved_locally:
            switch_to_info_view()

    # -----------------------------------------------------------------------
    # ابزار مشترک ذخیره فایل‌های گزارش (PDF و CSV) در پوشه Downloads گوشی
    # -----------------------------------------------------------------------
    async def _save_exported_file_to_downloads(
        file_bytes: bytes,
        filename: str,
        extension: str,
        dialog_title: str,
    ) -> None:
        backup_path = await asyncio.to_thread(
            save_pdf_backup_in_app_storage,
            file_bytes,
            filename,
        )
        wrote_direct, saved_path, _err = await asyncio.to_thread(
            try_write_pdf_to_downloads,
            file_bytes,
            filename,
        )
        if wrote_direct and saved_path is not None:
            show_snackbar(f"فایل «{filename}» با موفقیت در پوشه Downloads ذخیره شد.")
            return

        try:
            saved_uri = await file_picker.save_file(
                dialog_title=dialog_title,
                file_name=filename,
                initial_directory="/storage/emulated/0/Download",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=[extension],
                src_bytes=file_bytes,
            )
            if saved_uri:
                show_snackbar(f"فایل «{filename}» با موفقیت ذخیره شد.")
            else:
                show_snackbar(
                    f"ذخیره در Downloads لغو شد (نسخه پشتیبان در {backup_path.name} موجود است).",
                    is_error=True,
                )
        except Exception as picker_exc:
            logger.warning("SAF save_file error: %s", picker_exc)
            show_snackbar(
                f"گزارش در حافظه برنامه ذخیره شد: {backup_path}",
                is_error=False,
            )

    # -----------------------------------------------------------------------
    # پنجره جزئیات تخلف (تمامی دکمه‌ها در یک ردیف + دکمه «گزارش» PDF در پوشه دانلود)
    # -----------------------------------------------------------------------
    async def open_violation_detail_dialog(record: ViolationRecord) -> None:
        """نمایش جزئیات کامل تخلف با تاریخ شمسی، موقعیت GPS، تصاویر، دکمه‌های تک‌ردیفه و خروجی PDF."""
        plate_bytes = await asyncio.to_thread(read_violation_image_bytes, record.plate_image_file)
        evidence_bytes = await asyncio.to_thread(
            read_violation_image_bytes,
            record.evidence_image_file,
        )

        jalali_date_str = to_persian_digits(record.jalali_date or "---")
        if record.jalali_weekday:
            jalali_date_str = f"{record.jalali_weekday} ({jalali_date_str})"
        jalali_time_str = to_persian_digits(record.jalali_time or "---")

        p = get_theme_palette(settings.dark_mode)
        synced_color = ft.Colors.GREEN_400 if p.dark_mode else ft.Colors.GREEN_700
        unsynced_color = ft.Colors.ORANGE_300 if p.dark_mode else ft.Colors.ORANGE_800

        detail_controls: list[ft.Control] = [
            _info_row("پلاک خودرو", to_persian_digits(record.plate)),
            _info_row("کد لاتین پلاک", record.plate_latin or "---"),
            _info_row("نوع تخلف", record.violation_type, value_color=p.violation_text),
            _info_row("تاریخ شمسی", jalali_date_str, value_color=p.accent_text),
            _info_row("ساعت ثبت", jalali_time_str, value_color=p.accent_text),
            _info_row(
                "موقعیت مکانی (GPS)",
                record.formatted_gps,
                value_color=synced_color if record.has_gps else p.text_muted,
            ),
            _info_row(
                "وضعیت ارسال",
                "ارسال‌شده به سرور" if record.synced else "ذخیره‌شده روی گوشی (ارسال‌نشده)",
                value_color=synced_color if record.synced else unsynced_color,
            ),
        ]
        if record.has_gps:
            detail_controls.append(
                _info_row("لینک نقشه", record.google_maps_url, value_color=p.accent_text)
            )
        if record.description:
            detail_controls.append(_info_row("توضیحات", record.description))
        if record.server_response:
            detail_controls.append(_info_row("پاسخ سرور", record.server_response[:160]))

        if plate_bytes:
            detail_controls.extend(
                [
                    ft.Divider(height=10, color=p.border),
                    ft.Text("تصویر برش‌خورده پلاک:", size=12, weight=ft.FontWeight.BOLD),
                    ft.Container(
                        alignment=ft.Alignment.CENTER,
                        bgcolor=ft.Colors.BLACK,
                        border_radius=ft.BorderRadius.all(8),
                        padding=ft.Padding.all(4),
                        content=ft.Image(
                            src=plate_bytes,
                            height=85,
                            fit=ft.BoxFit.CONTAIN,
                            border_radius=ft.BorderRadius.all(6),
                        ),
                    ),
                ]
            )

        if evidence_bytes:
            detail_controls.extend(
                [
                    ft.Divider(height=10, color=p.border),
                    ft.Text("تصویر مستندات تخلف:", size=12, weight=ft.FontWeight.BOLD),
                    ft.Container(
                        alignment=ft.Alignment.CENTER,
                        bgcolor=ft.Colors.BLACK,
                        border_radius=ft.BorderRadius.all(8),
                        padding=ft.Padding.all(4),
                        content=ft.Image(
                            src=evidence_bytes,
                            height=180,
                            fit=ft.BoxFit.CONTAIN,
                            border_radius=ft.BorderRadius.all(6),
                        ),
                    ),
                ]
            )

        if not plate_bytes and not evidence_bytes:
            detail_controls.append(
                ft.Text(
                    "تصویری برای این تخلف پیوست نشده است.",
                    size=11,
                    color=p.text_muted,
                )
            )

        pdf_progress = ft.ProgressRing(width=13, height=13, stroke_width=2, visible=False)
        pdf_btn_ref: list[ft.FilledButton] = []

        async def handle_export_pdf(_ev) -> None:
            btn = pdf_btn_ref[0]
            btn.disabled = True
            pdf_progress.visible = True
            btn.update()
            try:
                filename = build_violation_pdf_filename(record)
                pdf_bytes = await asyncio.to_thread(
                    generate_violation_pdf_bytes,
                    record,
                    plate_bytes,
                    evidence_bytes,
                )
                await _save_exported_file_to_downloads(
                    pdf_bytes,
                    filename,
                    "pdf",
                    "ذخیره گزارش PDF تخلف در پوشه دانلود",
                )
            except Exception as exc:
                logger.exception("PDF report export failed")
                show_snackbar(f"خطا در ساخت گزارش PDF: {exc}", is_error=True)
            finally:
                pdf_progress.visible = False
                btn.disabled = False
                try:
                    btn.update()
                except Exception:
                    pass

        async def handle_retry_sync(_ev) -> None:
            page.pop_dialog()
            res = await api_client.retry_violation_sync(record)
            show_snackbar(res.message, is_error=not res.success)
            await reload_history_panel()

        async def handle_delete(_ev) -> None:
            page.pop_dialog()
            deleted = await asyncio.to_thread(delete_violation_record, record.id)
            show_snackbar(
                "گزارش تخلف حذف شد." if deleted else "حذف گزارش ناموفق بود.",
                is_error=not deleted,
            )
            await reload_history_panel()

        # تمامی دکمه‌های بخش جزئیات تخلف صراحتاً درون یک ردیف واحد (Single Row)
        report_pdf_btn = ft.FilledButton(
            tooltip="تولید گزارش PDF و ذخیره در پوشه دانلود",
            content=ft.Row(
                tight=True,
                spacing=3,
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    pdf_progress,
                    ft.Icon(ft.Icons.PICTURE_AS_PDF, size=14),
                    ft.Text("گزارش", size=11, weight=ft.FontWeight.BOLD),
                ],
            ),
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.TEAL_700,
                color=ft.Colors.WHITE,
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=ft.Padding.symmetric(horizontal=6, vertical=9),
            ),
            expand=True,
            on_click=handle_export_pdf,
        )
        pdf_btn_ref.append(report_pdf_btn)

        single_row_buttons: list[ft.Control] = [report_pdf_btn]

        if not record.synced:
            single_row_buttons.append(
                ft.FilledButton(
                    tooltip="ارسال مجدد به سرور",
                    content=ft.Row(
                        tight=True,
                        spacing=3,
                        alignment=ft.MainAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.Icons.CLOUD_UPLOAD, size=14),
                            ft.Text("ارسال", size=11, weight=ft.FontWeight.BOLD),
                        ],
                    ),
                    style=ft.ButtonStyle(
                        bgcolor=ft.Colors.BLUE_700,
                        color=ft.Colors.WHITE,
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.Padding.symmetric(horizontal=6, vertical=9),
                    ),
                    expand=True,
                    on_click=handle_retry_sync,
                )
            )

        single_row_buttons.extend(
            [
                ft.FilledButton(
                    tooltip="حذف گزارش تخلف",
                    content=ft.Row(
                        tight=True,
                        spacing=3,
                        alignment=ft.MainAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.Icons.DELETE_OUTLINE, size=14),
                            ft.Text("حذف", size=11, weight=ft.FontWeight.BOLD),
                        ],
                    ),
                    style=ft.ButtonStyle(
                        bgcolor=ft.Colors.RED_700,
                        color=ft.Colors.WHITE,
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.Padding.symmetric(horizontal=6, vertical=9),
                    ),
                    expand=True,
                    on_click=handle_delete,
                ),
                ft.FilledButton(
                    tooltip="بستن پنجره",
                    content=ft.Row(
                        tight=True,
                        spacing=3,
                        alignment=ft.MainAxisAlignment.CENTER,
                        controls=[
                            ft.Icon(ft.Icons.CLOSE, size=14),
                            ft.Text("بستن", size=11, weight=ft.FontWeight.BOLD),
                        ],
                    ),
                    style=ft.ButtonStyle(
                        bgcolor=ft.Colors.BLUE_GREY_700,
                        color=ft.Colors.WHITE,
                        shape=ft.RoundedRectangleBorder(radius=8),
                        padding=ft.Padding.symmetric(horizontal=6, vertical=9),
                    ),
                    expand=True,
                    on_click=lambda _: page.pop_dialog(),
                ),
            ]
        )

        # قرار دادن ردیف دکمه‌ها در پایین بدنه دیالوگ باعث می‌شود OverflowBar فلاتر
        # هرگز آن‌ها را به دو ردیف عمودی نشکند و همیشه در یک ردیف افقی بمانند.
        page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("جزئیات تخلف ثبت‌شده", size=16, weight=ft.FontWeight.BOLD),
                content_padding=ft.Padding.only(left=14, right=14, top=10, bottom=12),
                content=ft.Container(
                    width=350,
                    content=ft.Column(
                        tight=True,
                        spacing=10,
                        controls=[
                            ft.Container(
                                height=380,
                                content=ft.Column(
                                    spacing=8,
                                    scroll=ft.ScrollMode.AUTO,
                                    controls=detail_controls,
                                ),
                            ),
                            ft.Divider(height=1, color=ft.Colors.BLUE_GREY_800),
                            ft.Row(
                                spacing=5,
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                controls=single_row_buttons,
                            ),
                        ],
                    ),
                ),
                actions=[],
            )
        )

    # -----------------------------------------------------------------------
    # عملیات گروهی صفحه بازبینی تخلفات (ارسال یکجا + PDF گروهی + اکسل CSV)
    # -----------------------------------------------------------------------
    async def handle_sync_all_violations(pending_records: list[ViolationRecord]) -> None:
        total = len(pending_records)
        if total == 0:
            show_snackbar("همه تخلفات این فهرست قبلاً ارسال شده‌اند.")
            return

        sent_ok = 0
        failed = 0
        violation_history_panel.set_batch_busy(
            True,
            f"در حال ارسال ۱ از {to_persian_digits(str(total))}...",
        )
        violation_history_panel.update()
        try:
            for idx, rec in enumerate(pending_records, start=1):
                violation_history_panel.set_batch_busy(
                    True,
                    f"در حال ارسال {to_persian_digits(str(idx))} از {to_persian_digits(str(total))}...",
                )
                violation_history_panel.update()
                res = await api_client.retry_violation_sync(rec)
                if res.success:
                    sent_ok += 1
                else:
                    failed += 1
        finally:
            violation_history_panel.set_batch_busy(False, "")
            await reload_history_panel()

        if failed == 0:
            show_snackbar(
                f"همه {to_persian_digits(str(sent_ok))} تخلف با موفقیت به سرور ارسال شدند."
            )
        else:
            show_snackbar(
                f"ارسال یکجا تمام شد: {to_persian_digits(str(sent_ok))} موفق، {to_persian_digits(str(failed))} ناموفق.",
                is_error=sent_ok == 0,
            )

    async def handle_export_batch_pdf(records: list[ViolationRecord], filter_summary: str) -> None:
        if not records:
            show_snackbar("تخلفی برای خروجی PDF گروهی وجود ندارد.", is_error=True)
            return
        violation_history_panel.set_batch_busy(True, "در حال ساخت گزارش PDF گروهی...")
        violation_history_panel.update()
        try:
            filename = build_batch_export_filename("pdf")
            pdf_bytes = await asyncio.to_thread(
                generate_batch_violations_pdf_bytes,
                records,
                filter_summary=filter_summary,
            )
            await _save_exported_file_to_downloads(
                pdf_bytes,
                filename,
                "pdf",
                "ذخیره گزارش PDF گروهی تخلفات در پوشه دانلود",
            )
        except Exception as exc:
            logger.exception("Batch PDF export failed")
            show_snackbar(f"خطا در ساخت PDF گروهی: {exc}", is_error=True)
        finally:
            violation_history_panel.set_batch_busy(False, "")
            try:
                violation_history_panel.update()
            except Exception:
                pass

    async def handle_export_batch_csv(records: list[ViolationRecord]) -> None:
        if not records:
            show_snackbar("تخلفی برای خروجی اکسل CSV وجود ندارد.", is_error=True)
            return
        violation_history_panel.set_batch_busy(True, "در حال ساخت فایل اکسل (CSV)...")
        violation_history_panel.update()
        try:
            filename = build_batch_export_filename("csv", prefix="تخلفات-ثبت-شده")
            csv_bytes = await asyncio.to_thread(generate_batch_violations_csv_bytes, records)
            await _save_exported_file_to_downloads(
                csv_bytes,
                filename,
                "csv",
                "ذخیره فایل اکسل CSV تخلفات در پوشه دانلود",
            )
        except Exception as exc:
            logger.exception("Batch CSV export failed")
            show_snackbar(f"خطا در ساخت فایل CSV: {exc}", is_error=True)
        finally:
            violation_history_panel.set_batch_busy(False, "")
            try:
                violation_history_panel.update()
            except Exception:
                pass

    violation_panel = ViolationRegistrationPanel(
        page=page,
        on_back=switch_to_info_view,
        on_submit=handle_violation_submit,
        on_capture_evidence=handle_capture_evidence_photo,
        on_pick_evidence=handle_pick_evidence_photo,
        on_refresh_gps=refresh_violation_panel_gps,
        dark_mode=settings.dark_mode,
    )
    violation_history_panel = ViolationHistoryPanel(
        on_back=switch_to_info_view,
        on_refresh=reload_history_panel,
        on_select_record=open_violation_detail_dialog,
        on_sync_all=handle_sync_all_violations,
        on_export_batch_pdf=handle_export_batch_pdf,
        on_export_batch_csv=handle_export_batch_csv,
        dark_mode=settings.dark_mode,
    )
    # ارجاع صفحه ثبت تخلف به سرویس GPS برای نمایش خودکار موقعیت کش‌شده
    gps_service.panel = violation_panel

    register_violation_btn.on_click = switch_to_violation_view
    review_violations_btn.on_click = switch_to_history_view

    # -----------------------------------------------------------------------
    # اعمال پویای تم روز (روشن) و تم شب (تیره) روی کل اپلیکیشن
    # -----------------------------------------------------------------------
    login_screen_ref: list[LoginScreen] = []

    def apply_app_theme(dark_mode: bool, *, update_page: bool = True) -> None:
        settings.dark_mode = bool(dark_mode)
        p = get_theme_palette(settings.dark_mode)
        page.theme_mode = ft.ThemeMode.DARK if settings.dark_mode else ft.ThemeMode.LIGHT
        page.bgcolor = p.page_bg

        bottom_section_container.bgcolor = p.panel_bg
        bottom_section_container.border = ft.Border(top=ft.BorderSide(width=1, color=p.border))
        server_info_card.bgcolor = p.card_bg
        server_info_card.border = ft.Border.all(1, p.border)
        fullscreen_overlay.bgcolor = p.page_bg
        evidence_capture_overlay.bgcolor = p.overlay_scrim
        evidence_capture_overlay.border = ft.Border(top=ft.BorderSide(1, p.border))
        evidence_prompt_text.color = p.text_primary
        initial_guide_text.color = p.text_secondary

        violation_panel.apply_theme(settings.dark_mode)
        violation_history_panel.apply_theme(settings.dark_mode)
        if login_screen_ref:
            login_screen_ref[0].apply_theme(settings.dark_mode)

        render_server_info_panel()
        if update_page:
            page.update()

    # -----------------------------------------------------------------------
    # دیالوگ تنظیمات (سوییچ‌های استعلام، اثر انگشت، تم روز/شب + آدرس سرور)
    # -----------------------------------------------------------------------
    def open_settings_dialog(_e=None) -> None:
        prev_dark_mode = settings.dark_mode
        p = get_theme_palette(settings.dark_mode)

        base_url_tf = ft.TextField(label="آدرس پایه سرور API", value=settings.api_base_url)
        check_path_tf = ft.TextField(label="مسیر استعلام پلاک", value=settings.api_check_path)
        viol_path_tf = ft.TextField(label="مسیر ثبت تخلف", value=settings.api_violation_path)

        switches_card = ft.Container(
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            border_radius=ft.BorderRadius.all(10),
            bgcolor=p.card_bg,
            border=ft.Border.all(1, p.border),
        )

        def _on_theme_toggle(ev: ft.ControlEvent) -> None:
            is_dark = bool(ev.control.value)
            new_p = get_theme_palette(is_dark)
            switches_card.bgcolor = new_p.card_bg
            switches_card.border = ft.Border.all(1, new_p.border)
            apply_app_theme(is_dark, update_page=True)

        auto_query_switch = ft.Switch(
            label="استعلام خودکار",
            value=settings.auto_query_enabled,
        )
        all_plates_switch = ft.Switch(
            label="استعلام تمام پلاکها",
            value=settings.query_all_plates,
        )
        biometric_switch = ft.Switch(
            label="ورود با اثر انگشت",
            value=settings.biometric_login_enabled,
        )
        battery_switch = ft.Switch(
            label="مدیریت باتری",
            value=settings.battery_saver,
        )
        theme_switch = ft.Switch(
            label="تم شب / روز",
            value=settings.dark_mode,
            on_change=_on_theme_toggle,
        )

        switches_card.content = ft.Column(
            spacing=4,
            controls=[
                auto_query_switch,
                all_plates_switch,
                biometric_switch,
                battery_switch,
                theme_switch,
            ],
        )

        def _cancel_settings_click(_ev) -> None:
            if settings.dark_mode != prev_dark_mode:
                apply_app_theme(prev_dark_mode, update_page=False)
            page.pop_dialog()
            page.update()

        def _save_settings_click(_ev) -> None:
            settings.api_base_url = (base_url_tf.value or "http://myweb.web.com").strip()
            settings.api_check_path = (check_path_tf.value or "/api/plate/check").strip()
            settings.api_violation_path = (viol_path_tf.value or "/api/violation").strip()
            settings.auto_query_enabled = bool(auto_query_switch.value)
            settings.query_all_plates = bool(all_plates_switch.value)
            settings.biometric_login_enabled = bool(biometric_switch.value)
            settings.battery_saver = bool(battery_switch.value)
            settings.dark_mode = bool(theme_switch.value)
            save_settings(settings)
            api_client.clear_cache()
            if login_screen_ref:
                login_screen_ref[0].set_biometric_enabled(settings.biometric_login_enabled)
            page.pop_dialog()
            apply_app_theme(settings.dark_mode, update_page=True)
            show_snackbar("تنظیمات با موفقیت ذخیره شد.")

        page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("تنظیمات سامانه"),
                content=ft.Container(
                    width=340,
                    content=ft.Column(
                        tight=True,
                        spacing=10,
                        scroll=ft.ScrollMode.AUTO,
                        controls=[
                            switches_card,
                            ft.Divider(height=6, color=p.border),
                            base_url_tf,
                            check_path_tf,
                            viol_path_tf,
                        ],
                    ),
                ),
                actions=[
                    ft.TextButton("انصراف", on_click=_cancel_settings_click),
                    ft.FilledButton("ذخیره تنظیمات", on_click=_save_settings_click),
                ],
            )
        )

    settings_btn.on_click = open_settings_dialog

    # -----------------------------------------------------------------------
    # پایپ‌لاین استنتاج غیرهمزمان و استعلام دستی/خودکار پلاک
    # -----------------------------------------------------------------------
    def _decode_and_infer_sync(frame_bytes: bytes, rotation_degrees: int) -> FrameInferenceResult:
        with Image.open(io.BytesIO(frame_bytes)) as img:
            rgb = img.convert("RGB")
        if rotation_degrees:
            rgb = rotate_pil_clockwise(rgb, rotation_degrees)
        return engine.process_image(rgb)

    async def query_plate_server(
        plate: PlateDetection,
        *,
        force_refresh: bool = False,
    ) -> ApiCheckResult | None:
        plate_key = plate.plate_latin or plate.plate_persian
        if not plate_key:
            return None
        if not force_refresh and plate_key in state.inflight_queries:
            return None

        state.inflight_queries.add(plate_key)
        try:
            result = await api_client.check_plate(
                plate_persian=plate.plate_persian,
                plate_latin=plate.plate_latin,
                confidence=plate.confidence,
                force_refresh=force_refresh,
            )
            state.plate_statuses[plate_key] = result.status

            if state.selected_plate is not None:
                selected_key = state.selected_plate.plate_latin or state.selected_plate.plate_persian
                if selected_key == plate_key:
                    state.last_api_result = result
                    if state.screen_mode == "info":
                        render_server_info_panel()
                        bottom_section_container.update()
                        # در صورت پاسخ yes (مجوز)، پاسخ سرور در پنجره جدید نمایش داده می‌شود
                        if result.status == PlateCheckStatus.APPROVED:
                            show_server_response_dialog(result)

            refresh_overlay_boxes()
            return result
        finally:
            state.inflight_queries.discard(plate_key)

    async def handle_manual_query_click(_e=None) -> None:
        """ارسال دستی پلاک شناسایی‌شده به سرور هنگام فشردن دکمه «استعلام پلاک»."""
        if state.manual_query_busy:
            return
        plate = state.selected_plate
        if plate is None or not (plate.plate_latin or plate.plate_persian):
            show_snackbar(
                "پلاکی شناسایی نشده است؛ ابتدا دوربین را به سمت پلاک خودرو بگیرید.",
                is_error=True,
            )
            return

        if not should_query_plate(
            settings,
            chars_persian=plate.chars_persian,
            plate_persian=plate.plate_persian,
            plate_latin=plate.plate_latin,
        ):
            show_snackbar(
                "سوییچ «استعلام تمام پلاکها» غیرفعال است؛ فقط پلاک‌های عمومی (حرف «ع») استعلام می‌شوند.",
                is_error=True,
            )
            return

        plate_key = plate.plate_latin or plate.plate_persian
        state.manual_query_busy = True
        state.plate_statuses[plate_key] = PlateCheckStatus.PENDING
        state.last_api_result = None
        query_plate_progress.visible = True
        query_plate_btn.disabled = True
        render_server_info_panel()
        bottom_section_container.update()
        refresh_overlay_boxes()

        try:
            res = await query_plate_server(plate, force_refresh=True)
            if res is not None and res.error_message:
                show_snackbar(res.error_message, is_error=True)
            elif res is not None and res.status == PlateCheckStatus.REJECTED:
                show_snackbar("نتیجه استعلام: غیرمجاز (NO)", is_error=True)
            elif res is not None and res.status == PlateCheckStatus.APPROVED:
                show_snackbar("نتیجه استعلام: مجاز (تأیید سرور)")
        finally:
            state.manual_query_busy = False
            query_plate_progress.visible = False
            query_plate_btn.disabled = False
            if state.screen_mode == "info":
                render_server_info_panel()
                bottom_section_container.update()

    query_plate_btn.on_click = handle_manual_query_click

    async def process_camera_frame(frame_bytes: bytes) -> None:
        try:
            rotation_deg = 0
            if page.platform == ft.PagePlatform.ANDROID and state.selected_camera is not None:
                rotation_deg = compute_android_rotation_degrees(
                    sensor_orientation=state.selected_camera.sensor_orientation,
                    device_orientation_degrees=_device_orientation_to_degrees(state.device_orientation),
                    is_front_camera=(
                        state.selected_camera.lens_direction == fc.CameraLensDirection.FRONT
                    ),
                )

            state.last_raw_frame_bytes = frame_bytes
            state.last_raw_frame_rotation = rotation_deg

            # در حالت صفحات تمام‌صفحه یا کادربندی عکس مستندات، فقط آخرین فریم ذخیره می‌شود
            if state.screen_mode != "info":
                return

            result = await asyncio.to_thread(_decode_and_infer_sync, frame_bytes, rotation_deg)
            state.last_inference_ms = result.inference_ms
            if result.inference_ms > 0:
                state.fps = 1000.0 / max(1.0, result.inference_ms)
            hud_fps_text.value = to_persian_digits(f"{state.fps:.1f} FPS ({int(result.inference_ms)}ms)")
            hud_fps_text.update()

            # اعمال فیلتر تثبیت چندفریمی (Multi-Frame Consensus) روی پلاک اصلی
            stabilized_primary: StabilizedPlateResult | None = None
            if result.plates:
                primary = result.plates[0]
                if not engine.single_stage_mode and primary.complete and len(primary.chars_persian) == 8:
                    stabilized_primary = plate_stabilizer.update(
                        primary.chars_persian,
                        primary.confidence,
                    )
                    if stabilized_primary is not None:
                        primary.chars_persian = list(stabilized_primary.chars_persian)
                        primary.plate_persian = stabilized_primary.plate_persian
                        primary.plate_latin = stabilized_primary.plate_latin
                        primary.confidence = stabilized_primary.confidence
                        state.last_stabilized = stabilized_primary
                else:
                    state.last_stabilized = None

            state.active_plates = result.plates

            # ثبت رنگ آبی اولیه؛ ارسال خودکار فقط پس از تثبیت چندفریمی (Multi-Frame Consensus) و تطابق با فیلتر پلاک
            for idx_p, plate in enumerate(result.plates):
                plate_key = plate.plate_latin or plate.plate_persian
                if not plate_key:
                    continue

                if not engine.single_stage_mode and not plate.complete:
                    state.plate_statuses[plate_key] = PlateCheckStatus.PENDING
                    continue

                # پلاک اصلی در حالت دومرحله‌ای تا تأیید چندفریمی نباید استعلام خودکار ارسال کند
                confirmed_for_auto = True
                if idx_p == 0 and not engine.single_stage_mode and stabilized_primary is not None:
                    confirmed_for_auto = stabilized_primary.is_confirmed

                cached = api_client.get_cached(plate_key)
                if cached is not None:
                    state.plate_statuses[plate_key] = cached.status
                elif plate_key not in state.plate_statuses:
                    state.plate_statuses[plate_key] = PlateCheckStatus.PENDING
                    if (
                        confirmed_for_auto
                        and settings.auto_query_enabled
                        and should_query_plate(
                            settings,
                            chars_persian=plate.chars_persian,
                            plate_persian=plate.plate_persian,
                            plate_latin=plate.plate_latin,
                        )
                    ):
                        asyncio.create_task(query_plate_server(plate))
                elif (
                    confirmed_for_auto
                    and plate_key not in state.inflight_queries
                    and state.plate_statuses.get(plate_key) == PlateCheckStatus.PENDING
                    and settings.auto_query_enabled
                    and should_query_plate(
                        settings,
                        chars_persian=plate.chars_persian,
                        plate_persian=plate.plate_persian,
                        plate_latin=plate.plate_latin,
                    )
                ):
                    asyncio.create_task(query_plate_server(plate))

            # انتخاب اصلی‌ترین پلاک دیده‌شده برای نمایش در بخش پایین صفحه
            if result.plates:
                primary = result.plates[0]
                primary_key = primary.plate_latin or primary.plate_persian
                prev_key = (
                    (state.selected_plate.plate_latin or state.selected_plate.plate_persian)
                    if state.selected_plate
                    else ""
                )
                state.selected_plate = primary
                if primary_key != prev_key:
                    state.last_api_result = api_client.get_cached(primary_key) if primary_key else None
                if state.screen_mode == "info":
                    render_server_info_panel()
                    bottom_section_container.update()

            refresh_overlay_boxes()
        except Exception as exc:
            logger.exception("Frame processing error: %s", exc)
        finally:
            state.inference_busy = False

    def handle_stream_image(e: fc.CameraImageEvent) -> None:
        if state.is_paused or not engine.is_ready:
            return
        now = time.monotonic()
        if state.inference_busy or (now - state.last_inference_ts) < settings.min_inference_interval_sec:
            return
        state.inference_busy = True
        state.last_inference_ts = now
        asyncio.create_task(process_camera_frame(e.bytes))

    camera_control.on_stream_image = handle_stream_image

    def handle_camera_state_change(e: fc.CameraStateEvent) -> None:
        if e.device_orientation is not None:
            state.device_orientation = e.device_orientation
        if e.has_error and e.error_description:
            hud_status_text.value = f"خطای دوربین: {e.error_description}"
            hud_status_text.update()

    camera_control.on_state_change = handle_camera_state_change

    # -----------------------------------------------------------------------
    # کنترل‌های توقف اسکن و چراغ قوه
    # -----------------------------------------------------------------------
    def toggle_pause(_e=None) -> None:
        state.is_paused = not state.is_paused
        if not state.is_paused:
            plate_stabilizer.reset()
        pause_btn.icon = ft.Icons.PLAY_CIRCLE_OUTLINE if state.is_paused else ft.Icons.PAUSE_CIRCLE_OUTLINE
        hud_status_text.value = "اسکن متوقف شد" if state.is_paused else "در حال اسکن زنده..."
        pause_btn.update()
        hud_status_text.update()

    async def toggle_flash(_e=None) -> None:
        if not state.is_camera_ready:
            return
        try:
            state.torch_on = not state.torch_on
            mode = fc.FlashMode.TORCH if state.torch_on else fc.FlashMode.OFF
            await camera_control.set_flash_mode(mode)
            flash_btn.icon = ft.Icons.FLASH_ON if state.torch_on else ft.Icons.FLASH_OFF
            flash_btn.update()
        except Exception as exc:
            logger.warning("Flash toggle unsupported: %s", exc)

    pause_btn.on_click = toggle_pause
    flash_btn.on_click = toggle_flash

    # -----------------------------------------------------------------------
    # حالت تست شبیه‌سازی برای اجرا روی کامپیوتر (دسکتاپ) بدون دوربین اندروید
    # -----------------------------------------------------------------------
    async def run_demo_detection(is_no: bool) -> None:
        demo_chars = ["1", "2", "ع", "3", "4", "5", "1", "1"]
        demo_crop = await asyncio.to_thread(_create_demo_jpeg, (235, 238, 245), (280, 72))
        plate_stabilizer.reset()
        plate_stabilizer.update(demo_chars, 0.92)
        state.last_stabilized = plate_stabilizer.update(demo_chars, 0.96)
        demo_plate = PlateDetection(
            norm_box=(0.2, 0.38, 0.8, 0.56),
            pixel_box=(128, 180, 512, 270),
            confidence=0.94,
            plate_persian="".join(demo_chars),
            plate_latin="12EIN34511",
            chars_persian=demo_chars,
            complete=True,
            plate_crop_jpeg=demo_crop,
        )
        state.active_plates = [demo_plate]
        state.selected_plate = demo_plate
        state.plate_statuses[demo_plate.plate_latin] = PlateCheckStatus.PENDING
        state.last_api_result = None
        if state.screen_mode == "info":
            render_server_info_panel()
            bottom_section_container.update()
        refresh_overlay_boxes()

        # در حالت دستی، فقط پلاک شناسایی می‌شود و کادر آبی می‌ماند تا کاربر «استعلام پلاک» بزند؛
        # ولی اگر دکمه شبیه‌سازی پاسخ زده شد، نتیجه آزمایشی نمایش داده می‌شود.
        await asyncio.sleep(0.7)

        state.plate_statuses[demo_plate.plate_latin] = (
            PlateCheckStatus.REJECTED if is_no else PlateCheckStatus.APPROVED
        )
        state.last_api_result = ApiCheckResult(
            plate_persian=demo_plate.plate_persian,
            plate_latin=demo_plate.plate_latin,
            status=state.plate_statuses[demo_plate.plate_latin],
            raw_response="NO" if is_no else "YES",
            details={
                "پاسخ سرور": "NO" if is_no else "OK",
                "وضعیت مجوز": "غیرمجاز (تخلف)" if is_no else "مجاز برای تردد",
                "ناوگان": "اتوبوس بین‌شهری (تست شبیه‌ساز)",
            },
            latency_ms=42.0,
            checked_at=time.strftime("%H:%M:%S"),
        )
        if state.screen_mode == "info":
            render_server_info_panel()
            bottom_section_container.update()
        refresh_overlay_boxes()

    # -----------------------------------------------------------------------
    # راه‌اندازی مدل ONNX و دوربین اندروید
    # -----------------------------------------------------------------------
    async def init_camera_pipeline(_e=None) -> None:
        retry_camera_btn.visible = False
        camera_placeholder_text.value = "در حال دریافت مجوز دوربین..."
        camera_fallback_overlay.visible = True
        page.update()

        is_mobile_or_web = page.platform in (
            ft.PagePlatform.ANDROID,
            ft.PagePlatform.IOS,
        ) or page.web

        if not is_mobile_or_web:
            camera_placeholder_text.value = (
                "حالت دسکتاپ: دوربین زنده Flet روی اندروید و iOS فعال است.\n"
                "برای بررسی رنگ کادرها (آبی → قرمز/سبز)، ثبت تخلف و گزارش PDF از دکمه‌های زیر استفاده کنید:"
            )
            desktop_demo_controls.visible = True
            hud_status_text.value = "حالت پیش‌نمایش دسکتاپ"
            page.update()
            return

        if page.platform in (ft.PagePlatform.ANDROID, ft.PagePlatform.IOS):
            try:
                perm = await permission_handler.request(fph.Permission.CAMERA)
                if perm not in (fph.PermissionStatus.GRANTED, fph.PermissionStatus.LIMITED):
                    camera_placeholder_text.value = (
                        "دسترسی به دوربین تأیید نشد.\n"
                        "برای اسکن زنده پلاک، لطفاً مجوز دوربین را فعال کنید."
                    )
                    retry_camera_btn.visible = True
                    hud_status_text.value = "نیاز به مجوز دوربین"
                    page.update()
                    return
            except Exception as exc:
                logger.warning("Permission request error: %s", exc)

        try:
            state.cameras = await camera_control.get_available_cameras()
            if not state.cameras:
                camera_placeholder_text.value = "هیچ دوربینی روی دستگاه شناسایی نشد."
                retry_camera_btn.visible = True
                page.update()
                return

            back_cam = next(
                (c for c in state.cameras if c.lens_direction == fc.CameraLensDirection.BACK),
                state.cameras[0],
            )
            state.selected_camera = back_cam

            await camera_control.initialize(
                description=back_cam,
                resolution_preset=fc.ResolutionPreset.MEDIUM,
                enable_audio=False,
                image_format_group=fc.ImageFormatGroup.JPEG,
            )

            try:
                await camera_control.lock_capture_orientation(ft.DeviceOrientation.PORTRAIT_UP)
            except Exception:
                pass

            state.is_camera_ready = True
            camera_fallback_overlay.visible = False

            if await camera_control.supports_image_streaming():
                await camera_control.start_image_stream()
                state.is_streaming = True
                hud_status_text.value = (
                    "در حال اسکن زنده..." if engine.is_ready else "دوربین فعال (مدل ONNX یافت نشد)"
                )
            else:
                hud_status_text.value = "استریم زنده روی این دوربین پشتیبانی نشد"
            page.update()

            # پیش‌گرم‌سازی بی‌صدای GPS (فقط در صورت داشتن مجوز قبلی) برای آماده‌بودن فوری هنگام ثبت تخلف
            asyncio.create_task(fetch_current_gps(request_if_needed=False))
        except Exception as exc:
            logger.exception("Camera initialization failed")
            camera_placeholder_text.value = f"خطا در راه‌اندازی دوربین:\n{exc}"
            retry_camera_btn.visible = True
            page.update()

    retry_camera_btn.on_click = init_camera_pipeline

    async def preload_models_in_background() -> None:
        """بارگذاری بی‌صدای مدل‌های ONNX و شمارنده تخلفات همزمان با نمایش صفحه ورود."""
        await refresh_violation_counter(update_ui=False)

        loaded = await asyncio.to_thread(engine.load)
        if loaded:
            mode_label = "تک‌مرحله‌ای" if engine.single_stage_mode else "دومرحله‌ای YOLO26"
            hud_status_text.value = f"مدل آماده ({mode_label})"
        else:
            hud_status_text.value = "مدل ONNX در پوشه weights نیست"
            if engine.load_error:
                server_details_column.controls = [
                    ft.Text(engine.load_error, size=12, color=ft.Colors.AMBER_400)
                ]
        render_server_info_panel()
        if state.is_authenticated:
            page.update()

    # -----------------------------------------------------------------------
    # احراز هویت بیومتریک (اثر انگشت گوشی) با flet-local-auth
    # -----------------------------------------------------------------------
    async def handle_biometric_auth() -> tuple[bool, str]:
        if local_auth is None or fla is None:
            return False, "ماژول احراز هویت بیومتریک روی این دستگاه در دسترس نیست."

        if page.platform in (ft.PagePlatform.LINUX,) or page.web:
            return False, "ورود با اثر انگشت فقط روی گوشی اندروید/iOS پشتیبانی می‌شود."

        try:
            supported = await local_auth.is_device_supported()
            can_check = await local_auth.can_check_biometrics()
            if not (supported or can_check):
                return False, "این گوشی فاقد سخت‌افزار اثر انگشت یا قفل بیومتریک فعال است."

            ok = await local_auth.authenticate(
                "برای ورود به پلاک‌خوان هوشمند، اثر انگشت خود را اسکن کنید",
                biometric_only=False,
                android_messages=fla.AndroidAuthMessages(
                    sign_in_title="ورود با اثر انگشت",
                    sign_in_hint="حسگر اثر انگشت گوشی را لمس کنید",
                    cancel_button="انصراف",
                ),
            )
            if ok:
                return True, "احراز هویت بیومتریک تأیید شد."
            return False, "تأیید اثر انگشت انجام نشد."
        except Exception as exc:
            err_code = getattr(getattr(exc, "code", None), "value", "") or str(exc)
            if "cancel" in err_code.lower():
                return False, ""
            if "notEnrolled" in err_code or "noCredentials" in err_code:
                return False, "ابتدا اثر انگشت خود را در تنظیمات امنیتی گوشی ثبت کنید."
            if "lockout" in err_code.lower():
                return False, "تلاش‌های ناموفق زیاد بود؛ لحظاتی دیگر مجدد امتحان کنید."
            logger.warning("Biometric authentication error: %s", exc)
            return False, f"خطا در احراز هویت اثر انگشت: {err_code}"

    # -----------------------------------------------------------------------
    # چیدمان نهایی: صفحه ورود (لاگین) -> ورود به لایه اصلی دوبخشی + صفحات تمام‌صفحه
    # -----------------------------------------------------------------------
    main_split_view = ft.Column(
        expand=True,
        spacing=0,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        controls=[
            top_camera_section,
            bottom_section_container,
        ],
    )

    main_app_stack = ft.Stack(
        expand=True,
        fit=ft.StackFit.EXPAND,
        controls=[
            main_split_view,
            fullscreen_overlay,
            evidence_capture_overlay,
        ],
    )

    app_root_container = ft.Container(expand=True)

    async def handle_login_success(username: str) -> None:
        state.is_authenticated = True
        state.current_username = username
        app_root_container.content = main_app_stack
        render_server_info_panel()
        page.update()

        if not state.camera_started:
            state.camera_started = True
            await init_camera_pipeline()

    login_screen = LoginScreen(
        on_login_success=handle_login_success,
        on_biometric_login=handle_biometric_auth,
        biometric_enabled=settings.biometric_login_enabled,
        dark_mode=settings.dark_mode,
    )
    login_screen_ref.append(login_screen)
    app_root_container.content = login_screen

    page.add(
        ft.SafeArea(
            expand=True,
            content=app_root_container,
        )
    )

    # پیش‌بارگذاری سریع مدل‌های ONNX در پس‌زمینه هنگام نمایش صفحه ورود
    asyncio.create_task(preload_models_in_background())

    # در صورت فعال بودن ورود با اثر انگشت روی گوشی، پنجره اثر انگشت به‌صورت خودکار باز می‌شود
    if settings.biometric_login_enabled and page.platform in (
        ft.PagePlatform.ANDROID,
        ft.PagePlatform.IOS,
    ):
        async def _auto_biometric_prompt() -> None:
            await asyncio.sleep(0.35)
            if not state.is_authenticated:
                await login_screen.trigger_biometric_login()

        asyncio.create_task(_auto_biometric_prompt())


if __name__ == "__main__":
    ft.run(main)
