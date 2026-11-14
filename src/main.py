"""نقطه ورود اصلی اپلیکیشن اندروید «پلاک‌خوان هوشمند» با Flet 1.0 + ONNX Runtime.

ساختار صفحه اصلی:
- بخش بالا (55% صفحه): نمایش زنده دوربین گوشی + رسم کادرهای رنگی (آبی -> قرمز/سبز)
- بخش پایین (45% صفحه):
    1) حالت پیش‌فرض: نمایش گرافیکی استاندارد پلاک ایران، اطلاعات دریافتی از سرور myweb.web.com
       و دکمه‌های «ثبت تخلف» و «بازبینی تخلفات»
    2) حالت ثبت تخلف: فرم ثبت تخلف همراه با پیوست تصویر برش‌خورده پلاک و عکس مستندات (دوربین/گالری)
    3) حالت بازبینی تخلفات: فهرست تخلفات ثبت‌شده با امکان مشاهده تصاویر، ارسال مجدد و حذف
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

# اطمینان از در دسترس بودن ماژول‌های داخل پوشه src در همه محیط‌های اجرا
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AppSettings, to_persian_digits
from core.geometry import compute_android_rotation_degrees, rotate_pil_clockwise
from core.plate_detector import FrameInferenceResult, PersianPlateEngine, PlateDetection
from services.api_client import (
    ApiCheckResult,
    PlateApiClient,
    PlateCheckStatus,
    ViolationRecord,
    delete_violation_record,
    load_violation_records,
    read_violation_image_bytes,
)
from ui.components import (
    build_iranian_plate_widget,
    build_overlay_shapes,
    status_to_badge_info,
)
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

    # آخرین تشخیص‌ها و وضعیت سرور برای هر پلاک
    active_plates: list[PlateDetection] = field(default_factory=list)
    plate_statuses: dict[str, PlateCheckStatus] = field(default_factory=dict)
    selected_plate: PlateDetection | None = None
    last_api_result: ApiCheckResult | None = None

    # حالت بخش پایینی صفحه: "info" | "violation" | "history"
    bottom_mode: str = "info"


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
    page.title = "پلاک‌خوان هوشمند"
    page.rtl = True
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.spacing = 0
    page.bgcolor = ft.Colors.GREY_950 if hasattr(ft.Colors, "GREY_950") else "#0D1117"
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.BLUE)

    settings = AppSettings()
    state = AppRuntimeState()
    engine = PersianPlateEngine(settings=settings)
    api_client = PlateApiClient(settings=settings)
    permission_handler = fph.PermissionHandler()
    file_picker = ft.FilePicker()
    if hasattr(page, "services"):
        page.services.append(file_picker)

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
        tooltip="تنظیمات آدرس سرور",
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
            controls=[
                ft.Row(
                    spacing=8,
                    controls=[
                        ft.Icon(ft.Icons.MEMORY, size=16, color=ft.Colors.LIGHT_BLUE_300),
                        hud_status_text,
                        ft.Text("|", color=ft.Colors.WHITE_38),
                        hud_fps_text,
                    ],
                ),
                ft.Row(
                    spacing=0,
                    controls=[flash_btn, pause_btn, settings_btn],
                ),
            ],
        ),
    )

    top_camera_section = ft.Container(
        expand=5,
        bgcolor=ft.Colors.BLACK,
        content=ft.Stack(
            expand=True,
            controls=[
                camera_control,
                camera_fallback_overlay,
                top_hud_bar,
            ],
        ),
    )

    # -----------------------------------------------------------------------
    # کنترل‌های بخش پایین صفحه (اطلاعات سرور <-> ثبت تخلف <-> بازبینی تخلفات)
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
        padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        border_radius=ft.BorderRadius.all(10),
        bgcolor=ft.Colors.BLUE_GREY_800,
        content=ft.Row(
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[status_chip_icon, status_chip_label],
        ),
    )

    server_details_column = ft.Column(
        spacing=6,
        controls=[
            ft.Text(
                "دوربین گوشی را به سمت پلاک خودرو بگیرید. پس از تشخیص محلی (کادر آبی)، "
                "نتیجه استعلام از سرور myweb.web.com در این بخش نمایش داده می‌شود.",
                size=12,
                color=ft.Colors.WHITE_70,
            )
        ],
    )

    server_info_card = ft.Container(
        expand=True,
        padding=ft.Padding.all(10),
        border_radius=ft.BorderRadius.all(10),
        bgcolor="#161B22",
        border=ft.Border.all(1, ft.Colors.BLUE_GREY_800),
        content=ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            controls=[server_details_column],
        ),
    )

    register_violation_btn = ft.FilledButton(
        content=ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=6,
            controls=[
                ft.Icon(ft.Icons.REPORT_PROBLEM, size=19),
                ft.Text("ثبت تخلف", size=14, weight=ft.FontWeight.BOLD),
            ],
        ),
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.RED_600,
            color=ft.Colors.WHITE,
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding.symmetric(horizontal=10, vertical=13),
        ),
        expand=3,
    )

    review_violations_label = ft.Text("بازبینی تخلفات", size=13, weight=ft.FontWeight.BOLD)
    review_violations_btn = ft.FilledButton(
        content=ft.Row(
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=6,
            controls=[
                ft.Icon(ft.Icons.HISTORY, size=19),
                review_violations_label,
            ],
        ),
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.BLUE_GREY_700,
            color=ft.Colors.WHITE,
            shape=ft.RoundedRectangleBorder(radius=12),
            padding=ft.Padding.symmetric(horizontal=10, vertical=13),
        ),
        expand=2,
    )

    server_info_view = ft.Container(
        expand=True,
        padding=ft.Padding.all(12),
        content=ft.Column(
            spacing=10,
            expand=True,
            controls=[
                plate_widget_holder,
                status_chip,
                server_info_card,
                ft.Row(
                    spacing=8,
                    controls=[
                        register_violation_btn,
                        review_violations_btn,
                    ],
                ),
            ],
        ),
    )

    # -----------------------------------------------------------------------
    # توابع کمکی بروزرسانی رابط کاربری و تعویض بخش پایینی صفحه
    # -----------------------------------------------------------------------
    def show_snackbar(message: str, is_error: bool = False) -> None:
        page.show_dialog(
            ft.SnackBar(
                content=ft.Text(message, color=ft.Colors.WHITE),
                bgcolor=ft.Colors.RED_700 if is_error else ft.Colors.GREEN_700,
            )
        )

    async def refresh_violation_counter(update_ui: bool = True) -> list[ViolationRecord]:
        records = await asyncio.to_thread(load_violation_records)
        count = len(records)
        review_violations_label.value = (
            f"بازبینی تخلفات ({to_persian_digits(str(count))})"
            if count > 0
            else "بازبینی تخلفات"
        )
        if update_ui and state.bottom_mode == "info":
            review_violations_label.update()
        return records

    def refresh_overlay_boxes() -> None:
        mirror_x = (
            state.selected_camera is not None
            and state.selected_camera.lens_direction == fc.CameraLensDirection.FRONT
        )
        overlay_canvas.shapes = build_overlay_shapes(
            plates=state.active_plates,
            plate_statuses=state.plate_statuses,
            canvas_width=state.canvas_width,
            canvas_height=state.canvas_height,
            mirror_x=mirror_x,
        )
        overlay_canvas.update()

    def render_server_info_panel() -> None:
        plate = state.selected_plate
        api_res = state.last_api_result

        if plate is None:
            plate_widget_holder.content = build_iranian_plate_widget([], PlateCheckStatus.PENDING)
            status_chip.bgcolor = ft.Colors.BLUE_GREY_800
            status_chip_icon.name = ft.Icons.SEARCH
            status_chip_label.value = "در انتظار تشخیص پلاک خودرو..."
            return

        plate_key = plate.plate_latin or plate.plate_persian
        current_status = state.plate_statuses.get(plate_key, PlateCheckStatus.PENDING)
        plate_widget_holder.content = build_iranian_plate_widget(plate.chars_persian, current_status)

        title, bg_color, icon_name = status_to_badge_info(current_status)
        status_chip.bgcolor = bg_color
        status_chip_icon.name = icon_name
        status_chip_label.value = title

        rows: list[ft.Control] = [
            _info_row("پلاک تشخیص‌داده‌شده", plate.display_text),
            _info_row("کد لاتین ارسالی به سرور", plate.plate_latin or "---"),
            _info_row("دقت مدل محلی", to_persian_digits(f"{int(plate.confidence * 100)}٪")),
        ]

        if api_res is not None:
            rows.append(ft.Divider(height=8, color=ft.Colors.BLUE_GREY_800))
            if api_res.error_message:
                rows.append(_info_row("وضعیت ارتباط", api_res.error_message, value_color=ft.Colors.ORANGE_300))
            else:
                rows.append(_info_row("زمان استعلام", to_persian_digits(api_res.checked_at)))
                rows.append(_info_row("سرعت پاسخ سرور", to_persian_digits(f"{int(api_res.latency_ms)} میلی‌ثانیه")))
                for key, val in api_res.details.items():
                    rows.append(_info_row(str(key), str(val)))
        else:
            if not engine.single_stage_mode and not plate.complete:
                rows.append(_info_row("وضعیت", "در حال خوانش کامل ۸ کاراکتر پلاک (کادر آبی)..."))
            else:
                rows.append(_info_row("آدرس API", settings.check_url))
                rows.append(_info_row("وضعیت", "در حال ارسال پلاک به سرور..."))

        server_details_column.controls = rows

    def _info_row(label: str, value: str, value_color: str = ft.Colors.WHITE) -> ft.Row:
        return ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.START,
            controls=[
                ft.Text(f"{label}:", size=12, color=ft.Colors.WHITE_60),
                ft.Text(
                    value,
                    size=12,
                    color=value_color,
                    weight=ft.FontWeight.W_600,
                    text_align=ft.TextAlign.LEFT,
                    expand=True,
                ),
            ],
        )

    def switch_to_info_view() -> None:
        state.bottom_mode = "info"
        bottom_section_container.content = server_info_view
        render_server_info_panel()
        bottom_section_container.update()

    def switch_to_violation_view(_e=None) -> None:
        state.bottom_mode = "violation"
        plate_text = state.selected_plate.display_text if state.selected_plate else ""
        crop_bytes = (
            state.selected_plate.plate_crop_jpeg
            if state.selected_plate is not None
            else None
        )
        violation_panel.prepare(
            plate_display=plate_text,
            has_crop=crop_bytes is not None,
            plate_crop_bytes=crop_bytes,
        )
        bottom_section_container.content = violation_panel
        bottom_section_container.update()

    async def handle_capture_evidence_photo() -> bytes | None:
        """ثبت عکس مستندات از دوربین زنده (با پشتیبان آخرین فریم دوربین در صورت محدودیت سخت‌افزاری)."""
        if state.is_camera_ready:
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

            if captured_bytes:
                show_snackbar("عکس مستندات با دوربین ثبت شد.")
                return captured_bytes
            show_snackbar("ثبت عکس با دوربین ناموفق بود؛ می‌توانید از گالری انتخاب کنید.", is_error=True)
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
        violation_type: str,
        notes: str,
        attach_crop: bool,
        evidence_bytes: bytes | None,
    ) -> None:
        plate_latin = state.selected_plate.plate_latin if state.selected_plate else ""
        crop_bytes = (
            state.selected_plate.plate_crop_jpeg
            if (attach_crop and state.selected_plate is not None)
            else None
        )
        result = await api_client.submit_violation(
            plate_persian=plate_text,
            plate_latin=plate_latin,
            violation_type=violation_type,
            description=notes,
            plate_crop_jpeg=crop_bytes,
            evidence_image_jpeg=evidence_bytes,
        )
        await refresh_violation_counter(update_ui=False)
        show_snackbar(result.message, is_error=not result.success)
        if result.success or result.saved_locally:
            switch_to_info_view()

    async def reload_history_panel() -> None:
        records = await refresh_violation_counter(update_ui=False)
        violation_history_panel.set_records(records)
        if state.bottom_mode == "history":
            bottom_section_container.update()

    async def switch_to_history_view(_e=None) -> None:
        state.bottom_mode = "history"
        records = await refresh_violation_counter(update_ui=False)
        violation_history_panel.set_records(records)
        bottom_section_container.content = violation_history_panel
        bottom_section_container.update()

    async def open_violation_detail_dialog(record: ViolationRecord) -> None:
        """نمایش جزئیات کامل یک تخلف ثبت‌شده شامل تصویر برش پلاک، عکس مستندات، ارسال مجدد و حذف."""
        plate_bytes = await asyncio.to_thread(read_violation_image_bytes, record.plate_image_file)
        evidence_bytes = await asyncio.to_thread(
            read_violation_image_bytes,
            record.evidence_image_file,
        )

        detail_controls: list[ft.Control] = [
            _info_row("پلاک", record.plate),
            _info_row("نوع تخلف", record.violation_type, value_color=ft.Colors.AMBER_300),
            _info_row("زمان ثبت", to_persian_digits(record.local_time or "---")),
            _info_row(
                "وضعیت ارسال",
                "ارسال‌شده به سرور" if record.synced else "ذخیره‌شده روی گوشی (ارسال‌نشده)",
                value_color=ft.Colors.GREEN_400 if record.synced else ft.Colors.ORANGE_300,
            ),
        ]
        if record.description:
            detail_controls.append(_info_row("توضیحات", record.description))
        if record.server_response:
            detail_controls.append(_info_row("پاسخ سرور", record.server_response[:160]))

        if plate_bytes:
            detail_controls.extend(
                [
                    ft.Divider(height=10, color=ft.Colors.BLUE_GREY_800),
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
                    ft.Divider(height=10, color=ft.Colors.BLUE_GREY_800),
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
                    color=ft.Colors.WHITE_60,
                )
            )

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

        dialog_actions: list[ft.Control] = [
            ft.TextButton(
                "حذف",
                style=ft.ButtonStyle(color=ft.Colors.RED_400),
                on_click=handle_delete,
            ),
        ]
        if not record.synced:
            dialog_actions.append(
                ft.FilledButton(
                    "ارسال مجدد به سرور",
                    style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
                    on_click=handle_retry_sync,
                )
            )
        dialog_actions.append(
            ft.FilledButton("بستن", on_click=lambda _: page.pop_dialog())
        )

        page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("جزئیات تخلف ثبت‌شده"),
                content=ft.Container(
                    width=340,
                    content=ft.Column(
                        tight=True,
                        spacing=8,
                        scroll=ft.ScrollMode.AUTO,
                        controls=detail_controls,
                    ),
                ),
                actions=dialog_actions,
            )
        )

    violation_panel = ViolationRegistrationPanel(
        on_back=switch_to_info_view,
        on_submit=handle_violation_submit,
        on_capture_evidence=handle_capture_evidence_photo,
        on_pick_evidence=handle_pick_evidence_photo,
    )
    violation_history_panel = ViolationHistoryPanel(
        on_back=switch_to_info_view,
        on_refresh=reload_history_panel,
        on_select_record=open_violation_detail_dialog,
    )

    register_violation_btn.on_click = switch_to_violation_view
    review_violations_btn.on_click = switch_to_history_view

    bottom_section_container = ft.Container(
        expand=4,
        bgcolor="#0F141B",
        border=ft.Border(top=ft.BorderSide(width=1, color=ft.Colors.BLUE_GREY_800)),
        content=server_info_view,
    )

    # -----------------------------------------------------------------------
    # دیالوگ تنظیمات سریع سرور (آدرس API قابل تغییر در صورت نیاز)
    # -----------------------------------------------------------------------
    def open_settings_dialog(_e=None) -> None:
        base_url_tf = ft.TextField(label="آدرس پایه سرور API", value=settings.api_base_url)
        check_path_tf = ft.TextField(label="مسیر استعلام پلاک", value=settings.api_check_path)
        viol_path_tf = ft.TextField(label="مسیر ثبت تخلف", value=settings.api_violation_path)

        def save_settings(_ev) -> None:
            settings.api_base_url = (base_url_tf.value or "http://myweb.web.com").strip()
            settings.api_check_path = (check_path_tf.value or "/api/plate/check").strip()
            settings.api_violation_path = (viol_path_tf.value or "/api/violation").strip()
            api_client.clear_cache()
            page.pop_dialog()
            show_snackbar("تنظیمات سرور ذخیره شد.")

        page.show_dialog(
            ft.AlertDialog(
                title=ft.Text("تنظیمات اتصال به سرور"),
                content=ft.Column(
                    tight=True,
                    spacing=10,
                    controls=[base_url_tf, check_path_tf, viol_path_tf],
                ),
                actions=[
                    ft.TextButton("انصراف", on_click=lambda _: page.pop_dialog()),
                    ft.FilledButton("ذخیره", on_click=save_settings),
                ],
            )
        )

    settings_btn.on_click = open_settings_dialog

    # -----------------------------------------------------------------------
    # پایپ‌لاین استنتاج غیرهمزمان روی فریم‌های زنده دوربین
    # -----------------------------------------------------------------------
    def _decode_and_infer_sync(frame_bytes: bytes, rotation_degrees: int) -> FrameInferenceResult:
        with Image.open(io.BytesIO(frame_bytes)) as img:
            rgb = img.convert("RGB")
        if rotation_degrees:
            rgb = rotate_pil_clockwise(rgb, rotation_degrees)
        return engine.process_image(rgb)

    async def query_plate_server(plate: PlateDetection) -> None:
        plate_key = plate.plate_latin or plate.plate_persian
        if not plate_key:
            return

        result = await api_client.check_plate(
            plate_persian=plate.plate_persian,
            plate_latin=plate.plate_latin,
            confidence=plate.confidence,
        )
        state.plate_statuses[plate_key] = result.status

        if state.selected_plate is not None:
            selected_key = state.selected_plate.plate_latin or state.selected_plate.plate_persian
            if selected_key == plate_key:
                state.last_api_result = result
                if state.bottom_mode == "info":
                    render_server_info_panel()
                    bottom_section_container.update()

        refresh_overlay_boxes()

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

            result = await asyncio.to_thread(_decode_and_infer_sync, frame_bytes, rotation_deg)
            state.last_inference_ms = result.inference_ms
            if result.inference_ms > 0:
                state.fps = 1000.0 / max(1.0, result.inference_ms)
            hud_fps_text.value = to_persian_digits(f"{state.fps:.1f} FPS ({int(result.inference_ms)}ms)")
            hud_fps_text.update()

            state.active_plates = result.plates

            # ثبت رنگ آبی اولیه برای پلاک‌های جدید و ارسال غیرهمزمان پلاک کامل (۸ کاراکتر) به سرور
            for plate in result.plates:
                plate_key = plate.plate_latin or plate.plate_persian
                if not plate_key:
                    continue

                # در حالت دومرحله‌ای تا تکمیل ۸ کاراکتر پلاک، کادر آبی می‌ماند تا استعلام ناقص به سرور ارسال نشود
                if not engine.single_stage_mode and not plate.complete:
                    state.plate_statuses[plate_key] = PlateCheckStatus.PENDING
                    continue

                cached = api_client.get_cached(plate_key)
                if cached is not None:
                    state.plate_statuses[plate_key] = cached.status
                elif plate_key not in state.plate_statuses:
                    # بلافاصله پس از تشخیص محلی، رنگ کادر آبی می‌شود و سپس به سرور استعلام می‌رود
                    state.plate_statuses[plate_key] = PlateCheckStatus.PENDING
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
                if state.bottom_mode == "info":
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
            # حذف فریم‌های اضافی (Backpressure) جهت حفظ روانی ۶۰ فریم بر ثانیه دوربین
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
        demo_chars = ["1", "2", "ب", "3", "4", "5", "1", "1"]
        demo_crop = await asyncio.to_thread(_create_demo_jpeg, (235, 238, 245), (280, 72))
        demo_plate = PlateDetection(
            norm_box=(0.2, 0.38, 0.8, 0.56),
            pixel_box=(128, 180, 512, 270),
            confidence=0.94,
            plate_persian="".join(demo_chars),
            plate_latin="12B34511",
            chars_persian=demo_chars,
            complete=True,
            plate_crop_jpeg=demo_crop,
        )
        state.active_plates = [demo_plate]
        state.selected_plate = demo_plate
        state.plate_statuses[demo_plate.plate_latin] = PlateCheckStatus.PENDING
        state.last_api_result = None
        if state.bottom_mode == "info":
            render_server_info_panel()
            bottom_section_container.update()
        refresh_overlay_boxes()

        # مکث کوتاه برای نمایش کادر آبی پیش از پاسخ سرور
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
                "خودرو": "پژو ۲۰۷ سفید (تست شبیه‌ساز)",
            },
            latency_ms=42.0,
            checked_at=time.strftime("%H:%M:%S"),
        )
        if state.bottom_mode == "info":
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
                "برای بررسی رنگ کادرها (آبی → قرمز/سبز)، ثبت تخلف و بازبینی تخلفات از دکمه‌های زیر استفاده کنید:"
            )
            desktop_demo_controls.visible = True
            hud_status_text.value = "حالت پیش‌نمایش دسکتاپ"
            page.update()
            return

        # درخواست مجوز زمان اجرای دوربین روی اندروید
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

            # انتخاب خودکار دوربین عقب گوشی
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
        except Exception as exc:
            logger.exception("Camera initialization failed")
            camera_placeholder_text.value = f"خطا در راه‌اندازی دوربین:\n{exc}"
            retry_camera_btn.visible = True
            page.update()

    retry_camera_btn.on_click = init_camera_pipeline

    async def bootstrap_app() -> None:
        # 1. بارگذاری شمارنده تخلفات ثبت‌شده قبلی
        await refresh_violation_counter(update_ui=False)

        # 2. بارگذاری غیرهمزمان مدل‌های ONNX در ترد جداگانه تا رابط کاربری قفل نشود
        loaded = await asyncio.to_thread(engine.load)
        if loaded:
            mode_label = "تک‌مرحله‌ای" if engine.single_stage_mode else "دومرحله‌ای YOLO26"
            hud_status_text.value = f"مدل آماده ({mode_label})"
        else:
            hud_status_text.value = "مدل ONNX در پوشه weights نیست"
            if engine.load_error:
                server_details_column.controls = [
                    ft.Text(engine.load_error, size=12, color=ft.Colors.AMBER_300)
                ]
        page.update()

        # 3. فعال‌سازی دوربین زنده گوشی
        await init_camera_pipeline()

    # -----------------------------------------------------------------------
    # چیدمان نهایی صفحه اصلی (دو بخش بالا و پایین)
    # -----------------------------------------------------------------------
    page.add(
        ft.SafeArea(
            expand=True,
            content=ft.Column(
                expand=True,
                spacing=0,
                controls=[
                    top_camera_section,
                    bottom_section_container,
                ],
            ),
        )
    )

    await bootstrap_app()


if __name__ == "__main__":
    ft.run(main)
