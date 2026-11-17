"""تولید گزارش رسمی PDF فارسی از تخلف ثبت‌شده (صفحه A4 استاندارد با تصویر پلاک و مستندات).

ویژگی‌ها:
- بدون هیچ کتابخانه PDF یا Rust اضافی (فقط با Pillow که از قبل در اپلیکیشن موجود است)
- استفاده از موتور شکل‌دهی فارسی خالص پایتون (`core.persian_text`) و `ImageFont.Layout.BASIC`
  برای خروجی ۱۰۰٪ یکسان و بی‌نقص در اندروید و دسکتاپ
- طراحی رسمی A4 شامل: سربرگ با آیکون برنامه، پلاک گرافیکی ایران، جدول کامل اطلاعات تخلف با تاریخ شمسی و ساعت،
  تصویر برش‌خورده پلاک و تصویر مستندات
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from config import (
    APP_DIR,
    get_writable_data_dir,
    parse_iranian_plate_text,
    to_persian_digits,
)
from core.jalali import format_jalali_date, format_jalali_time, now_iran_datetime
from core.persian_text import to_visual_rtl, wrap_persian_lines
from services.api_client import ViolationRecord

logger = logging.getLogger(__name__)

# ابعاد استاندارد A4 در دقت 150 DPI (210mm x 297mm)
PAGE_W = 1240
PAGE_H = 1754
PAGE_DPI = 150.0
MARGIN = 64

# پالت رنگی رسمی گزارش
COLOR_BG = (255, 255, 255)
COLOR_PRIMARY = (0, 101, 160)        # آبی برند مطابق آیکون 2.png
COLOR_PRIMARY_DARK = (11, 61, 117)
COLOR_PLATE_BLUE = (13, 71, 161)
COLOR_TEXT_MAIN = (15, 23, 42)
COLOR_TEXT_MUTED = (71, 85, 105)
COLOR_BORDER = (203, 213, 225)
COLOR_ROW_ALT = (248, 250, 252)
COLOR_LABEL_BG = (238, 245, 252)
COLOR_GREEN = (22, 163, 74)
COLOR_GREEN_BG = (220, 252, 231)
COLOR_ORANGE = (217, 119, 6)
COLOR_ORANGE_BG = (254, 243, 199)
COLOR_RED = (220, 38, 38)


def build_violation_pdf_filename(record: ViolationRecord) -> str:
    """نام فایل PDF را بر اساس اطلاعات پلاک خودرو و سازگار با فایل‌سیستم اندروید می‌سازد.

    مثال: «۱۲ ب ۳۴۵ | ایران ۱۱» -> «۱۲-ب-۳۴۵-ایران-۱۱.pdf»
    """
    raw = (record.plate or record.plate_latin or "").strip()
    # حذف کاراکترهای غیرمجاز فایل‌سیستم اندروید/ویندوز
    cleaned = raw.replace("|", " ").replace("/", "-").replace("\\", "-")
    cleaned = re.sub(r'[<>:"*?]+', "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned).strip("-._ ")

    if not cleaned or cleaned in ("---", "پلاک-ثبت-نشده"):
        cleaned = f"violation-{record.id}"

    # محدودسازی طول نام فایل
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip("-._ ")

    return f"{cleaned}.pdf"


def _find_font_paths() -> tuple[Path | None, Path | None]:
    """فونت فارسی مناسب (اولویت: Vazirmatn، سپس DejaVuSans، سپس فونت‌های سیستمی اندروید/لینوکس)."""
    fonts_dir = APP_DIR / "assets" / "fonts"
    regular_candidates = [
        fonts_dir / "Vazirmatn-Regular.ttf",
        fonts_dir / "Vazir-Regular.ttf",
        fonts_dir / "DejaVuSans.ttf",
        Path("/system/fonts/NotoNaskhArabic-Regular.ttf"),
        Path("/system/fonts/NotoSansArabic-Regular.ttf"),
        Path("/system/fonts/DroidNaskh-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    bold_candidates = [
        fonts_dir / "Vazirmatn-Bold.ttf",
        fonts_dir / "Vazir-Bold.ttf",
        fonts_dir / "DejaVuSans-Bold.ttf",
        Path("/system/fonts/NotoNaskhArabic-Bold.ttf"),
        Path("/system/fonts/NotoSansArabic-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/tahomabd.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]

    reg = next((p for p in regular_candidates if p.is_file()), None)
    bold = next((p for p in bold_candidates if p.is_file()), None)
    return reg, bold or reg


def _load_font(path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is not None:
        try:
            # استفاده از Layout.BASIC برای یکسان‌سازی ۱۰۰٪ رندر در اندروید و دسکتاپ همراه با core.persian_text
            return ImageFont.truetype(
                str(path),
                size=size,
                layout_engine=ImageFont.Layout.BASIC,
            )
        except Exception as exc:
            logger.warning("Failed to load font '%s': %s", path, exc)
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if not text:
        return 0.0
    bbox = draw.textbbox((0, 0), text, font=font)
    return float(bbox[2] - bbox[0])


def _draw_rtl_text(
    draw: ImageDraw.ImageDraw,
    x_right: float,
    y_top: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    """متن فارسی را تراز از راست در مختصات (x_right, y_top) رسم می‌کند."""
    visual = to_visual_rtl(text)
    w = _text_width(draw, visual, font)
    draw.text((x_right - w, y_top), visual, font=font, fill=fill)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    x_center: float,
    y_center: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    *,
    apply_rtl: bool = True,
) -> None:
    visual = to_visual_rtl(text) if apply_rtl else text
    bbox = draw.textbbox((0, 0), visual, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x_center - w / 2 - bbox[0], y_center - h / 2 - bbox[1]), visual, font=font, fill=fill)


def _draw_iranian_plate_graphic(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    width: int,
    height: int,
    plate_text: str,
    font_digits_lg: ImageFont.ImageFont,
    font_letter_lg: ImageFont.ImageFont,
    font_iran_label: ImageFont.ImageFont,
    font_flag: ImageFont.ImageFont,
) -> None:
    """گرافیک دقیق پلاک استاندارد ایران را (نوار آبی سمت چپ، ایران سمت راست) روی PDF رسم می‌کند."""
    # کادر سفید اصلی پلاک با حاشیه مشکی
    draw.rounded_rectangle(
        (x0, y0, x0 + width, y0 + height),
        radius=16,
        fill=(255, 255, 255),
        outline=(15, 23, 42),
        width=4,
    )

    # ۱. نوار آبی سمت چپ (I.R. IRAN + پرچم ایران)
    strip_w = int(width * 0.125)
    draw.rounded_rectangle(
        (x0 + 3, y0 + 3, x0 + strip_w, y0 + height - 3),
        radius=13,
        fill=COLOR_PLATE_BLUE,
    )
    # پوشاندن گوشه راست نوار آبی برای صاف شدن خط اتصال
    draw.rectangle((x0 + strip_w - 14, y0 + 3, x0 + strip_w, y0 + height - 3), fill=COLOR_PLATE_BLUE)

    # پرچم سه‌رنگ ایران بالای نوار آبی
    flag_x0 = x0 + 16
    flag_x1 = x0 + strip_w - 14
    flag_y = y0 + 16
    band_h = 9
    draw.rectangle((flag_x0, flag_y, flag_x1, flag_y + band_h), fill=(22, 163, 74))
    draw.rectangle((flag_x0, flag_y + band_h, flag_x1, flag_y + 2 * band_h), fill=(255, 255, 255))
    draw.rectangle((flag_x0, flag_y + 2 * band_h, flag_x1, flag_y + 3 * band_h), fill=(220, 38, 38))

    _draw_centered_text(
        draw, x0 + strip_w / 2, y0 + height - 46, "I.R.", font_flag, (255, 255, 255), apply_rtl=False
    )
    _draw_centered_text(
        draw, x0 + strip_w / 2, y0 + height - 24, "IRAN", font_flag, (255, 255, 255), apply_rtl=False
    )

    # ۲. باکس «ایران» در سمت راست
    iran_w = int(width * 0.21)
    div_x = x0 + width - iran_w
    draw.line((div_x, y0 + 4, div_x, y0 + height - 4), fill=(15, 23, 42), width=3)

    left2, letter, mid3, iran2 = parse_iranian_plate_text(plate_text)

    _draw_centered_text(
        draw,
        div_x + iran_w / 2,
        y0 + 28,
        "ایران",
        font_iran_label,
        (15, 23, 42),
    )
    _draw_centered_text(
        draw,
        div_x + iran_w / 2,
        y0 + height * 0.66,
        iran2 or "--",
        font_digits_lg,
        (15, 23, 42),
        apply_rtl=False,
    )

    # ۳. بخش میانی: [۲ رقم چپ]  [حرف وسط]  [۳ رقم راست]
    mid_x0 = x0 + strip_w
    mid_w = div_x - mid_x0
    cy = y0 + height / 2

    if left2 and letter and mid3:
        _draw_centered_text(
            draw, mid_x0 + mid_w * 0.20, cy, left2, font_digits_lg, (15, 23, 42), apply_rtl=False
        )
        _draw_centered_text(
            draw, mid_x0 + mid_w * 0.48, cy - 4, letter, font_letter_lg, (15, 23, 42), apply_rtl=True
        )
        _draw_centered_text(
            draw, mid_x0 + mid_w * 0.78, cy, mid3, font_digits_lg, (15, 23, 42), apply_rtl=False
        )
    else:
        fallback = to_persian_digits(plate_text or "---")
        _draw_centered_text(
            draw, mid_x0 + mid_w / 2, cy, fallback, font_letter_lg, (15, 23, 42), apply_rtl=True
        )


def _paste_image_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    title: str,
    image_bytes: bytes | None,
    empty_text: str,
    font_title: ImageFont.ImageFont,
    font_empty: ImageFont.ImageFont,
) -> None:
    """کادر نمایش تصویر (برش پلاک یا مستندات) را با حفظ نسبت ابعاد و کیفیت بالا روی صفحه می‌چسباند."""
    # کادر کارت
    draw.rounded_rectangle(
        (x0, y0, x0 + w, y0 + h),
        radius=14,
        fill=(248, 250, 252),
        outline=COLOR_BORDER,
        width=2,
    )
    # نوار عنوان بالای کارت
    header_h = 46
    draw.rounded_rectangle(
        (x0, y0, x0 + w, y0 + header_h),
        radius=14,
        fill=COLOR_LABEL_BG,
        outline=COLOR_BORDER,
        width=2,
    )
    draw.rectangle((x0 + 2, y0 + header_h - 14, x0 + w - 2, y0 + header_h), fill=COLOR_LABEL_BG)
    draw.line((x0, y0 + header_h, x0 + w, y0 + header_h), fill=COLOR_BORDER, width=2)

    _draw_centered_text(draw, x0 + w / 2, y0 + header_h / 2, title, font_title, COLOR_PRIMARY_DARK)

    # ناحیه داخلی تصویر
    pad = 14
    img_box_x = x0 + pad
    img_box_y = y0 + header_h + pad
    img_box_w = w - 2 * pad
    img_box_h = h - header_h - 2 * pad

    draw.rounded_rectangle(
        (img_box_x, img_box_y, img_box_x + img_box_w, img_box_y + img_box_h),
        radius=10,
        fill=(15, 23, 42) if image_bytes else (241, 245, 249),
    )

    if not image_bytes:
        _draw_centered_text(
            draw,
            img_box_x + img_box_w / 2,
            img_box_y + img_box_h / 2,
            empty_text,
            font_empty,
            COLOR_TEXT_MUTED,
        )
        return

    try:
        with Image.open(io.BytesIO(image_bytes)) as src_img:
            img = ImageOps.exif_transpose(src_img).convert("RGB")
            fitted = ImageOps.contain(img, (img_box_w - 8, img_box_h - 8), Image.Resampling.LANCZOS)
            px = img_box_x + (img_box_w - fitted.width) // 2
            py = img_box_y + (img_box_h - fitted.height) // 2
            canvas.paste(fitted, (px, py))
    except Exception as exc:
        logger.warning("Could not render image in PDF card '%s': %s", title, exc)
        _draw_centered_text(
            draw,
            img_box_x + img_box_w / 2,
            img_box_y + img_box_h / 2,
            "خطا در خواندن فایل تصویر",
            font_empty,
            COLOR_RED,
        )


def generate_violation_pdf_image(
    record: ViolationRecord,
    plate_image_bytes: bytes | None = None,
    evidence_image_bytes: bytes | None = None,
) -> Image.Image:
    """صفحه A4 گزارش تخلف را به‌صورت تصویر RGB با وضوح 150 DPI تولید می‌کند."""
    reg_path, bold_path = _find_font_paths()

    f_title = _load_font(bold_path, 34)
    f_subtitle = _load_font(reg_path, 20)
    f_section = _load_font(bold_path, 24)
    f_label = _load_font(bold_path, 21)
    f_value = _load_font(reg_path, 21)
    f_small = _load_font(reg_path, 17)
    f_small_bold = _load_font(bold_path, 18)
    f_plate_digits = _load_font(bold_path, 52)
    f_plate_letter = _load_font(bold_path, 48)
    f_plate_iran = _load_font(bold_path, 20)
    f_plate_flag = _load_font(bold_path, 16)

    page = Image.new("RGB", (PAGE_W, PAGE_H), COLOR_BG)
    draw = ImageDraw.Draw(page)

    now_ir = now_iran_datetime()
    generated_jalali = to_persian_digits(
        f"{format_jalali_date(now_ir)} - {format_jalali_time(now_ir)}"
    )

    # =========================================================================
    # ۱. سربرگ رسمی (Header Banner)
    # =========================================================================
    header_h = 176
    draw.rectangle((0, 0, PAGE_W, header_h), fill=COLOR_PRIMARY)
    draw.rectangle((0, header_h - 8, PAGE_W, header_h), fill=COLOR_PRIMARY_DARK)

    # لوگوی اپلیکیشن (سمت راست سربرگ)
    icon_size = 112
    icon_x = PAGE_W - MARGIN - icon_size
    icon_y = (header_h - 8 - icon_size) // 2
    icon_path = APP_DIR / "assets" / "icon.png"
    if icon_path.is_file():
        try:
            with Image.open(icon_path) as ico:
                ico_rgba = ico.convert("RGBA").resize((icon_size, icon_size), Image.Resampling.LANCZOS)
                # هاله سفید ظریف دور آیکون
                draw.ellipse(
                    (icon_x - 4, icon_y - 4, icon_x + icon_size + 4, icon_y + icon_size + 4),
                    fill=(255, 255, 255),
                )
                page.paste(ico_rgba, (icon_x, icon_y), ico_rgba)
        except Exception as exc:
            logger.debug("Skipping PDF header icon: %s", exc)

    # عنوان اصلی و زیرعنوان
    title_right_x = icon_x - 24
    _draw_rtl_text(
        draw, title_right_x, 38, "گزارش رسمی ثبت تخلف خودرو", f_title, (255, 255, 255)
    )
    _draw_rtl_text(
        draw,
        title_right_x,
        96,
        "سامانه هوشمند پلاک‌خوان و مستندسازی تخلفات",
        f_subtitle,
        (219, 234, 254),
    )

    # کادر شناسه گزارش در سمت چپ سربرگ
    meta_box_w = 390
    meta_box_h = 98
    meta_box_x = MARGIN
    meta_box_y = 36
    draw.rounded_rectangle(
        (meta_box_x, meta_box_y, meta_box_x + meta_box_w, meta_box_y + meta_box_h),
        radius=12,
        fill=COLOR_PRIMARY_DARK,
        outline=(147, 197, 253),
        width=2,
    )
    id_display = record.id if len(record.id) <= 24 else f"{record.id[:21]}..."
    f_meta_id = _load_font(bold_path, 15)
    f_meta_date = _load_font(reg_path, 15)
    _draw_centered_text(
        draw,
        meta_box_x + meta_box_w / 2,
        meta_box_y + 30,
        f"شناسه: {id_display}",
        f_meta_id,
        (255, 255, 255),
    )
    _draw_centered_text(
        draw,
        meta_box_x + meta_box_w / 2,
        meta_box_y + 68,
        f"صدور: {generated_jalali}",
        f_meta_date,
        (191, 219, 254),
    )

    # =========================================================================
    # ۲. کارت پلاک گرافیکی ایران
    # =========================================================================
    y = header_h + 34
    plate_card_h = 228
    draw.rounded_rectangle(
        (MARGIN, y, PAGE_W - MARGIN, y + plate_card_h),
        radius=16,
        fill=(248, 250, 252),
        outline=COLOR_BORDER,
        width=2,
    )

    plate_w = 660
    plate_h = 128
    plate_x = (PAGE_W - plate_w) // 2
    plate_y = y + 22
    _draw_iranian_plate_graphic(
        draw,
        plate_x,
        plate_y,
        plate_w,
        plate_h,
        record.plate,
        f_plate_digits,
        f_plate_letter,
        f_plate_iran,
        f_plate_flag,
    )

    # ردیف زیر پلاک: کد لاتین + وضعیت همگام‌سازی
    pill_y = plate_y + plate_h + 18
    status_text = "وضعیت: ارسال و ثبت‌شده در سرور" if record.synced else "وضعیت: ذخیره محلی (در انتظار ارسال)"
    status_fg = COLOR_GREEN if record.synced else COLOR_ORANGE
    status_bg = COLOR_GREEN_BG if record.synced else COLOR_ORANGE_BG

    pill_w = 430
    pill_h = 40
    pill_x_right = PAGE_W - MARGIN - 28
    draw.rounded_rectangle(
        (pill_x_right - pill_w, pill_y, pill_x_right, pill_y + pill_h),
        radius=20,
        fill=status_bg,
        outline=status_fg,
        width=2,
    )
    _draw_centered_text(
        draw, pill_x_right - pill_w / 2, pill_y + pill_h / 2, status_text, f_small_bold, status_fg
    )

    latin_label = f"کد لاتین پلاک: {record.plate_latin or '---'}"
    latin_w = 340
    draw.rounded_rectangle(
        (MARGIN + 28, pill_y, MARGIN + 28 + latin_w, pill_y + pill_h),
        radius=20,
        fill=(255, 255, 255),
        outline=COLOR_BORDER,
        width=2,
    )
    _draw_centered_text(
        draw, MARGIN + 28 + latin_w / 2, pill_y + pill_h / 2, latin_label, f_small_bold, COLOR_TEXT_MAIN
    )

    # =========================================================================
    # ۳. جدول اطلاعات کامل تخلف (شامل تاریخ شمسی و ساعت)
    # =========================================================================
    y = y + plate_card_h + 34
    _draw_rtl_text(draw, PAGE_W - MARGIN, y, "مشخصات کامل تخلف ثبت‌شده", f_section, COLOR_PRIMARY_DARK)
    y += 40
    draw.line((MARGIN, y, PAGE_W - MARGIN, y), fill=COLOR_PRIMARY, width=3)
    y += 14

    jalali_date_display = to_persian_digits(record.jalali_date) if record.jalali_date else "---"
    if record.jalali_weekday:
        jalali_date_full = f"{record.jalali_weekday} ({jalali_date_display})"
    else:
        jalali_date_full = jalali_date_display

    jalali_time_display = to_persian_digits(record.jalali_time) if record.jalali_time else "---"
    if record.has_gps:
        gps_display = f"{record.formatted_gps}\n{record.google_maps_url}"
    else:
        gps_display = "ثبت نشده"
    server_resp_display = record.server_response.strip() if record.server_response else "---"
    notes_display = record.description.strip() if record.description else "توضیحاتی ثبت نشده است."

    table_rows: list[tuple[str, str]] = [
        ("شماره پلاک خودرو", to_persian_digits(record.plate)),
        ("نوع تخلف", record.violation_type),
        ("تاریخ شمسی تخلف", jalali_date_full),
        ("ساعت وقوع تخلف", jalali_time_display),
        ("موقعیت مکانی (GPS)", gps_display),
        ("پاسخ سرور", server_resp_display),
        ("توضیحات تکمیلی", notes_display),
    ]

    table_w = PAGE_W - 2 * MARGIN
    label_col_w = 270
    value_col_w = table_w - label_col_w
    split_x = PAGE_W - MARGIN - label_col_w

    table_top = y
    for idx, (label, raw_val) in enumerate(table_rows):
        wrapped = wrap_persian_lines(
            raw_val,
            measure_fn=lambda s: _text_width(draw, s, f_value),
            max_width=value_col_w - 42,
            max_lines=3 if label == "توضیحات تکمیلی" else 2,
        ) or ["---"]
        line_h = 32
        row_h = max(50, len(wrapped) * line_h + 18)

        row_bg = COLOR_ROW_ALT if idx % 2 == 1 else (255, 255, 255)
        # پس‌زمینه ردیف
        draw.rectangle((MARGIN, y, split_x, y + row_h), fill=row_bg)
        draw.rectangle((split_x, y, PAGE_W - MARGIN, y + row_h), fill=COLOR_LABEL_BG)
        # خطوط جداکننده
        draw.line((MARGIN, y + row_h, PAGE_W - MARGIN, y + row_h), fill=COLOR_BORDER, width=1)

        # متن عنوان ستون راست
        _draw_rtl_text(
            draw,
            PAGE_W - MARGIN - 18,
            y + 14,
            label,
            f_label,
            COLOR_PRIMARY_DARK,
        )

        # متن مقدار ستون چپ (راست‌چین)
        for line_i, line_txt in enumerate(wrapped):
            _draw_rtl_text(
                draw,
                split_x - 20,
                y + 14 + line_i * line_h,
                line_txt,
                f_value,
                COLOR_TEXT_MAIN,
            )

        y += row_h

    # کادر خارجی و خط عمودی جدول
    draw.rectangle((MARGIN, table_top, PAGE_W - MARGIN, y), outline=COLOR_BORDER, width=2)
    draw.line((split_x, table_top, split_x, y), fill=COLOR_BORDER, width=2)

    # =========================================================================
    # ۴. بخش مستندات تصویری (تصویر برش پلاک + تصویر مستندات تخلف)
    # =========================================================================
    y += 34
    _draw_rtl_text(draw, PAGE_W - MARGIN, y, "مستندات تصویری تخلف", f_section, COLOR_PRIMARY_DARK)
    y += 40
    draw.line((MARGIN, y, PAGE_W - MARGIN, y), fill=COLOR_PRIMARY, width=3)
    y += 20

    footer_reserved = 90
    available_h = max(320, PAGE_H - footer_reserved - y)
    card_h = min(510, available_h)
    gap = 28
    card_w = (table_w - gap) // 2

    # کارت سمت راست: تصویر برش‌خورده پلاک
    _paste_image_card(
        page,
        draw,
        x0=MARGIN + card_w + gap,
        y0=y,
        w=card_w,
        h=card_h,
        title="تصویر برش‌خورده پلاک",
        image_bytes=plate_image_bytes,
        empty_text="تصویر برش پلاک ثبت نشده است",
        font_title=f_small_bold,
        font_empty=f_small,
    )

    # کارت سمت چپ: تصویر مستندات تخلف
    _paste_image_card(
        page,
        draw,
        x0=MARGIN,
        y0=y,
        w=card_w,
        h=card_h,
        title="تصویر مستندات تخلف",
        image_bytes=evidence_image_bytes,
        empty_text="تصویر مستندات ضمیمه نشده است",
        font_title=f_small_bold,
        font_empty=f_small,
    )

    # =========================================================================
    # ۵. پاورقی (Footer)
    # =========================================================================
    footer_y = PAGE_H - 64
    draw.line((MARGIN, footer_y - 14, PAGE_W - MARGIN, footer_y - 14), fill=COLOR_BORDER, width=2)
    _draw_rtl_text(
        draw,
        PAGE_W - MARGIN,
        footer_y,
        "تولیدشده توسط اپلیکیشن پلاک‌خوان هوشمند | نسخه اندروید",
        f_small,
        COLOR_TEXT_MUTED,
    )
    draw.text(
        (MARGIN, footer_y),
        to_visual_rtl("صفحه ۱ از ۱"),
        font=f_small,
        fill=COLOR_TEXT_MUTED,
    )

    return page


def generate_violation_pdf_bytes(
    record: ViolationRecord,
    plate_image_bytes: bytes | None = None,
    evidence_image_bytes: bytes | None = None,
) -> bytes:
    """فایل کامل PDF را در حافظه تولید کرده و بایت‌های آن را برمی‌گرداند."""
    page_image = generate_violation_pdf_image(
        record=record,
        plate_image_bytes=plate_image_bytes,
        evidence_image_bytes=evidence_image_bytes,
    )
    buf = io.BytesIO()
    page_image.save(
        buf,
        format="PDF",
        resolution=PAGE_DPI,
        title=f"Violation Report - {record.plate}",
        author="Persian Plate Recognition App",
    )
    return buf.getvalue()


def get_downloads_candidates() -> list[Path]:
    """مسیرهای احتمالی پوشه Downloads در اندروید و دسکتاپ را برمی‌گرداند."""
    candidates: list[Path] = []
    # مسیر استاندارد پوشه Download در اندروید
    android_dl = Path("/storage/emulated/0/Download")
    android_dl_alt = Path("/sdcard/Download")
    if os.path.isdir("/storage/emulated/0") or os.path.isdir("/sdcard"):
        candidates.extend([android_dl, android_dl_alt])
    else:
        candidates.append(Path.home() / "Downloads")
    return candidates


def save_pdf_backup_in_app_storage(pdf_bytes: bytes, filename: str) -> Path:
    """یک نسخه پشتیبان مطمئن در حافظه داخلی اپلیکیشن ذخیره می‌کند."""
    reports_dir = get_writable_data_dir() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    target = reports_dir / Path(filename).name
    target.write_bytes(pdf_bytes)
    return target


def try_write_pdf_to_downloads(pdf_bytes: bytes, filename: str) -> tuple[bool, Path | None, str]:
    """تلاش برای ذخیره مستقیم فایل PDF در پوشه Downloads گوشی/سیستم.

    در اندروید ۱۰ به بالا (Scoped Storage)، نوشتن مستقیم در /storage/emulated/0/Download
    معمولاً با PermissionError مواجه می‌شود؛ در این صورت (False, None, reason) برمی‌گردد تا
    رابط کاربری از طریق دیالوگ ذخیره اندروید (`FilePicker.save_file(..., src_bytes=...)`)
    فایل را بدون نیاز به مجوز خاص مستقیماً در Downloads ذخیره کند.
    """
    safe_name = Path(filename).name
    last_error = "پوشه Downloads قابل دسترس نیست."

    for dl_dir in get_downloads_candidates():
        try:
            dl_dir.mkdir(parents=True, exist_ok=True)
            dest = dl_dir / safe_name
            dest.write_bytes(pdf_bytes)
            return True, dest, str(dest)
        except OSError as exc:
            last_error = str(exc)
            logger.info("Direct Downloads write to '%s' unavailable (%s)", dl_dir, exc)

    return False, None, last_error


# =============================================================================
# خروجی گروهی تخلفات (گزارش PDF چندصفحه‌ای تجمیعی + فایل CSV سازگار با اکسل)
# =============================================================================


def build_batch_export_filename(ext: str = "pdf", prefix: str = "گزارش-تجمیعی-تخلفات") -> str:
    """نام فایل خروجی گروهی را بر اساس تاریخ و ساعت شمسی می‌سازد."""
    now_ir = now_iran_datetime()
    date_part = format_jalali_date(now_ir, persian_digits=False).replace("/", "-")
    time_part = format_jalali_time(now_ir, persian_digits=False).replace(":", "")
    clean_ext = ext.lstrip(".").lower() or "pdf"
    return f"{prefix}-{date_part}-{time_part}.{clean_ext}"


def generate_batch_violations_csv_bytes(records: list[ViolationRecord]) -> bytes:
    """فایل CSV با کدگذاری UTF-8 BOM (بازشدن بدون بهم‌ریختگی فارسی در Excel) تولید می‌کند."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "ردیف",
            "شناسه تخلف",
            "شماره پلاک فارسی",
            "کد لاتین پلاک",
            "نوع تخلف",
            "تاریخ شمسی",
            "روز هفته",
            "ساعت ثبت",
            "عرض جغرافیایی (Lat)",
            "طول جغرافیایی (Lon)",
            "دقت GPS (متر)",
            "لینک نقشه",
            "وضعیت ارسال به سرور",
            "پاسخ سرور",
            "توضیحات تکمیلی",
        ]
    )

    for idx, rec in enumerate(records, start=1):
        writer.writerow(
            [
                idx,
                rec.id,
                rec.plate,
                rec.plate_latin,
                rec.violation_type,
                rec.jalali_date,
                rec.jalali_weekday,
                rec.jalali_time,
                f"{rec.latitude:.6f}" if rec.latitude is not None else "",
                f"{rec.longitude:.6f}" if rec.longitude is not None else "",
                f"{rec.location_accuracy_m:.1f}" if rec.location_accuracy_m is not None else "",
                rec.google_maps_url,
                "ارسال‌شده" if rec.synced else "در انتظار ارسال",
                rec.server_response,
                rec.description,
            ]
        )

    # utf-8-sig (BOM) باعث نمایش صحیح حروف فارسی در Microsoft Excel می‌شود
    return buf.getvalue().encode("utf-8-sig")


def generate_batch_violations_pdf_images(
    records: list[ViolationRecord],
    *,
    filter_summary: str = "",
) -> list[Image.Image]:
    """گزارش تجمیعی چندصفحه‌ای A4 از تخلفات فیلترشده را تولید می‌کند."""
    reg_path, bold_path = _find_font_paths()

    f_title = _load_font(bold_path, 32)
    f_subtitle = _load_font(reg_path, 19)
    f_kpi_num = _load_font(bold_path, 28)
    f_kpi_lbl = _load_font(reg_path, 16)
    f_th = _load_font(bold_path, 17)
    f_cell = _load_font(reg_path, 16)
    f_cell_bold = _load_font(bold_path, 16)
    f_cell_sm = _load_font(reg_path, 14)
    f_footer = _load_font(reg_path, 16)

    now_ir = now_iran_datetime()
    generated_jalali = to_persian_digits(
        f"{format_jalali_date(now_ir)} - {format_jalali_time(now_ir)}"
    )

    table_w = PAGE_W - 2 * MARGIN  # 1112px
    # ستون‌ها از راست به چپ: ردیف، پلاک، نوع تخلف و توضیحات، تاریخ/ساعت، GPS، وضعیت
    col_specs = [
        ("ردیف", 56),
        ("شماره پلاک", 216),
        ("نوع تخلف و توضیحات", 340),
        ("تاریخ و ساعت شمسی", 190),
        ("موقعیت GPS", 196),
        ("وضعیت", 114),
    ]
    assert sum(w for _, w in col_specs) == table_w

    # محاسبه بازه‌های افقی هر ستون از راست به چپ
    col_bounds: list[tuple[int, int, str]] = []
    cursor_right = PAGE_W - MARGIN
    for title, width in col_specs:
        col_bounds.append((cursor_right - width, cursor_right, title))
        cursor_right -= width

    pages: list[Image.Image] = []

    def _new_page(is_first: bool) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        img = Image.new("RGB", (PAGE_W, PAGE_H), COLOR_BG)
        drw = ImageDraw.Draw(img)

        if is_first:
            header_h = 152
            drw.rectangle((0, 0, PAGE_W, header_h), fill=COLOR_PRIMARY)
            drw.rectangle((0, header_h - 8, PAGE_W, header_h), fill=COLOR_PRIMARY_DARK)

            icon_size = 96
            icon_x = PAGE_W - MARGIN - icon_size
            icon_y = (header_h - 8 - icon_size) // 2
            icon_path = APP_DIR / "assets" / "icon.png"
            if icon_path.is_file():
                try:
                    with Image.open(icon_path) as ico:
                        ico_rgba = ico.convert("RGBA").resize(
                            (icon_size, icon_size), Image.Resampling.LANCZOS
                        )
                        drw.ellipse(
                            (icon_x - 3, icon_y - 3, icon_x + icon_size + 3, icon_y + icon_size + 3),
                            fill=(255, 255, 255),
                        )
                        img.paste(ico_rgba, (icon_x, icon_y), ico_rgba)
                except Exception:
                    pass

            _draw_rtl_text(
                drw, icon_x - 20, 30, "گزارش تجمیعی تخلفات ثبت‌شده", f_title, (255, 255, 255)
            )
            sub = filter_summary.strip() or "سامانه هوشمند پلاک‌خوان و مستندسازی تخلفات"
            _draw_rtl_text(drw, icon_x - 20, 86, sub, f_subtitle, (219, 234, 254))

            # تاریخ صدور سمت چپ سربرگ
            meta_w = 310
            drw.rounded_rectangle(
                (MARGIN, 34, MARGIN + meta_w, 114),
                radius=12,
                fill=COLOR_PRIMARY_DARK,
                outline=(147, 197, 253),
                width=2,
            )
            _draw_centered_text(
                drw, MARGIN + meta_w / 2, 58, "تاریخ صدور گزارش", f_cell_sm, (191, 219, 254)
            )
            _draw_centered_text(
                drw, MARGIN + meta_w / 2, 90, generated_jalali, f_cell_bold, (255, 255, 255)
            )

            # کارت‌های آمار خلاصه (KPI)
            y_kpi = header_h + 24
            kpi_h = 84
            kpi_gap = 16
            kpi_w = (table_w - 3 * kpi_gap) // 4
            total_n = len(records)
            synced_n = sum(1 for r in records if r.synced)
            pending_n = total_n - synced_n
            gps_n = sum(1 for r in records if r.has_gps)

            kpis = [
                ("کل تخلفات", to_persian_digits(str(total_n)), COLOR_PRIMARY, COLOR_LABEL_BG),
                ("ارسال‌شده به سرور", to_persian_digits(str(synced_n)), COLOR_GREEN, COLOR_GREEN_BG),
                ("در انتظار ارسال", to_persian_digits(str(pending_n)), COLOR_ORANGE, COLOR_ORANGE_BG),
                ("دارای موقعیت GPS", to_persian_digits(str(gps_n)), COLOR_PRIMARY_DARK, COLOR_ROW_ALT),
            ]
            # رسم از راست به چپ
            for k_idx, (lbl, val, fg_col, bg_col) in enumerate(kpis):
                kx1 = PAGE_W - MARGIN - k_idx * (kpi_w + kpi_gap)
                kx0 = kx1 - kpi_w
                drw.rounded_rectangle(
                    (kx0, y_kpi, kx1, y_kpi + kpi_h),
                    radius=12,
                    fill=bg_col,
                    outline=COLOR_BORDER,
                    width=2,
                )
                _draw_centered_text(drw, (kx0 + kx1) / 2, y_kpi + 30, val, f_kpi_num, fg_col, apply_rtl=False)
                _draw_centered_text(drw, (kx0 + kx1) / 2, y_kpi + 64, lbl, f_kpi_lbl, COLOR_TEXT_MAIN)

            y_start = y_kpi + kpi_h + 24
        else:
            drw.rectangle((0, 0, PAGE_W, 74), fill=COLOR_PRIMARY)
            _draw_rtl_text(
                drw,
                PAGE_W - MARGIN,
                22,
                "ادامه گزارش تجمیعی تخلفات ثبت‌شده",
                f_subtitle,
                (255, 255, 255),
            )
            drw.text((MARGIN, 26), to_visual_rtl(generated_jalali), font=f_cell_sm, fill=(219, 234, 254))
            y_start = 98

        # هدر جدول
        th_h = 46
        drw.rectangle((MARGIN, y_start, PAGE_W - MARGIN, y_start + th_h), fill=COLOR_PRIMARY_DARK)
        for cx0, cx1, title in col_bounds:
            _draw_centered_text(drw, (cx0 + cx1) / 2, y_start + th_h / 2, title, f_th, (255, 255, 255))
            drw.line((cx0, y_start, cx0, y_start + th_h), fill=(100, 140, 190), width=1)
        drw.rectangle(
            (MARGIN, y_start, PAGE_W - MARGIN, y_start + th_h),
            outline=COLOR_PRIMARY_DARK,
            width=2,
        )

        return img, drw, y_start + th_h

    cur_img, cur_draw, y = _new_page(is_first=True)
    max_body_y = PAGE_H - 92

    if not records:
        row_h = 84
        cur_draw.rectangle((MARGIN, y, PAGE_W - MARGIN, y + row_h), fill=COLOR_ROW_ALT, outline=COLOR_BORDER, width=2)
        _draw_centered_text(
            cur_draw,
            PAGE_W / 2,
            y + row_h / 2,
            "هیچ تخلفی مطابق فیلتر انتخاب‌شده یافت نشد.",
            f_subtitle,
            COLOR_TEXT_MUTED,
        )
        pages.append(cur_img)
    else:
        for row_idx, rec in enumerate(records, start=1):
            vtype_lines = wrap_persian_lines(
                rec.violation_type,
                measure_fn=lambda s: _text_width(cur_draw, s, f_cell_bold),
                max_width=col_specs[2][1] - 20,
                max_lines=2,
            ) or ["---"]
            note_lines = (
                wrap_persian_lines(
                    rec.description.strip(),
                    measure_fn=lambda s: _text_width(cur_draw, s, f_cell_sm),
                    max_width=col_specs[2][1] - 20,
                    max_lines=2,
                )
                if rec.description.strip()
                else []
            )
            plate_lines = wrap_persian_lines(
                to_persian_digits(rec.plate),
                measure_fn=lambda s: _text_width(cur_draw, s, f_cell_bold),
                max_width=col_specs[1][1] - 16,
                max_lines=2,
            ) or ["---"]

            content_lines = max(
                len(vtype_lines) + len(note_lines),
                len(plate_lines),
                2,
            )
            row_h = max(62, content_lines * 24 + 20)

            if y + row_h > max_body_y:
                pages.append(cur_img)
                cur_img, cur_draw, y = _new_page(is_first=False)

            bg = COLOR_ROW_ALT if row_idx % 2 == 0 else (255, 255, 255)
            cur_draw.rectangle((MARGIN, y, PAGE_W - MARGIN, y + row_h), fill=bg, outline=COLOR_BORDER, width=1)
            for cx0, _cx1, _t in col_bounds:
                cur_draw.line((cx0, y, cx0, y + row_h), fill=COLOR_BORDER, width=1)

            # ۰) ردیف
            c0x0, c0x1, _ = col_bounds[0]
            _draw_centered_text(
                cur_draw,
                (c0x0 + c0x1) / 2,
                y + row_h / 2,
                to_persian_digits(str(row_idx)),
                f_cell_bold,
                COLOR_PRIMARY_DARK,
                apply_rtl=False,
            )

            # ۱) شماره پلاک
            c1x0, c1x1, _ = col_bounds[1]
            for li, ltxt in enumerate(plate_lines):
                _draw_centered_text(
                    cur_draw, (c1x0 + c1x1) / 2, y + 22 + li * 24, ltxt, f_cell_bold, COLOR_TEXT_MAIN
                )

            # ۲) نوع تخلف + توضیحات
            c2x0, c2x1, _ = col_bounds[2]
            ty = y + 10
            for ltxt in vtype_lines:
                _draw_rtl_text(cur_draw, c2x1 - 10, ty, ltxt, f_cell_bold, COLOR_TEXT_MAIN)
                ty += 23
            for ltxt in note_lines:
                _draw_rtl_text(cur_draw, c2x1 - 10, ty, ltxt, f_cell_sm, COLOR_TEXT_MUTED)
                ty += 20

            # ۳) تاریخ و ساعت شمسی
            c3x0, c3x1, _ = col_bounds[3]
            _draw_centered_text(
                cur_draw,
                (c3x0 + c3x1) / 2,
                y + 22,
                to_persian_digits(rec.jalali_date or "---"),
                f_cell_bold,
                COLOR_TEXT_MAIN,
                apply_rtl=False,
            )
            _draw_centered_text(
                cur_draw,
                (c3x0 + c3x1) / 2,
                y + 46,
                to_persian_digits(rec.jalali_time or "---"),
                f_cell_sm,
                COLOR_TEXT_MUTED,
                apply_rtl=False,
            )

            # ۴) موقعیت GPS
            c4x0, c4x1, _ = col_bounds[4]
            if rec.has_gps:
                _draw_centered_text(
                    cur_draw,
                    (c4x0 + c4x1) / 2,
                    y + 22,
                    f"{rec.latitude:.5f}, {rec.longitude:.5f}",
                    f_cell_sm,
                    COLOR_PRIMARY_DARK,
                    apply_rtl=False,
                )
                acc_str = (
                    f"دقت: ±{to_persian_digits(str(int(round(rec.location_accuracy_m))))} متر"
                    if rec.location_accuracy_m
                    else "ثبت GPS"
                )
                _draw_centered_text(
                    cur_draw, (c4x0 + c4x1) / 2, y + 46, acc_str, f_cell_sm, COLOR_TEXT_MUTED
                )
            else:
                _draw_centered_text(
                    cur_draw, (c4x0 + c4x1) / 2, y + row_h / 2, "ثبت نشده", f_cell_sm, COLOR_TEXT_MUTED
                )

            # ۵) وضعیت ارسال
            c5x0, c5x1, _ = col_bounds[5]
            st_lbl = "ارسال‌شده" if rec.synced else "در انتظار"
            st_fg = COLOR_GREEN if rec.synced else COLOR_ORANGE
            st_bg = COLOR_GREEN_BG if rec.synced else COLOR_ORANGE_BG
            cur_draw.rounded_rectangle(
                (c5x0 + 10, y + (row_h - 32) // 2, c5x1 - 10, y + (row_h + 32) // 2),
                radius=14,
                fill=st_bg,
                outline=st_fg,
                width=1,
            )
            _draw_centered_text(
                cur_draw, (c5x0 + c5x1) / 2, y + row_h / 2, st_lbl, f_cell_sm, st_fg
            )

            y += row_h

        pages.append(cur_img)

    # درج پاورقی و شماره صفحه روی تمام صفحات
    total_pages = len(pages)
    for p_idx, p_img in enumerate(pages, start=1):
        p_draw = ImageDraw.Draw(p_img)
        footer_y = PAGE_H - 64
        p_draw.line((MARGIN, footer_y - 14, PAGE_W - MARGIN, footer_y - 14), fill=COLOR_BORDER, width=2)
        _draw_rtl_text(
            p_draw,
            PAGE_W - MARGIN,
            footer_y,
            "گزارش تجمیعی اپلیکیشن پلاک‌خوان هوشمند | نسخه اندروید",
            f_footer,
            COLOR_TEXT_MUTED,
        )
        page_num_str = to_persian_digits(f"صفحه {p_idx} از {total_pages}")
        p_draw.text((MARGIN, footer_y), to_visual_rtl(page_num_str), font=f_footer, fill=COLOR_TEXT_MUTED)

    return pages


def generate_batch_violations_pdf_bytes(
    records: list[ViolationRecord],
    *,
    filter_summary: str = "",
) -> bytes:
    """فایل PDF چندصفحه‌ای گزارش تجمیعی تخلفات را تولید کرده و بایت‌های آن را برمی‌گرداند."""
    pages = generate_batch_violations_pdf_images(records, filter_summary=filter_summary)
    buf = io.BytesIO()
    pages[0].save(
        buf,
        format="PDF",
        resolution=PAGE_DPI,
        save_all=True,
        append_images=pages[1:],
        title="Batch Violations Report",
        author="Persian Plate Recognition App",
    )
    return buf.getvalue()

