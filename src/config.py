"""تنظیمات مرکزی اپلیکیشن تشخیص پلاک هوشمند.

تمامی کلاس‌های کاراکتر پلاک ایرانی دقیقاً منطبق بر مخزن:
https://github.com/shahabbai/Persian_Plate_Recognition (ir_lpr.yaml - 39 کلاس)
تنظیم شده‌اند.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# مسیر پوشه weights (سازگار با اجرای اندروید در بسته Flet و اجرای محلی دسکتاپ)
# ---------------------------------------------------------------------------
APP_DIR: Path = Path(__file__).resolve().parent


def find_weights_dir() -> Path:
    """پوشه weights حاوی فایل‌های .onnx را پیدا می‌کند."""
    candidates = [
        APP_DIR / "weights",
        APP_DIR.parent / "weights",
        Path.cwd() / "weights",
        Path.cwd() / "src" / "weights",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.onnx")):
            return candidate
    return candidates[0]


def get_writable_data_dir() -> Path:
    """مسیر قابل نوشتن برای ذخیره لاگ تخلف‌ها در اندروید و دسکتاپ."""
    flet_storage = os.getenv("FLET_APP_STORAGE_DATA")
    if flet_storage:
        path = Path(flet_storage)
    else:
        path = Path.cwd() / "app_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# نام فایل‌های پیش‌فرض مدل ONNX (خروجی مخزن Persian_Plate_Recognition)
# ---------------------------------------------------------------------------
DEFAULT_CARS_MODEL_NAME = "cars_best_v26.onnx"
DEFAULT_CHARS_MODEL_NAME = "chars_best_v26.onnx"

# شناسه‌های کلاس مدل مرحله اول (data.yaml)
CLASS_ID_CAR = 0
CLASS_ID_PLATE = 1

# 39 کلاس کاراکتر پلاک ایرانی دقیقاً مطابق ir_lpr.yaml در مخزن مرجع
CHAR_NAMES_PERSIAN: tuple[str, ...] = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "D", "S", "الف", "ب", "ت", "تشریفات", "ث", "ج", "د", "ز",
    "س", "ش", "ص", "ط", "ظ", "ع", "ف", "ق", "ل", "م",
    "ن", "ه", "ه‌", "و", "پ", "ژ", "ک", "گ", "ی",
)

CHAR_NAMES_LATIN: tuple[str, ...] = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "D", "S", "A", "B", "T", "CER", "S", "J", "D", "Z",
    "S", "SH", "SAD", "T", "ZA", "EIN", "F", "GH", "L", "M",
    "N", "H", "H2", "V", "P", "ZHE", "K", "G", "Y",
)

# تبدیل ارقام لاتین به فارسی برای نمایش خوانا در رابط کاربری
_LATIN_TO_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

# فهرست انواع تخلف جهت نمایش در صفحه ثبت تخلف
VIOLATION_TYPES: tuple[str, ...] = (
    "توقف ممنوع / پارک دوبل",
    "ورود غیرمجاز به محدوده",
    "نداشتن مجوز تردد",
    "سرعت غیرمجاز",
    "عبور از چراغ قرمز",
    "نداشتن معاینه فنی معتبر",
    "پلاک مخدوش یا ناخوانا",
    "سایر موارد",
)


@dataclass
class AppSettings:
    """تنظیمات قابل تغییر در زمان اجرا."""

    api_base_url: str = "http://myweb.web.com"
    api_check_path: str = "/api/plate/check"
    api_violation_path: str = "/api/violation"
    api_timeout_seconds: float = 5.0
    api_cache_ttl_seconds: float = 15.0

    # پارامترهای مدل هوش مصنوعی (قابل تنظیم برای توازن سرعت/دقت)
    input_size: int = 640
    cars_conf_threshold: float = 0.55
    chars_conf_threshold: float = 0.35
    nms_iou_threshold: float = 0.45
    expected_plate_chars: int = 8
    ort_threads: int = 4

    # حداقل فاصله زمانی (ثانیه) بین دو پردازش پیاپی فریم دوربین برای بهینه‌سازی باتری و سرعت
    min_inference_interval_sec: float = 0.12

    @property
    def check_url(self) -> str:
        base = self.api_base_url.rstrip("/")
        path = self.api_check_path if self.api_check_path.startswith("/") else f"/{self.api_check_path}"
        return f"{base}{path}"

    @property
    def violation_url(self) -> str:
        base = self.api_base_url.rstrip("/")
        path = (
            self.api_violation_path
            if self.api_violation_path.startswith("/")
            else f"/{self.api_violation_path}"
        )
        return f"{base}{path}"


def to_persian_digits(text: str) -> str:
    """ارقام انگلیسی یک رشته را به فارسی تبدیل می‌کند."""
    return text.translate(_LATIN_TO_PERSIAN_DIGITS)


def format_iranian_plate(chars: list[str]) -> tuple[str, str, str, str]:
    """کاراکترهای مرتب‌شده از چپ به راست پلاک را به اجزای استاندارد پلاک ایران تفکیک می‌کند.

    قالب استاندارد پلاک سواری ایران (8 کاراکتر از چپ به راست):
    [2 رقم اول] [حرف فارسی] [3 رقم وسط] [2 رقم کد شهر (ایران)]
    مثال: ['1','2','ب','3','4','5','6','7'] -> ('۱۲', 'ب', '۳۴۵', '۶۷')
    """
    if len(chars) == 8:
        left2 = to_persian_digits("".join(chars[0:2]))
        letter = chars[2]
        mid3 = to_persian_digits("".join(chars[3:6]))
        iran2 = to_persian_digits("".join(chars[6:8]))
        return left2, letter, mid3, iran2

    joined = to_persian_digits("".join(chars))
    return joined, "", "", ""
