"""تنظیمات مرکزی اپلیکیشن تشخیص پلاک هوشمند.

تمامی کلاس‌های کاراکتر پلاک ایرانی دقیقاً منطبق بر مخزن:
https://github.com/shahabbai/Persian_Plate_Recognition (ir_lpr.yaml - 39 کلاس)
تنظیم شده‌اند.
"""

from __future__ import annotations

import json
import os
import re
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

# تبدیل ارقام لاتین و عربی به فارسی و بالعکس
_LATIN_TO_PERSIAN_DIGITS = str.maketrans("0123456789٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹۰۱۲۳۴۵۶۷۸۹")
_PERSIAN_TO_LATIN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# حروف مجاز پلاک‌های ایران برای انتخاب در ورودی پلاک فرم ثبت تخلف
PLATE_LETTERS_PERSIAN: tuple[str, ...] = (
    "الف", "ب", "پ", "ت", "ث", "ج", "د", "ز", "ژ",
    "س", "ش", "ص", "ط", "ظ", "ع", "ف", "ق", "ک", "گ",
    "ل", "م", "ن", "و", "ه", "ه‌", "ی", "D", "S", "تشریفات",
)

PERSIAN_TO_LATIN_LETTER: dict[str, str] = {
    persian: latin
    for persian, latin in zip(CHAR_NAMES_PERSIAN[10:], CHAR_NAMES_LATIN[10:])
}

# فهرست انواع تخلف جهت نمایش در صفحه ثبت تخلف
VIOLATION_TYPES: tuple[str, ...] = (
    "نداشتن صورت وضعیت مسافری",
    "مغایرت تعداد/اسامی مسافر",
    "مسافرگیری بیش از ظرفیت",
    "مغایرت مفاد صورت وضعیت",
    "صورت وضعیت غیر برخط",
    "دریافت کرایه نامتعارف",
    "نداشتن لباس فرم",
    "نداشتن بارنامه/ CMR",
    "اضافه تناژ",
    "عدم مهار صحیح بار",
    "نقص فنی ناوگان",
    "نداشتن تجهیزات ایمنی/کپسول",
    "حرکت خارج از مسیر راهنامه/پروانه",
    "نداشتن پروانه بار ترافیکی/اسکورت",
    "سایر",
)


@dataclass
class AppSettings:
    """تنظیمات قابل تغییر در زمان اجرا."""

    api_base_url: str = "http://myweb.web.com"
    api_check_path: str = "/api/plate/check"
    api_violation_path: str = "/api/violation"
    api_timeout_seconds: float = 5.0
    api_cache_ttl_seconds: float = 15.0

    # حالت استعلام خودکار از سرور: به‌صورت پیش‌فرض غیرفعال است و فقط با فعال‌سازی کاربر انجام می‌شود
    auto_query_enabled: bool = False

    # نوع استعلام پلاک: به‌صورت پیش‌فرض غیرفعال (فقط پلاک‌های عمومی حرف «ع» استعلام می‌شوند)
    query_all_plates: bool = False

    # تم ظاهری برنامه: True = تم شب (تیره)، False = تم روز (روشن)
    dark_mode: bool = True

    # ورود با اثر انگشت (احراز هویت بیومتریک گوشی): پیش‌فرض غیرفعال
    biometric_login_enabled: bool = False

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


def get_settings_file_path(data_dir: Path | None = None) -> Path:
    base_dir = data_dir if data_dir is not None else get_writable_data_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "settings.json"


def load_settings(data_dir: Path | None = None) -> AppSettings:
    """تنظیمات ذخیره‌شده کاربر را می‌خواند؛ در نبود فایل، مقادیر پیش‌فرض (استعلام خودکار غیرفعال) برمی‌گردد."""
    settings = AppSettings()
    path = get_settings_file_path(data_dir)
    if not path.is_file():
        return settings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return settings
        if isinstance(data.get("api_base_url"), str) and data["api_base_url"].strip():
            settings.api_base_url = data["api_base_url"].strip()
        if isinstance(data.get("api_check_path"), str) and data["api_check_path"].strip():
            settings.api_check_path = data["api_check_path"].strip()
        if isinstance(data.get("api_violation_path"), str) and data["api_violation_path"].strip():
            settings.api_violation_path = data["api_violation_path"].strip()
        if isinstance(data.get("auto_query_enabled"), bool):
            settings.auto_query_enabled = data["auto_query_enabled"]
        if isinstance(data.get("query_all_plates"), bool):
            settings.query_all_plates = data["query_all_plates"]
        if isinstance(data.get("dark_mode"), bool):
            settings.dark_mode = data["dark_mode"]
        if isinstance(data.get("biometric_login_enabled"), bool):
            settings.biometric_login_enabled = data["biometric_login_enabled"]
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings: AppSettings, data_dir: Path | None = None) -> None:
    """تنظیمات کاربر را به‌صورت دائمی در حافظه اپلیکیشن ذخیره می‌کند."""
    path = get_settings_file_path(data_dir)
    payload = {
        "api_base_url": settings.api_base_url,
        "api_check_path": settings.api_check_path,
        "api_violation_path": settings.api_violation_path,
        "auto_query_enabled": bool(settings.auto_query_enabled),
        "query_all_plates": bool(settings.query_all_plates),
        "dark_mode": bool(settings.dark_mode),
        "biometric_login_enabled": bool(settings.biometric_login_enabled),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_public_ein_plate(
    chars_persian: list[str] | None = None,
    plate_persian: str = "",
    plate_latin: str = "",
) -> bool:
    """بررسی می‌کند آیا پلاک جزو پلاک‌های حمل‌ونقل عمومی (دارای حرف «ع») است یا خیر."""
    if chars_persian:
        if len(chars_persian) == 8:
            return chars_persian[2].strip() == "ع"
        if any(ch.strip() == "ع" for ch in chars_persian):
            return True

    if "ع" in (plate_persian or ""):
        return True
    if "EIN" in (plate_latin or "").upper():
        return True
    return False


def should_query_plate(
    settings: AppSettings,
    chars_persian: list[str] | None = None,
    plate_persian: str = "",
    plate_latin: str = "",
) -> bool:
    """بر اساس سوییچ «استعلام تمام پلاکها» تعیین می‌کند آیا پلاک مجاز به استعلام از سرور است."""
    if settings.query_all_plates:
        return True
    return is_public_ein_plate(
        chars_persian=chars_persian,
        plate_persian=plate_persian,
        plate_latin=plate_latin,
    )


def to_persian_digits(text: str) -> str:
    """ارقام انگلیسی/عربی یک رشته را به فارسی تبدیل می‌کند."""
    return text.translate(_LATIN_TO_PERSIAN_DIGITS)


def to_latin_digits(text: str) -> str:
    """ارقام فارسی/عربی یک رشته را به انگلیسی تبدیل می‌کند."""
    return text.translate(_PERSIAN_TO_LATIN_DIGITS)


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


def parse_iranian_plate_text(plate_text: str) -> tuple[str, str, str, str]:
    """رشته نمایشی پلاک (مانند «۱۲ ب ۳۴۵ | ایران ۱۱») را به (دو رقم، حرف، سه رقم، کد ایران) تفکیک می‌کند.

    اگر قالب استاندارد نباشد، tuple شامل رشته‌های خالی یا بخش‌های موجود برمی‌گرداند.
    """
    raw = to_latin_digits((plate_text or "").strip())
    if not raw:
        return "", "", "", ""

    # حذف کلمه «ایران» و جداکننده‌های رایج
    cleaned = raw.replace("ایران", " ").replace("|", " ").replace("-", " ")
    # الگو: دو رقم + حرف فارسی/لاتین + سه رقم + دو رقم کد ایران
    match = re.match(
        r"^\s*(\d{2})\s*([^\d\s]+)\s*(\d{3})\s*(\d{2})\s*$",
        cleaned,
    )
    if match:
        return (
            to_persian_digits(match.group(1)),
            match.group(2).strip(),
            to_persian_digits(match.group(3)),
            to_persian_digits(match.group(4)),
        )
    return "", "", "", ""


def compose_iranian_plate(
    left2: str,
    letter: str,
    mid3: str,
    iran2: str,
) -> tuple[str, str, list[str]]:
    """اجزای چهارگانه پلاک ایران را به (رشته نمایشی فارسی، کد لاتین، لیست ۸ کاراکتر) تبدیل می‌کند."""
    l2_lat = re.sub(r"\D", "", to_latin_digits(left2))[:2]
    m3_lat = re.sub(r"\D", "", to_latin_digits(mid3))[:3]
    i2_lat = re.sub(r"\D", "", to_latin_digits(iran2))[:2]
    let_clean = (letter or "").strip()

    l2_fa = to_persian_digits(l2_lat)
    m3_fa = to_persian_digits(m3_lat)
    i2_fa = to_persian_digits(i2_lat)

    parts_main = " ".join(p for p in (l2_fa, let_clean, m3_fa) if p)
    if i2_fa:
        display_persian = f"{parts_main} | ایران {i2_fa}" if parts_main else f"ایران {i2_fa}"
    else:
        display_persian = parts_main

    let_lat = PERSIAN_TO_LATIN_LETTER.get(let_clean, let_clean.upper())
    plate_latin = f"{l2_lat}{let_lat}{m3_lat}{i2_lat}"

    chars_8: list[str] = []
    if len(l2_lat) == 2 and let_clean and len(m3_lat) == 3 and len(i2_lat) == 2:
        chars_8 = [l2_lat[0], l2_lat[1], let_clean, m3_lat[0], m3_lat[1], m3_lat[2], i2_lat[0], i2_lat[1]]

    return display_persian, plate_latin, chars_8

