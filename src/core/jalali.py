"""تبدیل دقیق تاریخ میلادی به تاریخ شمسی (جلالی) بدون نیاز به کتابخانه خارجی.

الگوریتم استاندارد تقویم هجری شمسی (بر پایه تقویم رسمی ایران / Borkowski):
- پشتیبانی دقیق از سال‌های کبیسه (از جمله کبیسه ۳۳ ساله مانند ۱۴۰۳)
- استفاده از منطقه زمانی رسمی ایران (UTC+03:30) بدون نیاز به بسته tzdata در اندروید
- تولید قالب‌های عددی (1405/08/23)، ساعت (15:42:18) و نوشتاری (شنبه ۲۳ آبان ۱۴۰۵)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# منطقه زمانی رسمی ایران (ایران از سال ۱۴۰۱ تغییر ساعت تابستانی ندارد و همیشه UTC+03:30 است)
IRAN_TIMEZONE = timezone(timedelta(hours=3, minutes=30), name="Asia/Tehran")

JALALI_MONTH_NAMES: tuple[str, ...] = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)

# نگاشت weekday() در پایتون (0=Monday .. 6=Sunday) به نام روزهای هفته فارسی
PERSIAN_WEEKDAY_NAMES: tuple[str, ...] = (
    "دوشنبه",
    "سه‌شنبه",
    "چهارشنبه",
    "پنج‌شنبه",
    "جمعه",
    "شنبه",
    "یکشنبه",
)

_LATIN_TO_PERSIAN = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """تاریخ میلادی (سال، ماه، روز) را به تاریخ شمسی (سال، ماه، روز) تبدیل می‌کند."""
    g_d_m = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        355666
        + (365 * gy)
        + ((gy2 + 3) // 4)
        - ((gy2 + 99) // 100)
        + ((gy2 + 399) // 400)
        + gd
        + g_d_m[gm - 1]
    )
    jy = -1595 + (33 * (days // 12053))
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + (days // 31)
        jd = 1 + (days % 31)
    else:
        jm = 7 + ((days - 186) // 30)
        jd = 1 + ((days - 186) % 30)
    return jy, jm, jd


def now_iran_datetime() -> datetime:
    """زمان فعلی را به وقت رسمی ایران (UTC+03:30) برمی‌گرداند."""
    return datetime.now(timezone.utc).astimezone(IRAN_TIMEZONE)


def format_jalali_date(dt: datetime, persian_digits: bool = False) -> str:
    """تاریخ شمسی را به قالب YYYY/MM/DD برمی‌گرداند."""
    jy, jm, jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
    out = f"{jy:04d}/{jm:02d}/{jd:02d}"
    return out.translate(_LATIN_TO_PERSIAN) if persian_digits else out


def format_jalali_time(dt: datetime, persian_digits: bool = False) -> str:
    """ساعت را به قالب HH:MM:SS برمی‌گرداند."""
    out = f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
    return out.translate(_LATIN_TO_PERSIAN) if persian_digits else out


def format_jalali_verbose_date(dt: datetime, persian_digits: bool = True) -> str:
    """تاریخ شمسی نوشتاری (مثال: شنبه ۲۳ آبان ۱۴۰۵) برمی‌گرداند."""
    jy, jm, jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
    weekday = PERSIAN_WEEKDAY_NAMES[dt.weekday()]
    month_name = JALALI_MONTH_NAMES[jm - 1]
    out = f"{weekday} {jd} {month_name} {jy}"
    return out.translate(_LATIN_TO_PERSIAN) if persian_digits else out


def iso_to_jalali_parts(iso_timestamp: str) -> tuple[str, str, str]:
    """یک رشته زمان ISO (حتی برای رکوردهای قدیمی) را به (تاریخ شمسی، ساعت، تاریخ نوشتاری) تبدیل می‌کند."""
    cleaned = (iso_timestamp or "").strip()
    if not cleaned:
        return "", "", ""
    try:
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IRAN_TIMEZONE)
        else:
            dt = dt.astimezone(IRAN_TIMEZONE)
        return (
            format_jalali_date(dt, persian_digits=False),
            format_jalali_time(dt, persian_digits=False),
            format_jalali_verbose_date(dt, persian_digits=True),
        )
    except ValueError:
        return "", "", ""
