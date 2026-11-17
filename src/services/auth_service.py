"""سرویس احراز هویت اپلیکیشن پلاک‌خوان هوشمند.

روش کار:
- تابع خالی `authenticate_custom(username, password)` برای پیاده‌سازی روش احراز هویت
  اصلی در آینده (مثلاً اتصال به API سرور، توکن JWT یا دیتابیس) در نظر گرفته شده است.
- تا زمانی که `authenticate_custom` خالی باشد (مقدار `None` برگرداند)، کاربر به‌طور موقت
  با نام کاربری `admin` و رمز عبور `1` وارد برنامه می‌شود.
"""

from __future__ import annotations

TEMP_USERNAME = "admin"
TEMP_PASSWORD = "1"


async def authenticate_custom(username: str, password: str) -> bool | None:
    """تابع خالی احراز هویت برای پیاده‌سازی در آینده.

    برای فعال‌سازی احراز هویت واقعی، کد بررسی نام کاربری و رمز عبور را درون این تابع بنویسید
    و در صورت معتبر بودن `True` و در غیر این صورت `False` برگردانید.
    """
    pass


async def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    """احراز هویت کاربر برای ورود به اپلیکیشن."""
    clean_user = (username or "").strip()
    clean_pass = password or ""

    if not clean_user or not clean_pass:
        return False, "لطفاً نام کاربری و رمز عبور را وارد کنید."

    # اولویت با تابع احراز هویت اختصاصی (در صورت تکمیل شدن در آینده)
    custom_result = await authenticate_custom(clean_user, clean_pass)
    if custom_result is not None:
        if custom_result:
            return True, "ورود با موفقیت انجام شد."
        return False, "نام کاربری یا رمز عبور نامعتبر است."

    # ورود موقت پیش‌فرض تا زمان پیاده‌سازی تابع احراز هویت اصلی
    if clean_user == TEMP_USERNAME and clean_pass == TEMP_PASSWORD:
        return True, "ورود با موفقیت انجام شد."

    return False, "نام کاربری یا رمز عبور نادرست است."
