"""سرویس احراز هویت اپلیکیشن پلاک‌خوان هوشمند."""

from __future__ import annotations

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "1"


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    """احراز هویت کاربر با نام کاربری admin و رمز عبور 1."""
    clean_user = (username or "").strip()
    clean_pass = password or ""

    if not clean_user or not clean_pass:
        return False, "لطفاً نام کاربری و رمز عبور را وارد کنید."

    if clean_user == DEFAULT_USERNAME and clean_pass == DEFAULT_PASSWORD:
        return True, "ورود با موفقیت انجام شد."

    return False, "نام کاربری یا رمز عبور نادرست است."
